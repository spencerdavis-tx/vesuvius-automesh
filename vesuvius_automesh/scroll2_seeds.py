"""Scroll 2 seed sweep: survey-recipe sampler + run_sweep wrapper.

Seeding recipe: seeds = supported-preds positives, sampled proportionally to local
density (a per-192^3-chunk positive-count map, array `chunk_pos` in an
npz you compute over your preds volume), stratified in z, restricted to a radial band (mid/outer
first — compressed cores are fusion-prone), silent zones skipped implicitly
(their chunk density ~ 0). Default window: z chunks 13..36 (z_L0 ~2500-7000),
radial fraction 0.40-1.05 of the per-slab 95th-percentile mass radius.

The scroll3 SeedSampler (good-cube maps) stays untouched; this module plugs a
Scroll 2 sampler into tracer_sweep.run_sweep via sampler_factory.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import zarr

from .config import DATA_ROOT
from .tracer_sweep import run_sweep

CACHE2 = DATA_ROOT / "cache" / "scroll2"
DEFAULT_VOL = CACHE2 / "preds_supported.zarr"
DEFAULT_CHUNK_POS = CACHE2 / "l0_stream_stats.npz"


def weighted_quantile(vals: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(vals)
    cw = np.cumsum(weights[order])
    if cw[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cw[-1], cw, vals[order]))


class Scroll2Sampler:
    """Density-weighted chunk sampler over the supported preds volume."""

    def __init__(
        self,
        rng: random.Random,
        *,
        vol_dir: Path = DEFAULT_VOL,
        chunk_pos_npz: Path = DEFAULT_CHUNK_POS,
        z_chunk_range: tuple[int, int] = (13, 36),
        radial_range: tuple[float, float] = (0.40, 1.05),
    ):
        self.rng = rng
        self.arr = zarr.open_group(str(vol_dir), mode="r")["0"]
        self.chunk = int(self.arr.chunks[0])
        counts = np.load(chunk_pos_npz)["chunk_pos"].astype(np.float64)
        zlo, zhi = z_chunk_range
        if not (0 <= zlo <= zhi < counts.shape[0]):
            raise ValueError(f"bad z_chunk_range {z_chunk_range}")

        yy, xx = np.meshgrid(np.arange(counts.shape[1]),
                             np.arange(counts.shape[2]), indexing="ij")
        elig = np.zeros(counts.shape, dtype=bool)
        for z in range(zlo, zhi + 1):
            w = counts[z]
            tot = w.sum()
            if tot <= 0:
                continue
            cy = float((w * yy).sum() / tot)
            cx = float((w * xx).sum() / tot)
            r = np.hypot(yy - cy, xx - cx)
            pos = w > 0
            r95 = weighted_quantile(r[pos], w[pos], 0.95)
            if not np.isfinite(r95) or r95 <= 0:
                continue
            frac = r / r95
            elig[z] = pos & (frac >= radial_range[0]) & (frac <= radial_range[1])

        w = np.where(elig, counts, 0.0).ravel()
        if w.sum() <= 0:
            raise RuntimeError("no eligible chunks for scroll2 seeding")
        self.flat_idx = np.flatnonzero(w > 0)
        p = w[self.flat_idx]
        self.probs = p / p.sum()
        self.grid_shape = counts.shape
        self.stats = {
            "eligible_chunks": int(len(self.flat_idx)),
            "eligible_pos_voxels": int(w.sum()),
            "z_chunk_range": list(z_chunk_range),
            "radial_range": list(radial_range),
        }

    def sample(self) -> tuple[int, int, int] | None:
        """One candidate seed as (x, y, z) preds-voxel ints, or None."""
        flat = int(np.random.default_rng(self.rng.randrange(2**32)).choice(
            self.flat_idx, p=self.probs))
        iz, iy, ix = np.unravel_index(flat, self.grid_shape)
        c = self.chunk
        z0, y0, x0 = int(iz) * c, int(iy) * c, int(ix) * c
        blk = np.asarray(self.arr[z0 : z0 + c, y0 : y0 + c, x0 : x0 + c])
        pts = np.argwhere(blk > 0)
        if len(pts) == 0:  # chunk emptied by the coarse mask; skip
            return None
        dz, dy, dx = pts[self.rng.randrange(len(pts))]
        return (x0 + int(dx), y0 + int(dy), z0 + int(dz))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tgt-dir", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--target-net-cm2", type=float, required=True)
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--vol", type=Path, default=DEFAULT_VOL)
    ap.add_argument("--chunk-pos", type=Path, default=DEFAULT_CHUNK_POS)
    ap.add_argument("--z-chunk-lo", type=int, default=13)
    ap.add_argument("--z-chunk-hi", type=int, default=36)
    ap.add_argument("--radial-lo", type=float, default=0.40)
    ap.add_argument("--radial-hi", type=float, default=1.05)
    ap.add_argument("--batch-size", type=int, default=12)
    args = ap.parse_args()

    holder: dict = {}

    def factory(rng: random.Random) -> Scroll2Sampler:
        s = Scroll2Sampler(
            rng, vol_dir=args.vol, chunk_pos_npz=args.chunk_pos,
            z_chunk_range=(args.z_chunk_lo, args.z_chunk_hi),
            radial_range=(args.radial_lo, args.radial_hi),
        )
        holder["stats"] = s.stats
        print(f"[scroll2-seeds] sampler: {json.dumps(s.stats)}", flush=True)
        return s

    out = run_sweep(
        "scroll2", args.tgt_dir, args.params,
        target_net_cm2=args.target_net_cm2,
        max_attempts=args.max_attempts,
        seed=args.seed,
        log_path=args.log,
        batch_size=args.batch_size,
        sampler_factory=factory,
        vol_dir=args.vol,
    )
    out["sampler_stats"] = holder.get("stats")
    (args.tgt_dir / "sweep_ledger.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "patches"}, indent=1))


if __name__ == "__main__":
    main()
