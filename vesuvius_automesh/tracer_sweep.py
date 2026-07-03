"""Seed-sweep orchestrator for vc_grow_seg_from_seed coverage growth.

Loop: sample a seed voxel from "good" cubes (supported preds, off-swirl,
outside existing traced surfaces via KD-tree), launch an explicit-seed trace
into a shared patch dir, account NET-new area (fraction of new patch points
far from previously traced points), repeat until the net-area target or the
attempt budget is reached.

Phantom-zone protection: seeds are drawn only from preds & (CT>5) voxels in
swirl-guarded good cubes (the tracer's own random_seed mode would happily
trace the SAM2 phantom halo).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from .config import AUTOMESH_DATA, CUBE, SCROLLS
from .export_cubes import open_ct_aligned, open_preds_l0
from .select_regions import load_good_mask

# villa volume-cartographer binary; either on PATH or via VC_GROW_SEG_BIN.
VC_BIN = Path(os.environ.get("VC_GROW_SEG_BIN", "vc_grow_seg_from_seed"))
PATCH_PREFIX = "auto_grown"


def load_patch_points(patch_dir: Path, stride: int = 2) -> np.ndarray:
    """(N,3) zyx preds-voxel points of a tifxyz patch (strided)."""
    try:
        x = tifffile.imread(patch_dir / "x.tif")[::stride, ::stride]
        y = tifffile.imread(patch_dir / "y.tif")[::stride, ::stride]
        z = tifffile.imread(patch_dir / "z.tif")[::stride, ::stride]
    except (OSError, ValueError):
        return np.empty((0, 3))
    pts = np.stack([z, y, x], axis=-1).reshape(-1, 3).astype(np.float64)
    ok = np.isfinite(pts).all(axis=1) & (pts > -0.5).all(axis=1)
    return pts[ok]


def existing_patch_dirs(tgt_dir: Path) -> list[Path]:
    return sorted(
        d for d in tgt_dir.iterdir()
        if d.is_dir() and d.name.startswith(PATCH_PREFIX)
        and (d / "x.tif").exists()
    ) if tgt_dir.exists() else []


class SeedSampler:
    """Random supported-preds voxels from good cubes."""

    def __init__(self, scroll: str, rng: random.Random):
        self.paths = SCROLLS[scroll]
        self.preds = open_preds_l0(self.paths)
        self.ct = open_ct_aligned(self.paths) if self.paths.ct_support else None
        good, occ = load_good_mask()
        idx = np.argwhere(good & (occ > 0.03))
        if len(idx) == 0:
            raise RuntimeError("no good cubes to seed from")
        self.cubes = idx
        self.rng = rng

    def sample(self) -> tuple[int, int, int] | None:
        """One candidate seed as (x, y, z) preds-voxel ints, or None."""
        cz, cy, cx = self.cubes[self.rng.randrange(len(self.cubes))]
        z0, y0, x0 = int(cz) * CUBE, int(cy) * CUBE, int(cx) * CUBE
        p = np.asarray(self.preds[z0 : z0 + CUBE, y0 : y0 + CUBE, x0 : x0 + CUBE])
        m = p > 0
        if self.ct is not None:
            c = np.asarray(self.ct[z0 : z0 + CUBE, y0 : y0 + CUBE, x0 : x0 + CUBE])
            m &= c > 5
        pts = np.argwhere(m)
        if len(pts) == 0:
            return None
        dz, dy, dx = pts[self.rng.randrange(len(pts))]
        return (x0 + int(dx), y0 + int(dy), z0 + int(dz))


def run_sweep(
    scroll: str,
    tgt_dir: Path,
    params_path: Path,
    *,
    target_net_cm2: float,
    max_attempts: int = 400,
    min_seed_dist_vox: float = 25.0,
    novelty_dist_vox: float = 20.0,
    min_patch_cm2: float = 0.3,
    seed: int = 1234,
    log_path: Path | None = None,
    batch_size: int = 12,
    batch_seed_sep_vox: float = 400.0,
    trace_timeout_s: int = 3600,
    sampler_factory=None,
    vol_dir: Path | None = None,
) -> dict:
    """Batched: K concurrent single-core traces per round (one vc_grow uses
    ~1 core; the host has ~32)."""
    rng = random.Random(seed)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    sampler = sampler_factory(rng) if sampler_factory else SeedSampler(scroll, rng)
    if vol_dir is None:
        vol_dir = AUTOMESH_DATA / scroll / "preds_supported.zarr"

    ledger: list[dict] = []
    all_pts: list[np.ndarray] = []
    for d in existing_patch_dirs(tgt_dir):
        pts = load_patch_points(d)
        if len(pts):
            all_pts.append(pts)
    tree = cKDTree(np.vstack(all_pts)) if all_pts else None
    net_total = 0.0
    attempts = 0
    t0 = time.time()

    while net_total < target_net_cm2 and attempts < max_attempts:
        # ---- assemble a batch of well-separated fresh seeds
        batch: list[tuple[int, int, int]] = []
        tries = 0
        while len(batch) < batch_size and tries < 200:
            tries += 1
            cand = sampler.sample()
            if cand is None:
                continue
            czyx = np.array([[cand[2], cand[1], cand[0]]])
            if tree is not None:
                d, _ = tree.query(czyx, k=1)
                if d[0] < min_seed_dist_vox:
                    continue
            if any(np.linalg.norm(czyx[0] - np.array([b[2], b[1], b[0]])) <
                   batch_seed_sep_vox for b in batch):
                continue
            batch.append(cand)
        if not batch:
            print("[sweep] no fresh seeds found; stopping")
            break
        attempts += len(batch)

        before = set(p.name for p in existing_patch_dirs(tgt_dir))
        log = open(log_path, "a") if log_path else subprocess.DEVNULL
        procs = []
        for cand in batch:
            cmd = [str(VC_BIN), "-v", str(vol_dir), "-t", str(tgt_dir),
                   "-p", str(params_path),
                   "-s", str(cand[0]), str(cand[1]), str(cand[2])]
            procs.append(subprocess.Popen(cmd, stdout=log,
                                          stderr=subprocess.STDOUT))
        deadline = time.time() + trace_timeout_s
        for pr in procs:
            try:
                pr.wait(timeout=max(10, deadline - time.time()))
            except subprocess.TimeoutExpired:
                pr.kill()
                print("[sweep] trace timed out; killed")
        if log_path:
            log.close()

        new_dirs = [d for d in existing_patch_dirs(tgt_dir)
                    if d.name not in before]
        if not new_dirs:
            print(f"[sweep] batch of {len(batch)}: no new patches")
            continue
        for nd in sorted(new_dirs):
            try:
                meta = json.loads((nd / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            area = float(meta.get("area_cm2", 0.0))
            pts = load_patch_points(nd)
            if area < min_patch_cm2 or len(pts) == 0:
                print(f"[sweep] {nd.name}: tiny ({area:.2f} cm2) — not counted")
                continue
            if tree is not None and len(pts):
                d, _ = tree.query(pts, k=1)
                novel = float((d > novelty_dist_vox).mean())
            else:
                novel = 1.0
            net = area * novel
            net_total += net
            all_pts.append(pts)
            tree = cKDTree(np.vstack(all_pts))
            row = {"patch": nd.name, "area_cm2": round(area, 3),
                   "novel_fraction": round(novel, 3), "net_cm2": round(net, 3)}
            ledger.append(row)
            print(f"[sweep] {nd.name}: {area:.2f} cm2, novel {novel:.0%} -> "
                  f"net {net:.2f} | cumulative {net_total:.1f}/{target_net_cm2} cm2 "
                  f"({attempts} attempts, {(time.time()-t0)/60:.0f} min)")
        (tgt_dir / "sweep_ledger.json").write_text(json.dumps(
            {"partial": True, "net_total_cm2": round(net_total, 2),
             "patches": ledger}, indent=1))

    out = {
        "scroll": scroll,
        "tgt_dir": str(tgt_dir),
        "target_net_cm2": target_net_cm2,
        "net_total_cm2": round(net_total, 2),
        "attempts": attempts,
        "n_patches": len(ledger),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "patches": ledger,
    }
    (tgt_dir / "sweep_ledger.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--tgt-dir", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--target-net-cm2", type=float, required=True)
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    out = run_sweep(
        args.scroll, args.tgt_dir, args.params,
        target_net_cm2=args.target_net_cm2,
        max_attempts=args.max_attempts,
        seed=args.seed,
        log_path=args.log,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "patches"}, indent=1))


if __name__ == "__main__":
    main()
