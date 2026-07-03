"""Select cube-grid regions for meshing from the survey cube maps.

Inputs (survey/ in this repo, produced by the 2026-06-11 preds survey):
- cube_maps.npz: npos/nct/njoint/ones arrays, shape (66,31,31) int32 — counts
  per 128-cube out of <=32^3 coarse samples (preds-positive, CT-supported,
  joint, total in-volume samples).
- existing_cubes.npy: (1853,3) int zyx cube indices covered by the two traced
  segments (67 cm² baseline) — excluded so new area complements them.

Strategy: greedy box packing. Score each candidate interior box by the count
of "good" cubes (supported occupancy in [min_occ, max_occ], support >= 0.5,
not existing, cube-z in [8, 58]); emit boxes best-first with no overlap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parents[1] / "survey"


def load_good_mask(
    *,
    min_occ: float = 0.01,
    max_occ: float = 0.50,
    min_support: float = 0.5,
    z_lo: int = 8,
    z_hi: int = 59,
    exclude_existing: bool = True,
    min_radius_cubes: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (good (66,31,31) bool, occ (66,31,31) float supported-occupancy).

    min_radius_cubes excludes the central swirl: cubes whose (y,x) center is
    within that many cube-widths of the per-slice occupancy centroid (the
    survey rejected swirl cubes as merge-prone; outer wraps first).
    """
    d = np.load(SURVEY / "cube_maps.npz")
    npos, njoint, ones = d["npos"], d["njoint"], d["ones"]
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = np.where(ones > 0, njoint / np.maximum(ones, 1), 0.0)
        support = np.where(npos > 0, njoint / np.maximum(npos, 1), 0.0)
    good = (occ >= min_occ) & (occ <= max_occ) & (support >= min_support)
    good[:z_lo] = False
    good[z_hi:] = False
    if min_radius_cubes > 0:
        w = njoint.sum(axis=0).astype(np.float64)  # (y,x) occupancy mass
        yy, xx = np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]),
                             indexing="ij")
        cy = float((w * yy).sum() / max(w.sum(), 1))
        cx = float((w * xx).sum() / max(w.sum(), 1))
        r = np.sqrt((yy + 0.5 - cy - 0.5) ** 2 + (xx + 0.5 - cx - 0.5) ** 2)
        good &= (r >= min_radius_cubes)[None, :, :]
    if exclude_existing:
        ex = np.load(SURVEY / "existing_cubes.npy")
        good[ex[:, 0], ex[:, 1], ex[:, 2]] = False
    return good, occ


def greedy_boxes(
    good: np.ndarray,
    *,
    box: tuple[int, int, int] = (8, 8, 8),
    n_boxes: int = 12,
    min_good: int = 60,
) -> list[dict]:
    """Greedily place non-overlapping boxes maximizing good-cube count."""
    g = good.astype(np.int32)
    bz, by, bx = box
    # integral image for O(1) box sums
    ii = np.zeros((g.shape[0] + 1, g.shape[1] + 1, g.shape[2] + 1), dtype=np.int64)
    ii[1:, 1:, 1:] = g.cumsum(0).cumsum(1).cumsum(2)

    def box_sum(z, y, x):
        return int(
            ii[z + bz, y + by, x + bx] - ii[z, y + by, x + bx]
            - ii[z + bz, y, x + bx] - ii[z + bz, y + by, x]
            + ii[z, y, x + bx] + ii[z, y + by, x] + ii[z + bz, y, x]
            - ii[z, y, x]
        )

    taken = np.zeros_like(good, dtype=bool)
    out: list[dict] = []
    for _ in range(n_boxes):
        best, best_pos = -1, None
        for z in range(good.shape[0] - bz + 1):
            for y in range(good.shape[1] - by + 1):
                for x in range(good.shape[2] - bx + 1):
                    if taken[z : z + bz, y : y + by, x : x + bx].any():
                        continue
                    s = box_sum(z, y, x)
                    if s > best:
                        best, best_pos = s, (z, y, x)
        if best_pos is None or best < min_good:
            break
        z, y, x = best_pos
        taken[z : z + bz, y : y + by, x : x + bx] = True
        out.append(
            {
                "cube_lo": [z, y, x],
                "cube_hi": [z + bz, y + by, x + bx],
                "n_good_cubes": best,
            }
        )
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box", nargs=3, type=int, default=[8, 8, 8])
    ap.add_argument("--n-boxes", type=int, default=12)
    ap.add_argument("--min-good", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    good, occ = load_good_mask()
    boxes = greedy_boxes(good, box=tuple(args.box), n_boxes=args.n_boxes,
                         min_good=args.min_good)
    for i, b in enumerate(boxes):
        lo, hi = b["cube_lo"], b["cube_hi"]
        mean_occ = float(occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].mean())
        b["region"] = f"r{i:02d}_z{lo[0]}y{lo[1]}x{lo[2]}"
        b["mean_supported_occ"] = round(mean_occ, 4)
    payload = {
        "total_good_cubes": int(good.sum()),
        "boxes": boxes,
    }
    text = json.dumps(payload, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
