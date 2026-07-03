"""Export 128^3 prediction cubes from a preds zarr into scrollfiesta's grid format.

Output: <grid_dir>/cubes_PRED/z#####_y#####_x#####.tif  (multipage uint8,
128 pages of 128x128, strip-encoded, 0=background 255=recto surface), plus
<grid_dir>/export_manifest.json.

Phantom filtering (scroll3 m7 preds): the organizers' preds contain solid
false-positive slabs where the masked CT is exactly 0 (LOG 2026-06-10; survey
2026-06-11: ~70% of positive voxels are phantom). We therefore mask
preds &= (CT > ct_threshold) voxel-wise and skip cubes whose positive voxels
have CT-support fraction < min_support.

Scrollfiesta contract (verified against its source, see TRACK_LOG 2026-06-11):
- filenames encode the cube's absolute voxel origin on a 128-multiple lattice;
- neighbours are probed at origin +/- 128 per axis; missing neighbours are
  zero-filled, so callers should export a 1-cube sacrificial rim around any
  region of interest;
- TIFFs must be 8-bit single-sample strip-encoded; 16-bit input causes a heap
  overrun in TiffIO_load (it never checks BITSPERSAMPLE).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage

from .config import CUBE, SCROLLS, ScrollPaths

NECK_HALO = 8  # slab halo voxels so morphology is correct at cube borders


@dataclass(frozen=True)
class ExportStats:
    n_written: int
    n_skipped_empty: int
    n_skipped_support: int
    pos_voxels_written: int


def open_preds_l0(paths: ScrollPaths) -> zarr.Array:
    """Open the preds zarr's level-0 array (grid == CT level ct_align_level)."""
    grp = zarr.open_group(str(paths.preds_zarr), mode="r")
    return grp["0"]


def open_ct_aligned(paths: ScrollPaths) -> zarr.Array:
    """Open the CT level whose grid matches preds L0."""
    grp = zarr.open_group(str(paths.ct_zarr), mode="r")
    return grp[str(paths.ct_align_level)]


def cube_origin_name(z0: int, y0: int, x0: int) -> str:
    return f"z{z0:05d}_y{y0:05d}_x{x0:05d}"


def read_block_padded(arr: zarr.Array, z0: int, y0: int, x0: int, size: int) -> np.ndarray:
    """Read a size^3 block at (z0,y0,x0), zero-padding outside the array."""
    out = np.zeros((size, size, size), dtype=arr.dtype)
    z1, y1, x1 = z0 + size, y0 + size, x0 + size
    cz0, cy0, cx0 = max(z0, 0), max(y0, 0), max(x0, 0)
    cz1 = min(z1, arr.shape[0])
    cy1 = min(y1, arr.shape[1])
    cx1 = min(x1, arr.shape[2])
    if cz1 <= cz0 or cy1 <= cy0 or cx1 <= cx0:
        return out
    block = np.asarray(arr[cz0:cz1, cy0:cy1, cx0:cx1])
    out[cz0 - z0 : cz1 - z0, cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = block
    return out


def neck_cut(mask: np.ndarray, *, erode_iters: int = 1) -> tuple[np.ndarray, int]:
    """Break zero-gap fusions between touching wraps.

    The m7 preds fuse compressed wraps with NO voxel gap at contact points;
    scrollfiesta's sheet-split oracle detects multi-sheet components by gaps
    along rays, so zero-gap fusions pass straight through to the weld and
    percolate into giant multi-wrap blobs (pilot1: 84% of faces in one
    component). Fix at the mask level: erode (necks <= ~2*iters vox vanish,
    sheet cores ~4 vox survive) -> 26-conn label -> assign every mask voxel
    to its nearest surviving core (EDT watershed-back) -> delete voxels on
    inter-label interfaces (both sides -> ~2-voxel gap, wider than the BPA
    ball so meshing cannot re-bridge).

    Returns (cut mask, n_voxels_removed).
    """
    if not mask.any():
        return mask, 0
    eroded = ndimage.binary_erosion(mask, iterations=erode_iters)
    if not eroded.any():
        return mask, 0  # too thin to erode: leave untouched
    labels, n_labels = ndimage.label(eroded, structure=np.ones((3, 3, 3), bool))
    if n_labels <= 1:
        return mask, 0
    idx = ndimage.distance_transform_edt(
        ~eroded, return_distances=False, return_indices=True
    )
    w = labels[tuple(idx)]
    w[~mask] = 0
    interface = np.zeros_like(mask)
    for axis in range(3):
        a = np.diff(w, axis=axis) != 0
        both = (
            np.take(w, range(0, w.shape[axis] - 1), axis=axis) > 0
        ) & (np.take(w, range(1, w.shape[axis]), axis=axis) > 0)
        hit = a & both
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, w.shape[axis] - 1)
        sl_hi[axis] = slice(1, w.shape[axis])
        interface[tuple(sl_lo)] |= hit
        interface[tuple(sl_hi)] |= hit
    out = mask & ~interface
    return out, int(mask.sum() - out.sum())


def write_cube_tiff(path: Path, mask: np.ndarray) -> None:
    """Write a 128^3 uint8 mask as a strip-encoded multipage TIFF (0/255)."""
    if mask.shape != (CUBE, CUBE, CUBE) or mask.dtype != np.uint8:
        raise ValueError(f"bad cube array {mask.shape} {mask.dtype}")
    tifffile.imwrite(
        path,
        mask,
        photometric="minisblack",
        compression="adobe_deflate",
        rowsperstrip=CUBE,
    )


def export_region(
    scroll: str,
    cube_lo: tuple[int, int, int],
    cube_hi: tuple[int, int, int],
    grid_dir: Path,
    *,
    rim: int = 1,
    ct_threshold: int = 5,
    min_support: float = 0.5,
    min_pos_voxels: int = 200,
    cut_necks: bool = True,
    erode_iters: int = 1,
) -> ExportStats:
    """Export cubes for cube-index box [cube_lo, cube_hi) plus a rim.

    cube_lo/cube_hi are (z,y,x) cube indices (origin = index * 128). The rim
    cubes give interior cubes their full 13-voxel halo; they are exported like
    any other cube and flagged "rim" in the manifest so downstream accounting
    can separate them.
    """
    paths = SCROLLS[scroll]
    preds = open_preds_l0(paths)
    ct = open_ct_aligned(paths) if paths.ct_support else None
    if ct is not None and tuple(ct.shape) != tuple(preds.shape):
        raise ValueError(f"CT/preds grid mismatch: {ct.shape} vs {preds.shape}")

    cubes_dir = grid_dir / "cubes_PRED"
    cubes_dir.mkdir(parents=True, exist_ok=True)

    lo = np.array(cube_lo) - rim
    hi = np.array(cube_hi) + rim
    lo = np.maximum(lo, 0)
    n_grid = -(-np.array(preds.shape) // CUBE)  # ceil-div: total cube grid
    hi = np.minimum(hi, n_grid)

    manifest: dict = {
        "scroll": scroll,
        "preds_zarr": str(paths.preds_zarr),
        "ct_zarr": str(paths.ct_zarr) if ct is not None else None,
        "ct_threshold": ct_threshold if ct is not None else None,
        "min_support": min_support if ct is not None else None,
        "min_pos_voxels": min_pos_voxels,
        "cube_lo_interior": list(map(int, cube_lo)),
        "cube_hi_interior": list(map(int, cube_hi)),
        "rim": rim,
        "cube_lo_export": lo.tolist(),
        "cube_hi_export": hi.tolist(),
        "cut_necks": cut_necks,
        "erode_iters": erode_iters if cut_necks else None,
        "created_unix": time.time(),
        "cubes": [],
    }

    n_written = n_empty = n_support = 0
    pos_total = 0
    cut_total = 0
    halo = NECK_HALO if cut_necks else 0
    size = CUBE + 2 * halo
    for cz in range(lo[0], hi[0]):
        for cy in range(lo[1], hi[1]):
            for cx in range(lo[2], hi[2]):
                z0, y0, x0 = cz * CUBE, cy * CUBE, cx * CUBE
                p = read_block_padded(preds, z0 - halo, y0 - halo, x0 - halo, size)
                pos = p > 0
                core = (slice(halo, halo + CUBE),) * 3
                npos = int(pos[core].sum())
                if npos < min_pos_voxels:
                    n_empty += 1
                    continue
                support = None
                if ct is not None:
                    c = read_block_padded(ct, z0 - halo, y0 - halo, x0 - halo, size)
                    supported = pos & (c > ct_threshold)
                    nsup = int(supported[core].sum())
                    support = nsup / npos
                    if support < min_support:
                        n_support += 1
                        manifest["cubes"].append(
                            {
                                "id": cube_origin_name(z0, y0, x0),
                                "cube_idx": [cz, cy, cx],
                                "status": "skipped_phantom",
                                "pos_raw": npos,
                                "support": round(support, 4),
                            }
                        )
                        continue
                    pos = supported
                    npos = nsup
                    if npos < min_pos_voxels:
                        n_empty += 1
                        continue
                if cut_necks:
                    pos, _ = neck_cut(pos, erode_iters=erode_iters)
                n_before_core = npos
                pos = pos[core]
                npos = int(pos.sum())
                n_cut = n_before_core - npos
                cut_total += n_cut
                if npos < min_pos_voxels:
                    n_empty += 1
                    continue
                mask = np.where(pos, np.uint8(255), np.uint8(0))
                write_cube_tiff(cubes_dir / f"{cube_origin_name(z0, y0, x0)}.tif", mask)
                interior = all(
                    cube_lo[i] <= (cz, cy, cx)[i] < cube_hi[i] for i in range(3)
                )
                manifest["cubes"].append(
                    {
                        "id": cube_origin_name(z0, y0, x0),
                        "cube_idx": [cz, cy, cx],
                        "status": "written",
                        "role": "interior" if interior else "rim",
                        "pos_voxels": npos,
                        "neck_cut_voxels": n_cut if cut_necks else None,
                        "support": round(support, 4) if support is not None else None,
                    }
                )
                n_written += 1
                pos_total += npos

    stats = ExportStats(n_written, n_empty, n_support, pos_total)
    manifest["stats"] = {
        "written": n_written,
        "skipped_empty": n_empty,
        "skipped_phantom": n_support,
        "pos_voxels_written": pos_total,
        "neck_cut_voxels": cut_total if cut_necks else None,
    }
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=1))
    return stats


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--cube-lo", nargs=3, type=int, required=True, metavar=("CZ", "CY", "CX"))
    ap.add_argument("--cube-hi", nargs=3, type=int, required=True, metavar=("CZ", "CY", "CX"))
    ap.add_argument("--grid-dir", type=Path, required=True)
    ap.add_argument("--rim", type=int, default=1)
    ap.add_argument("--ct-threshold", type=int, default=5)
    ap.add_argument("--min-support", type=float, default=0.5)
    ap.add_argument("--no-neck-cut", action="store_true")
    ap.add_argument("--erode-iters", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    stats = export_region(
        args.scroll,
        tuple(args.cube_lo),
        tuple(args.cube_hi),
        args.grid_dir,
        rim=args.rim,
        ct_threshold=args.ct_threshold,
        min_support=args.min_support,
        cut_necks=not args.no_neck_cut,
        erode_iters=args.erode_iters,
    )
    print(
        f"written={stats.n_written} empty={stats.n_skipped_empty} "
        f"phantom={stats.n_skipped_support} pos_voxels={stats.pos_voxels_written} "
        f"elapsed={time.time() - t0:.1f}s -> {args.grid_dir}"
    )


if __name__ == "__main__":
    main()
