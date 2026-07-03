"""Render a VC tifxyz patch (vc_grow_seg_from_seed output) via first_letters.

A tifxyz dir holds x.tif / y.tif / z.tif float coordinate bands on a quad
grid (volume voxel coords of the traced volume = our preds grid), meta.json
with {"scale": [sx, sy]} mapping grid steps to nominal voxels, and an
optional mask.tif. The coordinate maps ARE the renderer's pos-grid — no
flattening or rasterization needed: upsample the coordinate field to the
render pitch, derive normals from grid derivatives, then reuse the proven
recenter -> 66-layer render -> QC chain (first_letters modules vendored
under _vendor/first_letters).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from .config import LEDGER_PATH, QC_REFERENCE_LAYER, SCROLLS
from ._vendor.first_letters import mesh_render, qc, render, zarrio
from .render_driver import preds_to_l0


def load_tifxyz(patch_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (pos_grid (H,W,3) float64 zyx preds-voxels, valid (H,W), meta)."""
    x = tifffile.imread(patch_dir / "x.tif").astype(np.float64)
    y = tifffile.imread(patch_dir / "y.tif").astype(np.float64)
    z = tifffile.imread(patch_dir / "z.tif").astype(np.float64)
    meta = json.loads((patch_dir / "meta.json").read_text())
    pos = np.stack([z, y, x], axis=-1)
    valid = np.isfinite(pos).all(axis=-1) & (pos > -0.5).all(axis=-1)
    mask_path = patch_dir / "mask.tif"
    if mask_path.exists():
        m = tifffile.imread(mask_path)
        valid &= m > 0
    return pos, valid, meta


def upsample_grid(
    pos: np.ndarray, valid: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-upsample the coordinate field by `factor` per axis."""
    h, w = valid.shape
    hh = max(2, int(round((h - 1) * factor)) + 1)
    ww = max(2, int(round((w - 1) * factor)) + 1)
    rr, cc = np.meshgrid(
        np.linspace(0.0, h - 1.0, hh), np.linspace(0.0, w - 1.0, ww), indexing="ij"
    )
    out = np.empty((hh, ww, 3), dtype=np.float64)
    fill = np.where(valid[..., None], pos, 0.0)
    for k in range(3):
        out[..., k] = map_coordinates(fill[..., k], [rr, cc], order=1, mode="nearest")
    vmap = map_coordinates(valid.astype(np.float32), [rr, cc], order=1, mode="nearest")
    return out, vmap > 0.999  # only pixels fully inside valid quads


def grid_normals(pos_um: np.ndarray) -> np.ndarray:
    """Unit normals of a (H,W,3) coordinate grid via central differences."""
    du = np.gradient(pos_um, axis=0)
    dv = np.gradient(pos_um, axis=1)
    n = np.cross(du.reshape(-1, 3), dv.reshape(-1, 3)).reshape(pos_um.shape)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    bad = norm[..., 0] < 1e-12
    n[bad] = (0.0, 0.0, 1.0)
    norm[bad] = 1.0
    return n / norm


def prefetch_level_chunks(
    ct, cache, pos_um: np.ndarray, valid: np.ndarray, level: int,
    *, margin_um: float = 400.0, threads: int = 48,
) -> int:
    """Warm the ChunkCache with all `level` chunks under a window's bbox.

    Volume.sample fetches chunk misses serially (latency-bound over S3, ~2%
    CPU workers). Fetching the window's chunk set with a thread pool first,
    then seeding cache.get with the prefetched block, is ~20x faster and
    needs no first_letters changes.
    """
    from concurrent.futures import ThreadPoolExecutor

    pts = pos_um[valid]
    if len(pts) == 0:
        return 0
    info = ct.level_info(level)
    arr = ct._levels[level]  # read-only use of the public-enough mapping
    cz, cy, cx = info.chunks
    # Only chunks the SURFACE touches (its AABB is mostly empty space for a
    # curved sheet): chunk ids of strided surface points, dilated by 1 chunk
    # to cover the 66-layer band + recenter shifts.
    sub = pts[:: max(1, len(pts) // 200_000)]
    cv = ct.to_level(sub, level)
    ids = np.unique((cv // (cz, cy, cx)).astype(np.int64), axis=0)
    n_grid = np.maximum((np.array(info.shape) + (cz, cy, cx) - 1)
                        // (cz, cy, cx), 1)
    touched = set()
    for base in ids:
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    t = (int(base[0]) + dz, int(base[1]) + dy, int(base[2]) + dx)
                    if all(0 <= t[k] < n_grid[k] for k in range(3)):
                        touched.add(t)
    keys = []
    for iz, iy, ix in sorted(touched):
        key = (ct.name, level, iz, iy, ix)
        if key not in cache._store:
            keys.append((key, iz, iy, ix))

    def fetch_one(item):
        key, iz, iy, ix = item
        c0 = np.array([iz * cz, iy * cy, ix * cx])
        c1 = np.minimum(c0 + (cz, cy, cx), info.shape)
        blk = np.asarray(arr[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]])
        return key, blk

    n = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for key, blk in ex.map(fetch_one, keys):
            cache.get(key, lambda b=blk: b)
            n += 1
    return n


def render_patch_dir(
    patch_dir: Path,
    scroll: str,
    out_dir: Path,
    *,
    seg_id: str | None = None,
    level: int = 1,
    recenter_level: int = 2,
    pitch_um: float | None = None,
    ct=None,
    cache=None,
    max_px: int = 16000,
) -> dict:
    paths = SCROLLS[scroll]
    if ct is None:
        ledger = zarrio.load_ledger(LEDGER_PATH)
        ct = zarrio.open_volume(paths.ledger_ct_key, ledger=ledger,
                                local_root=paths.ct_zarr)
    if cache is None:
        cache = zarrio.ChunkCache(max_bytes=30 * 2**30)
    pitch = pitch_um or ct.level_voxel_um(level)
    seg_id = seg_id or patch_dir.name

    pos_preds, valid, meta = load_tifxyz(patch_dir)
    if not valid.any():
        raise ValueError(f"{seg_id}: empty tifxyz mask")

    # Grid step measured from the coordinate maps themselves (meta "scale"
    # is grid-units-per-voxel, e.g. 0.05 for step_size 20 — but measuring is
    # robust to convention drift): median 3D distance between valid
    # neighbouring grid cells, in preds voxels.
    steps = []
    for axis in (0, 1):
        d = np.diff(pos_preds, axis=axis)
        ok = valid & np.roll(valid, -1, axis=axis)
        ok = ok.take(range(d.shape[axis]), axis=axis)
        if ok.any():
            steps.append(np.median(np.linalg.norm(d[ok], axis=-1)))
    if not steps:
        raise ValueError(f"{seg_id}: no valid grid adjacency")
    step_vox = float(np.median(steps))
    if not (1.0 <= step_vox <= 200.0):
        raise ValueError(f"{seg_id}: implausible grid step {step_vox} vox")

    preds_voxel_um = ct.voxel_um * (2 ** paths.ct_align_level)
    factor = step_vox * preds_voxel_um / pitch

    # Big traces are rendered as fixed-size windows of the quad grid (the
    # production pattern: 20x25mm crops), each its own villa-layout patch.
    win_px = min(max_px, 5200)  # ~25mm at 4.8um — production crop scale
    win_grid = max(8, int(win_px / factor))  # grid cells per window side
    h_g, w_g = valid.shape
    base_meta = {
        "scroll": scroll,
        "source": "auto-mesh vc_grow_seg_from_seed tifxyz",
        "patch_dir": str(patch_dir),
        "tifxyz_meta": {k: meta[k] for k in ("scale", "area_cm2", "uuid", "bbox")
                        if k in meta},
        "grid_step_vox": step_vox,
        "preds_zarr": str(paths.preds_zarr),
        "frame": "tifxyz preds-grid -> L0 = 4v+1.5 -> x ct.voxel_um",
        "ct_ledger_id": paths.ledger_ct_key,
        "git_sha": zarrio.git_sha(),
    }
    ref = qc.reference_stats(QC_REFERENCE_LAYER, px_um=2.0)

    windows = []
    for gi in range(0, h_g, win_grid):
        for gj in range(0, w_g, win_grid):
            v_win = valid[gi : gi + win_grid + 1, gj : gj + win_grid + 1]
            if v_win.sum() < 16:  # needs a few quads to be worth rendering
                continue
            windows.append((gi, gj))
    n_skipped = 0
    rows = []
    for gi, gj in windows:
        wid = f"{seg_id}_w{gi:04d}_{gj:04d}" if len(windows) > 1 else seg_id
        pos_w = pos_preds[gi : gi + win_grid + 1, gj : gj + win_grid + 1]
        val_w = valid[gi : gi + win_grid + 1, gj : gj + win_grid + 1]
        pos_up, valid_up = upsample_grid(pos_w, val_w, factor)
        if not valid_up.any():
            n_skipped += 1
            continue
        h, w = valid_up.shape
        pos_um = preds_to_l0(pos_up, paths.ct_align_level) * ct.voxel_um
        nrm = grid_normals(pos_um)

        t_pf = time.time()
        n_pf = prefetch_level_chunks(ct, cache, pos_um, valid_up, level)
        if n_pf:
            print(f"  [prefetch] {n_pf} L{level} chunks in {time.time()-t_pf:.0f}s")
        # Scroll 2 has no local CT cache: the recenter level also streams from
        # S3, and recenter's serial chunk misses would dominate wall clock.
        # Prefetch it too (no-op on scrolls whose recenter level is local).
        if (recenter_level != level and
                getattr(ct, "level_sources", {}).get(recenter_level) == "s3"):
            t_pf = time.time()
            n_pf = prefetch_level_chunks(ct, cache, pos_um, valid_up,
                                         recenter_level)
            if n_pf:
                print(f"  [prefetch] {n_pf} L{recenter_level} chunks "
                      f"in {time.time()-t_pf:.0f}s")

        rec = mesh_render.recenter_grid(
            pos_um.astype(np.float32), nrm.astype(np.float32), valid_up, ct,
            level=recenter_level, cache=cache, pitch_um=pitch,
        )
        pos_render = (pos_um.astype(np.float32)
                      + rec.shift_full[..., None] * nrm.astype(np.float32))
        result = mesh_render.render_grid(
            pos_render, nrm.astype(np.float32), valid_up, ct,
            level=level, pitch_um=pitch, u0_um=0.0, v0_um=0.0, cache=cache,
            tile_px=512, verbose=False,
        )
        seg_dir = render.write_patch(
            out_dir, wid, result,
            {**base_meta, "window_grid_origin": [gi, gj], "recenter": rec.stats},
        )
        qcd = qc.evaluate_patch(
            result.layers, result.mask, shift_um=rec.shift_coarse,
            merge_fraction=rec.merge_fraction, ref=ref, px_um=result.pitch_um,
        )
        qc.write_qc(seg_dir, qcd, result.layers, result.mask)
        mask_px = int(result.mask.sum())
        # Acceptance = kill-gate-1 AND recenter lock-on. A cross-wrap-cutting
        # surface can fool the coherence gate (sheet cross-sections are
        # locally coherent) but cannot fool the band finder: found_fraction
        # 0.54 on the no-direction-field smoke trace vs 0.97-1.00 on real
        # sheets (2026-06-12).
        found = float(rec.stats.get("found_fraction") or 0.0)
        rows.append({
            "seg_id": wid,
            "grid": [h, w],
            "masked_area_cm2": round(mask_px * (pitch / 1e4) ** 2, 4),
            "qc_pass": bool(qcd["pass"]) and found >= 0.9,
            "qc_gates_pass": bool(qcd["pass"]),
            "qc": {k: qcd[k] for k in ("coherence_ratio", "band_contrast",
                                       "merge_fraction", "mask_fraction")
                   if k in qcd},
            "recenter_found": rec.stats.get("found_fraction"),
        })
        print(f"  [{wid}] qc_pass={rows[-1]['qc_pass']} "
              f"masked {rows[-1]['masked_area_cm2']} cm2 {rows[-1]['qc']}")

    pass_area = sum(r["masked_area_cm2"] for r in rows if r["qc_pass"])
    return {
        "seg_id": seg_id,
        "tifxyz_area_cm2": meta.get("area_cm2"),
        "n_windows": len(rows),
        "n_windows_skipped": n_skipped,
        "masked_area_cm2": round(sum(r["masked_area_cm2"] for r in rows), 4),
        "qc_pass_area_cm2": round(pass_area, 4),
        "qc_pass": bool(rows) and all(r["qc_pass"] for r in rows),
        "windows": rows,
        "fetched_gib": round(cache.fetched_bytes / 2**30, 2),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patches", type=Path, required=True,
                    help="tifxyz patch dir, or a dir of patch dirs")
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--recenter-level", type=int, default=2,
                    help="CT level for band recentering (2 = local on scroll3)")
    ap.add_argument("--pitch-um", type=float, default=None)
    args = ap.parse_args()

    if (args.patches / "x.tif").exists():
        dirs = [args.patches]
    else:
        dirs = sorted(d for d in args.patches.iterdir()
                      if (d / "x.tif").exists())
    print(f"{len(dirs)} tifxyz patches")

    paths = SCROLLS[args.scroll]
    ledger = zarrio.load_ledger(LEDGER_PATH)
    ct = zarrio.open_volume(paths.ledger_ct_key, ledger=ledger,
                            local_root=paths.ct_zarr)
    cache = zarrio.ChunkCache(max_bytes=30 * 2**30)

    rows = []
    t0 = time.time()
    for d in dirs:
        try:
            row = render_patch_dir(d, args.scroll, args.out, level=args.level,
                                   recenter_level=args.recenter_level,
                                   pitch_um=args.pitch_um, ct=ct, cache=cache)
        except (ValueError, KeyError, OSError) as e:
            row = {"seg_id": d.name, "error": str(e)}
            print(f"[{d.name}] FAILED: {e}")
        rows.append(row)
        if "qc_pass_area_cm2" in row:
            print(f"[{row['seg_id']}] windows={row['n_windows']} "
                  f"qc-pass {row['qc_pass_area_cm2']} of "
                  f"{row['masked_area_cm2']} cm2")

    total_pass = sum(r.get("qc_pass_area_cm2", 0) for r in rows)
    summary = {
        "patches_root": str(args.patches),
        "pitch_um": args.pitch_um,
        "level": args.level,
        "n_rendered": sum(1 for r in rows if "qc_pass" in r),
        "n_failed": sum(1 for r in rows if "error" in r),
        "qc_pass_area_cm2": round(total_pass, 3),
        "elapsed_s": round(time.time() - t0, 1),
        "patches": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "render_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "patches"},
                     indent=1))


if __name__ == "__main__":
    main()
