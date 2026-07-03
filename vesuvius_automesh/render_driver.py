"""Render a scrollfiesta welded mesh through the first_letters pipeline.

Chain per connected component:
  welded.obj (verts zyx, preds-grid voxels) -> L0 voxel coords (4v + 1.5)
  -> world um (x 2.399) -> SLIM UV -> rasterize -> recenter -> 66-layer render
  -> villa layout + qc.json under results/auto-mesh/<scroll>/<region>/.

The first_letters render/QC modules are vendored under
vesuvius_automesh/_vendor/first_letters (see attribution headers there).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .config import LEDGER_PATH, QC_REFERENCE_LAYER, SCROLLS
from .flatten import FlattenResult
from .mesh_io import Component, load_obj_zyx, mesh_area_voxels, split_components
from ._vendor.first_letters import mesh_render, qc, render, zarrio
MAX_SLIM_VERTS = 1_500_000  # above this, decimate -> flatten -> transfer UVs


def preds_to_l0(verts_zyx: np.ndarray, align_level: int) -> np.ndarray:
    """Preds-grid voxel coords -> CT L0 voxel coords (zarrio convention:
    level-L voxel i covers L0 [i*s,(i+1)*s), center i*s + (s-1)/2)."""
    s = float(2**align_level)
    return verts_zyx * s + (s - 1.0) / 2.0


def flatten_component(comp: Component, timeout_s: int = 3600) -> FlattenResult:
    """SLIM-flatten in a crash-isolated child process.

    igl segfaults on some pathological welded geometry; a child process turns
    that into a catchable failure instead of killing the whole batch.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.npz"
        out_path = Path(td) / "out.npz"
        np.savez(in_path, V=comp.verts_zyx, F=comp.faces,
                 max_slim_verts=MAX_SLIM_VERTS)
        proc = subprocess.run(
            [sys.executable, "-m", "vesuvius_automesh.flatten_worker",
             str(in_path), str(out_path)],
            capture_output=True, text=True, timeout=timeout_s,
            env={**os.environ,
                 "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        )
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            raise ValueError(
                f"flatten worker rc={proc.returncode}: {' | '.join(tail) or 'crashed'}"
            )
        d = np.load(out_path)
        return FlattenResult(
            uv=d["uv"],
            convergence=list(d["convergence"]),
            area_ratio_p5=float(d["area_ratio_p5"]),
            area_ratio_median=float(d["area_ratio_median"]),
            area_ratio_p95=float(d["area_ratio_p95"]),
            method=str(d["method"]),
        )


def render_component(
    comp: Component,
    flat: FlattenResult,
    ct,
    cache,
    *,
    seg_id: str,
    out_dir: Path,
    level: int,
    pitch_um: float,
    base_meta: dict,
    max_px: int = 16000,
) -> dict:
    """Rasterize + recenter + render + QC one component. Returns summary row."""
    align = base_meta["preds_align_ct_level"]
    v_l0 = preds_to_l0(comp.verts_zyx, align)
    pos_um = v_l0 * ct.voxel_um
    nrm = mesh_render.vertex_normals(v_l0, comp.faces)

    scale = mesh_render.uv_scale_um(pos_um, comp.faces, flat.uv, comp.faces)
    uv_um = flat.uv * np.array([scale.s_u_um, scale.s_v_um])
    # Rotate UVs to principal axes (isometric; tightens the render bbox that
    # SLIM's arbitrary rotation would otherwise waste).
    centered = uv_um - uv_um.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    uv_um = centered @ vt.T
    lo = uv_um.min(axis=0)
    hi = uv_um.max(axis=0)
    shape = (
        int((hi[0] - lo[0]) / pitch_um) + 1,
        int((hi[1] - lo[1]) / pitch_um) + 1,
    )
    if shape[0] > max_px or shape[1] > max_px:
        raise ValueError(f"{seg_id}: render grid {shape} exceeds max_px={max_px}")

    uvs_px = (uv_um - lo) / pitch_um
    pos_grid, nrm_grid, covered = mesh_render.rasterize(
        uvs_px, comp.faces, comp.faces, pos_um, nrm, shape
    )
    if not covered.any():
        raise ValueError(f"{seg_id}: rasterization produced empty coverage")

    rec = mesh_render.recenter_grid(
        pos_grid, nrm_grid, covered, ct, level=level, cache=cache, pitch_um=pitch_um
    )
    pos_render = pos_grid + rec.shift_full[..., None] * nrm_grid
    result = mesh_render.render_grid(
        pos_render,
        nrm_grid,
        covered,
        ct,
        level=level,
        pitch_um=pitch_um,
        u0_um=float(lo[0]),
        v0_um=float(lo[1]),
        cache=cache,
        tile_px=512,
        verbose=True,
    )

    area_mesh_cm2 = mesh_area_voxels(pos_um / 1e4, comp.faces)  # um->cm
    meta = {
        **base_meta,
        "component_verts": int(len(comp.verts_zyx)),
        "component_faces": int(len(comp.faces)),
        "mesh_area_cm2": round(area_mesh_cm2, 4),
        "flatten": {
            "method": flat.method,
            "area_ratio_p5": round(flat.area_ratio_p5, 4),
            "area_ratio_median": round(flat.area_ratio_median, 4),
            "area_ratio_p95": round(flat.area_ratio_p95, 4),
            "final_rel_step": flat.convergence[-1] if flat.convergence else None,
        },
        "uv_scale": {
            "s_u_um": scale.s_u_um,
            "s_v_um": scale.s_v_um,
            **{k: v for k, v in scale.stats.items()},
        },
        "recenter": rec.stats,
    }
    seg_dir = render.write_patch(out_dir, seg_id, result, meta)

    ref = qc.reference_stats(QC_REFERENCE_LAYER, px_um=2.0)
    qcd = qc.evaluate_patch(
        result.layers,
        result.mask,
        shift_um=rec.shift_coarse,
        merge_fraction=rec.merge_fraction,
        ref=ref,
        px_um=result.pitch_um,
    )
    qc.write_qc(seg_dir, qcd, result.layers, result.mask)

    mask_px = int(result.mask.sum())
    return {
        "seg_id": seg_id,
        "verts": int(len(comp.verts_zyx)),
        "faces": int(len(comp.faces)),
        "grid": list(shape),
        "mesh_area_cm2": round(area_mesh_cm2, 4),
        "masked_area_cm2": round(mask_px * (pitch_um / 1e4) ** 2, 4),
        "qc_pass": bool(qcd["pass"]),
        "qc": {k: qcd[k] for k in ("coherence_ratio", "band_contrast", "merge_fraction", "mask_fraction") if k in qcd},
        "fetched_gib": round(cache.fetched_bytes / 2**30, 2),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--welded", type=Path, required=True)
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", type=int, default=1, help="CT texture/recenter level")
    ap.add_argument("--pitch-um", type=float, default=None)
    ap.add_argument("--min-faces", type=int, default=500)
    ap.add_argument("--max-comps", type=int, default=None)
    ap.add_argument("--comps", nargs="*", type=int, default=None,
                    help="explicit component indices (size order) to render")
    ap.add_argument("--min-area-cm2", type=float, default=0.05,
                    help="skip components below this 3D mesh area")
    args = ap.parse_args()

    paths = SCROLLS[args.scroll]
    ledger = zarrio.load_ledger(LEDGER_PATH)
    ct = zarrio.open_volume(
        paths.ledger_ct_key, ledger=ledger, local_root=paths.ct_zarr
    )
    cache = zarrio.ChunkCache(max_bytes=16 * 2**30)
    pitch = args.pitch_um or ct.level_voxel_um(args.level)

    V, F = load_obj_zyx(args.welded)
    comps = split_components(V, F, min_faces=args.min_faces)
    print(f"welded: {len(V)} verts, {len(F)} faces -> {len(comps)} components "
          f"(>= {args.min_faces} faces)")

    base_meta = {
        "scroll": args.scroll,
        "region": args.region,
        "source": "auto-mesh scrollfiesta welded mesh",
        "welded_obj": str(args.welded),
        "preds_zarr": str(paths.preds_zarr),
        "preds_align_ct_level": paths.ct_align_level,
        "frame": "welded zyx preds-grid -> L0 = 4v+1.5 -> x ct.voxel_um",
        "ct_ledger_id": paths.ledger_ct_key,
        "git_sha": zarrio.git_sha(),
    }

    rows = []
    t0 = time.time()
    for i, comp in enumerate(comps):
        if args.comps is not None and i not in args.comps:
            continue
        if args.max_comps is not None and i >= args.max_comps:
            break
        seg_id = f"{args.region}_c{i:03d}"
        area3d = mesh_area_voxels(
            preds_to_l0(comp.verts_zyx, paths.ct_align_level) * ct.voxel_um / 1e4,
            comp.faces,
        )
        if area3d < args.min_area_cm2:
            print(f"[{seg_id}] skip: area {area3d:.3f} cm2 < {args.min_area_cm2}")
            continue
        try:
            tf0 = time.time()
            flat = flatten_component(comp)
            print(f"[{seg_id}] flatten {time.time()-tf0:.0f}s "
                  f"area_ratio med {flat.area_ratio_median:.3f} "
                  f"[{flat.area_ratio_p5:.3f},{flat.area_ratio_p95:.3f}] {flat.method}")
            # Fail fast on hopeless distortion (multi-wrap merged blobs are
            # intrinsically non-developable; rendering them wastes hours of
            # S3 streaming only for QC to reject the smear).
            if not (0.5 <= flat.area_ratio_median <= 2.0) or (
                flat.area_ratio_p95 > 5.0 * max(flat.area_ratio_p5, 1e-6)
            ):
                raise ValueError(
                    f"flatten distortion hopeless (med {flat.area_ratio_median:.3f}, "
                    f"p5 {flat.area_ratio_p5:.3f}, p95 {flat.area_ratio_p95:.3f}) — "
                    "likely multi-wrap merged component"
                )
            row = render_component(
                comp, flat, ct, cache,
                seg_id=seg_id, out_dir=args.out, level=args.level,
                pitch_um=pitch, base_meta=base_meta,
            )
        except Exception as e:  # noqa: BLE001 — batch must survive any component
            row = {"seg_id": seg_id, "error": f"{type(e).__name__}: {e}",
                   "verts": int(len(comp.verts_zyx)),
                   "faces": int(len(comp.faces))}
            print(f"[{seg_id}] FAILED: {row['error']}")
        rows.append(row)
        if "qc_pass" in row:
            print(f"[{seg_id}] qc_pass={row['qc_pass']} "
                  f"masked {row['masked_area_cm2']} cm2 {row['qc']}")

    total_pass = sum(r.get("masked_area_cm2", 0) for r in rows if r.get("qc_pass"))
    summary = {
        "region": args.region,
        "welded": str(args.welded),
        "pitch_um": pitch,
        "level": args.level,
        "n_components_rendered": sum(1 for r in rows if "qc_pass" in r),
        "n_failed": sum(1 for r in rows if "error" in r),
        "qc_pass_area_cm2": round(total_pass, 3),
        "elapsed_s": round(time.time() - t0, 1),
        "components": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "render_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "components"}, indent=1))


if __name__ == "__main__":
    main()
