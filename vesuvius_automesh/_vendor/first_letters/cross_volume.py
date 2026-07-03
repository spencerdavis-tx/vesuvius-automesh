# Vendored from the author's private `first_letters` renderer (module first_letters.cross_volume),
# (c) 2026 Spencer Davis, published here under the MIT license of
# vesuvius-automesh. Vendored (rather than imported) so this repo is
# self-contained; only path defaults were adjusted for the public layout.
"""Cross-volume rendering: surfaces extracted on the preds grid, sampled from
a different (finer) CT volume via that volume's transform.json.

PHerc0139 chain (validated 2026-06-10, results/first-letters/phase2/
pherc0139_control_2p4/landmark_validation.json): the 2.4um volume's
transform.json registers 2.4um-zarr voxel xyz (moving) -> fixed frame
"SCROLLS_HEL_4.681um_..._binmean_2_PHerc_0139_TA_0001_masked". The "binmean_2"
fixed frame IS the 9.362um grid of pherc0139_ct_9p4um / pherc0139_preds_m7_L0:
the affine's linear column scales are 0.2570-0.2572 (implied fixed voxel
9.33um ~ 2 x 4.681um) and all 14 fixed landmarks land on bright papyrus in
the 9.36um CT as-is, while x2 / x2 +-0.5-voxel variants land on CT == 0.
So preds-grid coords map *identically* (x1, no offset) into the fixed frame.
Landmark residuals: median 2.05 fixed-vox (19.2um), max 5.36 (50.2um) --
well inside the +/-200um re-centering band.

Chain: preds world um (zyx) -> preds L0 voxel zyx -> xyz -> [== fixed frame]
-> inv(transform 3x4, homogenized) -> render-CT voxel xyz -> zyx -> world um.
Normals map through the inverse-transpose (mesh_render helpers reused).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh_render import load_transform, normals_through_affine, recenter_grid, render_grid
from .render import RenderResult
from .sheet import SheetSurface
from .zarrio import ChunkCache, Volume

# zyx <-> xyz axis swap as a homogeneous matrix (self-inverse).
_SWAP = np.eye(4)[[2, 1, 0, 3]]


def _diag4(scale: np.ndarray | float, offset: np.ndarray | float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = np.diag(np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,)))
    m[:3, 3] = np.broadcast_to(np.asarray(offset, dtype=np.float64), (3,))
    return m


@dataclass(frozen=True)
class CrossVolumeTransform:
    """Affine taking preds-frame world um (zyx) to render-CT world um (zyx)."""

    affine: np.ndarray  # (4, 4)
    meta: dict

    def points(self, world_um: np.ndarray) -> np.ndarray:
        """Map (..., 3) preds-frame world-um points into the render-CT frame."""
        pts = np.asarray(world_um, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError(f"points must be (..., 3), got {pts.shape}")
        return pts @ self.affine[:3, :3].T + self.affine[:3, 3]

    def normals(self, normals: np.ndarray) -> np.ndarray:
        """Map (..., 3) unit normals; renormalized in the render-CT frame."""
        n = np.asarray(normals, dtype=np.float64)
        if n.shape[-1] != 3:
            raise ValueError(f"normals must be (..., 3), got {n.shape}")
        flat = normals_through_affine(self.affine, n.reshape(-1, 3))
        return flat.reshape(n.shape)


def resolve_transform_path(cache_path: Path | str, url: str | None = None) -> Path:
    """Local transform.json path, downloading from `url` on first use."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return cache_path
    if url is None:
        raise FileNotFoundError(f"transform.json not found: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
        data = r.read()
    json.loads(data)  # validate JSON before caching
    cache_path.write_bytes(data)
    return cache_path


def preds_to_ct_transform(
    transform_path: Path | str,
    *,
    preds_voxel_um: float,
    ct_voxel_um: float,
    preds_origin_um: np.ndarray | float = 0.0,
    ct_origin_um: np.ndarray | float = 0.0,
    fixed_from_preds_scale: float = 1.0,
    fixed_from_preds_offset: float = 0.0,
) -> CrossVolumeTransform:
    """Build the preds-world -> render-CT-world affine from a transform.json
    whose 3x4 matrix maps render-CT voxel xyz -> fixed-frame voxel xyz, with
    fixed = preds-grid * fixed_from_preds_scale + fixed_from_preds_offset
    (PHerc0139: scale 1, offset 0 -- see module docstring).
    """
    if preds_voxel_um <= 0 or ct_voxel_um <= 0:
        raise ValueError("voxel sizes must be positive")
    if fixed_from_preds_scale <= 0:
        raise ValueError(
            f"fixed_from_preds_scale must be positive, got {fixed_from_preds_scale}")
    t44 = load_transform(transform_path)  # render-CT voxel xyz -> fixed xyz
    world_to_preds_vox = _diag4(
        1.0 / preds_voxel_um,
        -np.broadcast_to(np.asarray(preds_origin_um, dtype=np.float64), (3,))
        / preds_voxel_um,
    )
    preds_to_fixed = _diag4(fixed_from_preds_scale, fixed_from_preds_offset)
    ct_vox_to_world = _diag4(ct_voxel_um, ct_origin_um)
    affine = (ct_vox_to_world @ _SWAP @ np.linalg.inv(t44)
              @ preds_to_fixed @ _SWAP @ world_to_preds_vox)
    meta = {
        "transform_path": str(transform_path),
        "transform_matrix_3x4": t44[:3].tolist(),
        "fixed_from_preds_scale": fixed_from_preds_scale,
        "fixed_from_preds_offset": fixed_from_preds_offset,
        "preds_voxel_um": float(preds_voxel_um),
        "ct_voxel_um": float(ct_voxel_um),
        "world_affine_4x4": affine.tolist(),
    }
    return CrossVolumeTransform(affine=affine, meta=meta)


# ---------------------------------------------------------------- rendering


@dataclass(frozen=True)
class CrossRenderOutput:
    result: RenderResult
    recenter: object  # mesh_render.GridRecenterResult
    meta: dict


def render_surface_cross(
    surface: SheetSurface,
    xform: CrossVolumeTransform,
    ct: Volume,
    *,
    level: int,
    pitch_um: float,
    cache: ChunkCache,
    recenter_level: int = 1,
    halfspan_um: float = 200.0,
    max_shift_um: float = 150.0,
    n_layers: int = 66,
    center_layer: int = 32,
    tile_px: int = 256,
    verbose: bool = False,
) -> CrossRenderOutput:
    """Render a preds-frame SheetSurface from a different CT volume.

    The surface (extracted on the preds grid) is evaluated on the output pixel
    grid, mapped into the render-CT frame with `xform`, re-centered on the CT
    bright band along the mapped normals (+/-halfspan_um absorbs the
    registration residual), then n_layers are sampled at pitch_um.
    """
    if pitch_um <= 0:
        raise ValueError(f"pitch_um must be positive, got {pitch_um}")
    if not surface.valid.any():
        raise ValueError("surface has no valid heightfield bins to render")
    vi = np.flatnonzero(surface.valid.any(axis=1))
    vj = np.flatnonzero(surface.valid.any(axis=0))
    u_px = np.arange(surface.u[vi[0]], surface.u[vi[-1]] + pitch_um / 2, pitch_um)
    v_px = np.arange(surface.v[vj[0]], surface.v[vj[-1]] + pitch_um / 2, pitch_um)
    uu, vv = np.meshgrid(u_px, v_px, indexing="ij")

    pos = xform.points(surface.points(uu, vv)).astype(np.float32)
    nrm = xform.normals(surface.normals(uu, vv)).astype(np.float32)
    del uu, vv

    # surface-bin validity, nearest-bin lookup per output pixel (as in render)
    bin_i = np.clip(np.round((u_px - surface.u[0]) / surface.grid_um).astype(int),
                    0, len(surface.u) - 1)
    bin_j = np.clip(np.round((v_px - surface.v[0]) / surface.grid_um).astype(int),
                    0, len(surface.v) - 1)
    covered = surface.valid[bin_i[:, None], bin_j[None, :]]

    rec = recenter_grid(
        pos, nrm, covered, ct, level=recenter_level, cache=cache,
        pitch_um=pitch_um, halfspan_um=halfspan_um, max_shift_um=max_shift_um,
        verbose=verbose)
    pos_render = pos + rec.shift_full[..., None] * nrm
    result = render_grid(
        pos_render, nrm, covered, ct, level=level, pitch_um=pitch_um,
        u0_um=float(u_px[0]), v0_um=float(v_px[0]),
        n_layers=n_layers, center_layer=center_layer,
        cache=cache, tile_px=tile_px, verbose=verbose)
    meta = {
        "cross_volume": xform.meta,
        "recenter": rec.stats,
        "recenter_merge_fraction": rec.merge_fraction,
        "normals_provenance": "PCA seed frame + bicubic spline heightfield on "
                              "the preds grid, mapped by the cross-volume "
                              "affine (inverse-transpose for normals), "
                              "recentered on the render-CT bright band",
    }
    return CrossRenderOutput(result=result, recenter=rec, meta=meta)
