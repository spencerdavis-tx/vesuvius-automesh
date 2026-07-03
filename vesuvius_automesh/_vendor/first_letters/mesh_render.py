# Vendored from the author's private `first_letters` renderer (module first_letters.mesh_render),
# (c) 2026 Spencer Davis, published here under the MIT license of
# vesuvius-automesh. Vendored (rather than imported) so this repo is
# self-contained; only path defaults were adjusted for the public layout.
"""Route 1 mesh-transform rendering (MESH_RENDER_PLAN.md).

Renders villa-layout surface-volume patches from human-traced segment meshes
instead of home-rolled sheet extraction:

1. Load the OBJ pair: `<id>_normalized.obj` carries the 3D vertices
   (7.91um-volume voxel xyz) and faces; `<id>_flattened.obj` (byte-identical
   `v`/`f` records, verified 2026-06-10) carries the UV parameterization in
   its `vt` records, normalized to [0,1] per axis.
2. Per-vertex normals from the normalized mesh geometry; vertices map through
   the inverse of transform.json's 3x4 affine (new-2.4um zarr voxel xyz ->
   7.91um voxel xyz) and normals through its inverse-transpose, renormalized.
3. The UV domain's physical scale (um per UV unit, per axis) is recovered
   from per-face Jacobians d(world um)/d(uv) -- median column norms -- since
   the vt records are normalized, not in voxel units.
4. Faces are rasterized onto the output pixel grid (pitch = target CT level's
   voxel size) with barycentric interpolation of positions and normals.
5. The surface is re-centered on the CT bright band along the normals with a
   widened +/-200um window (registration residual ~100um RMS), then 66 layers
   are sampled (reusing zarrio sampling, render.write_patch and qc).

Grid convention: image rows follow u, columns follow v.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .recenter import _band_centroid
from .render import RenderResult
from .zarrio import ChunkCache, Volume

_EPS_BBOX = 1e-9    # pixel-coordinate slop when snapping triangle bboxes
_EPS_WEIGHT = 1e-7  # barycentric inclusion tolerance (edges/vertices count)
_EPS_DENOM = 1e-12  # smaller |2*area| in px^2 -> degenerate face, skipped


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[mesh_render] {msg}", file=sys.stderr)


# -------------------------------------------------------------- OBJ loading


@dataclass(frozen=True)
class Mesh:
    verts: np.ndarray     # (Nv, 3) float64, xyz
    uvs: np.ndarray       # (Nt, 2) float64, normalized UV units
    faces_v: np.ndarray   # (F, 3) int64, vertex indices
    faces_vt: np.ndarray  # (F, 3) int64, UV indices


def load_obj(path: Path | str) -> Mesh:
    """Parse an OBJ with `v`, `vt` and triangular `f v/vt ...` records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OBJ not found: {path}")
    vs: list[list[str]] = []
    vts: list[list[str]] = []
    fv: list[list[str]] = []
    fvt: list[list[str]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append(line.split()[1:4])
            elif line.startswith("vt "):
                vts.append(line.split()[1:3])
            elif line.startswith("f "):
                corners = [p.split("/") for p in line.split()[1:]]
                if len(corners) != 3 or any(len(p) < 2 or not p[1] for p in corners):
                    raise ValueError(
                        f"{path.name}: only triangular `f v/vt` faces supported, "
                        f"got {line.strip()!r}")
                fv.append([p[0] for p in corners])
                fvt.append([p[1] for p in corners])
    if not vs or not fv or not vts:
        raise ValueError(f"{path.name}: missing v/vt/f records")
    mesh = Mesh(
        verts=np.asarray(vs, dtype=np.float64),
        uvs=np.asarray(vts, dtype=np.float64),
        faces_v=np.asarray(fv, dtype=np.int64) - 1,
        faces_vt=np.asarray(fvt, dtype=np.int64) - 1,
    )
    if mesh.faces_v.min() < 0 or mesh.faces_v.max() >= len(mesh.verts):
        raise ValueError(f"{path.name}: face vertex index out of range")
    if mesh.faces_vt.min() < 0 or mesh.faces_vt.max() >= len(mesh.uvs):
        raise ValueError(f"{path.name}: face UV index out of range")
    return mesh


def load_mesh_pair(seg_dir: Path | str, segment: str) -> Mesh:
    """Combine `<segment>_normalized.obj` (verts) + `<segment>_flattened.obj`
    (UV). Vertex counts and face topology must match exactly."""
    seg_dir = Path(seg_dir)
    normalized = load_obj(seg_dir / f"{segment}_normalized.obj")
    flattened = load_obj(seg_dir / f"{segment}_flattened.obj")
    if len(normalized.verts) != len(flattened.verts):
        raise ValueError(
            f"{segment}: vertex count mismatch between normalized "
            f"({len(normalized.verts)}) and flattened ({len(flattened.verts)})")
    if (normalized.faces_v.shape != flattened.faces_v.shape
            or not np.array_equal(normalized.faces_v, flattened.faces_v)):
        raise ValueError(
            f"{segment}: face topology differs between normalized and flattened")
    return Mesh(verts=normalized.verts, uvs=flattened.uvs,
                faces_v=normalized.faces_v, faces_vt=flattened.faces_vt)


# ------------------------------------------------------ normals + transform


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex unit normals from face geometry."""
    p = verts[faces]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # |fn| = 2 * area
    acc = np.zeros_like(verts, dtype=np.float64)
    for k in range(3):
        np.add.at(acc, faces[:, k], fn)
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    degenerate = norms[:, 0] < 1e-12
    acc[degenerate] = (0.0, 0.0, 1.0)  # isolated/cancelled verts: arbitrary unit
    norms[degenerate] = 1.0
    return acc / norms


def load_transform(path: Path | str) -> np.ndarray:
    """4x4 affine from transform.json: new-zarr voxel xyz -> 7.91um voxel xyz."""
    path = Path(path)
    matrix = np.asarray(json.loads(path.read_text())["transformation_matrix"],
                        dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"{path}: transformation_matrix must be 3x4, "
                         f"got {matrix.shape}")
    t44 = np.eye(4)
    t44[:3, :] = matrix
    return t44


def normals_through_affine(t44: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Normals of geometry transformed by `t44`: inverse-transpose of the 3x3
    part, renormalized. (Row form: n_out = n @ inv(A).)"""
    if t44.shape != (4, 4):
        raise ValueError(f"t44 must be 4x4, got {t44.shape}")
    out = normals @ np.linalg.inv(t44[:3, :3])
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        raise ValueError("normal collapsed to zero under the affine")
    return out / norms


# ----------------------------------------------------------------- UV scale


@dataclass(frozen=True)
class UVScale:
    s_u_um: float  # physical um per UV-u unit (median over faces)
    s_v_um: float
    stats: dict


def uv_scale_um(verts_um: np.ndarray, faces_v: np.ndarray,
                uvs: np.ndarray, faces_vt: np.ndarray) -> UVScale:
    """Recover the UV domain's physical scale from per-face Jacobians
    J = d(world um)/d(uv); s_u/s_v are median column norms."""
    p = verts_um[faces_v]
    uv = uvs[faces_vt]
    dp1, dp2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    duv1, duv2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    ok = np.abs(det) > 1e-15
    if ok.sum() < max(1, len(det) // 2):
        raise ValueError(
            f"degenerate UV parameterization: {int((~ok).sum())}/{len(det)} "
            "faces have ~zero UV area")
    d = det[ok][:, None]
    j_u = (dp1[ok] * duv2[ok][:, 1:2] - dp2[ok] * duv1[ok][:, 1:2]) / d
    j_v = (dp2[ok] * duv1[ok][:, 0:1] - dp1[ok] * duv2[ok][:, 0:1]) / d
    su = np.linalg.norm(j_u, axis=1)
    sv = np.linalg.norm(j_v, axis=1)
    cos_uv = np.abs(np.sum(j_u * j_v, axis=1) / np.maximum(su * sv, 1e-30))
    area_cm2 = float(np.linalg.norm(
        np.cross(dp1[ok], dp2[ok]), axis=1).sum() / 2.0 / 1e8)
    stats = {
        "n_faces": int(len(det)),
        "n_degenerate": int((~ok).sum()),
        "s_u_p5_um": float(np.percentile(su, 5)),
        "s_u_p95_um": float(np.percentile(su, 95)),
        "s_v_p5_um": float(np.percentile(sv, 5)),
        "s_v_p95_um": float(np.percentile(sv, 95)),
        "cos_uv_p95": float(np.percentile(cos_uv, 95)),
        "mesh_area_cm2": area_cm2,
    }
    return UVScale(s_u_um=float(np.median(su)), s_v_um=float(np.median(sv)),
                   stats=stats)


# -------------------------------------------------------------- rasterizing


def _face_bboxes(tri: np.ndarray, shape: tuple[int, int]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-face inclusive pixel bboxes (r0, r1, c0, c1), clipped to the grid."""
    h, w = shape
    r0 = np.maximum(np.ceil(tri[..., 0].min(axis=1) - _EPS_BBOX), 0).astype(np.int64)
    r1 = np.minimum(np.floor(tri[..., 0].max(axis=1) + _EPS_BBOX), h - 1).astype(np.int64)
    c0 = np.maximum(np.ceil(tri[..., 1].min(axis=1) - _EPS_BBOX), 0).astype(np.int64)
    c1 = np.minimum(np.floor(tri[..., 1].max(axis=1) + _EPS_BBOX), w - 1).astype(np.int64)
    return r0, r1, c0, c1


def _raster_chunk(tri: np.ndarray, denom: np.ndarray, bbox: tuple, p: np.ndarray,
                  n: np.ndarray, w: int, out: tuple[np.ndarray, ...]) -> None:
    """Rasterize one face chunk into the flat output buffers (last write wins)."""
    r0, r1, c0, c1 = bbox
    out_pos, out_nrm, out_cov = out
    ncols = c1 - c0 + 1
    counts = (r1 - r0 + 1) * ncols
    offsets = np.concatenate([[0], np.cumsum(counts)])
    fidx = np.repeat(np.arange(len(tri)), counts)
    k = np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts)
    pr = r0[fidx] + k // ncols[fidx]
    pc = c0[fidx] + k % ncols[fidx]

    a, b, c = tri[fidx, 0], tri[fidx, 1], tri[fidx, 2]
    d = denom[fidx]
    pu, pv = pr - a[:, 0], pc - a[:, 1]
    lb = (pu * (c[:, 1] - a[:, 1]) - pv * (c[:, 0] - a[:, 0])) / d
    lc = ((b[:, 0] - a[:, 0]) * pv - (b[:, 1] - a[:, 1]) * pu) / d
    la = 1.0 - lb - lc
    inside = (la >= -_EPS_WEIGHT) & (lb >= -_EPS_WEIGHT) & (lc >= -_EPS_WEIGHT)
    if not inside.any():
        return
    flat = (pr * w + pc)[inside]
    wa, wb, wc = la[inside, None], lb[inside, None], lc[inside, None]
    pp = p[fidx[inside]]
    nn = n[fidx[inside]]
    out_pos[flat] = (wa * pp[:, 0] + wb * pp[:, 1] + wc * pp[:, 2]).astype(np.float32)
    out_nrm[flat] = (wa * nn[:, 0] + wb * nn[:, 1] + wc * nn[:, 2]).astype(np.float32)
    out_cov[flat] = True


def rasterize(
    uvs_px: np.ndarray,
    faces_vt: np.ndarray,
    faces_v: np.ndarray,
    pos: np.ndarray,
    nrm: np.ndarray,
    shape: tuple[int, int],
    pair_budget: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a triangle mesh onto a pixel grid in UV space.

    uvs_px: (Nt, 2) UV coords in pixel units (row, col); pixel (i, j) center
    sits at (i, j). Returns (pos_grid (H,W,3) f32, unit nrm_grid (H,W,3) f32,
    covered (H,W) bool); positions/normals barycentrically interpolated.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError(f"shape must be positive, got {shape}")
    tri = uvs_px[faces_vt]  # (F, 3, 2)
    denom = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
             - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    r0, r1, c0, c1 = _face_bboxes(tri, shape)
    counts = np.maximum(r1 - r0 + 1, 0) * np.maximum(c1 - c0 + 1, 0)
    sel = np.flatnonzero((counts > 0) & (np.abs(denom) > _EPS_DENOM)
                         & (r1 >= r0) & (c1 >= c0))

    out_pos = np.zeros((h * w, 3), dtype=np.float32)
    out_nrm = np.zeros((h * w, 3), dtype=np.float32)
    out_cov = np.zeros(h * w, dtype=bool)
    cum = np.cumsum(counts[sel])
    start = 0
    while start < len(sel):
        base = int(cum[start - 1]) if start else 0
        end = int(np.searchsorted(cum, base + pair_budget)) + 1
        end = min(max(end, start + 1), len(sel))
        idx = sel[start:end]
        _raster_chunk(tri[idx], denom[idx], (r0[idx], r1[idx], c0[idx], c1[idx]),
                      pos[faces_v[idx]], nrm[faces_v[idx]], w,
                      (out_pos, out_nrm, out_cov))
        start = end

    norms = np.linalg.norm(out_nrm[out_cov], axis=1)
    bad = norms < 1e-9
    norms[bad] = 1.0
    out_nrm[out_cov] /= norms[:, None]
    if bad.any():  # interpolated normal cancelled out: drop those pixels
        cov_idx = np.flatnonzero(out_cov)
        out_cov[cov_idx[bad]] = False
    return (out_pos.reshape(h, w, 3), out_nrm.reshape(h, w, 3),
            out_cov.reshape(h, w))


# --------------------------------------------------------------- recentering


@dataclass(frozen=True)
class GridRecenterResult:
    shift_full: np.ndarray    # (H, W) float32, um along the normal
    shift_coarse: np.ndarray  # (Hc, Wc) float64, NaN where no band was found
    merge_fraction: float     # band thickness > 1.5x median (merge heuristic)
    stats: dict


def recenter_grid(
    pos: np.ndarray,
    nrm: np.ndarray,
    covered: np.ndarray,
    ct: Volume,
    *,
    level: int,
    cache: ChunkCache,
    pitch_um: float,
    halfspan_um: float = 200.0,
    max_shift_um: float = 150.0,
    target_step_um: float = 50.0,
    thickness_merge_factor: float = 1.5,
    batch: int = 4096,
    verbose: bool = False,
) -> GridRecenterResult:
    """Re-center a rasterized surface grid on the CT bright band.

    Profiles are sampled on a coarse (~target_step_um) subgrid, snapped with
    `recenter._band_centroid`, smoothed, and bilinearly upsampled to a
    full-resolution shift field (um along the per-pixel normal).
    """
    if halfspan_um <= 0 or max_shift_um <= 0 or pitch_um <= 0:
        raise ValueError("halfspan_um, max_shift_um and pitch_um must be positive")
    stride = max(1, round(target_step_um / pitch_um))
    sub_pos = pos[::stride, ::stride].astype(np.float64)
    sub_nrm = nrm[::stride, ::stride].astype(np.float64)
    sub_cov = covered[::stride, ::stride]
    step = ct.level_voxel_um(level)
    s_um = np.arange(-halfspan_um, halfspan_um + step / 2, step)

    shift = np.full(sub_cov.shape, np.nan)
    thickness = np.full(sub_cov.shape, np.nan)
    bins = np.argwhere(sub_cov)
    for b0 in range(0, len(bins), batch):
        bb = bins[b0:b0 + batch]
        p = sub_pos[bb[:, 0], bb[:, 1]]
        n = sub_nrm[bb[:, 0], bb[:, 1]]
        coords = p[:, None, :] + s_um[None, :, None] * n[:, None, :]
        values, ok = ct.sample(coords.reshape(-1, 3), level, cache)
        values = values.reshape(len(bb), len(s_um))
        ok = ok.reshape(len(bb), len(s_um))
        for (i, j), profile, prof_ok in zip(bb, values, ok, strict=True):
            if not prof_ok.all():
                continue
            band = _band_centroid(profile.astype(np.float64), s_um)
            if band is None or abs(band[0]) > max_shift_um:
                continue
            shift[i, j] = band[0]
            thickness[i, j] = band[1]
        _log(verbose, f"recenter: {min(b0 + batch, len(bins))}/{len(bins)} probes")

    found = np.isfinite(shift)
    n_found = int(found.sum())
    if n_found:
        t_found = thickness[found]
        merge_fraction = float(
            (t_found > thickness_merge_factor * np.median(t_found)).mean())
    else:
        merge_fraction = 1.0
    finite = shift[found]
    stats = {
        "n_probed": int(len(bins)),
        "n_found": n_found,
        "found_fraction": float(n_found / max(len(bins), 1)),
        "median_abs_um": float(np.median(np.abs(finite))) if n_found else None,
        "p95_abs_um": float(np.percentile(np.abs(finite), 95)) if n_found else None,
        "max_abs_um": float(np.max(np.abs(finite))) if n_found else None,
        "mean_um": float(np.mean(finite)) if n_found else None,
        "halfspan_um": halfspan_um,
        "max_shift_um": max_shift_um,
        "stride_px": stride,
        "profile_step_um": float(step),
        "level": level,
    }

    smoothed = gaussian_filter(np.where(found, shift, 0.0), sigma=1.0,
                               mode="nearest")
    hh, ww = covered.shape
    rr, cc = np.meshgrid(np.arange(hh, dtype=np.float64) / stride,
                         np.arange(ww, dtype=np.float64) / stride, indexing="ij")
    shift_full = map_coordinates(smoothed, [rr, cc], order=1,
                                 mode="nearest").astype(np.float32)
    return GridRecenterResult(shift_full=shift_full, shift_coarse=shift,
                              merge_fraction=merge_fraction, stats=stats)


# ------------------------------------------------------------ layer sampling


def render_grid(
    pos: np.ndarray,
    nrm: np.ndarray,
    covered: np.ndarray,
    ct: Volume,
    *,
    level: int,
    pitch_um: float,
    u0_um: float,
    v0_um: float,
    n_layers: int = 66,
    center_layer: int = 32,
    cache: ChunkCache,
    tile_px: int = 256,
    verbose: bool = False,
) -> RenderResult:
    """Sample n_layers along per-pixel normals of a rasterized surface grid."""
    if pitch_um <= 0:
        raise ValueError(f"pitch_um must be positive, got {pitch_um}")
    if not (0 <= center_layer < n_layers):
        raise ValueError(f"center_layer {center_layer} outside [0, {n_layers})")
    h, w = covered.shape
    offsets = (np.arange(n_layers, dtype=np.float64) - center_layer) * pitch_um
    layers = np.zeros((n_layers, h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    n_tiles = ((h + tile_px - 1) // tile_px) * ((w + tile_px - 1) // tile_px)
    done = 0
    for ti in range(0, h, tile_px):
        for tj in range(0, w, tile_px):
            done += 1
            cov = covered[ti:ti + tile_px, tj:tj + tile_px]
            if not cov.any():
                continue
            p = pos[ti:ti + tile_px, tj:tj + tile_px].astype(np.float64)
            n = nrm[ti:ti + tile_px, tj:tj + tile_px].astype(np.float64)
            th, tw = cov.shape
            coords = p[None] + offsets[:, None, None, None] * n[None]
            values, ok = ct.sample(coords.reshape(-1, 3), level, cache)
            values = values.reshape(n_layers, th, tw)
            ok = ok.reshape(n_layers, th, tw)
            layers[:, ti:ti + th, tj:tj + tw] = np.clip(
                np.rint(values), 0, 255).astype(np.uint8)
            mask[ti:ti + th, tj:tj + tw] = cov & ok.all(axis=0)
            if done % 20 == 0 or done == n_tiles:
                _log(verbose, f"render: tile {done}/{n_tiles}, "
                              f"fetched {cache.fetched_bytes / 2**30:.2f} GiB")
    u_um = u0_um + np.arange(h, dtype=np.float64) * pitch_um
    v_um = v0_um + np.arange(w, dtype=np.float64) * pitch_um
    return RenderResult(layers=layers, mask=mask, u_um=u_um, v_um=v_um,
                        pitch_um=pitch_um, level=level)


# -------------------------------------------------------------- orchestration


@dataclass(frozen=True)
class MeshRenderOutput:
    result: RenderResult
    recenter: GridRecenterResult
    scale: UVScale
    meta: dict


def _resolve_crop_um(
    uv_lo: np.ndarray, uv_hi: np.ndarray,
    crop_um: tuple[float, float, float, float] | None,
    crop_centered_mm: float | None,
) -> tuple[float, float, float, float]:
    """Crop rectangle (u0, v0, u1, v1) in um, clamped to the UV extent."""
    if crop_um is None and crop_centered_mm is not None:
        half = crop_centered_mm * 1000.0 / 2.0
        center = (uv_lo + uv_hi) / 2.0
        crop_um = (center[0] - half, center[1] - half,
                   center[0] + half, center[1] + half)
    if crop_um is None:
        return float(uv_lo[0]), float(uv_lo[1]), float(uv_hi[0]), float(uv_hi[1])
    u0 = max(float(crop_um[0]), float(uv_lo[0]))
    v0 = max(float(crop_um[1]), float(uv_lo[1]))
    u1 = min(float(crop_um[2]), float(uv_hi[0]))
    v1 = min(float(crop_um[3]), float(uv_hi[1]))
    if u1 <= u0 or v1 <= v0:
        raise ValueError(f"crop {crop_um} does not intersect the UV extent "
                         f"[{uv_lo}, {uv_hi}]")
    return u0, v0, u1, v1


def render_mesh_segment(
    seg_dir: Path | str,
    segment: str,
    transform_path: Path | str,
    ct: Volume,
    *,
    level: int,
    cache: ChunkCache,
    crop_um: tuple[float, float, float, float] | None = None,
    crop_centered_mm: float | None = None,
    recenter_level: int = 1,
    halfspan_um: float = 200.0,
    max_shift_um: float = 150.0,
    target_step_um: float = 50.0,
    n_layers: int = 66,
    center_layer: int = 32,
    tile_px: int = 256,
    verbose: bool = False,
) -> MeshRenderOutput:
    """Full Route 1 pipeline for one traced segment (see module docstring)."""
    t0 = time.monotonic()
    mesh = load_mesh_pair(seg_dir, segment)
    t44 = load_transform(transform_path)
    t_inv = np.linalg.inv(t44)
    verts_new_xyz = mesh.verts @ t_inv[:3, :3].T + t_inv[:3, 3]
    normals_new_xyz = normals_through_affine(
        t_inv, vertex_normals(mesh.verts, mesh.faces_v))
    pos_um = verts_new_xyz[:, ::-1] * ct.voxel_um  # zyx world um (zarrio frame)
    nrm_zyx = normals_new_xyz[:, ::-1]
    _log(verbose, f"{segment}: {len(mesh.verts)} verts, "
                  f"{len(mesh.faces_v)} faces loaded in {time.monotonic() - t0:.1f}s")

    scale = uv_scale_um(pos_um, mesh.faces_v, mesh.uvs, mesh.faces_vt)
    uv_um = mesh.uvs * (scale.s_u_um, scale.s_v_um)
    _log(verbose, f"UV scale: {scale.s_u_um / 1000:.2f} x "
                  f"{scale.s_v_um / 1000:.2f} mm ({scale.stats})")

    pitch_um = ct.level_voxel_um(level)
    u0, v0, u1, v1 = _resolve_crop_um(uv_um.min(axis=0), uv_um.max(axis=0),
                                      crop_um, crop_centered_mm)
    shape = (int((u1 - u0) / pitch_um) + 1, int((v1 - v0) / pitch_um) + 1)
    pos_grid, nrm_grid, cov = rasterize(
        (uv_um - (u0, v0)) / pitch_um, mesh.faces_vt, mesh.faces_v,
        pos_um, nrm_zyx, shape)
    _log(verbose, f"rasterized {shape[0]}x{shape[1]} px @ {pitch_um:.3f} um, "
                  f"coverage {cov.mean():.3f}")

    rec = recenter_grid(
        pos_grid, nrm_grid, cov, ct, level=recenter_level, cache=cache,
        pitch_um=pitch_um, halfspan_um=halfspan_um, max_shift_um=max_shift_um,
        target_step_um=target_step_um, verbose=verbose)
    _log(verbose, f"recenter: {rec.stats}")

    pos_render = pos_grid + rec.shift_full[..., None] * nrm_grid
    result = render_grid(
        pos_render, nrm_grid, cov, ct, level=level, pitch_um=pitch_um,
        u0_um=u0, v0_um=v0, n_layers=n_layers, center_layer=center_layer,
        cache=cache, tile_px=tile_px, verbose=verbose)
    _log(verbose, f"rendered {result.layers.shape}, mask {result.mask_fraction:.3f}, "
                  f"total {time.monotonic() - t0:.1f}s")

    meta = {
        "method": "mesh_transform_render",
        "segment_dir": str(seg_dir),
        "n_verts": int(len(mesh.verts)),
        "n_faces": int(len(mesh.faces_v)),
        "transform_path": str(transform_path),
        "transform_matrix_3x4": t44[:3].tolist(),
        "uv_scale": {"s_u_um": scale.s_u_um, "s_v_um": scale.s_v_um,
                     **scale.stats},
        "crop_um": [u0, v0, u1, v1],
        "coverage_fraction": float(cov.mean()),
        "grid_axes": "rows=u, cols=v (UV units from flattened.obj vt)",
        "recenter": rec.stats,
        "merge_fraction": rec.merge_fraction,
        "normals_provenance": "per-vertex area-weighted face normals of the "
                              "normalized mesh, mapped by inverse-transpose "
                              "of the new->791 affine inverse, "
                              "barycentric-interpolated, recentered on the "
                              "CT bright band",
    }
    return MeshRenderOutput(result=result, recenter=rec, scale=scale, meta=meta)
