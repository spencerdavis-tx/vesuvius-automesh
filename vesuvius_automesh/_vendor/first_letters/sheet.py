# Vendored from the author's private `first_letters` renderer (module first_letters.sheet),
# (c) 2026 Spencer Davis, published here under the MIT license of
# vesuvius-automesh. Vendored (rather than imported) so this repo is
# self-contained; only path defaults were adjusted for the public layout.
"""Sheet extraction: binary preds -> smoothed medial surface with normals.

Pipeline (per RENDERER_SPEC.md step 2):
1. Seed: connected-component the preds inside a small bbox at the candidate
   center; select the component through the center; PCA -> local (U,V,W) frame.
2. Heightfield: histogram occupied voxels into (u,v,w) bins on a ~50um grid,
   streamed in z-slabs so memory stays bounded.
3. Per-(u,v) w-runs; grow the sheet outward from the center, snapping each bin
   to the run nearest the locally predicted height (rejects neighboring wraps).
4. Merge heuristic: run thickness > 1.5x median sheet thickness -> flagged.
5. Robust fit: MAD outlier rejection + hole fill + light Gaussian smoothing +
   bicubic spline ("robust smoothing spline" per spec, implemented as
   reject-smooth-interpolate for determinism).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label

from .zarrio import ChunkCache, Volume

# Seed-candidate thinness test (distance transform, voxels): a papyrus sheet's
# 95th-percentile half-thickness. Measured on scroll3 preds L0 (9.6um): real
# sheets read 1.4-3.6, solid blobs/cores read >> 4.
THIN_DT_P95_VOX = 4.0
MIN_SEED_VOXELS = 20
_CC_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass
class SheetSurface:
    """Local medial surface: heightfield w = f(u,v) in an orthonormal frame.

    `axes` rows are the U, V, W unit vectors (world um, z/y/x order);
    a frame point (u,v,w) maps to world as origin + [u,v,w] @ axes.
    """

    origin: np.ndarray          # (3,) world um
    axes: np.ndarray            # (3,3) rows U, V, W
    u: np.ndarray               # (Nu,) frame um
    v: np.ndarray               # (Nv,)
    w: np.ndarray               # (Nu, Nv) fitted heights, um
    valid: np.ndarray           # (Nu, Nv) bool: measured (not hole-filled)
    merge: np.ndarray           # (Nu, Nv) bool: thickness merge flag
    thickness_um: np.ndarray    # (Nu, Nv) chosen-run thickness (NaN for holes)
    grid_um: float
    _spline: Any = None

    @property
    def merge_fraction(self) -> float:
        if not self.valid.any():
            return 1.0
        return float(self.merge[self.valid].mean())

    def heights(self, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return self._spline.ev(uu, vv)

    def points(self, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        """World-um points on the fitted surface; uu/vv broadcastable, um."""
        uu = np.asarray(uu, dtype=np.float64)
        vv = np.asarray(vv, dtype=np.float64)
        w = self._spline.ev(uu, vv)
        frame = np.stack([uu, vv, w], axis=-1)
        return self.origin + frame @ self.axes

    def normals(self, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        """Unit normals of the fitted surface (world um frame)."""
        uu = np.asarray(uu, dtype=np.float64)
        vv = np.asarray(vv, dtype=np.float64)
        fu = self._spline.ev(uu, vv, dx=1)
        fv = self._spline.ev(uu, vv, dy=1)
        n_frame = np.stack([-fu, -fv, np.ones_like(fu)], axis=-1)
        n_frame /= np.linalg.norm(n_frame, axis=-1, keepdims=True)
        return n_frame @ self.axes

    def with_heights(self, w_new: np.ndarray, valid_new: np.ndarray | None = None
                     ) -> SheetSurface:
        """Return a new surface with replaced height samples (refit spline)."""
        valid = self.valid if valid_new is None else valid_new
        w_fit, spline = _fit_spline(self.u, self.v, w_new, valid)
        return replace(self, w=w_fit, valid=valid, _spline=spline)


def _fill_holes(w: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Iterative neighbor-mean fill of NaN bins (diffusion inpainting)."""
    out = np.where(valid, w, np.nan)
    for _ in range(out.size):
        holes = np.isnan(out)
        if not holes.any():
            break
        padded = np.pad(out, 1, constant_values=np.nan)
        stack = np.stack([
            padded[1 + dz:1 + dz + out.shape[0], 1 + dx:1 + dx + out.shape[1]]
            for dz in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dx) != (0, 0)])
        cnt = np.isfinite(stack).sum(axis=0)
        neigh = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
        fill = holes & ~np.isnan(neigh)
        if not fill.any():
            raise ValueError("hole filling stalled: no valid bins at all?")
        out[fill] = neigh[fill]
    return out


def _fit_spline(u: np.ndarray, v: np.ndarray, w: np.ndarray, valid: np.ndarray,
                smooth_sigma_bins: float = 1.0):
    """Hole-fill + Gaussian smooth + bicubic spline. Returns (w_grid, spline)."""
    if valid.sum() < 16:
        raise ValueError(f"too few valid heightfield bins: {int(valid.sum())}")
    filled = _fill_holes(w, valid)
    smoothed = gaussian_filter(filled, sigma=smooth_sigma_bins, mode="nearest")
    spline = RectBivariateSpline(u, v, smoothed, kx=3, ky=3, s=0)
    return smoothed, spline


def _robust_eigh(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of a covariance matrix that never leaks LinAlgError.

    Symmetrizes first; falls back to SVD on eigh non-convergence; degenerate
    (non-finite) input raises the clean "not sheet-like" ValueError instead.
    Returns (eigenvalues ascending, eigenvectors as columns)."""
    sym = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(sym)):
        raise ValueError(
            "seed component is not sheet-like (degenerate covariance); "
            "try another region")
    try:
        return np.linalg.eigh(sym)
    except np.linalg.LinAlgError:
        try:
            u, s, _ = np.linalg.svd(sym)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "seed component is not sheet-like (covariance "
                f"eigendecomposition failed: {e}); try another region") from e
        return s[::-1], u[:, ::-1]  # symmetric PSD: singular values = eigvals


def _frame_from_voxels(preds: Volume, idx_lev: np.ndarray, level: int,
                       vox: float) -> tuple[np.ndarray, np.ndarray]:
    """PCA frame of a candidate voxel set; raises ValueError if not sheet-like.

    Returns (origin_world, axes) with axes rows U, V, W (W = sheet normal).
    The origin is anchored on the voxel nearest the centroid so the seed
    heightfield column is guaranteed to contain sheet voxels even when the
    component is curved (centroid of a curved sheet lies off the sheet)."""
    comp_world = preds.to_world(idx_lev.astype(np.float64), level)
    centroid = comp_world.mean(axis=0)
    d = comp_world - centroid
    cov = d.T @ d / max(len(d) - 1, 1)
    eigval, eigvec = _robust_eigh(cov)
    if eigval[1] < (2.0 * vox) ** 2:
        raise ValueError(
            f"seed component is not sheet-like (degenerate extent, "
            f"eigenvalues {eigval}); try another region")
    if eigval[0] > 0.25 * eigval[1]:
        raise ValueError(
            f"seed component is not sheet-like (eigenvalues {eigval}); "
            "try another region")
    axes = np.stack([eigvec[:, 2], eigvec[:, 1], eigvec[:, 0]])  # U, V, W rows
    if np.linalg.det(axes) < 0:
        axes[1] = -axes[1]
    origin = comp_world[np.argmin(np.sum(d**2, axis=1))]
    return origin, axes


def _dt_p95(comp: np.ndarray) -> float:
    """95th-percentile distance-transform value inside a component (voxels):
    a robust half-thickness. Thin sheets stay small even when curved."""
    pad = np.pad(comp, 1)
    return float(np.percentile(distance_transform_edt(pad)[pad], 95))


def _crop_to_bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a boolean mask to its bounding box; returns (crop, offset)."""
    idx = np.argwhere(mask)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], lo


def _nearest_first_labels(labels: np.ndarray, local_c: np.ndarray,
                          max_candidates: int) -> list[int]:
    """Component labels ordered by distance of their nearest voxel to the
    bbox center (alternate seed points within the bbox)."""
    occ_idx = np.argwhere(labels > 0)
    d2 = np.sum((occ_idx - local_c) ** 2, axis=1)
    lab_of = labels[tuple(occ_idx.T)]
    best = np.full(int(lab_of.max()) + 1, np.inf)
    np.minimum.at(best, lab_of, d2)
    order = np.argsort(best[1:], kind="stable") + 1  # skip background label 0
    return [int(lb) for lb in order[np.isfinite(best[order])][:max_candidates]]


def _try_component(preds: Volume, comp: np.ndarray, start: np.ndarray,
                   local_c: np.ndarray, level: int, vox: float,
                   errors: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt a seed frame from one connected component, retrying with
    progressively eroded thin subcomponents when the whole component fails
    (thin-but-curved multi-wrap components split at their bridges)."""
    crop, off = _crop_to_bbox(comp)
    n = int(crop.sum())
    if n < MIN_SEED_VOXELS:
        errors.append(f"component too small ({n} voxels)")
        return None
    if _dt_p95(crop) <= THIN_DT_P95_VOX:
        try:
            return _frame_from_voxels(preds, np.argwhere(crop) + off + start, level, vox)
        except ValueError as e:
            errors.append(str(e))
    else:
        errors.append(f"component fails thinness test ({n} voxels)")
    for it in (1, 2):
        eroded = binary_erosion(crop, structure=_CC_STRUCTURE.astype(bool),
                                iterations=it)
        if not eroded.any():
            break
        sub_labels, _ = label(eroded, structure=_CC_STRUCTURE)
        for lb in _nearest_first_labels(sub_labels, local_c - off, max_candidates=3):
            sub_idx = np.argwhere(sub_labels == lb)
            if len(sub_idx) < MIN_SEED_VOXELS or not _dt_p95(sub_labels == lb) <= THIN_DT_P95_VOX:
                continue
            try:
                return _frame_from_voxels(preds, sub_idx + off + start, level, vox)
            except ValueError as e:
                errors.append(f"erode{it}: {e}")
    return None


def _seed_frame(preds: Volume, center_world: np.ndarray, seed_radius_um: float,
                level: int, cache: ChunkCache,
                max_candidates: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """CC-label a seed bbox and fit a PCA frame, trying candidate components
    nearest the bbox center first and erosion-split thin subcomponents before
    failing. Returns (origin_world, axes) with axes rows U, V, W.
    """
    vox = preds.level_voxel_um(level)
    c = preds.to_level(center_world[None, :], level)[0]
    r = max(4, int(np.ceil(seed_radius_um / vox)))
    start = np.floor(c).astype(np.int64) - r
    stop = np.floor(c).astype(np.int64) + r + 1
    block = preds.read_block(level, start, stop, cache)
    occ = block > 0
    if not occ.any():
        raise ValueError(
            f"no preds voxels within {seed_radius_um}um of {center_world} (level {level})")
    labels, _ = label(occ, structure=_CC_STRUCTURE)
    local_c = c - start
    errors: list[str] = []
    for lb in _nearest_first_labels(labels, local_c, max_candidates):
        frame = _try_component(preds, labels == lb, start, local_c, level, vox, errors)
        if frame is not None:
            return frame
    raise ValueError(
        "seed component is not sheet-like (all candidates failed within "
        f"{seed_radius_um}um of {center_world}):\n    " + "\n    ".join(errors))


def _seed_frame_ladder(preds: Volume, center_world: np.ndarray,
                       seed_radius_um: float, level: int, cache: ChunkCache
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Try the requested seed radius, then halved radii: in dense regions a
    large seed bbox connected-components into a multi-wrap blob that fails the
    sheet-likeness check, while a smaller bbox isolates the local sheet."""
    errors: list[str] = []
    radius = seed_radius_um
    while radius >= preds.level_voxel_um(level) * 4:
        try:
            return _seed_frame(preds, center_world, radius, level, cache)
        except ValueError as e:
            errors.append(f"r={radius:.0f}um: {e}")
            radius /= 2.0
    raise ValueError("seed frame failed at all radii:\n  " + "\n  ".join(errors))


def _runs_of_column(col: np.ndarray, gap_bins: int = 1) -> list[tuple[int, int]]:
    """Contiguous nonzero runs [start, stop) in a w-histogram column, merging
    gaps of <= gap_bins empty bins (voxelization speckle)."""
    nz = np.flatnonzero(col)
    if nz.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    s = p = int(nz[0])
    for i in nz[1:]:
        if i - p <= gap_bins + 1:
            p = int(i)
        else:
            runs.append((s, p + 1))
            s = p = int(i)
    runs.append((s, p + 1))
    return runs


def extract_sheet(
    preds: Volume,
    *,
    center_world: np.ndarray,
    seed_radius_um: float,
    patch_radius_um: float,
    grid_um: float = 50.0,
    level: int = 0,
    cache: ChunkCache,
    w_halfspan_um: float = 1500.0,
    thickness_merge_factor: float = 1.5,
    track_tol_um: float = 250.0,
    slab_target_bytes: int = 128 << 20,
) -> SheetSurface:
    """Extract the medial surface of the sheet through `center_world`."""
    center_world = np.asarray(center_world, dtype=np.float64)
    if center_world.shape != (3,):
        raise ValueError(f"center_world must be (3,), got {center_world.shape}")
    for nm, val in [("seed_radius_um", seed_radius_um),
                    ("patch_radius_um", patch_radius_um), ("grid_um", grid_um)]:
        if val <= 0:
            raise ValueError(f"{nm} must be positive, got {val}")

    origin, axes = _seed_frame_ladder(preds, center_world, seed_radius_um, level, cache)

    vox = preds.level_voxel_um(level)
    w_bin = vox / 2.0
    half = patch_radius_um
    u_edges = np.arange(-half - grid_um / 2, half + grid_um, grid_um)
    centers = (u_edges[:-1] + u_edges[1:]) / 2.0
    nu = len(centers)
    nw = int(np.ceil(2 * w_halfspan_um / w_bin))
    hist = np.zeros((nu, nu, nw), dtype=np.uint16)

    # AABB (level coords) of the oriented patch box
    corners_frame = np.array([[su * (half + grid_um), sv * (half + grid_um), sw * w_halfspan_um]
                              for su in (-1, 1) for sv in (-1, 1) for sw in (-1, 1)])
    corners_world = origin + corners_frame @ axes
    corners_lev = preds.to_level(corners_world, level)
    shape = np.array(preds.level_info(level).shape)
    lo = np.maximum(np.floor(corners_lev.min(axis=0)).astype(np.int64) - 1, 0)
    hi = np.minimum(np.ceil(corners_lev.max(axis=0)).astype(np.int64) + 2, shape)
    if np.any(hi <= lo):
        raise ValueError("patch bbox lies outside the preds volume")

    slab_dz = max(8, int(slab_target_bytes // max(1, (hi[1] - lo[1]) * (hi[2] - lo[2]))))
    for z0 in range(int(lo[0]), int(hi[0]), slab_dz):
        z1 = min(z0 + slab_dz, int(hi[0]))
        block = preds.read_block(level, (z0, lo[1], lo[2]), (z1, hi[1], hi[2]), cache)
        idx = np.argwhere(block > 0).astype(np.float64)
        if idx.size == 0:
            continue
        idx += (z0, lo[1], lo[2])
        frame = (preds.to_world(idx, level) - origin) @ axes.T
        iu = np.floor((frame[:, 0] + half + grid_um / 2) / grid_um).astype(np.int64)
        iv = np.floor((frame[:, 1] + half + grid_um / 2) / grid_um).astype(np.int64)
        iw = np.floor((frame[:, 2] + w_halfspan_um) / w_bin).astype(np.int64)
        keep = ((iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nu) & (iw >= 0) & (iw < nw))
        np.add.at(hist, (iu[keep], iv[keep], iw[keep]), 1)

    # --- per-bin runs, then center-out growth picking the tracked sheet
    w_of = lambda b: (b + 0.5) * w_bin - w_halfspan_um  # noqa: E731
    runs_grid: list[list[list[tuple[int, int]]]] = [
        [_runs_of_column(hist[i, j]) for j in range(nu)] for i in range(nu)]

    chosen_w = np.full((nu, nu), np.nan)
    thickness = np.full((nu, nu), np.nan)
    iu0 = iv0 = int(np.argmin(np.abs(centers)))

    def run_stats(i: int, j: int, run: tuple[int, int]) -> tuple[float, float]:
        s, e = run
        counts = hist[i, j, s:e].astype(np.float64)
        wc = w_of(np.arange(s, e, dtype=np.float64) - 0.5 + 0.5)
        center = float(np.sum(wc * counts) / np.sum(counts))
        return center, (e - s) * w_bin

    if not runs_grid[iu0][iv0]:
        raise ValueError("no sheet voxels in the center heightfield bin")
    first = min(runs_grid[iu0][iv0], key=lambda r: abs(run_stats(iu0, iv0, r)[0]))
    chosen_w[iu0, iv0], thickness[iu0, iv0] = run_stats(iu0, iv0, first)

    # Grow outward to convergence: repeatedly sweep unchosen bins that have
    # runs, predicting w from already-chosen neighbors. Sweeping (rather than
    # a single BFS visit) lets bins be claimed once any neighbor is resolved,
    # so coverage is limited only by genuine gaps, not visit order.
    neigh8 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    candidates = deque(
        (i, j) for i in range(nu) for j in range(nu)
        if runs_grid[i][j] and (i, j) != (iu0, iv0))
    for _ in range(2 * nu):
        changed = False
        remaining: deque[tuple[int, int]] = deque()
        while candidates:
            i, j = candidates.popleft()
            preds_w = [chosen_w[i + a, j + b] for a, b in neigh8
                       if 0 <= i + a < nu and 0 <= j + b < nu
                       and np.isfinite(chosen_w[i + a, j + b])]
            if not preds_w:
                remaining.append((i, j))
                continue
            w_pred = float(np.mean(preds_w))
            stats = [run_stats(i, j, r) for r in runs_grid[i][j]]
            best = min(stats, key=lambda s: abs(s[0] - w_pred))
            if abs(best[0] - w_pred) <= track_tol_um:
                chosen_w[i, j], thickness[i, j] = best
                changed = True
            else:
                remaining.append((i, j))
        candidates = remaining
        if not changed or not candidates:
            break

    valid = np.isfinite(chosen_w)
    if valid.sum() < 16:
        raise ValueError(f"sheet tracking found only {int(valid.sum())} bins")

    # --- robust pass: reject bins far from a local smooth of the heightfield
    ref, _ = _fit_spline(centers, centers, chosen_w, valid)
    resid = np.abs(chosen_w - ref)
    mad = float(np.nanmedian(resid[valid])) + 1e-9
    valid &= ~(resid > max(6 * 1.4826 * mad, vox))

    med_t = float(np.nanmedian(thickness[valid]))
    merge = np.zeros((nu, nu), dtype=bool)
    merge[valid] = thickness[valid] > thickness_merge_factor * med_t

    w_fit, spline = _fit_spline(centers, centers, chosen_w, valid)
    return SheetSurface(
        origin=origin, axes=axes, u=centers.copy(), v=centers.copy(),
        w=w_fit, valid=valid, merge=merge, thickness_um=thickness,
        grid_um=grid_um, _spline=spline,
    )
