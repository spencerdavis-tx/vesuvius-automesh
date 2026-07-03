# Vendored from the author's private `first_letters` renderer (module first_letters.recenter),
# (c) 2026 Spencer Davis, published here under the MIT license of
# vesuvius-automesh. Vendored (rather than imported) so this repo is
# self-contained; only path defaults were adjusted for the public layout.
"""Subvoxel re-centering: snap each surface grid point to the CT bright-band
center along its normal (RENDERER_SPEC.md step 3).

For every valid heightfield bin we sample the CT intensity profile within
+/-halfspan_um along the local normal, threshold it at the midpoint of its
10th/90th percentiles, take the contiguous above-threshold band nearest the
current surface, and move to that band's intensity-weighted centroid
(weights = intensity - threshold, clipped at 0, to suppress baseline bias —
implementation choice; the spec only says "intensity-weighted centroid").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sheet import SheetSurface
from .zarrio import ChunkCache, Volume


@dataclass(frozen=True)
class RecenterResult:
    surface: SheetSurface       # adjusted surface (spline refit)
    shift_um: np.ndarray        # (Nu, Nv) signed shift along the normal; NaN if none
    adjusted: np.ndarray        # (Nu, Nv) bool: a band was found and applied

    @property
    def shift_stats(self) -> dict:
        s = self.shift_um[self.adjusted]
        if s.size == 0:
            return {"n": 0}
        return {
            "n": int(s.size),
            "mean_um": float(np.mean(s)),
            "median_abs_um": float(np.median(np.abs(s))),
            "p95_abs_um": float(np.percentile(np.abs(s), 95)),
            "max_abs_um": float(np.max(np.abs(s))),
        }


def _band_centroid(profile: np.ndarray, s_um: np.ndarray,
                   min_range: float = 5.0) -> tuple[float, float] | None:
    """(centroid_um, thickness_um) of the above-threshold band nearest 0."""
    p10, p90 = np.percentile(profile, [10, 90])
    if p90 - p10 < min_range:
        return None  # flat profile: all papyrus or all air; nothing to center on
    thr = 0.5 * (p10 + p90)
    above = profile > thr
    if not above.any():
        return None
    # contiguous runs of `above`
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    starts = np.concatenate([[0] if above[0] else [], edges[~above[edges]] + 1]).astype(int)
    stops = np.concatenate([edges[above[edges]] + 1, [len(above)] if above[-1] else []]
                           ).astype(int)
    best: tuple[float, int, int] | None = None
    for s, e in zip(starts, stops, strict=True):
        d = 0.0 if s_um[s] <= 0.0 <= s_um[e - 1] else min(abs(s_um[s]), abs(s_um[e - 1]))
        if best is None or d < best[0]:
            best = (d, s, e)
    _, s, e = best
    w = np.clip(profile[s:e] - thr, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        return None
    step = float(s_um[1] - s_um[0]) if len(s_um) > 1 else 0.0
    return float(np.sum(s_um[s:e] * w) / total), (e - s) * step


def recenter_surface(
    surface: SheetSurface,
    ct: Volume,
    *,
    level: int,
    cache: ChunkCache,
    halfspan_um: float = 60.0,
    step_um: float | None = None,
    max_shift_um: float | None = None,
) -> RecenterResult:
    """Re-center every valid grid point of `surface` on the CT bright band."""
    if halfspan_um <= 0:
        raise ValueError(f"halfspan_um must be positive, got {halfspan_um}")
    step = step_um if step_um is not None else ct.level_voxel_um(level)
    if step <= 0:
        raise ValueError(f"step_um must be positive, got {step}")
    max_shift = max_shift_um if max_shift_um is not None else halfspan_um / 2.0

    nu, nv = surface.w.shape
    uu, vv = np.meshgrid(surface.u, surface.v, indexing="ij")
    pts = surface.points(uu, vv)            # (Nu, Nv, 3)
    nrm = surface.normals(uu, vv)           # (Nu, Nv, 3)
    s_um = np.arange(-halfspan_um, halfspan_um + step / 2, step)
    ns = len(s_um)

    bins = np.argwhere(surface.valid)
    coords = (pts[bins[:, 0], bins[:, 1], None, :]
              + s_um[None, :, None] * nrm[bins[:, 0], bins[:, 1], None, :])
    values, ok = ct.sample(coords.reshape(-1, 3), level, cache)
    values = values.reshape(len(bins), ns)
    ok = ok.reshape(len(bins), ns)

    shift = np.full((nu, nv), np.nan)
    adjusted = np.zeros((nu, nv), dtype=bool)
    for (i, j), profile, prof_ok in zip(bins, values, ok, strict=True):
        if not prof_ok.all():
            continue  # profile leaves the volume; leave the point as-is
        band = _band_centroid(profile.astype(np.float64), s_um)
        if band is None or abs(band[0]) > max_shift:
            continue
        shift[i, j] = band[0]
        adjusted[i, j] = True

    # shift moves the point along the normal; convert to a height change via the
    # normal's w-component (u,v drift is second order for near-W normals).
    n_w = nrm @ surface.axes[2]
    w_new = np.where(adjusted, surface.w + np.nan_to_num(shift) * n_w, surface.w)
    return RecenterResult(
        surface=surface.with_heights(w_new),
        shift_um=shift,
        adjusted=adjusted,
    )
