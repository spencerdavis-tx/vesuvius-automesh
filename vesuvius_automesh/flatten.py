"""Free-boundary SLIM UV parameterization for welded sheet meshes.

Recipe distilled from ThaumatoAnakalyptor's slim_uv.py (libigl 2.5 class API)
ported to the libigl 2.6.x function API (igl.slim_precompute / slim_solve),
validated on this Mac at 1M verts (~56s/iter, area-ratio median ~1.01).

dtype gotchas (libigl 2.6.2 nanobind): harmonic/boundary_loop want int64 F;
slim_precompute wants float64 F-ordered V/uv + int32 F-ordered F;
map_vertices_to_circle wants int32 boundary indices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlattenResult:
    uv: np.ndarray  # (N,2) float64, scaled so UV units ~= 3D units (voxels)
    convergence: list[float]  # relative UV motion per solve block
    area_ratio_p5: float
    area_ratio_median: float
    area_ratio_p95: float
    method: str


def _tri_areas_2d(uv: np.ndarray, F: np.ndarray) -> np.ndarray:
    e1 = uv[F[:, 1]] - uv[F[:, 0]]
    e2 = uv[F[:, 2]] - uv[F[:, 0]]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def _tri_areas_3d(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def flatten_slim(
    V: np.ndarray,
    F: np.ndarray,
    *,
    iters: int = 30,
    block: int = 5,
    converge_rel: float = 1e-4,
) -> FlattenResult:
    """Harmonic-to-circle IC + free-boundary SLIM (symmetric Dirichlet).

    V may be in any consistent 3D frame (we use zyx voxels; UVs are frame-
    agnostic up to reflection). Returns UVs rescaled to 3D units.
    Raises ValueError on degenerate input or solver collapse.
    """
    import igl

    if len(F) < 3:
        raise ValueError("too few faces to flatten")
    V_in = np.ascontiguousarray(V, dtype=np.float64)
    F_in = np.ascontiguousarray(F, dtype=np.int64)

    # Drop degenerate (zero-area) faces — they NaN the harmonic cotangent
    # weights (pilot1: 7 zero-area faces killed 6/8 components) — and
    # renumber away any verts left unreferenced.
    keep = _tri_areas_3d(V_in, F_in) > 1e-9
    if keep.sum() < 3:
        raise ValueError("all faces degenerate")
    Vr, Fr, IM, _ = igl.remove_unreferenced(V_in, F_in[keep])
    V = np.ascontiguousarray(Vr, dtype=np.float64)
    F64 = np.ascontiguousarray(Fr, dtype=np.int64)

    def _lscm_ic(bnd_arr: np.ndarray) -> np.ndarray:
        b = np.array([bnd_arr[0], bnd_arr[len(bnd_arr) // 2]], dtype=np.int64)
        bc = np.array([[0.0, 0.0], [1.0, 0.0]])
        uv, _q = igl.lscm(V, F64, b, bc)  # returns (uv, Q) — verified empirically
        if uv is None or not np.all(np.isfinite(np.asarray(uv, dtype=np.float64))):
            raise ValueError("LSCM fallback IC failed")
        return np.asarray(uv, dtype=np.float64)

    bnd = igl.boundary_loop(F64)
    if bnd is None or len(bnd) < 3:
        raise ValueError("no boundary loop — closed or non-manifold component")
    bnd_uv = igl.map_vertices_to_circle(np.asfortranarray(V), bnd.astype(np.int32))
    uv0 = igl.harmonic(V, F64, bnd.astype(np.int64), np.ascontiguousarray(bnd_uv), 1)
    if not np.all(np.isfinite(uv0)):
        uv0 = _lscm_ic(bnd)

    # Guard: harmonic IC can fold; SLIM needs a mostly-injective start. If
    # >25% of triangles are flipped, fall back to LSCM as IC.
    a2d = _tri_areas_2d(uv0, F64)
    if (np.sign(a2d) != np.sign(np.median(a2d))).mean() > 0.25:
        uv0 = _lscm_ic(bnd)

    data = igl.slim_precompute(
        np.asfortranarray(V),
        np.asfortranarray(F.astype(np.int32)),
        np.asfortranarray(uv0.astype(np.float64)),
        igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
        np.zeros(0, dtype=np.int32),
        np.asfortranarray(np.zeros((0, 2), dtype=np.float64)),
        0.0,
    )
    # igl 2.6.2's SLIMData exposes no energy; converge on relative UV motion.
    uv_best = np.asarray(uv0, dtype=np.float64)
    rel_steps: list[float] = []
    done = 0
    while done < iters:
        n = min(block, iters - done)
        uv = igl.slim_solve(data, n)
        done += n
        if uv is None or not np.all(np.isfinite(uv)):
            break  # roll back to uv_best (slim_uv.py NaN guard)
        uv = np.asarray(uv, dtype=np.float64)
        diag = float(np.linalg.norm(uv.max(axis=0) - uv.min(axis=0)))
        step = float(np.abs(uv - uv_best).mean()) / max(diag, 1e-12)
        rel_steps.append(step)
        uv_best = uv
        if step < converge_rel:
            break

    a3d = _tri_areas_3d(V, F64)
    a2d = np.abs(_tri_areas_2d(uv_best, F64))
    scale = float(np.sqrt(a3d.sum() / max(a2d.sum(), 1e-12)))
    uv_scaled = uv_best * scale

    ratio = (np.abs(_tri_areas_2d(uv_scaled, F64)) + 1e-12) / (a3d + 1e-12)
    ok = a3d > np.percentile(a3d, 1)  # ignore degenerate slivers in stats
    p5, med, p95 = np.percentile(ratio[ok], [5, 50, 95])

    # Scatter UVs back to the original vertex numbering (verts referenced
    # only by dropped degenerate faces get (0,0); no surviving face uses
    # them, so the rasterizer never reads those values).
    uv_full = np.zeros((len(V_in), 2), dtype=np.float64)
    old_idx = np.flatnonzero(np.asarray(IM) >= 0)
    uv_full[old_idx] = uv_scaled[np.asarray(IM)[old_idx]]
    return FlattenResult(
        uv=uv_full,
        convergence=rel_steps,
        area_ratio_p5=float(p5),
        area_ratio_median=float(med),
        area_ratio_p95=float(p95),
        method="harmonic+slim_sd",
    )
