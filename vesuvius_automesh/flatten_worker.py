"""Subprocess entry for crash-isolated flattening.

igl can segfault outright on pathological welded geometry (observed: silent
death of a whole render batch on a merged-blob component). Running each
flatten in a child process contains native crashes: the parent sees a
nonzero/signal exit instead of dying.

CLI: python -m vesuvius_automesh.flatten_worker <in.npz> <out.npz>
  in.npz:  V (N,3) f64, F (M,3) i64, max_slim_verts ()
  out.npz: uv (N,2) f64, area_ratio_p5/median/p95 (), method (str), convergence
"""
from __future__ import annotations

import sys

import numpy as np


def flatten_for_npz(V: np.ndarray, F: np.ndarray, max_slim_verts: int):
    import igl

    from .flatten import FlattenResult, flatten_slim

    if len(V) <= max_slim_verts:
        return flatten_slim(V, F)

    n_target = int(len(F) * (max_slim_verts / len(V)))
    ok, Vd, Fd, _, _ = igl.qslim(V, F.astype(np.int64), n_target)
    if not ok or len(Fd) < 100:
        raise ValueError(f"qslim failed (ok={ok}, faces={len(Fd)})")
    flat = flatten_slim(Vd, Fd)
    _, fi, cp = igl.point_mesh_squared_distance(V, Vd, Fd.astype(np.int64))
    tri = Vd[Fd[fi]]
    bary = igl.barycentric_coordinates_tri(
        np.ascontiguousarray(cp),
        np.ascontiguousarray(tri[:, 0]),
        np.ascontiguousarray(tri[:, 1]),
        np.ascontiguousarray(tri[:, 2]),
    )
    uv_full = np.einsum("nk,nkd->nd", bary, flat.uv[Fd[fi]])
    return FlattenResult(
        uv=uv_full,
        convergence=flat.convergence,
        area_ratio_p5=flat.area_ratio_p5,
        area_ratio_median=flat.area_ratio_median,
        area_ratio_p95=flat.area_ratio_p95,
        method=flat.method + f"+qslim{len(Vd)}",
    )


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    d = np.load(in_path)
    flat = flatten_for_npz(d["V"], d["F"], int(d["max_slim_verts"]))
    np.savez(
        out_path,
        uv=flat.uv,
        area_ratio_p5=flat.area_ratio_p5,
        area_ratio_median=flat.area_ratio_median,
        area_ratio_p95=flat.area_ratio_p95,
        method=np.str_(flat.method),
        convergence=np.asarray(flat.convergence, dtype=np.float64),
    )


if __name__ == "__main__":
    main()
