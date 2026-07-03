"""Welded-OBJ loading and connected-component splitting.

Scrollfiesta's welded.obj stores vertex lines as (z, y, x) world voxel
coordinates in the preds grid (verified: scrollfiesta README L125-126 +
src/common/dump_obj.c L263-266). We keep zyx order throughout — it matches
zarr index order and the first_letters sampling contract (pos arrays are zyx
micrometers).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Component:
    """One connected component of a welded mesh, vertices renumbered."""

    verts_zyx: np.ndarray  # (N,3) float64, preds-grid voxel coords
    faces: np.ndarray  # (M,3) int64
    n_faces_total: int  # face count of the parent mesh (for accounting)


def load_obj_zyx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OBJ; returns (verts (N,3) float64 as stored, faces (M,3) int64 0-based).

    Handles 'f a b c', 'f a/b/c ...' and ignores everything else. Vertex rows
    are returned exactly as stored in the file (scrollfiesta: z y x). 'v'
    lines may carry per-vertex RGB (grid_weld writes 'v z y x r g b') — only
    the first 3 floats are taken.
    """
    verts: list[bytes] = []
    faces: list[bytes] = []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"v "):
                verts.append(b" ".join(line[2:].split()[:3]))
            elif line.startswith(b"f "):
                faces.append(line[2:])
    if not verts or not faces:
        raise ValueError(f"{path}: no verts or faces (v={len(verts)} f={len(faces)})")
    V = np.loadtxt(verts, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"{path}: unexpected vertex array shape {V.shape}")
    first = faces[0].split()
    if b"/" in first[0]:
        F = np.array(
            [[int(tok.split(b"/")[0]) for tok in ln.split()[:3]] for ln in faces],
            dtype=np.int64,
        )
    else:
        F = np.loadtxt(faces, dtype=np.int64).reshape(-1, 3)
    if F.min() < 1:
        raise ValueError(f"{path}: OBJ face indices must be 1-based, got min {F.min()}")
    return V, F - 1


def split_components(
    V: np.ndarray, F: np.ndarray, *, min_faces: int = 500
) -> list[Component]:
    """Split into face-connected components, largest first; drop tiny ones."""
    import igl

    _, C = igl.facet_components(F.astype(np.int64))
    comps: list[Component] = []
    for ci in np.unique(C):
        Fc = F[C == ci]
        if len(Fc) < min_faces:
            continue
        used = np.unique(Fc)
        remap = np.full(len(V), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        comps.append(
            Component(
                verts_zyx=np.ascontiguousarray(V[used]),
                faces=remap[Fc],
                n_faces_total=len(F),
            )
        )
    comps.sort(key=lambda c: len(c.faces), reverse=True)
    return comps


def mesh_area_voxels(verts: np.ndarray, faces: np.ndarray) -> float:
    """Total triangle area in (voxel-unit)^2 of the verts' own frame."""
    p0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - p0
    e2 = verts[faces[:, 2]] - p0
    return float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())
