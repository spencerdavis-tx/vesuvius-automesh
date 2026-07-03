# Vendored from the author's private `first_letters` renderer (module first_letters.zarrio),
# (c) 2026 Spencer Davis, published here under the MIT license of
# vesuvius-automesh. Vendored (rather than imported) so this repo is
# self-contained; only path defaults were adjusted for the public layout.
"""Multiscale zarr IO: local-or-S3 volumes by ledger id, chunk LRU cache,
world<->level coordinate transforms, trilinear sampling at float coords.

Coordinate conventions
----------------------
- All voxel arrays are (z, y, x).
- "World" coordinates are micrometers in the CT base grid frame: the center of
  CT level-0 voxel i sits at world i * voxel_um (+ per-volume origin_um).
- Pyramid level L voxel i covers level-0 voxels [i*s, (i+1)*s) with
  s = scale(L); its center sits at level-0 coordinate i*s + (s-1)/2.
- The preds grid is aligned to a CT pyramid level k, so the preds volume gets
  voxel_um = ct_base_um * 2**k and origin_um = (2**k - 1)/2 * ct_base_um.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

DEFAULT_CACHE_BYTES = 2 << 30

# Local cache roots per scroll, under $VESUVIUS_DATA_ROOT (see README);
# levels missing locally stream from S3 (anonymous) via the ledger.
_DATA_ROOT = Path(os.environ.get("VESUVIUS_DATA_ROOT", "data"))


def _cache(*parts: str) -> str:
    return str(_DATA_ROOT.joinpath("cache", *parts))


CACHE_ROOTS: dict[str, dict[str, str]] = {
    "scroll3": {
        "ct": _cache("scroll3", "ct.zarr"),
        "preds": _cache("scroll3", "preds_m7.zarr"),
    },
    "pherc0139": {
        "ct": _cache("pherc0139", "ct.zarr"),
        "preds": _cache("pherc0139", "preds_m7.zarr"),
    },
    "pherc0139_2p4": {
        "ct": _cache("pherc0139", "ct_2p4.zarr"),
        "preds": _cache("pherc0139", "preds_m7.zarr"),
        "preds_ct": _cache("pherc0139", "ct.zarr"),
    },
    "scroll1_2p4": {
        "ct": _cache("scroll1", "ct_2p4.zarr"),
    },
}

TRANSFORM_0139_2P4_URL = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/"
    "20260102150214-2.399um-0.2m-78keV-masked.zarr/transform.json"
)


@dataclass(frozen=True)
class ScrollSpec:
    ct_ledger_id: str          # CT volume textures are sampled from (render frame)
    # CT-only targets (mesh-rendered scrolls: surfaces come from traced meshes,
    # not preds) set preds_ledger_id=None; open_scroll then returns preds=None.
    preds_ledger_id: str | None = None
    preds_align_ct_level: int = 0  # preds L0 grid == preds-frame CT level k grid
    # Cross-volume targets only: the CT sharing the preds grid, plus the render
    # volume's transform.json (3x4 affine: render-CT voxel xyz -> preds/fixed
    # grid xyz). Validated for 0139 in cross_volume.py's module docstring.
    preds_ct_ledger_id: str | None = None
    transform_url: str | None = None


# Texture for the 0139 control comes from the 9.36um volume first (spec);
# its preds L0 grid == that volume's L0 grid. The 'pherc0139_2p4' target keeps
# sheet extraction on the 9.36um preds but samples texture from the 2.4um
# volume through its transform.json (Phase 2 in-domain control renders).
SCROLLS: dict[str, ScrollSpec] = {
    "scroll3": ScrollSpec("scroll3_ct_2p4um", "scroll3_preds_m7_L2", 2),
    "pherc0139": ScrollSpec("pherc0139_ct_9p4um", "pherc0139_preds_m7_L0", 0),
    "pherc0139_2p4": ScrollSpec(
        "pherc0139_ct_2p4um", "pherc0139_preds_m7_L0", 0,
        preds_ct_ledger_id="pherc0139_ct_9p4um",
        transform_url=TRANSFORM_0139_2P4_URL,
    ),
    # CT-only: GP segment meshes (volume 20230205180739, 7.91um frame) provide
    # the surfaces; its transform.json (validated by the author against 24
    # landmarks, 2026-06-10) maps new-2.4um zarr voxel xyz -> 7.91um voxel xyz
    # via p_old = M @ p_new24.
    "scroll1_2p4": ScrollSpec("scroll1_ct_2p4um_78kev"),
}


def repo_root() -> Path:
    """Repo root, derived from this file's location
    (repo/vesuvius_automesh/_vendor/first_letters/zarrio.py)."""
    return Path(__file__).resolve().parents[3]


def default_ledger_path() -> Path:
    return repo_root() / "config" / "level_ledger.json"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


class ChunkCache:
    """Byte-budgeted LRU cache of decoded chunk blocks."""

    def __init__(self, max_bytes: int = DEFAULT_CACHE_BYTES):
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.misses = 0           # chunk fetches (cache misses), for IO accounting
        self.fetched_bytes = 0    # decoded bytes fetched on misses
        self._store: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def get(self, key: tuple, fetch) -> np.ndarray:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        block = np.asarray(fetch())
        self.misses += 1
        self.fetched_bytes += block.nbytes
        self._store[key] = block
        self.current_bytes += block.nbytes
        while self.current_bytes > self.max_bytes and len(self._store) > 1:
            _, evicted = self._store.popitem(last=False)
            self.current_bytes -= evicted.nbytes
        return block


@dataclass(frozen=True)
class LevelInfo:
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    scale: float  # downsampling factor relative to this volume's level 0


class Volume:
    """A multiscale (z,y,x) volume; levels are any array-likes supporting
    numpy basic slicing plus .shape/.dtype (zarr arrays, numpy arrays, …)."""

    def __init__(
        self,
        name: str,
        levels: Mapping[int, Any],
        voxel_um: float,
        origin_um: float | np.ndarray = 0.0,
        scales: Mapping[int, float] | None = None,
        chunks_default: tuple[int, int, int] = (64, 64, 64),
    ):
        if not levels:
            raise ValueError("levels must be non-empty")
        if voxel_um <= 0:
            raise ValueError(f"voxel_um must be positive, got {voxel_um}")
        self.name = name
        self.voxel_um = float(voxel_um)
        self.origin_um = np.broadcast_to(np.asarray(origin_um, dtype=np.float64), (3,)).copy()
        self._levels = dict(levels)
        self._infos: dict[int, LevelInfo] = {}
        for lev, arr in self._levels.items():
            shape = tuple(int(s) for s in arr.shape)
            if len(shape) != 3:
                raise ValueError(f"level {lev} of {name} is not 3D: {shape}")
            chunks = tuple(int(c) for c in getattr(arr, "chunks", chunks_default))
            scale = float(scales[lev]) if scales is not None else float(2**lev)
            self._infos[lev] = LevelInfo(shape=shape, chunks=chunks, scale=scale)

    @property
    def levels(self) -> list[int]:
        return sorted(self._levels)

    def level_info(self, level: int) -> LevelInfo:
        if level not in self._infos:
            raise KeyError(f"volume {self.name} has no level {level}")
        return self._infos[level]

    def level_voxel_um(self, level: int) -> float:
        return self.voxel_um * self.level_info(level).scale

    # ---------------------------------------------------------- transforms

    def to_level(self, world_um: np.ndarray, level: int) -> np.ndarray:
        """World um -> continuous voxel coords at `level` (voxel centers at ints)."""
        s = self.level_info(level).scale
        x0 = (np.asarray(world_um, dtype=np.float64) - self.origin_um) / self.voxel_um
        return (x0 - (s - 1.0) / 2.0) / s

    def to_world(self, coords: np.ndarray, level: int) -> np.ndarray:
        s = self.level_info(level).scale
        x0 = np.asarray(coords, dtype=np.float64) * s + (s - 1.0) / 2.0
        return x0 * self.voxel_um + self.origin_um

    # ------------------------------------------------------------- reading

    def read_block(self, level: int, start, stop, cache: ChunkCache) -> np.ndarray:
        """Materialize [start, stop) at `level`, assembled from cached chunks.
        Out-of-bounds regions are zero-filled."""
        info = self.level_info(level)
        arr = self._levels[level]
        start = np.asarray(start, dtype=np.int64)
        stop = np.asarray(stop, dtype=np.int64)
        if np.any(stop <= start):
            raise ValueError(f"empty block: start={start}, stop={stop}")
        out = np.zeros(tuple(stop - start), dtype=arr.dtype)
        lo = np.maximum(start, 0)
        hi = np.minimum(stop, info.shape)
        if np.any(hi <= lo):
            return out
        cz, cy, cx = info.chunks
        for iz in range(int(lo[0] // cz), int((hi[0] - 1) // cz) + 1):
            for iy in range(int(lo[1] // cy), int((hi[1] - 1) // cy) + 1):
                for ix in range(int(lo[2] // cx), int((hi[2] - 1) // cx) + 1):
                    key = (self.name, level, iz, iy, ix)
                    c0 = np.array([iz * cz, iy * cy, ix * cx])
                    c1 = np.minimum(c0 + (cz, cy, cx), info.shape)

                    def fetch(c0=c0, c1=c1):
                        return arr[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]

                    blk = cache.get(key, fetch)
                    g0 = np.maximum(lo, c0)
                    g1 = np.minimum(hi, c1)
                    src = tuple(slice(int(a - b), int(c - b))
                                for a, b, c in zip(g0, c0, g1, strict=True))
                    dst = tuple(slice(int(a - b), int(c - b))
                                for a, b, c in zip(g0, start, g1, strict=True))
                    out[dst] = blk[src]
        return out

    # ------------------------------------------------------------ sampling

    def sample(
        self,
        world_um: np.ndarray,
        level: int,
        cache: ChunkCache,
        group_um: float = 2000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trilinear-sample at world points (N,3). Returns (values f32, valid bool).

        Points are grouped into coarse spatial cells; each group is served by a
        single block read, so memory stays bounded for scattered point sets.
        """
        world_um = np.asarray(world_um, dtype=np.float64)
        if world_um.ndim != 2 or world_um.shape[1] != 3:
            raise ValueError(f"world_um must be (N,3), got {world_um.shape}")
        info = self.level_info(level)
        coords = self.to_level(world_um, level)
        values = np.zeros(len(coords), dtype=np.float32)
        valid = np.all((coords >= 0.0) & (coords <= np.array(info.shape) - 1.0), axis=1)

        group_vox = max(2.0, group_um / self.level_voxel_um(level))
        cell = np.floor(coords / group_vox).astype(np.int64)
        # group points by coarse cell id
        order = np.lexsort((cell[:, 2], cell[:, 1], cell[:, 0]))
        cell_sorted = cell[order]
        boundaries = np.flatnonzero(np.any(np.diff(cell_sorted, axis=0) != 0, axis=1)) + 1
        for idx in np.split(order, boundaries):
            pts = coords[idx]
            start = np.floor(pts.min(axis=0)).astype(np.int64) - 1
            stop = np.floor(pts.max(axis=0)).astype(np.int64) + 3
            block = self.read_block(level, start, stop, cache)
            local = (pts - start).T
            values[idx] = map_coordinates(
                block.astype(np.float32), local, order=1, mode="nearest")
        return values, valid


# ------------------------------------------------------------------ loading


def load_ledger(path: Path | None = None) -> dict:
    path = path or default_ledger_path()
    if not path.exists():
        raise FileNotFoundError(f"level ledger not found: {path}")
    return json.loads(path.read_text())


def _local_level_complete(local_root: Path, level: int, ledger_level: dict) -> bool:
    """A local level is usable only if its file count matches the ledger's
    measured object count (an aws-sync in progress would otherwise read as
    silent zeros for missing chunks)."""
    lev_dir = local_root / str(level)
    if not (lev_dir / ".zarray").exists():
        return False
    expected = ledger_level.get("stored_objects")
    if expected is None:
        return False  # never measured -> don't trust a partial sync
    n = sum(1 for p in lev_dir.rglob("*") if p.is_file())
    return n >= int(expected)


def open_volume(
    ledger_id: str,
    *,
    ledger: dict | None = None,
    local_root: str | Path | None = None,
    voxel_um: float | None = None,
    origin_um: float | np.ndarray = 0.0,
) -> Volume:
    """Open a multiscale zarr by ledger id, per-level local-first with S3 fallback."""
    import zarr  # deferred: tests with numpy-backed volumes never need it

    ledger = ledger or load_ledger()
    entry = ledger["volumes"].get(ledger_id)
    if entry is None:
        raise KeyError(f"ledger has no volume {ledger_id}")
    local_root = Path(local_root) if local_root else None

    local_group = None
    if local_root is not None and (local_root / ".zattrs").exists():
        local_group = zarr.open_group(str(local_root), mode="r")
    s3_group = None  # opened lazily; S3 may never be needed

    levels: dict[int, Any] = {}
    scales: dict[int, float] = {}
    sources: dict[int, str] = {}
    for lev_str, lev_meta in entry["levels"].items():
        lev = int(lev_str)
        scales[lev] = float(lev_meta["scale_factor"])
        if (local_root is not None and local_group is not None
                and _local_level_complete(local_root, lev, lev_meta)):
            levels[lev] = local_group[lev_str]
            sources[lev] = "local"
        else:
            if s3_group is None:
                s3_group = zarr.open_group(
                    entry["s3_prefix"], mode="r", storage_options={"anon": True})
            levels[lev] = s3_group[lev_str]
            sources[lev] = "s3"
        ledger_shape = tuple(lev_meta["shape"])
        if tuple(levels[lev].shape) != ledger_shape:
            raise ValueError(
                f"{ledger_id} L{lev}: array shape {levels[lev].shape} != "
                f"ledger shape {ledger_shape}")

    if voxel_um is None:
        voxel_um = entry.get("base_voxel_um")
    if voxel_um is None:
        raise ValueError(f"{ledger_id}: voxel_um not in ledger; pass explicitly")
    vol = Volume(name=ledger_id, levels=levels, voxel_um=float(voxel_um),
                 origin_um=origin_um, scales=scales)
    vol.level_sources = sources  # type: ignore[attr-defined]  # provenance for meta.json
    return vol


@dataclass(frozen=True)
class ScrollData:
    name: str
    ct: Volume                  # render-frame CT (texture sampling)
    preds: Volume | None        # None for CT-only (mesh-rendered) targets
    spec: ScrollSpec
    preds_ct: Volume | None = None  # CT on the preds grid (== ct unless cross-volume)
    transform: Any = None           # CrossVolumeTransform: preds world -> ct world


def open_scroll(scroll: str, *, ledger_path: Path | None = None) -> ScrollData:
    """Open CT + preds for a named scroll; preds grid registered into the
    preds-frame CT world um. Cross-volume targets additionally carry the
    preds-grid CT and the preds-world -> render-CT-world transform.
    CT-only targets (preds_ledger_id=None) return preds=None."""
    if scroll not in SCROLLS:
        raise KeyError(f"unknown scroll {scroll!r}; known: {sorted(SCROLLS)}")
    spec = SCROLLS[scroll]
    ledger = load_ledger(ledger_path)
    roots = CACHE_ROOTS.get(scroll, {})
    ct = open_volume(spec.ct_ledger_id, ledger=ledger, local_root=roots.get("ct"))
    if spec.preds_ledger_id is None:
        return ScrollData(name=scroll, ct=ct, preds=None, spec=spec)
    if spec.preds_ct_ledger_id is None:
        preds_ct = ct
    else:
        preds_ct = open_volume(spec.preds_ct_ledger_id, ledger=ledger,
                               local_root=roots.get("preds_ct"))
    k = spec.preds_align_ct_level
    align_scale = preds_ct.level_info(k).scale
    preds = open_volume(
        spec.preds_ledger_id,
        ledger=ledger,
        local_root=roots.get("preds"),
        voxel_um=preds_ct.voxel_um * align_scale,
        origin_um=(align_scale - 1.0) / 2.0 * preds_ct.voxel_um,
    )
    ct_shape = preds_ct.level_info(k).shape
    preds_shape = preds.level_info(0).shape
    if ct_shape != preds_shape:
        raise ValueError(
            f"preds L0 grid {preds_shape} != CT L{k} grid {ct_shape} for {scroll}")
    transform = None
    if spec.transform_url is not None:
        from .cross_volume import preds_to_ct_transform, resolve_transform_path
        if "ct" not in roots:
            raise ValueError(f"{scroll}: cross-volume spec needs a 'ct' cache root "
                             "to hold transform.json")
        path = resolve_transform_path(Path(roots["ct"]) / "transform.json",
                                      url=spec.transform_url)
        transform = preds_to_ct_transform(
            path,
            preds_voxel_um=preds.voxel_um, ct_voxel_um=ct.voxel_um,
            preds_origin_um=preds.origin_um, ct_origin_um=ct.origin_um,
        )
    return ScrollData(name=scroll, ct=ct, preds=preds, spec=spec,
                      preds_ct=preds_ct, transform=transform)
