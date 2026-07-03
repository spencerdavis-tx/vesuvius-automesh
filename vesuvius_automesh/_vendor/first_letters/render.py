# Vendored from the author's private `first_letters` renderer (module first_letters.render),
# (c) 2026 Spencer Davis, published here under the MIT license of
# vesuvius-automesh. Vendored (rather than imported) so this repo is
# self-contained; only path defaults were adjusted for the public layout.
"""Layer sampling: flattened surface-volume patches in the villa ink-detection
layout (RENDERER_SPEC.md step 4).

66 layers along the normal, layer pitch = native CT voxel, layer `center_layer`
(32 by villa convention) on the sheet center. u,v pixel spacing = layer pitch
(isotropic render). Intensities are trilinear samples from the CT pyramid at
the requested level (already windowed to uint8 in the masked zarrs — pass
through). Output: layers/{NN}.tif (LZW) + <id>_mask.png + meta.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .sheet import SheetSurface
from .zarrio import ChunkCache, Volume


@dataclass(frozen=True)
class RenderResult:
    layers: np.ndarray   # (n_layers, H, W) uint8
    mask: np.ndarray     # (H, W) bool — all layers sampled in-bounds + surface valid
    u_um: np.ndarray     # (H,) frame-u of each row
    v_um: np.ndarray     # (W,) frame-v of each column
    pitch_um: float
    level: int

    @property
    def mask_fraction(self) -> float:
        return float(self.mask.mean())


def render_surface(
    surface: SheetSurface,
    ct: Volume,
    *,
    level: int,
    pitch_um: float,
    n_layers: int = 66,
    center_layer: int = 32,
    cache: ChunkCache,
    tile_px: int = 512,
) -> RenderResult:
    """Sample n_layers along surface normals onto an isotropic (u,v) pixel grid.

    The output grid is cropped to the bounding box of the surface's valid
    heightfield bins (its actual footprint): a partially extracted surface
    yields a smaller patch rather than a fixed grid padded with invalid pixels.
    """
    if pitch_um <= 0:
        raise ValueError(f"pitch_um must be positive, got {pitch_um}")
    if not (0 <= center_layer < n_layers):
        raise ValueError(f"center_layer {center_layer} outside [0, {n_layers})")
    if not surface.valid.any():
        raise ValueError("surface has no valid heightfield bins to render")

    vi = np.flatnonzero(surface.valid.any(axis=1))
    vj = np.flatnonzero(surface.valid.any(axis=0))
    u_px = np.arange(surface.u[vi[0]], surface.u[vi[-1]] + pitch_um / 2, pitch_um)
    v_px = np.arange(surface.v[vj[0]], surface.v[vj[-1]] + pitch_um / 2, pitch_um)
    h, w = len(u_px), len(v_px)
    layers = np.zeros((n_layers, h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    offsets = (np.arange(n_layers, dtype=np.float64) - center_layer) * pitch_um

    # surface-bin validity, nearest-bin lookup for each output pixel
    bin_i = np.clip(np.round((u_px - surface.u[0]) / surface.grid_um).astype(int),
                    0, len(surface.u) - 1)
    bin_j = np.clip(np.round((v_px - surface.v[0]) / surface.grid_um).astype(int),
                    0, len(surface.v) - 1)

    for ti in range(0, h, tile_px):
        for tj in range(0, w, tile_px):
            su = u_px[ti:ti + tile_px]
            sv = v_px[tj:tj + tile_px]
            uu, vv = np.meshgrid(su, sv, indexing="ij")
            pts = surface.points(uu, vv)          # (th, tw, 3)
            nrm = surface.normals(uu, vv)
            th, tw = uu.shape
            coords = (pts[None, :, :, :] + offsets[:, None, None, None] * nrm[None])
            vals, ok = ct.sample(coords.reshape(-1, 3), level, cache)
            vals = vals.reshape(n_layers, th, tw)
            ok = ok.reshape(n_layers, th, tw)
            layers[:, ti:ti + th, tj:tj + tw] = np.clip(np.rint(vals), 0, 255
                                                        ).astype(np.uint8)
            tile_valid = ok.all(axis=0)
            tile_valid &= surface.valid[bin_i[ti:ti + th][:, None],
                                        bin_j[tj:tj + tw][None, :]]
            mask[ti:ti + th, tj:tj + tw] = tile_valid

    return RenderResult(layers=layers, mask=mask, u_um=u_px, v_um=v_px,
                        pitch_um=pitch_um, level=level)


def write_patch(out_dir: Path, segment_id: str, result: RenderResult,
                meta: dict) -> Path:
    """Write villa layout: <out>/<id>/layers/{NN}.tif + <id>_mask.png + meta.json."""
    seg_dir = Path(out_dir) / segment_id
    layers_dir = seg_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(result.layers):
        tifffile.imwrite(layers_dir / f"{i:02d}.tif", layer, compression="lzw")
    mask_img = (result.mask.astype(np.uint8)) * 255
    Image.fromarray(mask_img).save(seg_dir / f"{segment_id}_mask.png")
    full_meta = {
        **meta,
        "segment_id": segment_id,
        "n_layers": int(result.layers.shape[0]),
        "height_px": int(result.layers.shape[1]),
        "width_px": int(result.layers.shape[2]),
        "pitch_um": result.pitch_um,
        "ct_level": result.level,
        "mask_fraction": result.mask_fraction,
    }
    (seg_dir / "meta.json").write_text(json.dumps(full_meta, indent=2))
    return seg_dir
