"""Regenerate the phantom-halo overlay figure with color legend + scale bar.

Same data and color semantics as tracks/auto-mesh/survey/slice_L2z1056_cubez33_mid.png:
- blue  = masked CT intensity (CT L4, 9.596*4 = 38.384 um/px grid)
- green = m7 surface preds where masked CT > 5 (CT-supported)
- red   = m7 surface preds where masked CT <= 5 (phantom: prediction on zero CT)
Axial plane z=1056 on the preds-L2 / CT-L4 grid ([2100, 986, 986]).

Additions vs the survey original: 2x nearest upscale, legend, 1 cm scale bar,
title/caption strip. Output is a drop-in replacement for
figures/phantom_halo_slice_L2z1056.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

import os

_DATA = Path(os.environ.get("VESUVIUS_DATA_ROOT", "data"))
CT_ZARR = _DATA / "cache" / "scroll3" / "ct.zarr"
PREDS_ZARR = _DATA / "cache" / "scroll3" / "preds_m7.zarr"
Z = 1056
UM_PER_PX = 9.596 * 4  # preds L2 == CT L4 grid
SCALE = 2  # upscale factor for legibility
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

GREEN = (0, 230, 90)
RED = (255, 40, 40)
BLUE_MAX = 235


def build_overlay() -> np.ndarray:
    ct = np.asarray(zarr.open(str(CT_ZARR), mode="r")["4"][Z])
    preds = np.asarray(zarr.open(str(PREDS_ZARR), mode="r")["2"][Z])
    pos = preds > 0
    supported = pos & (ct > 5)
    phantom = pos & (ct <= 5)

    rgb = np.zeros((*ct.shape, 3), dtype=np.uint8)
    # CT as blue background (perceptual stretch: sqrt)
    b = (np.sqrt(ct / 255.0) * BLUE_MAX).astype(np.uint8)
    rgb[..., 2] = b
    rgb[supported] = GREEN
    rgb[phantom] = RED

    frac = phantom.sum() / pos.sum()
    print(f"plane z={Z}: pos={pos.sum()} phantom_fraction={frac:.4f}")
    return rgb


def main(out_path: str) -> None:
    rgb = build_overlay()
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb).resize((w * SCALE, h * SCALE), Image.NEAREST)
    W, H = img.size

    title_h = 84
    canvas = Image.new("RGB", (W, H + title_h), (0, 0, 0))
    canvas.paste(img, (0, title_h))
    d = ImageDraw.Draw(canvas)
    f_title = ImageFont.truetype(FONT_BOLD, 30)
    f_sub = ImageFont.truetype(FONT, 22)
    f_leg = ImageFont.truetype(FONT, 24)

    d.text((16, 10), "Phantom halo: m7 surface predictions vs masked CT "
                     "(PHerc0332, axial z=1056, preds-L2 grid)",
           font=f_title, fill=(255, 255, 255))
    d.text((16, 48), "~67% of positive prediction voxels on this plane sit where "
                     "the SAM2-masked CT volume is exactly 0",
           font=f_sub, fill=(200, 200, 200))

    # Legend (top-left corner of the image area, which is empty/black)
    lx, ly = 24, title_h + 20
    entries = [
        (RED, "prediction on zero CT (phantom halo)"),
        (GREEN, "prediction, CT-supported (papyrus surface)"),
        ((70, 70, BLUE_MAX), "masked CT intensity (blue)"),
    ]
    box_w = 26
    pad = 12
    leg_h = len(entries) * 40 + 2 * pad
    leg_w = max(d.textlength(t, font=f_leg) for _, t in entries) + box_w + 3 * pad
    d.rectangle([lx - pad, ly - pad, lx - pad + leg_w, ly - pad + leg_h],
                fill=(15, 15, 15), outline=(120, 120, 120))
    for i, (color, label) in enumerate(entries):
        y = ly + i * 40
        d.rectangle([lx, y + 2, lx + box_w, y + box_w + 2], fill=color,
                    outline=(255, 255, 255))
        d.text((lx + box_w + pad, y), label, font=f_leg, fill=(255, 255, 255))

    # Scale bar: 1 cm (bottom-left)
    px_per_cm = 10_000 / UM_PER_PX * SCALE  # 10 mm in upscaled px
    bar_len = int(round(px_per_cm))
    bx, by = 30, H + title_h - 70
    d.rectangle([bx, by, bx + bar_len, by + 10], fill=(255, 255, 255))
    d.rectangle([bx, by - 8, bx + 4, by + 18], fill=(255, 255, 255))
    d.rectangle([bx + bar_len - 4, by - 8, bx + bar_len, by + 18],
                fill=(255, 255, 255))
    d.text((bx + bar_len // 2 - 40, by - 40), "1 cm", font=f_leg,
           fill=(255, 255, 255))
    d.text((bx, by + 22), f"{UM_PER_PX:.1f} um/px (native grid; shown 2x)",
           font=ImageFont.truetype(FONT, 18), fill=(180, 180, 180))

    canvas.save(out_path)
    print("wrote", out_path, canvas.size)


if __name__ == "__main__":
    main(sys.argv[1])
