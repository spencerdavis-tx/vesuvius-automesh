# Vendored from the author's private `first_letters` renderer (module first_letters.qc),
# (c) 2026 Spencer Davis, published here under the MIT license of
# vesuvius-automesh. Vendored (rather than imported) so this repo is
# self-contained; only path defaults were adjusted for the public layout.
"""Patch QC + kill-gate-1 evaluation (RENDERER_SPEC.md step 5).

Gate (decidable, no eyeballing): orientation coherence within 2x of the 343P
reference, band-profile contrast >= 1.05, merge-flag fraction < 45%, valid
mask >= 60%. Written to qc.json by `write_qc`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

Image.MAX_IMAGE_PIXELS = None  # 343P reference layer is 13924 x 8416

# Kill-gate-1 thresholds: engineering QC calibrated on the PHerc0139 control
# (known-ink in-situ scroll, organizer preds on its own 9.36um grid; 7 clean
# 5mm probe patches, 2026-06-10,
# results/first-letters/phase1/probe_pherc0139_control*):
#   coherence_ratio 1.555..1.905            -> 2x factor passed 7/7, kept
#   band_contrast   min 1.085, median 1.312 -> old 1.3 failed 4/7 clean patches
#   merge_fraction  median 0.202, max 0.403 -> old 0.10 failed 6/7
#   mask_fraction   min 0.671, median 0.864 -> old 0.90 failed 4/7
# In-situ wraps are tightly packed: outer layers hit neighboring papyrus, not
# air, so band contrast is structurally lower than on detached-fragment
# renders (flat/smeared renders still sit at ~1.0); real sheets carry cracks
# and holes, so sub-0.9 valid masks are normal; and the 1.5x-median-thickness
# merge flag fires routinely where wraps touch. The original thresholds were
# fragment-render guesses, not control-calibrated.
GATE_COHERENCE_FACTOR = 2.0
GATE_BAND_CONTRAST = 1.05
GATE_MERGE_FRACTION = 0.45
GATE_MASK_FRACTION = 0.60


def band_profile_contrast(layers: np.ndarray, mask: np.ndarray) -> float:
    """Mean center-layer intensity over mean of the outermost layers (0 and -1).
    Smeared renders flatten this toward 1."""
    if layers.ndim != 3:
        raise ValueError(f"layers must be (L,H,W), got {layers.shape}")
    if not mask.any():
        return 0.0
    center = layers[layers.shape[0] // 2].astype(np.float64)[mask].mean()
    outer = 0.5 * (layers[0].astype(np.float64)[mask].mean()
                   + layers[-1].astype(np.float64)[mask].mean())
    return float(center / max(outer, 1e-9))


def _coherence(img: np.ndarray, tensor_sigma_px: float) -> float:
    """Mean structure-tensor coherence (lam1-lam2)/(lam1+lam2) in [0,1]."""
    gy, gx = np.gradient(img.astype(np.float64))
    jyy = gaussian_filter(gy * gy, tensor_sigma_px)
    jxx = gaussian_filter(gx * gx, tensor_sigma_px)
    jxy = gaussian_filter(gx * gy, tensor_sigma_px)
    disc = np.sqrt((jyy - jxx) ** 2 + 4.0 * jxy**2)
    trace = jyy + jxx
    coh = np.where(trace > 1e-12, disc / (trace + 1e-12), 0.0)
    return float(coh.mean())


def _correlation_length_um(img: np.ndarray, px_um: float) -> float:
    """Radius (um) where the radially-averaged autocorrelation drops below 1/e."""
    f = img.astype(np.float64) - img.mean()
    acf = np.fft.irfft2(np.abs(np.fft.rfft2(f)) ** 2, s=f.shape)
    acf /= acf.flat[0]
    h, w = f.shape
    yy = np.minimum(np.arange(h), h - np.arange(h))[:, None]
    xx = np.minimum(np.arange(w), w - np.arange(w))[None, :]
    r = np.sqrt(yy**2 + xx**2).astype(np.int64)
    rmax = min(h, w) // 2
    prof = np.array([acf[r == k].mean() for k in range(rmax)])
    below = np.flatnonzero(prof < 1.0 / np.e)
    return float((below[0] if below.size else rmax) * px_um)


def orientation_stats(
    img: np.ndarray,
    *,
    px_um: float,
    tensor_sigma_um: float = 12.0,
    window_px: int = 1024,
    n_windows: int = 4,
) -> dict:
    """Fiber-texture stats of a 2D layer: orientation coherence + autocorrelation
    length. Averaged over up to `n_windows` deterministic interior windows."""
    if img.ndim != 2:
        raise ValueError(f"img must be 2D, got {img.shape}")
    if px_um <= 0:
        raise ValueError(f"px_um must be positive, got {px_um}")
    h, w = img.shape
    win = min(window_px, h, w)
    sigma_px = max(1.0, tensor_sigma_um / px_um)
    centers = [(h // 2, w // 2), (h // 4, w // 4), (h // 4, 3 * w // 4),
               (3 * h // 4, w // 2)][:n_windows]
    cohs, lens = [], []
    for cy, cx in centers:
        y0 = int(np.clip(cy - win // 2, 0, h - win))
        x0 = int(np.clip(cx - win // 2, 0, w - win))
        crop = img[y0:y0 + win, x0:x0 + win]
        cohs.append(_coherence(crop, sigma_px))
        lens.append(_correlation_length_um(crop, px_um))
    return {
        "coherence": float(np.mean(cohs)),
        "corr_length_um": float(np.mean(lens)),
        "window_px": int(win),
        "n_windows": len(centers),
        "px_um": px_um,
    }


def reference_stats(path: Path | str, px_um: float = 2.0) -> dict:
    """Stats of the known-good 2um render (PHerc0343P layer 32)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reference layer not found: {path}")
    img = np.asarray(Image.open(path).convert("L"))
    stats = orientation_stats(img, px_um=px_um)
    stats["source"] = str(path)
    return stats


def evaluate_patch(
    layers: np.ndarray,
    mask: np.ndarray,
    *,
    shift_um: np.ndarray,
    merge_fraction: float,
    ref: dict,
    px_um: float,
) -> dict:
    """Compute QC metrics + kill-gate-1 booleans for a rendered patch."""
    center = layers[layers.shape[0] // 2]
    stats = orientation_stats(center, px_um=px_um)
    contrast = band_profile_contrast(layers, mask)
    mask_fraction = float(mask.mean())
    coh_ratio = stats["coherence"] / max(ref["coherence"], 1e-9)

    finite = shift_um[np.isfinite(shift_um)]
    hist_counts, hist_edges = np.histogram(
        finite, bins=24, range=(-60.0, 60.0)) if finite.size else (np.zeros(24), np.zeros(25))

    gates = {
        "coherence_within_2x": bool(
            1.0 / GATE_COHERENCE_FACTOR <= coh_ratio <= GATE_COHERENCE_FACTOR),
        "band_contrast_ge_1p05": bool(contrast >= GATE_BAND_CONTRAST),
        "merge_fraction_lt_45pct": bool(merge_fraction < GATE_MERGE_FRACTION),
        "valid_mask_ge_60pct": bool(mask_fraction >= GATE_MASK_FRACTION),
    }
    return {
        "coherence": stats["coherence"],
        "coherence_ref": ref["coherence"],
        "coherence_ratio": float(coh_ratio),
        "corr_length_um": stats["corr_length_um"],
        "corr_length_ref_um": ref.get("corr_length_um"),
        "band_contrast": contrast,
        "merge_fraction": float(merge_fraction),
        "mask_fraction": mask_fraction,
        "recenter_shift_hist": {
            "counts": [int(c) for c in hist_counts],
            "edges_um": [float(e) for e in hist_edges],
            "n_adjusted": int(finite.size),
            "median_abs_um": float(np.median(np.abs(finite))) if finite.size else None,
        },
        "gates": gates,
        "pass": bool(all(gates.values())),
    }


def write_qc(out_dir: Path | str, qc: dict, layers: np.ndarray,
             mask: np.ndarray) -> None:
    """Write qc.json + a 4x-downsampled layer-32 preview PNG (preview only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qc.json").write_text(json.dumps(qc, indent=2))
    center = layers[layers.shape[0] // 2]
    preview = center[::4, ::4].copy()
    preview[~mask[::4, ::4]] = 0
    Image.fromarray(preview).save(out_dir / "preview.png")
