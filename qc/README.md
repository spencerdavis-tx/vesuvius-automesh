# Per-window QC records — Scroll 3 harvest (all 157 rendered windows)

One JSON row per rendered 25 mm window (passes and failures both), compiled from each
window's `qc.json`/`meta.json` and the topology re-gate ledger:

- `texture_gates` — the four calibrated texture/geometry gates (coherence_ratio,
  band_contrast, merge_fraction, mask_fraction) with per-gate booleans and the tier verdict.
- `recenter_found_fraction` — the surface-lock test (threshold 0.9).
- `render_qc_accepted` — tier-1 verdict (all four gates AND found_fraction ≥ 0.9);
  106 windows / 409.27 cm² pass.
- `topology_regate` — tier-2 independent verification for the 106 tier-1-accepted windows
  (prediction-support density ≥ 0.12, winding-walk backtrack ≤ 10°, window-level surface
  scoring cov@50 µm ≥ 0.80 AND mean distance ≤ 30 µm) with verdict VERIFIED /
  REJECTED-REGATE / UNVERIFIABLE; 61 windows / 279.38 cm² verify. `null` for windows the
  render-quality tier already rejected.

Totals: 157 windows, 586.7 cm² rendered; 106 / 409.27 cm² render-QC-accepted; 61 /
279.38 cm² verified (the headline number). Note: an early window-grid change in patch
…955704 left 7 duplicate window renders on disk; they are excluded here (157 = the
harvest ledger's rendered-window count).
