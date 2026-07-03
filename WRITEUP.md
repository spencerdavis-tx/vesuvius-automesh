# 409 cm² of QC-passing Scroll 3 (PHerc0332) surface with zero manual annotation and $0 GPU — plus a bug report on the public m7 surface predictions

## TL;DR

We built a fully automated chain — organizer surface-prediction zarr → CT-support masking →
`vc_grow_seg_from_seed` tracing → per-window 66-layer rendering → two-part quantitative
acceptance — that produced **409.3 cm² of QC-passing rendered surface on Scroll 3
(PHerc0332)**, about **6.1× the ~67 cm² of existing human-traced segments**, with **zero
manual annotation clicks and zero GPU spend** (Apple-silicon Mac CPU + anonymous S3
streaming only).

Along the way we found two things we believe are useful to others independently of the
area number:

1. **Bug report:** ~70% of the positive voxels in the public Scroll 3 m7 surface-prediction
   zarr are phantoms — solid prediction slabs sitting where the SAM2-masked CT volume is
   exactly 0 (a halo ring around the scroll plus end caps). The villa tracer is CT-blind
   and happily traces these shells. Numbers and a 10-line reproduction below.
2. **A two-part acceptance gate** (4 calibrated texture/geometry gates + a recenter
   `found_fraction ≥ 0.9` rule) that catches a failure mode the texture gates alone
   provably miss: locally coherent renders of scroll *cross-sections*. Calibration
   numbers below.

All code, configs, and per-window QC JSON are included. Engineer-to-engineer writeup;
no claims about ink or text — this is renderable *surface*, nothing more.

---

## 1. Context and motivation

The June 2026 PHerc. 1667 result (Angelotti et al.) makes clear that automated geometry —
segmentation/tracing of the papyrus surface — is the binding constraint on reading more
scrolls: 1667 still cost on the order of tens of hours of manual annotation per wrap, and
the released surface models leave a large gap to close. Scroll 3 (PHerc0332) has a public
2.4 µm scan and public organizer surface predictions, but (as of mid-June 2026) only
~67 cm² of human-traced, renderable segments on the data server (the two `samples` OBJs,
33.5 + 34.2 cm²).

Question we set out to answer: **how much QC-clean renderable surface can be harvested
from the *existing public* predictions with no human in the loop and no GPU?**

## 2. What was built

Pipeline (all steps programmatic, Mac CPU only, CT streamed anonymously from S3):

```
m7 preds zarr ──► CT-support masking ──► normal grids ──► seed-swept tracer ──► per-window
 (public S3)      (preds & CT>5)         (vc_gen_          (vc_grow_seg_        66-layer render
                                          normalgrids)      from_seed)          + 2-part QC gate
```

1. **`build_supported_preds.py`** — sparse voxel-wise mask of the preds zarr against the
   CT (`preds & (CT > 5)`), producing `preds_supported.zarr`. This keeps 30.2% of positive
   voxels — see the bug report (§4) for why the other 70% must go.
2. **`vc_gen_normalgrids`** (villa volume-cartographer) on the supported volume, 3 slice
   directions, ~25 min. Not optional: without normal grids the tracer's surface energy
   relaxes across wraps (villa docs say so; we confirmed by A/B — an early no-grids sweep
   produced 235 cm² of traces that were REJECTED wholesale by render-QC).
3. **`tracer_sweep.py`** — seed-sweep orchestrator around `vc_grow_seg_from_seed`: seeds
   drawn only from CT-supported preds voxels in "good" cubes (off-swirl, outside existing
   traced segments via KD-tree), net-new-area accounting to avoid re-tracing the same
   sheet, 12 concurrent single-core traces. Tracing is nearly free: minutes per patch.
4. **`render_tifxyz.py`** — the tracer's tifxyz output (x/y/z coordinate TIFFs on a quad
   grid) *is already a position grid*: upsample coordinates to the render pitch, derive
   normals from grid derivatives, then the standard recenter → 66-layer render. No
   flattening or rasterization stage needed. Renders are villa layout
   (`<id>/layers/00..65.tif` + mask + `meta.json` + `qc.json`), pitch 4.798 µm (CT level 1).
5. **`render_queue.py`** — daemon that renders patches in 25 mm windows and accepts area
   **per window** under the two-part gate (§5), maintaining a cumulative ledger.

The harvest framing is the design decision that made this work: the tracer free-grows past
data support in regional "continents" (per-patch on-sheet fraction ~40–70%), and rather
than fight that, we accept or reject *windows*. A wandering trace wastes trace area but
cannot poison an accepted window.

## 3. Result

| Metric | Value |
|---|---|
| Traced patches | 17 |
| Windows rendered (25 mm) | 157 |
| Windows accepted | 106 (68%) |
| Masked area rendered | 586.7 cm² |
| **Area accepted** | **409.3 cm²** |
| Area yield | 70% |
| Prior human-traced baseline | ~67 cm² (6.1×) |
| Median recenter found_fraction, passing windows | 0.989 |
| GPU spend | $0 |
| Manual annotation | 0 clicks |

Per-patch accepted area is bimodal: 7 patches in clean outer-wrap territory contributed
45–52 cm² each; 2 patches wandered entirely (0 cm²); the rest 2–18 cm². Seed placement
drives this.

Figures (in `figures/`):

- `render_preview_patch715_w0256_0128.png` — accepted window, 6.23 cm²,
  found_fraction 0.989. Clean cross-hatch papyrus fiber weave.
- `render_preview_patch589_w0129_0258.png` — accepted window, 6.22 cm²,
  found_fraction 0.996.
- `phantom_halo_slice_L2z1056.png` — axial cross-section overlay used in the phantom-halo
  forensics: the on-sheet prediction tracery follows the spiral wraps, while the thick
  ring *outside* the scroll body is prediction output sitting where the masked CT is
  exactly 0.

![accepted window, patch 715](figures/render_preview_patch715_w0256_0128.png)

![phantom halo overlay](figures/phantom_halo_slice_L2z1056.png)

Full per-window QC (`qc.json` with all gate values and the recenter shift histogram) ships
with every render; `harvest_ledger.json` is the cumulative ledger.

## 4. Contribution A — bug report: phantom positives in the public m7 surface predictions

**Artifact:**
`s3://vesuvius-challenge-open-data/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr`
(zarr v2, OME multiscale L0–L5, L0 shape [8398, 3941, 3941] on the CT-L2 / 9.596 µm grid,
uint8 0/255, chunks 192³; `metadata.json` inside the zarr dates the run 2026-05-13).

**Finding.** A large majority of the positive voxels are not papyrus surface. They are
solid slabs where the *SAM2-masked* CT volume (`20251211183505-2.399um-0.2m-78keV-masked.zarr`)
reads exactly 0 — a halo ring around the scroll cross-section plus end caps. Presumably
the surface model was run on (or blended over) unmasked data, or the mask used at
inference differs from the published masked volume; either way the published preds and
the published masked CT disagree about where the scroll *is*.

**Measured numbers** (survey 2026-06-11, `survey/preds_survey.json` in this repo):

- Global CT-supported share of positive voxels: **0.302** → **~70% phantom**.
- Full-resolution axial planes z ∈ {2000, 4224, 6000} (preds-L0 grid): phantom fraction
  0.681 / 0.672 / 0.646.
- Interior blocks (well inside the scroll): phantom fraction ≤ 6.1e-05 — where CT exists,
  the predictions are excellent recto-face skins. The bug is *only* in the zero-CT zones.
- Cube-level (128³) support is strongly bimodal: at a 0.5 support threshold, 10,915 cubes
  are ~all-phantom vs ~15.9k cubes ~fully supported, only ~536 in between — so cube-level
  filtering plus voxel-wise masking is cheap and clean.

**Why it matters downstream:** `vc_grow_seg_from_seed` does not consult the CT. Seeded
naively (its own `random_seed` mode, or any seed near the halo), it rides the phantom
shells: an early trace of ours had 63% of its lattice at CT ≤ 5. Anyone pointing the villa
tracer (or any CT-blind consumer, e.g. mesh extraction) at these preds as-is will burn
most of their compute on the halo.

**Reproduction (~10 lines):**

```python
import zarr, numpy as np
base = "s3://vesuvius-challenge-open-data/PHerc0332"
preds = zarr.open(f"{base}/representations/predictions/surfaces/"
                  "20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr", mode="r")["0"]
ct    = zarr.open(f"{base}/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr", mode="r")["2"]
# preds L0 and CT L2 share the same [8398, 3941, 3941] grid (9.596 um)
z = 4224
p, c = np.asarray(preds[z]) > 0, np.asarray(ct[z])
print("phantom fraction:", ((p & (c <= 5)).sum() / p.sum()))   # ~0.67
```

**Fix we used:** mask at the source — `supported = preds & (CT > 5)`, sparse chunk-wise
copy (`build_supported_preds.py`), then regenerate normal grids from the supported volume.
This fixed every downstream consumer at once. Suggested upstream fixes: AND the published
preds with the published mask before upload, or ship the phantom-fraction number in the
zarr's metadata so consumers know to mask.

## 5. Contribution B — the two-part acceptance gate (and the QC blind spot it closes)

Per 25 mm render window, acceptance requires **all four** calibrated texture/geometry
gates *and* a recentering-based surface-lock test:

| Gate | Threshold | What it catches |
|---|---|---|
| coherence_ratio | within 2× of human-render reference | non-papyrus / merged-blob texture |
| band_contrast | ≥ 1.05 | no sheet-normal intensity structure |
| merge_fraction | < 45% | double-surface / wrap-merge |
| mask_fraction | ≥ 60% | mostly-empty windows |
| **recenter found_fraction** | **≥ 0.9** | **off-surface / cross-cutting traces** |

The fifth rule exists because of a measured blind spot: our first tracer smoke test
rendered a scroll **cross-section** (the spiral is visible in the render) and *passed all
four texture gates* — fiber texture is locally coherent even when the surface cuts across
wraps. Merged-blob renders, by contrast, were correctly rejected by the coherence gate
(2.63/2.09 vs ≤ 2.0 cutoff). So the texture gates catch wrap-merging but not wrap-crossing.

`found_fraction` is the fraction of render-grid probes for which the subvoxel recentering
step locks onto a sheet-center within its search band. Calibration on this corpus:

- True on-sheet surfaces (human-traced renders + verified windows): 0.97–1.00.
- Wandering / cross-cutting traces: 0.36–0.54.
- Threshold 0.9 sits in a wide empty margin; passing windows median 0.989.

We report `qc_gates_pass` and the found-fraction separately in every `qc.json` for
honesty; a window is accepted only on both. Caveat: this calibration is on Scroll 3
against our own human-render references; it has not been validated on a known-ink control
scroll (see Limitations).

## 6. Reproduction

Hardware used: Apple-silicon Mac (CPU only), ~30 GB chunk cache, anonymous S3. No GPU.

```bash
# 0) Build villa volume-cartographer @ commit 4d0ce2881 (see Citations).
#    macOS notes: pass OpenCV_DIR for brew opencv; use a build dir whose path
#    contains NO SPACES (a space broke libbacktrace autoconf and PaStiX for us).

# 1) CT-supported preds (local copy of preds + CT L2 required, or point at S3)
python -m vesuvius_automesh.build_supported_preds --scroll scroll3 \
    --out .../preds_supported.zarr --ct-threshold 5 --workers 8

# 2) Normal grids on the supported volume (~25 min, 3 slice directions)
vc_gen_normalgrids ... preds_supported.zarr ... normal_grids_supported/

# 3) Seed sweep (params: tracer/params_seed_v3.json — step_size 20, generations 200,
#    normal_grid_path -> the grids from step 2)
python -m vesuvius_automesh.tracer_sweep --scroll scroll3 --tgt-dir .../patches_v2 \
    --params tracer/params_seed_v3.json --target-net-cm2 600

# 4) Render + harvest daemon (25mm windows, 2-part acceptance, cumulative ledger)
python -m vesuvius_automesh.render_queue --scroll scroll3 --patches-dir .../patches_v2 \
    --results-dir results/auto-mesh/scroll3/tracer_v2 --workers 5
```

Code: `vesuvius_automesh/` in this repo (MIT).
Render/QC internals reuse the author's `first_letters` renderer (recenter + 66-layer
sampling + kill-gate-1 QC), vendored under `vesuvius_automesh/_vendor/first_letters/`.

Performance reference points (observations, not promises): trace ≈ minutes/patch on one
core; the binding render cost was CT-L1 S3 latency, fixed by prefetching only
*surface-touched* chunks per window (a curved sheet's bounding box is ~80% empty:
~14–20k candidate chunks → ~3.1k fetched; ~10–16 min → ~2.5 min per window, 48 threads).

## 7. Honest limitations

1. **4.8 µm renders, not ink-ready.** We sample CT level 1 (4.798 µm) onto a 4.798 µm
   grid — sufficient for surface QC, coarser than the 2.4 µm native scan. Ink work would
   re-render at L0.
2. **30% of rendered area is rejected** (wandering windows). Recoverable in principle
   with fiber-direction fields / consensus tracing (organizer techniques we did not use).
3. **found_fraction ≥ 0.9 is calibrated, not control-validated.** Calibration is against
   human-traced Scroll 3 renders; it has not been validated on a scroll with confirmed
   ink. Treat accepted area as *renderable surface*, not ink-ready surface.
4. **Two of 17 patches yielded 0 cm²** — seed/terrain dependent; not retried.
5. **No novelty claim on components.** The tracer, normal grids, and render layout are
   all the organizers' tools; our contribution is the masking fix, the seed/harvest/
   acceptance machinery around them, and the measurements.
6. **No overlap/duplication audit across the 17 patches beyond the sweep's KD-tree
   net-area dedupe** (novelty measured at trace time, not re-verified post-render).

## 8. What this enables next

- **Scroll 2 (PHercParis3):** the pipeline is scroll-agnostic (config already wired);
  it needs only a surface-preds zarr for Scroll 2.
- **Ink screening on Scroll 3:** 409 cm² of QC'd surface is ~6× more substrate for any
  ink model than previously existed; re-render accepted windows at 2.4 µm first.
- **Geometry benchmarking:** the per-window QC records (gate values + recenter histograms
  for 157 windows, pass and fail) are a ready-made labeled set for anyone studying tracer
  failure modes.
- **Upstream:** masking the published preds (or publishing the phantom fraction) helps
  every downstream consumer of the m7 zarrs, not just us.

## 9. Citations, data attribution, licenses

- **Scroll data (PHerc0332 = "Scroll 3"):** EduceLab-Scrolls dataset — Parsons, S.,
  Parker, C. S., Chapman, C., Hayashida, M. & Seales, W. B. *EduceLab-Scrolls:
  Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT and Machine
  Learning.* arXiv:2304.02084 (2023). https://doi.org/10.48550/arXiv.2304.02084.
  Scrolls 1–4 scan data © EduceLab / The University of Kentucky. Used under the
  Vesuvius Challenge data agreement; CT volume:
  `s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr`.
- **Surface predictions (m7):**
  `s3://vesuvius-challenge-open-data/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr`
  (Vesuvius Challenge team; model `s3://scrollprize-reconstruction/models/m7_nnunet/` per
  the zarr's `metadata.json`).
- **First-scroll paper / pipeline context:** Angelotti, G., Parsons, S., Nicolardi, F.,
  Nader, Y., et al. *Complete virtual unwrapping and reading of a rolled Herculaneum
  papyrus.* arXiv:2606.29085 (2026). CC BY-NC 4.0.
- **Tracer and normal grids:** ScrollPrize *villa* monorepo,
  https://github.com/ScrollPrize/villa, commit
  `4d0ce2881752941ccb2ecb4b232bc12aecd8fdb5` (`vc_grow_seg_from_seed`,
  `vc_gen_normalgrids`; local build tweak: `vc_obj_uv_lift` gated behind
  `VC_BUILD_FLATBOI`, not needed for this pipeline).
- **ScrollFiesta** (explored for the meshing stage; superseded here by the tracer after
  we found its gap-based sheet oracle is structurally blind to the zero-gap wrap fusions
  in these preds): https://github.com/Hob3rMallow/scrollfiesta_public.
- Thanks to the Vesuvius Challenge team for open-sourcing the entire stack — every
  component we consumed (scans, predictions, tracer, render conventions) is theirs.

---

*Prepared 2026-07-03 from the run's `SUMMARY.md`, `harvest_ledger.json`, per-window
`qc.json`, and the track log (2026-06-11 → 06-13 entries). All numbers trace to those
artifacts; the phantom-fraction numbers were re-verified against the live public zarr
on 2026-07-03 (unchanged: planes z 2000/4224/6000 → 0.681/0.672/0.646; preds zarr S3
LastModified 2026-05-13, no re-versioning).*
