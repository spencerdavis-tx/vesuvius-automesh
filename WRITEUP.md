# 279 cm² of verified Scroll 3 surface, hands-free

*Spencer Davis — July 2026*

After the PHerc. 1667 paper, the picture is pretty clear: ink detection is in decent shape,
and the thing standing between us and the rest of the library is geometry — tracing the
rolled-up sheet, which still costs the team tens of hours of manual annotation per wrap.

So I wanted to know: using only what's already public — the team's Scroll 3 surface
predictions, their tracer, their render conventions — how much usable surface can you
harvest with **zero** human tracing?

The answer turned out to be **279 cm² of verified rendered surface** — roughly 4× the
human-traced segments currently on the data server for Scroll 3 — produced on a Mac CPU
streaming the CT from S3. No GPU, no annotation clicks. "Verified" means each window passed
both a render-quality gate and an independent topology check; the pipeline renders
substantially more than that, and the gates decide what counts.

![accepted window — clean fiber weave](figures/render_preview_patch715_w0256_0128.png)

## The idea: harvest, don't trace

The pipeline is the team's own tools in a row — surface predictions → `vc_grow_seg_from_seed`
→ standard 66-layer renders. What I added is glue and one design decision.

The tracer, left alone, wanders. It grows beautiful surfaces for a while and then drifts off
the sheet, and there's no point fighting it patch by patch — that's how you end up doing
manual annotation again. Instead: sweep many seeds, let every trace run, render *everything*
in 25 mm windows, and let a quality gate decide window by window what survives. A wandering
trace wastes some compute, but it can't sneak a bad window past the gate.

From 17 traces: 157 windows rendered, 106 passed the render-quality gate, and 61 of those
(279 cm²) passed the independent topology verification described below. Seven traces in
clean outer-wrap territory did most of the work; two wandered completely and contributed
nothing. That attrition is fine — the gates are what make the number honest, not the
tracer.

## The bug I hit on the way (please read this if you use the public predictions)

My first renders came out black, and I spent a while blaming my own code. The actual cause:
**about 70% of the positive voxels in the public Scroll 3 surface-prediction zarr sit where
the published masked CT volume is exactly zero** — a solid halo around the scroll plus end
caps. The predictions and the mask disagree about where the scroll *is*. It's not a Scroll 3
quirk either: PHerc1451, from the same batch run, spot-checks at ~60%.

![axial slice: red = prediction sitting on zero CT](figures/phantom_halo_slice_L2z1056.png)

Where the scan actually has material, the predictions are excellent — this is purely a
background/masking artifact. But the tracer never looks at the CT, so seeded naively it
happily traces those phantom shells (an early trace of mine had 63% of its points in the
halo). The one-line fix that unblocked everything: `supported = preds & (CT > 5)`, applied
before anything else touches the predictions.

Full numbers and a 10-line reproduction are in
[ScrollPrize/villa#1114](https://github.com/ScrollPrize/villa/issues/1114).

## The gate, and the blind spot that made it two-part

I started with four texture/geometry gates calibrated against human-traced renders
(coherence, band contrast, merge fraction, mask fraction). Then a smoke test produced a
render of a scroll **cross-section** — surface cutting straight across the wraps, spiral
plainly visible, obviously garbage — and it *passed all four gates*. Fiber texture is
locally coherent even when the geometry is completely wrong.

The first fix is a surface-lock test: during rendering, each probe tries to re-center onto
a sheet along its normal, and I track what fraction succeeds. Real surfaces score 0.97–1.00;
wandering or cross-cutting traces score 0.36–0.54; the threshold (0.9) sits in the empty
middle.

Surface-lock has its own scope limit, though: it assumes wraps are separated by air gaps. On
zero-gap *fused* terrain the probe can lock onto *some* sheet almost everywhere and the test
saturates. So accepted windows get a second, independent pass with topology instruments that
don't care about texture at all: winding-consistency of the traced surface (a real page
marches monotonically around the scroll; a wrap-hopper backtracks), agreement with the
surface-prediction volume where it has support, and window-level surface-distance scoring.
Only windows that pass both tiers count toward the headline — 61 of the 106
gate-accepted windows here. Every window ships with its own `qc.json` and topology verdict,
so you can audit any of this instead of taking my word for it.

## Caveats, honestly

The renders are 4.8 µm — good enough to judge surface quality, not what you'd feed an ink
model (the verified windows can be re-rendered at 2.4 µm from the same tifxyz). A little over half of the
rendered area fails one gate or the other and is excluded from the count; some of that is
probably recoverable with the team's fiber-direction techniques. The gate calibration comes from Scroll 3's own
human-traced segments, not from a known-ink control. And the two-tier design matters more,
not less, on other scrolls: the denser and more fused the terrain, the more the topology
tier does the real work — on such terrain, texture and surface-lock alone will over-accept.

And to be completely clear: **there are no ink claims here.** This is renderable surface,
nothing more.

## If you want to build on it

Everything is in this repo (MIT): the pipeline, configs, a walkthrough in the
[README](README.md), and per-window QC records for all 157 windows — the failures too, which
make a nice labeled set of tracer failure modes if automated segmentation is your thing.
The pipeline doesn't care which scroll it's pointed at; Scroll 2 is the obvious next target.
The exhaustive version of this writeup, with every number and reproduction command, is in
[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).

## Credit where it's due

Every component consumed here is the Vesuvius Challenge team's, released openly: the scans
(EduceLab-Scrolls, arXiv:2304.02084, © University of Kentucky), the surface predictions and
tracer (the [villa](https://github.com/ScrollPrize/villa) monorepo), and the context that
motivated all of it (Angelotti et al., arXiv:2606.29085, CC BY-NC 4.0). Full citations in
[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md#9-citations-data-attribution-licenses).
