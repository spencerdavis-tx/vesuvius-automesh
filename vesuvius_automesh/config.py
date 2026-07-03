"""Per-scroll data paths and grid constants for the auto-mesh pipeline.

All shapes/voxel sizes flow from config/level_ledger.json at run time where
possible; constants here are limited to paths and scroll wiring.

Data location: set the environment variable VESUVIUS_DATA_ROOT to the
directory that holds your local zarr caches (referred to as $DATA_DIR in the
README). Expected layout:

    $DATA_DIR/cache/scroll3/preds_m7.zarr     # m7 surface preds (S3 mirror)
    $DATA_DIR/cache/scroll3/ct.zarr           # masked CT levels 2+ (S3 mirror)
    $DATA_DIR/automesh/...                    # pipeline outputs (created)

Every volume can also stream from S3 (anonymous) via the ledger if a local
cache is absent — see _vendor/first_letters/zarrio.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CUBE = 128  # scrollfiesta cube edge, preds-grid voxels

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "config" / "level_ledger.json"
if not LEDGER_PATH.exists():  # non-editable install: resolve from CWD
    LEDGER_PATH = Path("config") / "level_ledger.json"

DATA_ROOT = Path(os.environ.get("VESUVIUS_DATA_ROOT", "data"))
AUTOMESH_DATA = DATA_ROOT / "automesh"

# Human-render reference for the QC gates (any 2 um/px human-verified fragment
# surface render works; we used PHerc0343P layer 32 from the data server).
QC_REFERENCE_LAYER = os.environ.get(
    "AUTOMESH_QC_REFERENCE",
    str(DATA_ROOT / "fragments2um" / "PHerc0343P" / "layers" / "32.png"),
)


@dataclass(frozen=True)
class ScrollPaths:
    """Local zarr paths for one scroll (preds grid == CT level ct_align_level)."""

    name: str
    preds_zarr: Path  # binary 0/255 surface preds, L0 == CT level `ct_align_level`
    ct_zarr: Path  # local CT cache (OME multiscale)
    ct_align_level: int  # CT level whose grid matches preds L0
    ledger_ct_key: str  # level_ledger key for the full CT pyramid (S3-backed)
    ct_support: bool  # apply (CT>5) phantom filtering in the exporter


SCROLLS = {
    "scroll3": ScrollPaths(
        name="scroll3",
        preds_zarr=DATA_ROOT / "cache" / "scroll3" / "preds_m7.zarr",
        ct_zarr=DATA_ROOT / "cache" / "scroll3" / "ct.zarr",
        ct_align_level=2,
        ledger_ct_key="scroll3_ct_2p4um",
        ct_support=True,
    ),
    "scroll2": ScrollPaths(
        name="scroll2",
        preds_zarr=DATA_ROOT / "cache" / "scroll2" / "preds_v1.zarr",
        ct_zarr=DATA_ROOT / "cache" / "scroll2" / "ct.zarr",  # may not exist locally
        ct_align_level=2,
        ledger_ct_key="scroll2_ct_2p4um",
        ct_support=False,  # supply your own preds for scroll2; see README
    ),
}
