"""Build a coarse-CT-supported copy of the Scroll 2 preds zarr (level 0 only).

Scroll 2 preds_v1 has essentially no phantom problem (support fraction 0.9969,
survey s9); the only pathology is ~13k floating dash-like speck components
>150um from any CT material (survey s11). Voxelwise CT L2 masking would need
TB-scale streaming for nothing — instead mask against the packed CT L5
occupancy (76.8um grid == preds L3), dilated 2 L5-voxels (the survey's exact
edge/floating split: keeps mask-edge sheets, kills floating specks), upsampled
8x to the preds L0 grid.

Output layout mirrors build_supported_preds (the Scroll 3 tracer input):
<out>.zarr/ with v2-format array "0" (192^3 zstd), .zgroup, .zattrs, and a VC
meta.json. Only chunks stored in the SOURCE are visited (sparse copy).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

from .build_supported_preds import stored_chunk_indices
from .config import SCROLLS

L0_TO_L5 = 8  # preds L0 == CT L2; CT L5 == CT L2 / 8


def load_coarse_occ(occ_npz: Path, *, dilate: int = 2) -> np.ndarray:
    """Unpack the survey's packed CT L5 occupancy and dilate it.

    dilate=2 (6-conn, like survey s11) keeps preds within ~154um of CT
    material ("edge") and removes only free-floating specks ("floating").
    """
    d = np.load(occ_npz)
    shape = tuple(int(s) for s in d["shape"])
    occ = np.unpackbits(d["packed"])[: int(np.prod(shape))].reshape(shape) > 0
    if dilate > 0:
        occ = ndimage.binary_dilation(occ, iterations=dilate)
    return occ


def build(scroll: str, out_root: Path, occ_npz: Path, *, dilate: int = 2,
          workers: int = 12) -> dict:
    paths = SCROLLS[scroll]
    src_grp = zarr.open_group(str(paths.preds_zarr), mode="r")
    src = src_grp["0"]
    occ = load_coarse_occ(occ_npz, dilate=dilate)
    exp_occ_shape = tuple(-(-s // L0_TO_L5) for s in src.shape)
    if occ.shape != exp_occ_shape:
        raise ValueError(f"occ grid {occ.shape} != ceil(src/8) {exp_occ_shape}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ".zgroup").write_text('{"zarr_format": 2}')
    (out_root / ".zattrs").write_text(json.dumps({
        "note": "coarse-CT-supported preds (preds & dilate%d(CT_L5_occ) x8), "
                "auto-mesh track L3.2" % dilate,
        "source": str(paths.preds_zarr),
        "occ_npz": str(occ_npz),
    }))
    import numcodecs

    dst = zarr.create_array(
        store=str(out_root), name="0", shape=src.shape, chunks=src.chunks,
        dtype="u1", zarr_format=2, fill_value=0, overwrite=True,
        compressors=numcodecs.Blosc(cname="zstd", clevel=1,
                                    shuffle=numcodecs.Blosc.SHUFFLE),
        chunk_key_encoding={"name": "v2", "separator": "/"},
    )

    cz, cy, cx = src.chunks
    chunks = stored_chunk_indices(Path(paths.preds_zarr) / "0")
    t0 = time.time()
    lock = threading.Lock()
    done = [0]
    raw = [0]
    kept = [0]
    n_full = [0]
    n_partial = [0]
    n_dropped = [0]

    def work(idx: tuple[int, int, int]) -> None:
        iz, iy, ix = idx
        z0, y0, x0 = iz * cz, iy * cy, ix * cx
        z1 = min(z0 + cz, src.shape[0])
        y1 = min(y0 + cy, src.shape[1])
        x1 = min(x0 + cx, src.shape[2])
        p = np.asarray(src[z0:z1, y0:y1, x0:x1])
        n_raw = int((p > 0).sum())
        if n_raw == 0:
            return
        b = occ[z0 // L0_TO_L5 : -(-z1 // L0_TO_L5),
                y0 // L0_TO_L5 : -(-y1 // L0_TO_L5),
                x0 // L0_TO_L5 : -(-x1 // L0_TO_L5)]
        if b.all():
            dst[z0:z1, y0:y1, x0:x1] = p
            n_kept, kind = n_raw, "full"
        elif not b.any():
            n_kept, kind = 0, "dropped"
        else:
            m = np.repeat(np.repeat(np.repeat(
                b, L0_TO_L5, 0), L0_TO_L5, 1), L0_TO_L5, 2)
            m = m[: z1 - z0, : y1 - y0, : x1 - x0]
            keep = (p > 0) & m
            n_kept, kind = int(keep.sum()), "partial"
            if n_kept:
                dst[z0:z1, y0:y1, x0:x1] = np.where(
                    keep, np.uint8(255), np.uint8(0))
        with lock:
            raw[0] += n_raw
            kept[0] += n_kept
            {"full": n_full, "dropped": n_dropped, "partial": n_partial}[kind][0] += 1
            done[0] += 1
            report = done[0] % 2000 == 0
        if report:
            print(f"  {done[0]}/{len(chunks)} chunks, "
                  f"kept {kept[0]/max(raw[0],1):.4f}, "
                  f"{time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, chunks))

    meta = {
        "uuid": out_root.stem,
        "name": f"{scroll} coarse-CT-supported surface preds (auto-mesh L3.2)",
        "type": "vol", "format": "zarr",
        "voxelsize": 9.596 if scroll == "scroll3" else 9.6,
        "slices": int(src.shape[0]), "height": int(src.shape[1]),
        "width": int(src.shape[2]), "min": 0.0, "max": 255.0,
        "mask": f"CT_L5_occ dilate{dilate} upsample{L0_TO_L5}",
        "supported_fraction": kept[0] / max(raw[0], 1),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=1))
    stats = {"chunks_visited": len(chunks), "raw_pos": raw[0],
             "kept_pos": kept[0],
             "kept_fraction": round(kept[0] / max(raw[0], 1), 6),
             "chunks_fully_kept": n_full[0], "chunks_partial": n_partial[0],
             "chunks_dropped": n_dropped[0],
             "dilate_l5": dilate,
             "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(stats))
    return stats


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--occ-npz", type=Path, required=True)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stats-out", type=Path, default=None)
    args = ap.parse_args()
    stats = build(args.scroll, args.out, args.occ_npz, dilate=args.dilate,
                  workers=args.workers)
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
