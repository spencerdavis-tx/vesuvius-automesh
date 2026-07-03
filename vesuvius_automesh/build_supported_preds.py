"""Build a CT-supported copy of the surface-preds zarr (level 0 only).

The organizers' m7 preds contain solid phantom slabs where the masked CT is 0
(~70% of positive voxels). The tracer is CT-blind and rides those phantom
shells (v2 smoke: 63% of lattice at CT<=5). Masking at the source — supported
= preds & (CT > threshold) — fixes every downstream consumer at once.

Output layout: <out>.zarr/ with v2-format array "0" (same chunking/codec
family as the source), .zgroup, minimal .zattrs, and a VC meta.json. Only
chunks stored in the SOURCE are written (sparse copy).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from .config import SCROLLS


def stored_chunk_indices(level_dir: Path) -> list[tuple[int, int, int]]:
    """Walk a v2 dim_separator='/' level dir: <z>/<y>/<x> chunk files."""
    out = []
    for zdir in level_dir.iterdir():
        if not zdir.name.isdigit():
            continue
        for ydir in zdir.iterdir():
            if not ydir.name.isdigit():
                continue
            for xf in ydir.iterdir():
                if xf.name.isdigit():
                    out.append((int(zdir.name), int(ydir.name), int(xf.name)))
    return out


def build(scroll: str, out_root: Path, *, ct_threshold: int = 5,
          workers: int = 8) -> dict:
    paths = SCROLLS[scroll]
    src_grp = zarr.open_group(str(paths.preds_zarr), mode="r")
    src = src_grp["0"]
    ct = zarr.open_group(str(paths.ct_zarr), mode="r")[str(paths.ct_align_level)]
    if tuple(ct.shape) != tuple(src.shape):
        raise ValueError(f"grid mismatch {ct.shape} vs {src.shape}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ".zgroup").write_text('{"zarr_format": 2}')
    (out_root / ".zattrs").write_text(json.dumps({
        "note": "CT-supported preds (preds & CT>%d), auto-mesh track" % ct_threshold,
        "source": str(paths.preds_zarr),
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
    done = [0]
    kept = [0]
    raw = [0]

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
        c = np.asarray(ct[z0:z1, y0:y1, x0:x1])
        m = (p > 0) & (c > ct_threshold)
        n_kept = int(m.sum())
        raw[0] += n_raw
        kept[0] += n_kept
        if n_kept:
            dst[z0:z1, y0:y1, x0:x1] = np.where(m, np.uint8(255), np.uint8(0))
        done[0] += 1
        if done[0] % 500 == 0:
            print(f"  {done[0]}/{len(chunks)} chunks, "
                  f"kept {kept[0]/max(raw[0],1):.3f}, "
                  f"{time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, chunks))

    meta = {
        "uuid": out_root.stem,
        "name": f"{scroll} CT-supported surface preds (auto-mesh)",
        "type": "vol", "format": "zarr",
        "voxelsize": 9.596 if scroll == "scroll3" else 9.6,
        "slices": int(src.shape[0]), "height": int(src.shape[1]),
        "width": int(src.shape[2]), "min": 0.0, "max": 255.0,
        "ct_threshold": ct_threshold,
        "supported_fraction": kept[0] / max(raw[0], 1),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=1))
    stats = {"chunks": len(chunks), "raw_pos": raw[0], "kept_pos": kept[0],
             "supported_fraction": round(kept[0] / max(raw[0], 1), 4),
             "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(stats))
    return stats


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ct-threshold", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    build(args.scroll, args.out, ct_threshold=args.ct_threshold,
          workers=args.workers)


if __name__ == "__main__":
    main()
