"""One-command region pipeline: export -> scrollfiesta -> render -> ledger.

Runs regions SERIALLY (grid_pipeline already saturates the machine). Each
region spec is {"region": str, "cube_lo": [z,y,x], "cube_hi": [z,y,x]}.

Resumable: a region whose render_summary.json exists is skipped; grid_pipeline
itself skips already-meshed cubes (marker = per-cube output in out/cubes).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import AUTOMESH_DATA, REPO_ROOT, SCROLLS
from .export_cubes import export_region

# Build https://github.com/Hob3rMallow/scrollfiesta_public and point
# SCROLLFIESTA_SRC at its src/ (needs grid_pipeline, cube_mesh, grid_weld).
SCROLLFIESTA = Path(os.environ.get(
    "SCROLLFIESTA_SRC", str(REPO_ROOT / "third_party" / "scrollfiesta_public" / "src")))
RESULTS_ROOT = REPO_ROOT / "results" / "auto-mesh"


def run_region(
    scroll: str,
    spec: dict,
    *,
    max_concurrent: int = 24,
    min_area_cm2: float = 0.05,
    render_level: int = 1,
) -> dict:
    region = spec["region"]
    grid_dir = AUTOMESH_DATA / scroll / region
    out_dir = grid_dir / "out"
    results_dir = RESULTS_ROOT / scroll / region
    summary_path = results_dir / "render_summary.json"
    if summary_path.exists():
        print(f"[{region}] render_summary.json exists -> skip")
        return json.loads(summary_path.read_text())

    t0 = time.time()
    if not (grid_dir / "export_manifest.json").exists():
        stats = export_region(
            scroll, tuple(spec["cube_lo"]), tuple(spec["cube_hi"]), grid_dir
        )
        print(f"[{region}] exported {stats.n_written} cubes "
              f"({stats.pos_voxels_written/1e6:.0f}M pos vox)")
    else:
        print(f"[{region}] export_manifest.json exists -> reuse")

    welded = out_dir / "welded.obj"
    if not welded.exists():
        # Mesh all cubes WITHOUT welding, retry crashed cubes once (rare
        # hole_fill crashes, ~0.4% in pilot), then weld manually.
        cmd = [
            str(SCROLLFIESTA / "grid_pipeline"), str(grid_dir), str(out_dir),
            "--halo", "13", "--qem", "--skip-weld",
            "--max-concurrent", str(max_concurrent),
            "--exe", str(SCROLLFIESTA / "cube_mesh"),
            "--weld", str(SCROLLFIESTA / "grid_weld"),
        ]
        log_path = grid_dir / "grid_pipeline.log"
        with open(log_path, "a") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"[{region}] grid_pipeline rc={rc}; see {log_path}")
        csv = out_dir / "pipeline_summary.csv"
        failed = []
        if csv.exists():
            for line in csv.read_text().splitlines()[1:]:
                cube_id, exit_code, _ = line.split(",")
                if exit_code != "0":
                    failed.append(cube_id)
        for cube_id in failed:
            print(f"[{region}] retrying crashed cube {cube_id}")
            with open(log_path, "a") as log:
                subprocess.run(
                    [str(SCROLLFIESTA / "cube_mesh"),
                     str(grid_dir / "cubes_PRED" / f"{cube_id}.tif"),
                     str(out_dir / "cubes" / f"{cube_id}.tif"),
                     "--halo", "13", "--dump-obj", str(out_dir / "dump"),
                     "--no-timeout"],
                    stdout=log, stderr=subprocess.STDOUT,
                )
        with open(log_path, "a") as log:
            rc = subprocess.run(
                [str(SCROLLFIESTA / "grid_weld"), str(out_dir / "dump"),
                 str(welded)],
                stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        if rc != 0 or not welded.exists():
            raise RuntimeError(f"[{region}] grid_weld rc={rc}; see {log_path}")
        print(f"[{region}] welded.obj done in {(time.time()-t0)/60:.0f} min "
              f"({len(failed)} cube retries)")

    cmd = [
        sys.executable, "-m", "vesuvius_automesh.render_driver",
        "--welded", str(welded), "--scroll", scroll, "--region", region,
        "--out", str(results_dir), "--level", str(render_level),
        "--min-area-cm2", str(min_area_cm2),
    ]
    log_path = grid_dir / "render_driver.log"
    with open(log_path, "a") as log:
        rc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ,
                 "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        ).returncode
    if rc != 0 or not summary_path.exists():
        raise RuntimeError(f"[{region}] render_driver rc={rc}; see {log_path}")
    summary = json.loads(summary_path.read_text())
    print(f"[{region}] rendered: {summary['n_components_rendered']} comps, "
          f"qc-pass {summary['qc_pass_area_cm2']} cm2, "
          f"total {(time.time()-t0)/60:.0f} min")
    return summary


def update_ledger(scroll: str) -> dict:
    """Aggregate all render_summary.json under results/auto-mesh/<scroll>/."""
    rows = []
    for p in sorted((RESULTS_ROOT / scroll).glob("*/render_summary.json")):
        s = json.loads(p.read_text())
        rows.append(
            {
                "region": s["region"],
                "qc_pass_area_cm2": s["qc_pass_area_cm2"],
                "n_components": s["n_components_rendered"],
                "n_failed": s["n_failed"],
            }
        )
    ledger = {
        "scroll": scroll,
        "updated_unix": time.time(),
        "total_qc_pass_cm2": round(sum(r["qc_pass_area_cm2"] for r in rows), 2),
        "baseline_traced_cm2": 67.0,
        "regions": rows,
    }
    (RESULTS_ROOT / scroll / "coverage_ledger.json").write_text(
        json.dumps(ledger, indent=1)
    )
    return ledger


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--regions-json", type=Path, required=True,
                    help="JSON with {boxes: [{region, cube_lo, cube_hi}...]}")
    ap.add_argument("--only", nargs="*", default=None,
                    help="region names to run (default: all in file order)")
    ap.add_argument("--max-concurrent", type=int, default=24)
    args = ap.parse_args()

    payload = json.loads(args.regions_json.read_text())
    boxes = payload["boxes"] if "boxes" in payload else payload
    for spec in boxes:
        if args.only and spec["region"] not in args.only:
            continue
        try:
            run_region(args.scroll, spec, max_concurrent=args.max_concurrent)
        except RuntimeError as e:
            print(f"REGION FAILED: {e}", file=sys.stderr)
        ledger = update_ledger(args.scroll)
        print(f"== cumulative qc-pass: {ledger['total_qc_pass_cm2']} cm2 "
              f"(target {3 * ledger['baseline_traced_cm2']:.0f}) ==")


if __name__ == "__main__":
    main()
