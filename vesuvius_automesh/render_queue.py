"""Render-queue daemon: watch a tifxyz patch dir, render new patches in
parallel worker subprocesses, keep a cumulative accepted-area ledger.

Acceptance per 25mm window = kill-gate-1 (4 calibrated gates) AND recenter
found_fraction >= 0.9. A patch is "done" when its render_summary fragment
exists under the results root.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import REPO_ROOT, SCROLLS

RESULTS_ROOT = REPO_ROOT / "results" / "auto-mesh"


def patch_done(results_dir: Path, patch_name: str) -> bool:
    return (results_dir / f"{patch_name}.row.json").exists()


def render_one(patch_dir: Path, scroll: str, results_dir: Path,
               log_dir: Path, pitch_um: float = 4.798) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u", "-m", "vesuvius_automesh.render_tifxyz",
        "--patches", str(patch_dir), "--scroll", scroll,
        "--out", str(results_dir / patch_dir.name),
        "--pitch-um", str(pitch_um),
    ]
    log = open(log_dir / f"{patch_dir.name}.log", "a")
    return subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )


def harvest_row(results_dir: Path, patch_name: str) -> dict | None:
    summary = results_dir / patch_name / "render_summary.json"
    if not summary.exists():
        return None
    s = json.loads(summary.read_text())
    n_windows = sum(p.get("n_windows", 0) for p in s.get("patches", []))
    n_win_pass = sum(
        1 for p in s.get("patches", []) for w in p.get("windows", [])
        if w.get("qc_pass")
    )
    row = {
        "patch": patch_name,
        "qc_pass_area_cm2": s.get("qc_pass_area_cm2", 0.0),
        "masked_area_cm2": sum(p.get("masked_area_cm2", 0.0)
                               for p in s.get("patches", [])),
        "n_windows": n_windows,
        "n_windows_pass": n_win_pass,
        "n_failed": s.get("n_failed", 0),
    }
    (results_dir / f"{patch_name}.row.json").write_text(json.dumps(row))
    return row


def update_ledger(results_dir: Path) -> dict:
    rows = [json.loads(p.read_text())
            for p in sorted(results_dir.glob("*.row.json"))]
    ledger = {
        "updated_unix": time.time(),
        "n_patches": len(rows),
        "accepted_cm2": round(sum(r["qc_pass_area_cm2"] for r in rows), 2),
        "target_cm2": 201.0,
        "rows": rows,
    }
    (results_dir / "harvest_ledger.json").write_text(json.dumps(ledger, indent=1))
    return ledger


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", required=True, choices=sorted(SCROLLS))
    ap.add_argument("--patches-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--pitch-um", type=float, default=4.798,
                    help="render pitch (scroll3 L1=4.798, scroll2 L1=4.8)")
    ap.add_argument("--idle-exit-min", type=float, default=180.0,
                    help="exit after this long with nothing to do")
    args = ap.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    running: dict[str, subprocess.Popen] = {}
    last_work = time.time()
    while True:
        # reap finished workers
        for name in list(running):
            if running[name].poll() is not None:
                row = harvest_row(results_dir, name)
                led = update_ledger(results_dir)
                print(f"[queue] {name}: "
                      f"{row['qc_pass_area_cm2'] if row else 'NO SUMMARY'} cm2 "
                      f"| cumulative accepted {led['accepted_cm2']} cm2",
                      flush=True)
                del running[name]
                last_work = time.time()
        # launch new work
        if args.patches_dir.exists():
            for d in sorted(args.patches_dir.iterdir()):
                if len(running) >= args.workers:
                    break
                if not d.is_dir() or not (d / "x.tif").exists():
                    continue
                if d.name in running or patch_done(results_dir, d.name):
                    continue
                # skip patches still being written (mtime very fresh)
                if time.time() - d.stat().st_mtime < 20:
                    continue
                running[d.name] = render_one(d, args.scroll, results_dir,
                                             log_dir, pitch_um=args.pitch_um)
                print(f"[queue] started {d.name} "
                      f"({len(running)}/{args.workers} workers)", flush=True)
                last_work = time.time()
        if not running and time.time() - last_work > args.idle_exit_min * 60:
            print("[queue] idle too long; exiting")
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
