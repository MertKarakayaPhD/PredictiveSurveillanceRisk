#!/usr/bin/env python3
"""
Benchmark compute backends on small metro runs for GPU-acceleration spike.

This script runs per-metro simulations with fixed small workload (default 50x4),
captures wall-clock runtime, and records the actual backend used from summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def latest_output_dir(output_root: Path, metro_id: str) -> Path | None:
    candidates = sorted(
        (p for p in output_root.glob(f"{metro_id}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_once(
    python_exe: str,
    config: str,
    camera_catalog_csv: str,
    metro_id: str,
    backend: str,
    workers: int,
    n_vehicles: int,
    n_trips: int,
    mp_chunksize: int,
    output_root: str,
    seed: int,
) -> dict[str, Any]:
    cmd = [
        python_exe,
        "scripts/run_metro_batch.py",
        "--config",
        config,
        "--metro-ids",
        metro_id,
        "--camera-catalog-csv",
        camera_catalog_csv,
        "--skip-preflight",
        "--force-rerun",
        "--n-vehicles",
        str(n_vehicles),
        "--n-trips",
        str(n_trips),
        "--workers",
        str(workers),
        "--mp-chunksize",
        str(mp_chunksize),
        "--blas-threads",
        "1",
        "--compute-backend",
        backend,
        "--seed",
        str(seed),
        "--output-root",
        output_root,
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.perf_counter() - t0

    out_dir = latest_output_dir(BASE_DIR / output_root, metro_id)
    summary: dict[str, Any] = {}
    if out_dir is not None:
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

    run_params = summary.get("run_parameters", {})
    predictability = summary.get("predictability", {})
    psr = summary.get("psr", {})
    u2 = summary.get("u2_random", {})

    return {
        "metro_id": metro_id,
        "requested_backend": backend,
        "actual_backend": str(run_params.get("compute_backend", "")),
        "workers": workers,
        "n_vehicles": n_vehicles,
        "n_trips": n_trips,
        "elapsed_sec": round(elapsed, 3),
        "exit_code": int(proc.returncode),
        "psr": psr.get("score"),
        "u2": u2.get("uniqueness"),
        "acc5": predictability.get("acc5_markov_order1"),
        "output_dir": str(out_dir) if out_dir else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark backend performance on small metro simulations.")
    parser.add_argument("--config", type=str, default="data/external/metro_batch/metros_us_32.json")
    parser.add_argument(
        "--camera-catalog-csv",
        type=str,
        default="data/external/camera_catalog/cameras_us_active.csv.gz",
    )
    parser.add_argument("--metro-ids", type=str, required=True, help="Comma-separated metro IDs")
    parser.add_argument("--backends", type=str, default="cpu,torch-cuda")
    parser.add_argument("--n-vehicles", type=int, default=50)
    parser.add_argument("--n-trips", type=int, default=4)
    parser.add_argument("--workers-cpu", type=int, default=12)
    parser.add_argument("--workers-gpu", type=int, default=1)
    parser.add_argument("--mp-chunksize", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="results/gpu_spike")
    parser.add_argument("--csv-out", type=str, default="results/gpu_spike/benchmark_gpu_spike.csv")
    args = parser.parse_args()

    metros = parse_csv_list(args.metro_ids)
    backends = parse_csv_list(args.backends)
    if not metros:
        raise ValueError("No metro IDs provided.")
    if not backends:
        raise ValueError("No backends provided.")

    results: list[dict[str, Any]] = []
    any_failure = False

    for metro_id in metros:
        for backend in backends:
            workers = args.workers_gpu if "cuda" in backend else args.workers_cpu
            print(f"\n[RUN] metro={metro_id} backend={backend} workers={workers}")
            row = run_once(
                python_exe=sys.executable,
                config=args.config,
                camera_catalog_csv=args.camera_catalog_csv,
                metro_id=metro_id,
                backend=backend,
                workers=workers,
                n_vehicles=args.n_vehicles,
                n_trips=args.n_trips,
                mp_chunksize=args.mp_chunksize,
                output_root=args.output_root,
                seed=args.seed,
            )
            results.append(row)
            print(
                f"[DONE] metro={metro_id} backend={backend} "
                f"actual={row['actual_backend']} elapsed={row['elapsed_sec']}s exit={row['exit_code']}"
            )
            if row["exit_code"] != 0:
                any_failure = True

    csv_path = (BASE_DIR / args.csv_out).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metro_id",
                "requested_backend",
                "actual_backend",
                "workers",
                "n_vehicles",
                "n_trips",
                "elapsed_sec",
                "exit_code",
                "psr",
                "u2",
                "acc5",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] Wrote benchmark CSV: {csv_path}")

    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

