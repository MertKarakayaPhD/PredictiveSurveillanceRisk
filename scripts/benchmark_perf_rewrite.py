#!/usr/bin/env python3
"""
Benchmark script for performance rewrite (cache-based) modes.

Modes:
- baseline: route cache OFF + node-camera cache OFF
- optimized: route cache ON + node-camera cache ON
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


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def latest_output_dir(output_root: Path, metro_id: str) -> Path | None:
    candidates = sorted(
        (p for p in output_root.glob(f"{metro_id}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_once(
    mode: str,
    metro_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/run_metro_batch.py",
        "--config",
        args.config,
        "--metro-ids",
        metro_id,
        "--camera-catalog-csv",
        args.camera_catalog_csv,
        "--skip-preflight",
        "--force-rerun",
        "--n-vehicles",
        str(args.n_vehicles),
        "--n-trips",
        str(args.n_trips),
        "--workers",
        str(args.workers),
        "--mp-chunksize",
        str(args.mp_chunksize),
        "--blas-threads",
        "1",
        "--route-cache-size",
        str(args.route_cache_size),
        "--node-camera-cache-size",
        str(args.node_camera_cache_size),
        "--seed",
        str(args.seed),
        "--output-root",
        args.output_root,
    ]

    if mode == "baseline":
        cmd.append("--disable-route-cache")
        cmd.append("--disable-node-camera-cache")
    elif mode == "optimized":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.perf_counter() - t0

    out_dir = latest_output_dir(BASE_DIR / args.output_root, metro_id)
    summary: dict[str, Any] = {}
    if out_dir is not None:
        sp = out_dir / "summary.json"
        if sp.exists():
            summary = json.loads(sp.read_text(encoding="utf-8"))

    run_params = summary.get("run_parameters", {})
    return {
        "metro_id": metro_id,
        "mode": mode,
        "elapsed_sec": round(elapsed, 3),
        "exit_code": int(proc.returncode),
        "workers": args.workers,
        "n_vehicles": args.n_vehicles,
        "n_trips": args.n_trips,
        "use_route_cache": run_params.get("use_route_cache"),
        "use_node_camera_cache": run_params.get("use_node_camera_cache"),
        "route_cache_size": run_params.get("route_cache_size"),
        "node_camera_cache_size": run_params.get("node_camera_cache_size"),
        "output_dir": str(out_dir) if out_dir else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs optimized cache modes.")
    parser.add_argument("--config", type=str, default="data/external/metro_batch/metros_us_32.json")
    parser.add_argument(
        "--camera-catalog-csv",
        type=str,
        default="data/external/camera_catalog/cameras_us_active.csv.gz",
    )
    parser.add_argument("--metro-ids", type=str, required=True, help="Comma-separated metro IDs.")
    parser.add_argument("--modes", type=str, default="baseline,optimized")
    parser.add_argument("--n-vehicles", type=int, default=50)
    parser.add_argument("--n-trips", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mp-chunksize", type=int, default=2)
    parser.add_argument("--route-cache-size", type=int, default=200000)
    parser.add_argument("--node-camera-cache-size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="results/perf_rewrite_bench")
    parser.add_argument("--csv-out", type=str, default="results/perf_rewrite_bench/benchmark_perf_rewrite.csv")
    args = parser.parse_args()

    metros = parse_csv(args.metro_ids)
    modes = parse_csv(args.modes)
    if not metros:
        raise ValueError("No metro IDs supplied.")
    if not modes:
        raise ValueError("No modes supplied.")

    rows: list[dict[str, Any]] = []
    any_fail = False
    for metro_id in metros:
        for mode in modes:
            print(f"[RUN] metro={metro_id} mode={mode}")
            row = run_once(mode=mode, metro_id=metro_id, args=args)
            rows.append(row)
            print(f"[DONE] metro={metro_id} mode={mode} elapsed={row['elapsed_sec']}s exit={row['exit_code']}")
            if row["exit_code"] != 0:
                any_fail = True

    out_csv = (BASE_DIR / args.csv_out).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metro_id",
                "mode",
                "elapsed_sec",
                "exit_code",
                "workers",
                "n_vehicles",
                "n_trips",
                "use_route_cache",
                "use_node_camera_cache",
                "route_cache_size",
                "node_camera_cache_size",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Benchmark CSV: {out_csv}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

