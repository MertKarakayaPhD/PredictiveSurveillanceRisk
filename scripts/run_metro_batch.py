#!/usr/bin/env python3
"""
Progressive metro-batch runner for ROI simulation.

Features:
1. Resume/skip metros with existing completed outputs
2. Run-level state tracking (JSON)
3. Per-metro logs and run summary
4. Log rotation with zip archive for old run folders
5. Optional preflight checks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "data" / "external" / "metro_batch" / "metros_us_32.json"
DEFAULT_CAMERA_CATALOG = BASE_DIR / "data" / "external" / "camera_catalog" / "cameras_us_active.csv.gz"
KNOWN_AADT_STATES = {"GA", "TN", "VA", "NC", "PA", "ME"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rotate_run_logs(log_root: Path, keep_last_runs: int) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        (p for p in log_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(run_dirs) <= keep_last_runs:
        return

    for run_dir in run_dirs[keep_last_runs:]:
        archive_path = Path(f"{run_dir}.zip")
        try:
            if not archive_path.exists():
                shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
            shutil.rmtree(run_dir)
        except Exception as exc:
            print(f"[WARN] Failed to rotate log folder {run_dir}: {exc}")


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def run_step(cmd: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {utc_now_iso()} =====\n")
        log_file.write("CMD: " + " ".join(cmd) + "\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)

        return proc.wait()


def parse_ring_breaks(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    out = sorted({float(p) for p in parts})
    if 0.0 not in out:
        out = [0.0] + out
    return out


def matches_signature(summary_path: Path, expected: dict[str, Any]) -> bool:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    run_params = summary.get("run_parameters", {})
    center = summary.get("center", {})
    summary_boundary = summary.get("boundary_geojson")
    expected_boundary = expected.get("boundary_geojson")
    summary_aadt = summary.get("aadt_path")
    expected_aadt = expected.get("aadt_path")
    summary_aadt_paths_input = [str(x) for x in summary.get("aadt_paths_input", [])]
    expected_aadt_paths_input = [str(x) for x in expected.get("aadt_paths_input", [])]
    checks = [
        int(run_params.get("n_vehicles", -1)) == int(expected["n_vehicles"]),
        int(run_params.get("n_trips", -1)) == int(expected["n_trips"]),
        int(run_params.get("seed", -1)) == int(expected["seed"]),
        int(run_params.get("k_shortest", -1)) == int(expected["k_shortest"]),
        abs(float(run_params.get("p_return", -1.0)) - float(expected["p_return"])) < 1e-12,
        abs(float(run_params.get("detection_radius_m", -1.0)) - float(expected["detection_radius_m"])) < 1e-9,
        [float(x) for x in run_params.get("ring_breaks_km", [])] == [float(x) for x in expected["ring_breaks_km"]],
        abs(float(summary.get("radius_km", -1.0)) - float(expected["radius_km"])) < 1e-9,
        abs(float(center.get("lat", -999.0)) - float(expected["center_lat"])) < 1e-9,
        abs(float(center.get("lon", -999.0)) - float(expected["center_lon"])) < 1e-9,
        str(summary_boundary or "") == str(expected_boundary or ""),
        str(summary_aadt or "") == str(expected_aadt or ""),
        summary_aadt_paths_input == expected_aadt_paths_input,
    ]
    return all(checks)


def find_latest_matching_output(output_root: Path, metro_id: str, expected: dict[str, Any]) -> Path | None:
    candidates = sorted(
        (p for p in output_root.glob(f"{metro_id}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        summary_path = c / "summary.json"
        if not summary_path.exists() or not (c / "psr.json").exists() or not (c / "ring_metrics.json").exists():
            continue
        if matches_signature(summary_path, expected):
            return c
    return None


def load_metros(config_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    metros = payload.get("metros")
    if not isinstance(metros, list) or not metros:
        raise ValueError(f"Invalid metro config (missing non-empty 'metros' list): {config_path}")
    return metros


def parse_selected_ids(raw: str) -> set[str] | None:
    if not raw.strip():
        return None
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run progressive multi-metro ROI batch with resume and logs.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--metro-ids", type=str, default="", help="Comma-separated metro ids to run from config.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N metros after filtering.")

    parser.add_argument("--output-root", type=str, default="results/metro_batch")
    parser.add_argument("--logs-root", type=str, default="logs/metro_batch")
    parser.add_argument("--keep-last-runs", type=int, default=8)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--camera-catalog-csv", type=str, default=str(DEFAULT_CAMERA_CATALOG))
    parser.add_argument("--n-vehicles", type=int, default=5000)
    parser.add_argument("--n-trips", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-shortest", type=int, default=3)
    parser.add_argument("--p-return", type=float, default=0.6)
    parser.add_argument("--detection-radius-m", type=float, default=100.0)
    parser.add_argument("--ring-breaks-km", type=str, default="0,5,10,20,40,80")
    parser.add_argument("--max-radius-km", type=float, default=80.0)
    parser.add_argument("--require-aadt", action="store_true")

    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--require-fresh-data", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--blas-threads", type=int, default=1)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if args.blas_threads <= 0:
        raise ValueError("--blas-threads must be >= 1")

    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = str(args.blas_threads)

    output_root = (BASE_DIR / args.output_root).resolve()
    logs_root = (BASE_DIR / args.logs_root).resolve()
    rotate_run_logs(logs_root, args.keep_last_runs)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    preflight_log = run_dir / "preflight.log"
    run_log = run_dir / "metro_batch.log"
    summary_log = run_dir / "summary.log"
    state_path = run_dir / "state.json"

    catalog_path = Path(args.camera_catalog_csv).expanduser().resolve() if args.camera_catalog_csv else None
    if catalog_path and not catalog_path.exists():
        raise FileNotFoundError(f"Camera catalog not found: {catalog_path}")

    metros = load_metros(config_path)
    selected_ids = parse_selected_ids(args.metro_ids)
    if selected_ids is not None:
        metros = [m for m in metros if str(m.get("id", "")).lower() in selected_ids]
    if args.limit > 0:
        metros = metros[: args.limit]
    if not metros:
        raise ValueError("No metros selected to run after filtering.")
    ring_breaks = parse_ring_breaks(args.ring_breaks_km)
    selected_states = sorted({str(m.get("primary_state", "")).upper() for m in metros if m.get("primary_state")})
    states_with_explicit_aadt = {
        str(m.get("primary_state", "")).upper()
        for m in metros
        if str(m.get("primary_state", "")).strip()
        and (
            str(m.get("aadt_path", "")).strip()
            or (
                isinstance(m.get("aadt_paths", []), list)
                and any(str(x).strip() for x in m.get("aadt_paths", []))
            )
            or (
                not isinstance(m.get("aadt_paths", []), list)
                and str(m.get("aadt_paths", "")).strip()
            )
        )
    }
    missing_aadt_states = [
        s for s in selected_states if s not in KNOWN_AADT_STATES and s not in states_with_explicit_aadt
    ]

    state: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": utc_now_iso(),
        "status": "running",
        "config_path": str(config_path),
        "output_root": str(output_root),
        "logs_root": str(logs_root),
        "params": {
            "n_vehicles": args.n_vehicles,
            "n_trips": args.n_trips,
            "workers": args.workers,
            "seed": args.seed,
            "k_shortest": args.k_shortest,
            "p_return": args.p_return,
            "detection_radius_m": args.detection_radius_m,
            "ring_breaks_km": args.ring_breaks_km,
            "max_radius_km": args.max_radius_km,
            "force_rerun": args.force_rerun,
            "require_aadt": args.require_aadt,
            "blas_threads": args.blas_threads,
        },
        "selected_states": selected_states,
        "missing_aadt_states": missing_aadt_states,
        "metros": {},
    }
    for m in metros:
        metro_id = str(m.get("id"))
        state["metros"][metro_id] = {
            "status": "pending",
            "started_at_utc": None,
            "ended_at_utc": None,
            "exit_code": None,
            "output_dir": None,
            "note": None,
        }
    save_state(state_path, state)

    if missing_aadt_states:
        msg = (
            "[WARN] Selected states without built-in AADT mapping in current pipeline: "
            + ", ".join(missing_aadt_states)
            + " (these will run with OSM proxy traffic unless you provide --aadt-path in per-ROI runs or extend mappings)."
        )
        print(msg)
        with run_log.open("a", encoding="utf-8") as rlog:
            rlog.write(msg + "\n")
        if args.require_aadt:
            state["status"] = "failed_preflight"
            state["ended_at_utc"] = utc_now_iso()
            save_state(state_path, state)
            return 1

    if not args.skip_preflight:
        freshness_json = run_dir / "data_freshness.json"
        fresh_cmd = [
            args.python_exe,
            "scripts/check_data_freshness.py",
            "--strict-missing",
            "--output-json",
            str(freshness_json),
        ]
        if args.require_fresh_data:
            fresh_cmd.append("--require-fresh")
        rc = run_step(fresh_cmd, preflight_log, BASE_DIR)
        if rc != 0:
            state["status"] = "failed_preflight"
            state["ended_at_utc"] = utc_now_iso()
            save_state(state_path, state)
            return rc

        rc = run_step(
            [args.python_exe, "scripts/download_aadt_data.py", "--validate"],
            preflight_log,
            BASE_DIR,
        )
        if rc != 0:
            state["status"] = "failed_preflight"
            state["ended_at_utc"] = utc_now_iso()
            save_state(state_path, state)
            return rc

        if args.run_tests:
            rc = run_step([args.python_exe, "-m", "pytest", "-q"], preflight_log, BASE_DIR)
            if rc != 0:
                state["status"] = "failed_preflight"
                state["ended_at_utc"] = utc_now_iso()
                save_state(state_path, state)
                return rc

    any_failure = False
    output_root.mkdir(parents=True, exist_ok=True)

    for m in metros:
        metro_id = str(m["id"])
        display_name = str(m.get("display_name", metro_id))
        lat = float(m["center_lat"])
        lon = float(m["center_lon"])
        radius_km = float(m.get("radius_km", 25.0))
        state_code = str(m.get("primary_state", "")).strip()
        boundary_geojson = str(m.get("boundary_geojson", "")).strip()
        aadt_path_value = str(m.get("aadt_path", "")).strip()
        aadt_paths_value = m.get("aadt_paths", [])
        boundary_path: Path | None = None
        aadt_path: Path | None = None
        aadt_paths: list[Path] = []

        if boundary_geojson:
            p = Path(boundary_geojson)
            boundary_path = p if p.is_absolute() else (BASE_DIR / p)
            boundary_path = boundary_path.expanduser().resolve()
            if not boundary_path.exists():
                state["metros"][metro_id]["status"] = "failed"
                state["metros"][metro_id]["note"] = f"Boundary file not found: {boundary_path}"
                state["metros"][metro_id]["ended_at_utc"] = utc_now_iso()
                any_failure = True
                save_state(state_path, state)
                if args.stop_on_error:
                    break
                continue

        if aadt_path_value:
            p = Path(aadt_path_value)
            aadt_path = p if p.is_absolute() else (BASE_DIR / p)
            aadt_path = aadt_path.expanduser().resolve()
            if not aadt_path.exists():
                state["metros"][metro_id]["status"] = "failed"
                state["metros"][metro_id]["note"] = f"AADT file not found: {aadt_path}"
                state["metros"][metro_id]["ended_at_utc"] = utc_now_iso()
                any_failure = True
                save_state(state_path, state)
                if args.stop_on_error:
                    break
                continue
        elif aadt_paths_value:
            raw_paths: list[str]
            if isinstance(aadt_paths_value, list):
                raw_paths = [str(x).strip() for x in aadt_paths_value if str(x).strip()]
            else:
                raw_paths = [str(aadt_paths_value).strip()] if str(aadt_paths_value).strip() else []

            missing = []
            for raw in raw_paths:
                p = Path(raw)
                resolved = p if p.is_absolute() else (BASE_DIR / p)
                resolved = resolved.expanduser().resolve()
                if not resolved.exists():
                    missing.append(str(resolved))
                else:
                    aadt_paths.append(resolved)

            if missing:
                state["metros"][metro_id]["status"] = "failed"
                state["metros"][metro_id]["note"] = f"AADT files not found: {', '.join(missing)}"
                state["metros"][metro_id]["ended_at_utc"] = utc_now_iso()
                any_failure = True
                save_state(state_path, state)
                if args.stop_on_error:
                    break
                continue

        if radius_km <= 0 or radius_km > args.max_radius_km:
            state["metros"][metro_id]["status"] = "failed"
            state["metros"][metro_id]["note"] = (
                f"Invalid radius_km={radius_km}; must be in (0, {args.max_radius_km}]"
            )
            state["metros"][metro_id]["ended_at_utc"] = utc_now_iso()
            any_failure = True
            save_state(state_path, state)
            if args.stop_on_error:
                break
            continue

        expected_signature = {
            "n_vehicles": args.n_vehicles,
            "n_trips": args.n_trips,
            "seed": args.seed,
            "k_shortest": args.k_shortest,
            "p_return": args.p_return,
            "detection_radius_m": args.detection_radius_m,
            "ring_breaks_km": ring_breaks,
            "radius_km": radius_km,
            "center_lat": lat,
            "center_lon": lon,
            "boundary_geojson": str(boundary_path) if boundary_path else "",
            "aadt_path": str(aadt_path) if aadt_path else "",
            "aadt_paths_input": [str(p) for p in aadt_paths],
        }
        existing = find_latest_matching_output(output_root, metro_id, expected_signature)
        if existing and not args.force_rerun:
            msg = f"[SKIP] {metro_id}: existing output {existing}"
            print(msg)
            with run_log.open("a", encoding="utf-8") as rlog:
                rlog.write(msg + "\n")
            state["metros"][metro_id]["status"] = "skipped_existing"
            state["metros"][metro_id]["output_dir"] = str(existing)
            state["metros"][metro_id]["note"] = "Existing completed output found"
            state["metros"][metro_id]["ended_at_utc"] = utc_now_iso()
            save_state(state_path, state)
            continue

        metro_log = run_dir / f"metro_{metro_id}.log"
        state["metros"][metro_id]["status"] = "running"
        state["metros"][metro_id]["started_at_utc"] = utc_now_iso()
        save_state(state_path, state)

        cmd = [
            args.python_exe,
            "scripts/run_roi_analysis.py",
            "--name",
            metro_id,
            "--center-lat",
            str(lat),
            "--center-lon",
            str(lon),
            "--radius-km",
            str(radius_km),
            "--n-vehicles",
            str(args.n_vehicles),
            "--n-trips",
            str(args.n_trips),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--k-shortest",
            str(args.k_shortest),
            "--p-return",
            str(args.p_return),
            "--detection-radius-m",
            str(args.detection_radius_m),
            "--ring-breaks-km",
            args.ring_breaks_km,
            "--output-root",
            str(output_root),
        ]
        if state_code:
            cmd.extend(["--state", state_code])
        if boundary_path is not None:
            cmd.extend(["--boundary-geojson", str(boundary_path)])
        if args.require_aadt:
            cmd.append("--require-aadt")
        if aadt_path is not None:
            cmd.extend(["--aadt-path", str(aadt_path)])
        elif aadt_paths:
            cmd.extend(["--aadt-paths", ",".join(str(p) for p in aadt_paths)])
        if catalog_path:
            cmd.extend(["--camera-catalog-csv", str(catalog_path)])

        banner = f"[RUN] {metro_id} | {display_name} | center=({lat},{lon}) radius_km={radius_km}"
        print(banner)
        with run_log.open("a", encoding="utf-8") as rlog:
            rlog.write(f"{banner}\n")

        if args.dry_run:
            with metro_log.open("a", encoding="utf-8") as log_file:
                log_file.write("DRY RUN COMMAND:\n")
                log_file.write(" ".join(cmd) + "\n")
            state["metros"][metro_id]["status"] = "dry_run"
            state["metros"][metro_id]["note"] = "No execution (--dry-run)"
            state["metros"][metro_id]["ended_at_utc"] = utc_now_iso()
            save_state(state_path, state)
            continue

        rc = run_step(cmd, metro_log, BASE_DIR)
        completed = find_latest_matching_output(output_root, metro_id, expected_signature)

        state["metros"][metro_id]["exit_code"] = rc
        state["metros"][metro_id]["output_dir"] = str(completed) if completed else None
        state["metros"][metro_id]["ended_at_utc"] = utc_now_iso()

        if rc == 0 and completed is not None:
            state["metros"][metro_id]["status"] = "completed"
            state["metros"][metro_id]["note"] = "OK"
        else:
            state["metros"][metro_id]["status"] = "failed"
            state["metros"][metro_id]["note"] = "Non-zero exit code or missing summary artifacts"
            any_failure = True

        save_state(state_path, state)
        if any_failure and args.stop_on_error:
            break

    lines = []
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Started: {state['started_at_utc']}")
    lines.append(f"Ended:   {utc_now_iso()}")
    lines.append("")
    lines.append("Per-metro status:")
    for metro_id, info in state["metros"].items():
        lines.append(
            f"{metro_id:32s} {info['status']:16s} exit={str(info['exit_code']):4s} output={str(info['output_dir'])}"
        )
    summary_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    state["ended_at_utc"] = utc_now_iso()
    state["status"] = "completed_with_failures" if any_failure else "completed"
    save_state(state_path, state)

    if any_failure:
        print(f"[WARN] Metro batch completed with failures. See: {summary_log}")
        return 1

    print(f"[INFO] Metro batch completed successfully. See: {summary_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
