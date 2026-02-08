#!/usr/bin/env python3
"""
Input data freshness and quality preflight checks.

Checks:
1. Required camera and AADT files exist
2. File age thresholds (warning level)
3. AADT column detectability using the same logic as runtime pipeline
4. Year-column sanity (warn if selected AADT column is not the latest year column)

Exit codes:
- 0: Checks passed (or warnings only)
- 1: Missing required data (or freshness required and stale data found)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd

from src.traffic_weights import detect_aadt_column


REGIONS = ["atlanta", "memphis", "richmond", "charlotte", "lehigh_valley", "maine"]
REGION_STATES = {
    "atlanta": "GA",
    "memphis": "TN",
    "richmond": "VA",
    "charlotte": "NC",
    "lehigh_valley": "PA",
    "maine": "ME",
}
AADT_PATHS = {
    "GA": "data/raw/aadt/georgia_aadt.shp",
    "TN": "data/raw/aadt/tennessee_aadt.shp",
    "VA": "data/raw/aadt/virginia_aadt.shp",
    "NC": "data/raw/aadt/north_carolina_aadt.shp",
    "PA": "data/raw/aadt/pennsylvania_aadt.shp",
    "ME": "data/raw/aadt/maine_aadt.shp",
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def file_age_days(path: Path) -> float:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    delta = now_utc() - modified
    return delta.total_seconds() / 86400.0


def extract_years_from_columns(columns: list[str]) -> list[int]:
    years: list[int] = []
    for col in columns:
        for y in re.findall(r"(?:19|20)\d{2}", col):
            years.append(int(y))
    return sorted(set(years))


def extract_year_from_column(col_name: str | None) -> int | None:
    if not col_name:
        return None
    matches = re.findall(r"(?:19|20)\d{2}", col_name)
    if not matches:
        return None
    return max(int(m) for m in matches)


@dataclass
class FileCheck:
    path: str
    exists: bool
    age_days: float | None = None
    status: str = "OK"
    warning: str | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def check_camera_files(base_dir: Path, max_age_days: int) -> dict[str, FileCheck]:
    out: dict[str, FileCheck] = {}
    for region in REGIONS:
        rel = f"data/raw/{region}_cameras.geojson"
        path = base_dir / rel
        if not path.exists():
            out[region] = FileCheck(path=rel, exists=False, status="ERROR", error="Missing camera GeoJSON")
            continue

        age = file_age_days(path)
        status = "WARN" if age > max_age_days else "OK"
        warn = (
            f"Camera map is {age:.1f} days old (threshold {max_age_days}d). Consider refreshing map data."
            if status == "WARN"
            else None
        )
        out[region] = FileCheck(path=rel, exists=True, age_days=age, status=status, warning=warn)
    return out


def check_aadt_files(base_dir: Path, max_age_days: int) -> dict[str, FileCheck]:
    out: dict[str, FileCheck] = {}
    for region in REGIONS:
        state = REGION_STATES[region]
        rel = AADT_PATHS[state]
        path = base_dir / rel
        if not path.exists():
            out[region] = FileCheck(path=rel, exists=False, status="ERROR", error="Missing AADT shapefile")
            continue

        age = file_age_days(path)
        status = "WARN" if age > max_age_days else "OK"
        warning: str | None = None
        details: dict[str, Any] = {"state": state}

        try:
            # Read just enough schema rows for column introspection.
            gdf = gpd.read_file(path, rows=5)
            col = detect_aadt_column(gdf)
            all_cols = list(gdf.columns)
            years = extract_years_from_columns(all_cols)
            latest_year = max(years) if years else None
            chosen_year = extract_year_from_column(col)

            details.update(
                {
                    "detected_aadt_column": col,
                    "latest_year_column": latest_year,
                    "detected_column_year": chosen_year,
                }
            )

            if not col:
                status = "ERROR"
                warning = None
                out[region] = FileCheck(
                    path=rel,
                    exists=True,
                    age_days=age,
                    status=status,
                    error="Could not detect usable AADT column",
                    details=details,
                )
                continue

            if status == "WARN":
                warning = f"AADT file is {age:.1f} days old (threshold {max_age_days}d)."

            # Year sanity warning: selected column is older than available year columns.
            if latest_year and chosen_year and chosen_year < latest_year:
                msg = (
                    f"Detected column '{col}' is year {chosen_year}, but newer year {latest_year} exists. "
                    "Verify intended AADT year selection."
                )
                warning = f"{warning} {msg}".strip() if warning else msg
                if status == "OK":
                    status = "WARN"

        except Exception as exc:
            status = "ERROR"
            out[region] = FileCheck(
                path=rel,
                exists=True,
                age_days=age,
                status=status,
                error=f"AADT file read/check failed: {exc}",
                details=details,
            )
            continue

        out[region] = FileCheck(
            path=rel,
            exists=True,
            age_days=age,
            status=status,
            warning=warning,
            details=details,
        )
    return out


def summarize(camera: dict[str, FileCheck], aadt: dict[str, FileCheck]) -> dict[str, Any]:
    all_checks = list(camera.values()) + list(aadt.values())
    n_error = sum(1 for c in all_checks if c.status == "ERROR")
    n_warn = sum(1 for c in all_checks if c.status == "WARN")
    return {
        "errors": n_error,
        "warnings": n_warn,
        "ok": len(all_checks) - n_error - n_warn,
        "total": len(all_checks),
    }


def print_report(camera: dict[str, FileCheck], aadt: dict[str, FileCheck], summary: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("DATA FRESHNESS & QUALITY CHECK")
    print("=" * 72)

    print("\nCamera files:")
    for region in REGIONS:
        c = camera[region]
        age = f"{c.age_days:.1f}d" if c.age_days is not None else "n/a"
        print(f"- {region:14s} [{c.status}] age={age:>7s}  path={c.path}")
        if c.warning:
            print(f"    warning: {c.warning}")
        if c.error:
            print(f"    error:   {c.error}")

    print("\nAADT files:")
    for region in REGIONS:
        c = aadt[region]
        age = f"{c.age_days:.1f}d" if c.age_days is not None else "n/a"
        col = c.details.get("detected_aadt_column")
        yr = c.details.get("detected_column_year")
        latest = c.details.get("latest_year_column")
        print(
            f"- {region:14s} [{c.status}] age={age:>7s}  col={str(col):<16s} "
            f"col_year={str(yr):<4s} latest={str(latest):<4s}  path={c.path}"
        )
        if c.warning:
            print(f"    warning: {c.warning}")
        if c.error:
            print(f"    error:   {c.error}")

    print("\nSummary:")
    print(
        f"- total={summary['total']}  ok={summary['ok']}  warnings={summary['warnings']}  errors={summary['errors']}"
    )
    print("=" * 72 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check camera/AADT data freshness before long runs.")
    parser.add_argument("--camera-max-age-days", type=int, default=180)
    parser.add_argument("--aadt-max-age-days", type=int, default=730)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Exit non-zero if any required input file is missing or unreadable.",
    )
    parser.add_argument(
        "--require-fresh",
        action="store_true",
        help="Exit non-zero on any warning (age/year warnings included).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    camera = check_camera_files(base_dir=base_dir, max_age_days=args.camera_max_age_days)
    aadt = check_aadt_files(base_dir=base_dir, max_age_days=args.aadt_max_age_days)
    summary = summarize(camera=camera, aadt=aadt)

    print_report(camera=camera, aadt=aadt, summary=summary)

    payload = {
        "generated_at_utc": now_utc().isoformat(),
        "camera": {k: asdict(v) for k, v in camera.items()},
        "aadt": {k: asdict(v) for k, v in aadt.items()},
        "summary": summary,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {out}")

    if args.require_fresh and (summary["warnings"] > 0 or summary["errors"] > 0):
        return 1
    if args.strict_missing and summary["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

