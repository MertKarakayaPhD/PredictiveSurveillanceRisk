#!/usr/bin/env python3
"""
Generate manuscript-eligible ALPR camera GeoJSON for a metro ROI.

Usage examples:
  python scripts/generate_metro_camera_geojson.py --metro-id philadelphia_pa
  python scripts/generate_metro_camera_geojson.py --name pittsburgh_pa --center-lat 40.4406 --center-lon -79.9959 --radius-km 30
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# Ensure project root imports work when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import query_alpr_cameras, save_cameras


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METRO_CONFIG = BASE_DIR / "data" / "external" / "metro_batch" / "metros_us_32.json"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    slug = slug.strip("_").lower()
    return slug or "metro"


def bbox_from_center_radius(center_lat: float, center_lon: float, radius_km: float) -> tuple[float, float, float, float]:
    if radius_km <= 0:
        raise ValueError("--radius-km must be > 0")

    dlat = radius_km / 111.0
    cos_lat = math.cos(math.radians(center_lat))
    if abs(cos_lat) < 1e-9:
        raise ValueError("Invalid latitude for bbox conversion")
    dlon = radius_km / (111.0 * cos_lat)
    return (center_lat - dlat, center_lon - dlon, center_lat + dlat, center_lon + dlon)


def load_metro_from_config(config_path: Path, metro_id: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    metros = payload.get("metros", [])
    if not isinstance(metros, list):
        raise ValueError(f"Invalid metro config format: {config_path}")

    target = metro_id.strip().lower()
    for m in metros:
        if str(m.get("id", "")).strip().lower() == target:
            return m
    raise ValueError(f"Metro id '{metro_id}' not found in config: {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manuscript-eligible metro camera GeoJSON from OSM/Overpass.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--metro-id", type=str, default="", help="Metro id from config (e.g., philadelphia_pa).")
    mode.add_argument("--name", type=str, default="", help="Custom metro name (used for output filename).")

    parser.add_argument("--config", type=str, default=str(DEFAULT_METRO_CONFIG), help="Metro config JSON path.")
    parser.add_argument("--center-lat", type=float, default=None, help="Center latitude (required with --name).")
    parser.add_argument("--center-lon", type=float, default=None, help="Center longitude (required with --name).")
    parser.add_argument("--radius-km", type=float, default=None, help="Radius in km (required with --name).")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output GeoJSON path (default: data/raw/<name>_cameras.geojson).",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Overpass timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Overpass retry count.")
    args = parser.parse_args()

    if args.metro_id:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        metro = load_metro_from_config(config_path, args.metro_id)
        name = slugify(str(metro.get("id", args.metro_id)))
        center_lat = float(metro["center_lat"])
        center_lon = float(metro["center_lon"])
        radius_km = float(metro["radius_km"])
    else:
        if args.center_lat is None or args.center_lon is None or args.radius_km is None:
            raise ValueError("--center-lat, --center-lon, and --radius-km are required with --name")
        name = slugify(args.name)
        center_lat = float(args.center_lat)
        center_lon = float(args.center_lon)
        radius_km = float(args.radius_km)

    south, west, north, east = bbox_from_center_radius(center_lat=center_lat, center_lon=center_lon, radius_km=radius_km)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (BASE_DIR / "data" / "raw" / f"{name}_cameras.geojson").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Name: {name}")
    print(f"[INFO] Center: ({center_lat}, {center_lon})")
    print(f"[INFO] Radius km: {radius_km}")
    print(f"[INFO] BBox (s,w,n,e): ({south}, {west}, {north}, {east})")
    print(f"[INFO] Output: {output_path}")

    gdf = query_alpr_cameras(
        bbox=(south, west, north, east),
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    save_cameras(gdf, output_path)
    print(f"[OK] Saved {len(gdf)} cameras to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

