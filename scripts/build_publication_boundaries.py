#!/usr/bin/env python3
"""
Build publication-grade metro boundary files from Census TIGER/Line CBSA polygons.

This script:
1. Loads a metro config (center/radius proxies)
2. Matches each metro display name to CBSA polygons
3. Exports per-metro boundary GeoJSON files
4. Writes an updated metro config with boundary_geojson + boundary metadata
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    v = value.lower().strip()
    v = re.sub(r"[^a-z0-9]+", "", v)
    return v


def haversine_km(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def boundary_centroid_and_area(gdf: gpd.GeoDataFrame) -> tuple[float, float, float]:
    utm = gdf.estimate_utm_crs()
    if utm is None:
        c = gdf.geometry.iloc[0].centroid
        return float(c.y), float(c.x), 0.0

    proj = gdf.to_crs(utm)
    c_proj = proj.geometry.iloc[0].centroid
    c_ll = gpd.GeoSeries([c_proj], crs=utm).to_crs("EPSG:4326").iloc[0]
    area_km2 = float(proj.geometry.iloc[0].area / 1e6)
    return float(c_ll.y), float(c_ll.x), area_km2


def boundary_max_radius_km(gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float) -> float:
    boundary = gdf.geometry.iloc[0].boundary
    coords: list[tuple[float, float]] = []

    if boundary.geom_type == "LineString":
        coords = [(float(x), float(y)) for x, y in boundary.coords]
    elif boundary.geom_type == "MultiLineString":
        for line in boundary.geoms:
            coords.extend((float(x), float(y)) for x, y in line.coords)

    if not coords:
        return 0.0

    lons = np.array([x for x, _ in coords], dtype=float)
    lats = np.array([y for _, y in coords], dtype=float)
    return float(haversine_km(center_lat, center_lon, lats, lons).max())


def build_cbsa_index(cbsa_gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in cbsa_gdf.iterrows():
        raw_name = str(row.get("NAME", "")).strip()
        core_name = raw_name.split(",")[0].strip()
        out.append(
            {
                "idx": idx,
                "name": raw_name,
                "core_name": core_name,
                "norm_name": normalize_text(raw_name),
                "norm_core": normalize_text(core_name),
                "geoid": str(row.get("GEOID", "")),
            }
        )
    return out


def best_match_cbsa(display_name: str, index_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    target = normalize_text(display_name)
    target_core = normalize_text(display_name.split(",")[0].strip())

    best: dict[str, Any] | None = None
    best_score = -1.0
    for item in index_rows:
        score_full = SequenceMatcher(None, target, item["norm_core"]).ratio()
        score_core = SequenceMatcher(None, target_core, item["norm_core"]).ratio()
        score = max(score_full, score_core)

        if item["norm_core"] == target or item["norm_core"] == target_core:
            score += 0.5
        elif item["norm_core"].startswith(target_core) or target_core.startswith(item["norm_core"]):
            score += 0.2

        if score > best_score:
            best_score = score
            best = item

    return best, best_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Build publication-grade metro boundaries from CBSA polygons.")
    parser.add_argument("--cbsa-shapefile", type=str, required=True, help="Path to tl_YYYY_us_cbsa.shp")
    parser.add_argument(
        "--metro-config",
        type=str,
        default=str(BASE_DIR / "data" / "external" / "metro_batch" / "metros_us_32.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(BASE_DIR / "data" / "external" / "metro_batch" / "boundaries_cbsa"),
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=str(BASE_DIR / "data" / "external" / "metro_batch" / "metros_us_32_publication.json"),
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=0.70,
        help="Minimum fuzzy-match score required for automatic CBSA assignment.",
    )
    parser.add_argument(
        "--max-radius-km",
        type=float,
        default=80.0,
        help="Clamp radius_km in output config to this value for operational safety.",
    )
    args = parser.parse_args()

    cbsa_path = Path(args.cbsa_shapefile).expanduser().resolve()
    metro_config_path = Path(args.metro_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    output_config_path = Path(args.output_config).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)

    if not cbsa_path.exists():
        raise FileNotFoundError(f"CBSA shapefile not found: {cbsa_path}")
    if not metro_config_path.exists():
        raise FileNotFoundError(f"Metro config not found: {metro_config_path}")

    cbsa = gpd.read_file(cbsa_path)
    if cbsa.crs is None:
        cbsa = cbsa.set_crs("EPSG:4269")
    cbsa = cbsa.to_crs("EPSG:4326")
    if "NAME" not in cbsa.columns:
        raise ValueError("CBSA shapefile must include 'NAME' column.")

    config = json.loads(metro_config_path.read_text(encoding="utf-8"))
    metros = config.get("metros", [])
    if not metros:
        raise ValueError("Metro config has empty 'metros' list.")

    cbsa_index = build_cbsa_index(cbsa)
    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "cbsa_shapefile": str(cbsa_path),
        "metro_config_input": str(metro_config_path),
        "output_config": str(output_config_path),
        "output_boundaries_dir": str(out_dir),
        "matches": [],
        "unmatched": [],
    }

    updated_metros: list[dict[str, Any]] = []
    for metro in metros:
        metro_id = str(metro.get("id", "")).strip()
        display_name = str(metro.get("display_name", metro_id)).strip()
        match, score = best_match_cbsa(display_name, cbsa_index)
        if match is None or score < args.min_match_score:
            report["unmatched"].append(
                {"metro_id": metro_id, "display_name": display_name, "best_score": score}
            )
            updated_metros.append(metro)
            continue

        row = cbsa.loc[match["idx"]]
        geom = row.geometry
        if geom is None or geom.is_empty:
            report["unmatched"].append(
                {"metro_id": metro_id, "display_name": display_name, "reason": "empty_geometry"}
            )
            updated_metros.append(metro)
            continue

        boundary_gdf = gpd.GeoDataFrame(
            [
                {
                    "metro_id": metro_id,
                    "metro_display_name": display_name,
                    "cbsa_name": str(row.get("NAME", "")),
                    "cbsa_geoid": str(row.get("GEOID", "")),
                    "geometry": geom,
                }
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        boundary_path = out_dir / f"{metro_id}.geojson"
        boundary_gdf.to_file(boundary_path, driver="GeoJSON")

        c_lat, c_lon, area_km2 = boundary_centroid_and_area(boundary_gdf[["geometry"]])
        max_radius_km = boundary_max_radius_km(boundary_gdf[["geometry"]], c_lat, c_lon)
        clamped_radius = min(max_radius_km, float(args.max_radius_km))

        rel_boundary_path = boundary_path.relative_to(BASE_DIR)
        metro_updated = dict(metro)
        metro_updated["center_lat"] = c_lat
        metro_updated["center_lon"] = c_lon
        metro_updated["radius_km"] = clamped_radius
        metro_updated["boundary_geojson"] = str(rel_boundary_path).replace("\\", "/")
        metro_updated["boundary"] = {
            "mode": "polygon_cbsa",
            "source": "census_tiger_cbsa",
            "cbsa_name": str(row.get("NAME", "")),
            "cbsa_geoid": str(row.get("GEOID", "")),
            "area_km2": area_km2,
            "max_radius_km": max_radius_km,
            "radius_clamped_km": clamped_radius,
        }
        updated_metros.append(metro_updated)

        report["matches"].append(
            {
                "metro_id": metro_id,
                "display_name": display_name,
                "cbsa_name": str(row.get("NAME", "")),
                "cbsa_geoid": str(row.get("GEOID", "")),
                "match_score": score,
                "boundary_geojson": str(rel_boundary_path).replace("\\", "/"),
                "center_lat": c_lat,
                "center_lon": c_lon,
                "area_km2": area_km2,
                "max_radius_km": max_radius_km,
                "radius_clamped_km": clamped_radius,
            }
        )

    updated_config = dict(config)
    updated_config["generated_on"] = datetime.now().strftime("%Y-%m-%d")
    updated_config["boundary_policy"] = {
        "mode": "polygon_cbsa",
        "source": "census_tiger_cbsa",
        "note": "Publication-grade metro boundaries derived from Census CBSA polygons.",
        "max_radius_km": float(args.max_radius_km),
    }
    updated_config["metros"] = updated_metros
    output_config_path.write_text(json.dumps(updated_config, indent=2), encoding="utf-8")

    report_path = output_config_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote publication config: {output_config_path}")
    print(f"[INFO] Wrote boundary-match report: {report_path}")
    print(f"[INFO] Matched metros: {len(report['matches'])}")
    print(f"[INFO] Unmatched metros: {len(report['unmatched'])}")

    return 0 if len(report["unmatched"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
