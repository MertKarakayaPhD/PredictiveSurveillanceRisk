#!/usr/bin/env python3
"""
Run traffic-weighted ALPR simulation + PSR for a custom ROI around coordinates.

Workflow:
1. Clip camera data to a user-provided circle ROI or polygon boundary
2. Resolve AADT input (existing local file, curated state file, or proxy-only fallback)
3. Run traffic-weighted road-network simulation
4. Compute U(2), predictability, and PSR
5. Save machine-readable outputs for reproducibility
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

# Ensure project root imports work when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_aadt_data import AADT_SOURCES
from scripts.road_network_simulation import (
    compute_random_point_uniqueness,
    simulate_road_network_trips,
)
from src.analysis import analyze_predictability, compute_psr
from src.prediction import MarkovPredictor, evaluate_predictor
from src.simulation import Trajectory
from src.traffic_weights import compute_traffic_weights_for_region, detect_aadt_column


BASE_DIR = Path(__file__).resolve().parent.parent
AADT_DIR = BASE_DIR / "data" / "raw" / "aadt"
DEFAULT_CAMERA_CATALOG = BASE_DIR / "data" / "external" / "camera_catalog" / "cameras_us_active.csv.gz"

AADT_PATHS = {
    "GA": AADT_DIR / "georgia_aadt.shp",
    "TN": AADT_DIR / "tennessee_aadt.shp",
    "VA": AADT_DIR / "virginia_aadt.shp",
    "NC": AADT_DIR / "north_carolina_aadt.shp",
    "PA": AADT_DIR / "pennsylvania_aadt.shp",
    "ME": AADT_DIR / "maine_aadt.shp",
}

STATE_NAME_TO_ABBR = {
    "georgia": "GA",
    "tennessee": "TN",
    "virginia": "VA",
    "north carolina": "NC",
    "pennsylvania": "PA",
    "maine": "ME",
}

DEFAULT_TRAFFIC_CONFIG = {
    "aadt_line_buffer_m": 50.0,
    "aadt_point_buffer_m": 100.0,
    "aadt_coverage_threshold": 0.005,
    "normalize_method": "log",
    "clip_percentile": 99.0,
    "traffic_blend_factor": 0.7,
    "lambda_traffic": 0.5,
    "min_trip_distance_m": 8047.0,
    "max_trip_distance_m": 16093.0,
}

HARD_MAX_RADIUS_KM = 80.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return slug.strip("_").lower() or "roi"


def normalize_state_code(state_value: str | None) -> str | None:
    if not state_value:
        return None
    s = state_value.strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return STATE_NAME_TO_ABBR.get(s.lower())


def infer_state_from_coordinates(lat: float, lon: float) -> str | None:
    """
    Best-effort reverse geocoding.

    Returns None if geopy/network is unavailable or state cannot be resolved.
    """
    try:
        from geopy.geocoders import Nominatim

        geocoder = Nominatim(user_agent="alpr_model_roi")
        location = geocoder.reverse((lat, lon), exactly_one=True, timeout=10)
        if not location:
            return None

        address = location.raw.get("address", {})
        code = address.get("state_code")
        if code:
            code = code.upper()
            if len(code) == 2:
                return code

        state_name = address.get("state")
        return normalize_state_code(state_name)
    except Exception:
        return None


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


def to_wgs84_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    return gdf


def load_boundary_geometry(boundary_geojson: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundary_geojson)
    if gdf.empty:
        raise ValueError(f"No features found in boundary file: {boundary_geojson}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    poly = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if poly.empty:
        raise ValueError("Boundary file must contain Polygon/MultiPolygon geometries.")

    geom = poly.union_all()
    if geom.is_empty:
        raise ValueError("Boundary geometry is empty after dissolve.")
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Boundary geometry must resolve to polygonal type, got: {geom.geom_type}")

    return gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")


def geometry_area_km2(geom_gdf: gpd.GeoDataFrame) -> float:
    if geom_gdf.empty:
        return 0.0
    utm = geom_gdf.estimate_utm_crs()
    if utm is None:
        return 0.0
    proj = geom_gdf.to_crs(utm)
    return float(proj.geometry.iloc[0].area / 1e6)


def boundary_centroid_lat_lon(geom_gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    if geom_gdf.empty:
        raise ValueError("Cannot compute centroid for empty geometry.")
    utm = geom_gdf.estimate_utm_crs()
    if utm is None:
        c = geom_gdf.geometry.iloc[0].centroid
        return float(c.y), float(c.x)
    proj = geom_gdf.to_crs(utm)
    centroid_proj = proj.geometry.iloc[0].centroid
    centroid_ll = gpd.GeoSeries([centroid_proj], crs=utm).to_crs("EPSG:4326").iloc[0]
    return float(centroid_ll.y), float(centroid_ll.x)


def boundary_max_radius_km(geom_gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float) -> float:
    if geom_gdf.empty:
        return 0.0
    boundary = geom_gdf.geometry.iloc[0].boundary
    coords: list[tuple[float, float]] = []

    if boundary.geom_type == "LineString":
        coords = [(float(x), float(y)) for x, y in boundary.coords]
    elif boundary.geom_type == "MultiLineString":
        for line in boundary.geoms:
            coords.extend((float(x), float(y)) for x, y in line.coords)

    if not coords:
        return 0.0

    lons = np.array([c[0] for c in coords], dtype=float)
    lats = np.array([c[1] for c in coords], dtype=float)
    return float(haversine_km(center_lat, center_lon, lats, lons).max())


def parse_ring_breaks_km(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    ring_breaks = sorted({float(p) for p in parts})
    if ring_breaks[0] != 0.0:
        ring_breaks = [0.0] + ring_breaks
    return ring_breaks


def build_ring_metrics(
    clipped_gdf: gpd.GeoDataFrame,
    trajectories: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float,
    n_vehicles: int,
    ring_breaks_km: list[float],
) -> dict[str, Any]:
    if clipped_gdf.empty:
        return {"ring_breaks_km": [], "per_ring": [], "gradient": {}}

    valid_breaks = sorted({b for b in ring_breaks_km if 0.0 <= b <= radius_km})
    if 0.0 not in valid_breaks:
        valid_breaks = [0.0] + valid_breaks
    if radius_km not in valid_breaks:
        valid_breaks.append(radius_km)
    if len(valid_breaks) < 2:
        valid_breaks = [0.0, radius_km]

    lats = clipped_gdf.geometry.y.to_numpy()
    lons = clipped_gdf.geometry.x.to_numpy()
    cam_dist_km = haversine_km(center_lat, center_lon, lats, lons)

    def ring_idx(dist_km: float) -> int:
        idx = int(np.searchsorted(valid_breaks, dist_km, side="right")) - 1
        if idx < 0:
            return 0
        if idx >= len(valid_breaks) - 1:
            return len(valid_breaks) - 2
        return idx

    n_rings = len(valid_breaks) - 1
    ring_camera_counts = [0] * n_rings
    cam_to_ring: dict[int, int] = {}
    for cam_id, d in enumerate(cam_dist_km.tolist()):
        ridx = ring_idx(d)
        cam_to_ring[cam_id] = ridx
        ring_camera_counts[ridx] += 1

    ring_hit_counts = [0] * n_rings
    ring_vehicle_seen: list[set[str]] = [set() for _ in range(n_rings)]
    total_hits = 0
    for traj in trajectories:
        vid = str(traj.get("vehicle_id", ""))
        hits = traj.get("camera_hits", [])
        seen_rings_for_vehicle: set[int] = set()
        for h in hits:
            if not isinstance(h, int):
                continue
            ridx = cam_to_ring.get(h)
            if ridx is None:
                continue
            ring_hit_counts[ridx] += 1
            total_hits += 1
            seen_rings_for_vehicle.add(ridx)
        for ridx in seen_rings_for_vehicle:
            ring_vehicle_seen[ridx].add(vid)

    per_ring: list[dict[str, Any]] = []
    for i in range(n_rings):
        r0 = valid_breaks[i]
        r1 = valid_breaks[i + 1]
        area_km2 = float(np.pi * (r1**2 - r0**2))
        camera_density = (ring_camera_counts[i] / area_km2) if area_km2 > 0 else 0.0
        hit_share = (ring_hit_counts[i] / total_hits) if total_hits > 0 else 0.0
        veh_obs_rate = (len(ring_vehicle_seen[i]) / n_vehicles) if n_vehicles > 0 else 0.0
        per_ring.append(
            {
                "ring_label": f"{r0:.1f}-{r1:.1f}km",
                "inner_km": r0,
                "outer_km": r1,
                "mid_km": (r0 + r1) / 2.0,
                "area_km2": area_km2,
                "n_cameras": int(ring_camera_counts[i]),
                "camera_density_per_km2": float(camera_density),
                "n_camera_hits": int(ring_hit_counts[i]),
                "camera_hit_share": float(hit_share),
                "n_vehicles_observed": int(len(ring_vehicle_seen[i])),
                "vehicle_observation_rate": float(veh_obs_rate),
            }
        )

    core = per_ring[0]
    periphery = per_ring[-1]
    x = np.array([r["mid_km"] for r in per_ring], dtype=float)
    y_density = np.array([r["camera_density_per_km2"] for r in per_ring], dtype=float)
    y_vehicle = np.array([r["vehicle_observation_rate"] for r in per_ring], dtype=float)
    y_hits = np.array([r["camera_hit_share"] for r in per_ring], dtype=float)

    def safe_slope(xv: np.ndarray, yv: np.ndarray) -> float:
        if len(xv) < 2 or np.allclose(xv, xv[0]):
            return 0.0
        return float(np.polyfit(xv, yv, 1)[0])

    gradient = {
        "core_ring": core["ring_label"],
        "periphery_ring": periphery["ring_label"],
        "core_minus_periphery_camera_density": float(
            core["camera_density_per_km2"] - periphery["camera_density_per_km2"]
        ),
        "core_minus_periphery_vehicle_observation_rate": float(
            core["vehicle_observation_rate"] - periphery["vehicle_observation_rate"]
        ),
        "core_minus_periphery_camera_hit_share": float(
            core["camera_hit_share"] - periphery["camera_hit_share"]
        ),
        "surveillance_gradient_index": float(
            core["vehicle_observation_rate"] - periphery["vehicle_observation_rate"]
        ),
        "slope_camera_density_per_km": safe_slope(x, y_density),
        "slope_vehicle_observation_rate_per_km": safe_slope(x, y_vehicle),
        "slope_camera_hit_share_per_km": safe_slope(x, y_hits),
    }

    return {
        "ring_breaks_km": valid_breaks,
        "per_ring": per_ring,
        "gradient": gradient,
        "total_camera_hits_counted": int(total_hits),
    }


def clip_cameras_to_radius(
    camera_geojson: Path,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    out_geojson: Path,
) -> tuple[gpd.GeoDataFrame, int]:
    gdf = gpd.read_file(camera_geojson)
    if gdf.empty:
        raise ValueError(f"No features found in camera file: {camera_geojson}")

    gdf = to_wgs84_points(gdf)
    if gdf.empty:
        raise ValueError("No Point geometries in camera file after filtering.")

    all_count = len(gdf)
    lats = gdf.geometry.y.to_numpy()
    lons = gdf.geometry.x.to_numpy()
    distances = haversine_km(center_lat, center_lon, lats, lons)
    gdf = gdf[distances <= radius_km].copy()

    if gdf.empty:
        raise ValueError(
            "No cameras remain inside ROI radius. Increase radius or provide a broader camera source file."
        )

    gdf = gdf.reset_index(drop=True)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf, all_count


def clip_cameras_to_boundary(
    camera_geojson: Path,
    boundary_gdf: gpd.GeoDataFrame,
    out_geojson: Path,
) -> tuple[gpd.GeoDataFrame, int]:
    gdf = gpd.read_file(camera_geojson)
    if gdf.empty:
        raise ValueError(f"No features found in camera file: {camera_geojson}")

    gdf = to_wgs84_points(gdf)
    if gdf.empty:
        raise ValueError("No Point geometries in camera file after filtering.")

    all_count = len(gdf)
    boundary_geom = boundary_gdf.geometry.iloc[0]
    gdf = gdf[gdf.geometry.intersects(boundary_geom)].copy()
    if gdf.empty:
        raise ValueError("No cameras intersect the boundary polygon.")

    gdf = gdf.reset_index(drop=True)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf, all_count


def clip_catalog_to_radius(
    camera_catalog_csv: Path,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    out_geojson: Path,
) -> tuple[gpd.GeoDataFrame, int]:
    df = pd.read_csv(camera_catalog_csv, compression="infer")
    if df.empty:
        raise ValueError(f"No rows in camera catalog: {camera_catalog_csv}")
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("Camera catalog must include 'lat' and 'lon' columns.")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        raise ValueError("Camera catalog has no valid lat/lon rows after numeric coercion.")

    all_count = len(df)
    distances = haversine_km(
        center_lat,
        center_lon,
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
    )
    roi_df = df[distances <= radius_km].copy()
    if roi_df.empty:
        raise ValueError(
            "No cameras remain inside ROI radius from catalog. Increase radius or verify coordinates."
        )

    roi_df = roi_df.reset_index(drop=True)
    geometry = gpd.points_from_xy(roi_df["lon"], roi_df["lat"])
    gdf = gpd.GeoDataFrame(roi_df, geometry=geometry, crs="EPSG:4326")

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf, all_count


def clip_catalog_to_boundary(
    camera_catalog_csv: Path,
    boundary_gdf: gpd.GeoDataFrame,
    out_geojson: Path,
) -> tuple[gpd.GeoDataFrame, int]:
    df = pd.read_csv(camera_catalog_csv, compression="infer")
    if df.empty:
        raise ValueError(f"No rows in camera catalog: {camera_catalog_csv}")
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("Camera catalog must include 'lat' and 'lon' columns.")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        raise ValueError("Camera catalog has no valid lat/lon rows after numeric coercion.")

    all_count = len(df)
    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    boundary_geom = boundary_gdf.geometry.iloc[0]
    gdf = gdf[gdf.geometry.intersects(boundary_geom)].copy()
    if gdf.empty:
        raise ValueError("No catalog cameras intersect the boundary polygon.")

    gdf = gdf.reset_index(drop=True)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf, all_count


def geometry_bbox_with_margin(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    margin_deg: float = 0.05,
) -> tuple[float, float, float, float]:
    west = minx - margin_deg
    south = miny - margin_deg
    east = maxx + margin_deg
    north = maxy + margin_deg
    return (west, south, east, north)


def camera_bbox_with_margin(gdf: gpd.GeoDataFrame, margin_deg: float = 0.05) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return geometry_bbox_with_margin(minx=minx, miny=miny, maxx=maxx, maxy=maxy, margin_deg=margin_deg)


def boundary_bbox_with_margin(boundary_gdf: gpd.GeoDataFrame, margin_deg: float = 0.05) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    return geometry_bbox_with_margin(minx=minx, miny=miny, maxx=maxx, maxy=maxy, margin_deg=margin_deg)


def convex_hull_area_km2(gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty:
        return 0.0
    utm = gdf.estimate_utm_crs()
    if utm is None:
        return 0.0
    gdf_proj = gdf.to_crs(utm)
    return float(gdf_proj.union_all().convex_hull.area / 1e6)


def compute_markov_acc5(trajectories: list[dict], seed: int, order: int = 1) -> dict[str, float]:
    traj_objs = [
        Trajectory(vehicle_id=t["vehicle_id"], camera_sequence=t.get("camera_hits", []))
        for t in trajectories
    ]
    traj_objs = [t for t in traj_objs if len(t.camera_sequence) >= (order + 1)]

    if len(traj_objs) < 4:
        return {
            "accuracy@5": 0.0,
            "coverage": 0.0,
            "n_predictions": 0,
            "n_no_prediction": 0,
        }

    rng = np.random.default_rng(seed)
    idx = np.arange(len(traj_objs))
    rng.shuffle(idx)
    split = max(1, int(0.8 * len(traj_objs)))
    train = [traj_objs[i] for i in idx[:split]]
    test = [traj_objs[i] for i in idx[split:]]
    if not test:
        test = train

    model = MarkovPredictor(order=order)
    model.fit(train)
    metrics = evaluate_predictor(model, test, k_values=[5], min_history=order)
    return {
        "accuracy@5": float(metrics.get("accuracy@5", 0.0)),
        "coverage": float(metrics.get("coverage", 0.0)),
        "n_predictions": int(metrics.get("n_predictions", 0)),
        "n_no_prediction": int(metrics.get("n_no_prediction", 0)),
    }


def print_aadt_instructions_for_state(state: str) -> None:
    info = AADT_SOURCES.get(state)
    if not info:
        print(f"[WARN] No curated AADT source entry for state '{state}'.")
        return
    print(f"[INFO] AADT download guidance for {state} ({info['region']} profile):")
    print(f"  Primary URL: {info['url']}")
    if info.get("alternative_url"):
        print(f"  Alternative: {info['alternative_url']}")
    print("  Expected file name in data/raw/aadt/:")
    print(f"    {info['expected_file']}")


def parse_path_list(value: str) -> list[Path]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    paths: list[Path] = []
    for p in parts:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"AADT path does not exist: {path}")
        paths.append(path)
    return paths


def build_combined_aadt_geojson(aadt_paths: list[Path], out_path: Path) -> tuple[Path, dict[str, Any]]:
    frames: list[gpd.GeoDataFrame] = []
    per_source: list[dict[str, Any]] = []

    for p in aadt_paths:
        gdf = gpd.read_file(p)
        if gdf.empty:
            per_source.append({"path": str(p), "used": False, "reason": "empty"})
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        col = detect_aadt_column(gdf)
        if not col:
            per_source.append({"path": str(p), "used": False, "reason": "no_aadt_column"})
            continue

        vals = pd.to_numeric(gdf[col], errors="coerce")
        part = gdf[["geometry"]].copy()
        part["AADT_COMBINED"] = vals
        part = part.dropna(subset=["AADT_COMBINED"])
        if part.empty:
            per_source.append({"path": str(p), "used": False, "reason": "no_numeric_aadt", "column": col})
            continue

        frames.append(part[["AADT_COMBINED", "geometry"]])
        per_source.append(
            {
                "path": str(p),
                "used": True,
                "column": col,
                "rows_used": int(len(part)),
            }
        )

    if not frames:
        raise ValueError("No usable AADT rows found across provided --aadt-paths inputs.")

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(out_path, driver="GeoJSON")
    meta = {
        "n_sources_provided": len(aadt_paths),
        "n_sources_used": int(sum(1 for x in per_source if x.get("used"))),
        "rows_combined": int(len(combined)),
        "sources": per_source,
    }
    return out_path, meta


def resolve_aadt_path(state: str | None, aadt_path_arg: str, require_aadt: bool) -> Path | None:
    if aadt_path_arg:
        p = Path(aadt_path_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--aadt-path does not exist: {p}")
        return p

    if state and state in AADT_PATHS:
        state_path = AADT_PATHS[state]
        if state_path.exists():
            return state_path
        print(f"[WARN] Expected AADT file missing for state {state}: {state_path}")
        print_aadt_instructions_for_state(state)
    elif state:
        print(f"[WARN] No built-in AADT path mapping for state {state}.")

    if require_aadt:
        raise FileNotFoundError("AADT required but not available. Provide --aadt-path or download state data first.")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run custom ROI traffic-weighted ALPR simulation and PSR."
    )
    parser.add_argument("--name", type=str, required=True, help="ROI name (used in output filenames)")
    parser.add_argument("--center-lat", type=float, default=None, help="ROI center latitude")
    parser.add_argument("--center-lon", type=float, default=None, help="ROI center longitude")
    parser.add_argument(
        "--radius-km",
        type=float,
        default=None,
        help=f"ROI radius in km (required for circle ROI; max {HARD_MAX_RADIUS_KM})",
    )
    parser.add_argument(
        "--boundary-geojson",
        type=str,
        default="",
        help="Optional polygon boundary GeoJSON for publication-grade ROI clipping.",
    )
    parser.add_argument(
        "--boundary-margin-deg",
        type=float,
        default=0.05,
        help="Extra lat/lon margin around boundary bbox when downloading OSM road network.",
    )
    parser.add_argument(
        "--camera-geojson",
        type=str,
        default="",
        help="Path to camera GeoJSON source to be clipped to ROI",
    )
    parser.add_argument(
        "--camera-catalog-csv",
        type=str,
        default="",
        help=(
            "Path to cached camera catalog CSV(.gz) with lat/lon columns "
            "(alternative to --camera-geojson). "
            "If omitted, defaults to data/external/camera_catalog/cameras_us_active.csv.gz when present."
        ),
    )
    parser.add_argument("--state", type=str, default="", help="State code (e.g., GA) or full state name")
    parser.add_argument(
        "--infer-state",
        action="store_true",
        help="Try reverse-geocoding center coordinates to infer state if --state not given",
    )
    parser.add_argument("--aadt-path", type=str, default="", help="Optional explicit AADT shapefile path")
    parser.add_argument(
        "--aadt-paths",
        type=str,
        default="",
        help="Optional comma-separated AADT shapefile paths (for cross-state metros).",
    )
    parser.add_argument(
        "--require-aadt",
        action="store_true",
        help="Fail if AADT cannot be resolved (otherwise fallback to OSM proxy-only)",
    )

    parser.add_argument("--n-vehicles", type=int, default=5000)
    parser.add_argument("--n-trips", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mp-chunksize", type=int, default=2, help="Multiprocessing chunksize for per-vehicle tasks.")
    parser.add_argument("--disable-route-cache", action="store_true", help="Disable OD route candidate cache.")
    parser.add_argument(
        "--disable-node-camera-cache",
        action="store_true",
        help="Disable node-level camera query cache.",
    )
    parser.add_argument("--route-cache-size", type=int, default=200000, help="Max OD cache entries per process.")
    parser.add_argument(
        "--node-camera-cache-size",
        type=int,
        default=200000,
        help="Max node camera cache entries per process.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-shortest", type=int, default=3)
    parser.add_argument("--p-return", type=float, default=0.6)
    parser.add_argument("--detection-radius-m", type=float, default=100.0)
    parser.add_argument(
        "--ring-breaks-km",
        type=str,
        default="0,5,10,20,40,80",
        help="Comma-separated ring breaks for core/periphery analysis (km).",
    )

    parser.add_argument("--output-root", type=str, default="results/custom_roi")
    args = parser.parse_args()

    use_boundary = bool(args.boundary_geojson)
    if args.radius_km is not None and args.radius_km <= 0:
        raise ValueError("--radius-km must be > 0 when provided")
    if not use_boundary:
        if args.center_lat is None or args.center_lon is None:
            raise ValueError("--center-lat/--center-lon are required when --boundary-geojson is not set")
        if args.radius_km is None:
            raise ValueError("--radius-km is required when --boundary-geojson is not set")
        if args.radius_km > HARD_MAX_RADIUS_KM:
            raise ValueError(f"--radius-km exceeds hard limit ({HARD_MAX_RADIUS_KM} km)")
    elif args.boundary_margin_deg < 0:
        raise ValueError("--boundary-margin-deg must be >= 0")
    if args.n_vehicles <= 0 or args.n_trips <= 0:
        raise ValueError("--n-vehicles and --n-trips must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")
    if args.mp_chunksize <= 0:
        raise ValueError("--mp-chunksize must be >= 1")
    if args.route_cache_size <= 0:
        raise ValueError("--route-cache-size must be >= 1")
    if args.node_camera_cache_size <= 0:
        raise ValueError("--node-camera-cache-size must be >= 1")

    roi_name = slugify(args.name)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (BASE_DIR / args.output_root / f"{roi_name}_{run_id}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clipped_camera_path = out_dir / "cameras_clipped.geojson"
    summary_path = out_dir / "summary.json"
    result_pkl_path = out_dir / "road_trajectories.pkl"
    psr_path = out_dir / "psr.json"
    ring_metrics_path = out_dir / "ring_metrics.json"

    print(f"[INFO] Output directory: {out_dir}")
    print("[INFO] Clipping cameras to ROI...")
    ring_breaks_km = parse_ring_breaks_km(args.ring_breaks_km)
    camera_source: Path | None = None
    catalog_source: Path | None = None
    boundary_source: Path | None = None
    boundary_gdf: gpd.GeoDataFrame | None = None
    boundary_area: float | None = None
    boundary_center_lat: float | None = None
    boundary_center_lon: float | None = None
    boundary_max_radius: float | None = None
    combined_aadt_meta: dict[str, Any] | None = None

    center_lat = args.center_lat
    center_lon = args.center_lon
    analysis_radius_km = args.radius_km

    if use_boundary:
        boundary_source = Path(args.boundary_geojson).expanduser().resolve()
        if not boundary_source.exists():
            raise FileNotFoundError(f"Boundary file not found: {boundary_source}")
        boundary_gdf = load_boundary_geometry(boundary_source)
        boundary_area = geometry_area_km2(boundary_gdf)
        boundary_center_lat, boundary_center_lon = boundary_centroid_lat_lon(boundary_gdf)
        boundary_max_radius = boundary_max_radius_km(
            geom_gdf=boundary_gdf,
            center_lat=boundary_center_lat,
            center_lon=boundary_center_lon,
        )
        if center_lat is None or center_lon is None:
            center_lat, center_lon = boundary_center_lat, boundary_center_lon
        if analysis_radius_km is None:
            analysis_radius_km = boundary_max_radius

    if args.camera_geojson:
        camera_source = Path(args.camera_geojson).expanduser().resolve()
        if not camera_source.exists():
            raise FileNotFoundError(f"Camera source file not found: {camera_source}")
        if boundary_gdf is not None:
            clipped_gdf, source_camera_count = clip_cameras_to_boundary(
                camera_geojson=camera_source,
                boundary_gdf=boundary_gdf,
                out_geojson=clipped_camera_path,
            )
        else:
            assert center_lat is not None and center_lon is not None and analysis_radius_km is not None
            clipped_gdf, source_camera_count = clip_cameras_to_radius(
                camera_geojson=camera_source,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=analysis_radius_km,
                out_geojson=clipped_camera_path,
            )
    elif args.camera_catalog_csv:
        catalog_source = Path(args.camera_catalog_csv).expanduser().resolve()
        if not catalog_source.exists():
            raise FileNotFoundError(f"Camera catalog file not found: {catalog_source}")
        if boundary_gdf is not None:
            clipped_gdf, source_camera_count = clip_catalog_to_boundary(
                camera_catalog_csv=catalog_source,
                boundary_gdf=boundary_gdf,
                out_geojson=clipped_camera_path,
            )
        else:
            assert center_lat is not None and center_lon is not None and analysis_radius_km is not None
            clipped_gdf, source_camera_count = clip_catalog_to_radius(
                camera_catalog_csv=catalog_source,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=analysis_radius_km,
                out_geojson=clipped_camera_path,
            )
    else:
        if DEFAULT_CAMERA_CATALOG.exists():
            catalog_source = DEFAULT_CAMERA_CATALOG
            print(f"[INFO] Using default cached camera catalog: {catalog_source}")
            if boundary_gdf is not None:
                clipped_gdf, source_camera_count = clip_catalog_to_boundary(
                    camera_catalog_csv=catalog_source,
                    boundary_gdf=boundary_gdf,
                    out_geojson=clipped_camera_path,
                )
            else:
                assert center_lat is not None and center_lon is not None and analysis_radius_km is not None
                clipped_gdf, source_camera_count = clip_catalog_to_radius(
                    camera_catalog_csv=catalog_source,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    radius_km=analysis_radius_km,
                    out_geojson=clipped_camera_path,
                )
        else:
            raise ValueError(
                "Provide either --camera-geojson or --camera-catalog-csv "
                "(or create default cache at data/external/camera_catalog/cameras_us_active.csv.gz)."
            )

    n_cameras = len(clipped_gdf)
    print(f"[INFO] Cameras in ROI: {n_cameras} / {source_camera_count}")

    if center_lat is None or center_lon is None:
        raise ValueError("Unable to determine ROI center coordinates.")
    if analysis_radius_km is None or analysis_radius_km <= 0:
        cam_dist = haversine_km(
            center_lat,
            center_lon,
            clipped_gdf.geometry.y.to_numpy(),
            clipped_gdf.geometry.x.to_numpy(),
        )
        analysis_radius_km = max(float(cam_dist.max()), 0.1)

    state = normalize_state_code(args.state)
    if state is None and args.infer_state:
        inferred = infer_state_from_coordinates(center_lat, center_lon)
        if inferred:
            state = inferred
            print(f"[INFO] Inferred state from coordinates: {state}")
        else:
            print("[WARN] Could not infer state from coordinates.")

    if args.aadt_path and args.aadt_paths:
        raise ValueError("Use either --aadt-path or --aadt-paths, not both.")

    if args.aadt_paths:
        aadt_paths = parse_path_list(args.aadt_paths)
        combined_aadt_path = out_dir / "aadt_combined.geojson"
        aadt_path, combined_aadt_meta = build_combined_aadt_geojson(aadt_paths, combined_aadt_path)
    else:
        aadt_path = resolve_aadt_path(state=state, aadt_path_arg=args.aadt_path, require_aadt=args.require_aadt)
    if aadt_path:
        print(f"[INFO] Using AADT shapefile: {aadt_path}")
    else:
        print("[WARN] Proceeding without AADT shapefile (OSM proxy-only traffic weights).")
        if state:
            print_aadt_instructions_for_state(state)

    print("[INFO] Building road network for ROI bounding box...")
    if boundary_gdf is not None:
        bbox = boundary_bbox_with_margin(boundary_gdf, margin_deg=args.boundary_margin_deg)
    else:
        bbox = camera_bbox_with_margin(clipped_gdf, margin_deg=0.05)
    G_road = ox.graph_from_bbox(bbox=bbox, network_type="drive", simplify=True)
    print(f"[INFO] Road network: {G_road.number_of_nodes()} nodes, {G_road.number_of_edges()} edges")

    print("[INFO] Computing traffic weights...")
    edge_traffic, node_traffic, traffic_meta = compute_traffic_weights_for_region(
        G=G_road,
        aadt_path=aadt_path,
        state=state,
        bbox=bbox,
        config={
            "aadt_line_buffer_m": DEFAULT_TRAFFIC_CONFIG["aadt_line_buffer_m"],
            "aadt_point_buffer_m": DEFAULT_TRAFFIC_CONFIG["aadt_point_buffer_m"],
            "aadt_coverage_threshold": DEFAULT_TRAFFIC_CONFIG["aadt_coverage_threshold"],
            "normalize_method": DEFAULT_TRAFFIC_CONFIG["normalize_method"],
            "clip_percentile": DEFAULT_TRAFFIC_CONFIG["clip_percentile"],
        },
    )
    print(
        "[INFO] Traffic metadata: "
        f"coverage={traffic_meta.get('aadt_coverage', 0):.1%}, "
        f"used_aadt={traffic_meta.get('used_aadt', False)}"
    )

    print("[INFO] Running road-network simulation...")
    simulation_result = simulate_road_network_trips(
        region=roi_name,
        camera_geojson_path=clipped_camera_path,
        n_vehicles=args.n_vehicles,
        n_trips_per_vehicle=args.n_trips,
        k_shortest=args.k_shortest,
        p_return=args.p_return,
        detection_radius_m=args.detection_radius_m,
        seed=args.seed,
        n_workers=args.workers,
        mp_chunksize=args.mp_chunksize,
        use_route_cache=not args.disable_route_cache,
        use_node_camera_cache=not args.disable_node_camera_cache,
        route_cache_size=args.route_cache_size,
        node_camera_cache_size=args.node_camera_cache_size,
        verbose=True,
        traffic_weights=node_traffic,
        edge_traffic_weights=edge_traffic,
        traffic_blend_factor=DEFAULT_TRAFFIC_CONFIG["traffic_blend_factor"],
        lambda_traffic=DEFAULT_TRAFFIC_CONFIG["lambda_traffic"],
        min_trip_distance_m=DEFAULT_TRAFFIC_CONFIG["min_trip_distance_m"],
        max_trip_distance_m=DEFAULT_TRAFFIC_CONFIG["max_trip_distance_m"],
    )
    if simulation_result is None:
        raise RuntimeError("Simulation returned None")

    print("[INFO] Computing U(2), predictability, and PSR...")
    u2 = compute_random_point_uniqueness(
        simulation_result["trajectories"], k=2, n_samples=1000, ordered=False, seed=args.seed
    )

    traj_objs = [
        Trajectory(vehicle_id=t["vehicle_id"], camera_sequence=t.get("camera_hits", []))
        for t in simulation_result["trajectories"]
    ]
    predictability = analyze_predictability(traj_objs)
    markov_metrics = compute_markov_acc5(simulation_result["trajectories"], seed=args.seed, order=1)

    area_km2 = boundary_area if (boundary_area is not None and boundary_area > 0) else convex_hull_area_km2(clipped_gdf)
    if area_km2 <= 0:
        area_km2 = np.pi * (analysis_radius_km**2)
    density = n_cameras / area_km2 if area_km2 > 0 else 0.0

    psr = compute_psr(
        predictability=predictability.get("Pi_max_population", 0.0),
        reidentification_rate=u2.get("uniqueness", 0.0),
        camera_density=density,
        achieved_accuracy=markov_metrics.get("accuracy@5"),
        n_cameras=n_cameras,
    )

    ring_metrics = build_ring_metrics(
        clipped_gdf=clipped_gdf,
        trajectories=simulation_result["trajectories"],
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=analysis_radius_km,
        n_vehicles=args.n_vehicles,
        ring_breaks_km=ring_breaks_km,
    )

    simulation_result["traffic_weight_metadata"] = traffic_meta
    simulation_result["uniqueness"] = {"u2_random": u2}
    simulation_result["predictability"] = predictability
    simulation_result["prediction_metrics"] = {"markov_order_1": markov_metrics}
    simulation_result["psr"] = psr
    simulation_result["ring_metrics"] = ring_metrics

    summary = {
        "generated_at_utc": utc_now_iso(),
        "name": roi_name,
        "center": {"lat": center_lat, "lon": center_lon},
        "radius_km": analysis_radius_km,
        "radius_input_km": args.radius_km,
        "state": state,
        "camera_source": str(camera_source) if camera_source else None,
        "camera_catalog_source": str(catalog_source) if catalog_source else None,
        "camera_catalog_paper_eligible": False if catalog_source else None,
        "boundary_geojson": str(boundary_source) if boundary_source else None,
        "boundary_mode": "polygon" if boundary_source else "circle",
        "boundary_area_km2": boundary_area,
        "boundary_centroid_lat": boundary_center_lat,
        "boundary_centroid_lon": boundary_center_lon,
        "boundary_max_radius_km": boundary_max_radius,
        "camera_file_clipped": str(clipped_camera_path),
        "n_cameras_source": source_camera_count,
        "n_cameras_roi": n_cameras,
        "area_km2": area_km2,
        "camera_density_per_km2": density,
        "aadt_path": str(aadt_path) if aadt_path else None,
        "aadt_paths_input": [p.strip() for p in args.aadt_paths.split(",") if p.strip()] if args.aadt_paths else [],
        "aadt_combined_meta": combined_aadt_meta,
        "traffic_weight_metadata": traffic_meta,
        "observation_stats": simulation_result.get("observation_stats", {}),
        "u2_random": u2,
        "predictability": {
            "pi_max_population": predictability.get("Pi_max_population", 0.0),
            "acc5_markov_order1": markov_metrics.get("accuracy@5", 0.0),
            "prediction_coverage": markov_metrics.get("coverage", 0.0),
        },
        "psr": {
            "score": psr.get("psr_score", 0.0),
            "interpretation": psr.get("interpretation"),
            "description": psr.get("description"),
            "components": psr.get("components", {}),
            "weights": psr.get("weights", {}),
        },
        "ring_metrics": ring_metrics,
        "run_parameters": {
            "n_vehicles": args.n_vehicles,
            "n_trips": args.n_trips,
            "workers": args.workers,
            "mp_chunksize": args.mp_chunksize,
            "use_route_cache": simulation_result.get("parameters", {}).get(
                "use_route_cache", not args.disable_route_cache
            ),
            "use_node_camera_cache": simulation_result.get("parameters", {}).get(
                "use_node_camera_cache", not args.disable_node_camera_cache
            ),
            "route_cache_size": simulation_result.get("parameters", {}).get("route_cache_size", args.route_cache_size),
            "node_camera_cache_size": simulation_result.get("parameters", {}).get(
                "node_camera_cache_size", args.node_camera_cache_size
            ),
            "seed": args.seed,
            "k_shortest": args.k_shortest,
            "p_return": args.p_return,
            "detection_radius_m": args.detection_radius_m,
            "ring_breaks_km": ring_breaks_km,
            "boundary_margin_deg": args.boundary_margin_deg,
        },
    }

    with open(result_pkl_path, "wb") as f:
        pickle.dump(simulation_result, f)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    psr_path.write_text(json.dumps(psr, indent=2), encoding="utf-8")
    ring_metrics_path.write_text(json.dumps(ring_metrics, indent=2), encoding="utf-8")

    print("[INFO] Completed custom ROI run.")
    print(f"[INFO] Result pickle: {result_pkl_path}")
    print(f"[INFO] Summary JSON: {summary_path}")
    print(f"[INFO] PSR JSON: {psr_path}")
    print(f"[INFO] Ring metrics JSON: {ring_metrics_path}")
    print(
        f"[INFO] PSR={psr.get('psr_score', 0.0):.3f} "
        f"[{psr.get('interpretation', 'N/A')}] "
        f"| P(>=1)={summary['observation_stats'].get('p_at_least_1', 0.0):.1%} "
        f"| U2={u2.get('uniqueness', 0.0):.1%}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
