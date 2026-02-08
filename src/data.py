"""
Data loading from OpenStreetMap via Overpass API.

Uses the same data source as DeFlock (deflock.me) for ALPR camera locations.
"""

import json
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import osmnx as ox
import requests
from shapely.geometry import Point

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Study region definitions
STUDY_REGIONS = {
    "atlanta": {
        "area_name": "Georgia",
        "admin_level": 4,
        "bbox": (33.5, -84.9, 34.2, -84.0),  # (south, west, north, east)
    },
    "lehigh_valley": {
        "area_name": None,
        "admin_level": None,
        "bbox": (40.4, -75.8, 40.8, -75.2),
    },
}


def query_alpr_cameras(
    bbox: Optional[tuple[float, float, float, float]] = None,
    area_name: Optional[str] = None,
    admin_level: int = 4,
    timeout: int = 120,
    max_retries: int = 3,
) -> gpd.GeoDataFrame:
    """
    Query ALPR cameras from OpenStreetMap via Overpass API.

    Args:
        bbox: (south, west, north, east) bounding box
        area_name: Name of admin area (e.g., "Georgia", "Pennsylvania")
        admin_level: OSM admin level (4=state, 6=county, 8=city)
        timeout: Query timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        GeoDataFrame with camera locations and attributes

    Raises:
        ValueError: If neither bbox nor area_name is provided
        requests.HTTPError: If API request fails after retries
    """
    if area_name:
        query = f'''
        [out:json][timeout:{timeout}];
        area["name"="{area_name}"]["admin_level"="{admin_level}"]->.searchArea;
        node["man_made"="surveillance"]["surveillance:type"="ALPR"](area.searchArea);
        out body geom;
        '''
    elif bbox:
        s, w, n, e = bbox
        query = f'''
        [out:json][timeout:{timeout}];
        node["man_made"="surveillance"]["surveillance:type"="ALPR"]({s},{w},{n},{e});
        out body geom;
        '''
    else:
        raise ValueError("Must provide either bbox or area_name")

    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=timeout + 30,
            )
            response.raise_for_status()
            data = response.json()
            return _parse_osm_response(data)
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                raise


def _parse_osm_response(data: dict) -> gpd.GeoDataFrame:
    """
    Parse Overpass API response into GeoDataFrame.

    Args:
        data: JSON response from Overpass API

    Returns:
        GeoDataFrame with parsed camera data
    """
    records = []
    for element in data.get("elements", []):
        if element["type"] == "node":
            record = {
                "osm_id": element["id"],
                "lat": element["lat"],
                "lon": element["lon"],
                "geometry": Point(element["lon"], element["lat"]),
            }
            # Extract all tags, replacing colons with underscores
            for key, value in element.get("tags", {}).items():
                record[key.replace(":", "_")] = value
            records.append(record)

    if not records:
        return gpd.GeoDataFrame(
            columns=["osm_id", "lat", "lon", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def load_road_network(
    gdf_cameras: gpd.GeoDataFrame,
    buffer_km: float = 1.0,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """
    Load road network covering the camera locations.

    Args:
        gdf_cameras: GeoDataFrame of camera locations
        buffer_km: Buffer around cameras to include roads (in km)
        network_type: OSMnx network type (default: "drive")

    Returns:
        NetworkX MultiDiGraph of road network
    """
    if gdf_cameras.empty:
        raise ValueError("Cannot load road network for empty camera GeoDataFrame")

    # Get bounding box with buffer
    bounds = gdf_cameras.total_bounds  # [minx, miny, maxx, maxy]
    buffer_deg = buffer_km / 111  # Approximate km to degrees

    bbox = (
        bounds[3] + buffer_deg,  # north
        bounds[1] - buffer_deg,  # south
        bounds[2] + buffer_deg,  # east
        bounds[0] - buffer_deg,  # west
    )

    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type=network_type,
        simplify=True,
    )
    return G


def load_study_regions(
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Load camera data for all study regions.

    Args:
        cache_dir: Directory for caching downloaded data
        use_cache: Whether to use cached data if available

    Returns:
        Dict mapping region name to GeoDataFrame of cameras
    """
    if cache_dir is None:
        cache_dir = Path("data/raw")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    regions = {}

    for region_name, config in STUDY_REGIONS.items():
        cache_file = cache_dir / f"{region_name}_cameras.geojson"

        if use_cache and cache_file.exists():
            print(f"Loading {region_name} from cache...")
            regions[region_name] = load_cameras(cache_file)
        else:
            print(f"Downloading {region_name} cameras...")
            if config["area_name"]:
                gdf = query_alpr_cameras(
                    area_name=config["area_name"],
                    admin_level=config["admin_level"],
                )
                # Filter to metro area bbox if provided
                if config["bbox"]:
                    s, w, n, e = config["bbox"]
                    gdf = gdf.cx[w:e, s:n]
            else:
                gdf = query_alpr_cameras(bbox=config["bbox"])

            regions[region_name] = gdf

            # Cache the result
            if not gdf.empty:
                save_cameras(gdf, cache_file)
                print(f"  Cached {len(gdf)} cameras to {cache_file}")

    return regions


def save_cameras(gdf: gpd.GeoDataFrame, path: Path | str) -> None:
    """
    Save camera data to GeoJSON.

    Args:
        gdf: GeoDataFrame of camera locations
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def load_cameras(path: Path | str) -> gpd.GeoDataFrame:
    """
    Load camera data from GeoJSON.

    Args:
        path: Input file path

    Returns:
        GeoDataFrame of camera locations
    """
    return gpd.read_file(path)


def get_camera_stats(gdf: gpd.GeoDataFrame) -> dict:
    """
    Compute basic statistics for a camera dataset.

    Args:
        gdf: GeoDataFrame of camera locations

    Returns:
        Dict with camera statistics
    """
    if gdf.empty:
        return {
            "n_cameras": 0,
            "operators": {},
            "coverage_area_km2": 0,
        }

    # Count by operator
    operator_col = "operator" if "operator" in gdf.columns else None
    if operator_col:
        operators = gdf[operator_col].value_counts().to_dict()
    else:
        operators = {"unknown": len(gdf)}

    # Compute coverage area (convex hull in UTM)
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    hull = gdf_proj.union_all().convex_hull
    area_km2 = hull.area / 1e6  # m² to km²

    return {
        "n_cameras": len(gdf),
        "operators": operators,
        "coverage_area_km2": round(area_km2, 2),
        "bounds": {
            "south": gdf.total_bounds[1],
            "west": gdf.total_bounds[0],
            "north": gdf.total_bounds[3],
            "east": gdf.total_bounds[2],
        },
    }


def find_sparse_regions(
    min_cameras: int = 10,
    max_cameras: int = 30,
    candidate_states: list[str] | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Search for regions with sparse ALPR coverage.

    Args:
        min_cameras: Minimum number of cameras
        max_cameras: Maximum number of cameras
        candidate_states: List of state names to search (default: selected US states)

    Returns:
        Dict mapping region name to GeoDataFrame of cameras
    """
    if candidate_states is None:
        # States likely to have some ALPR but not dense coverage
        candidate_states = [
            "Vermont",
            "Maine",
            "Wyoming",
            "Montana",
            "New Hampshire",
            "Idaho",
            "South Dakota",
            "North Dakota",
        ]

    sparse_regions = {}

    for state in candidate_states:
        try:
            print(f"Checking {state}...")
            gdf = query_alpr_cameras(area_name=state, admin_level=4)
            n_cameras = len(gdf)
            print(f"  Found {n_cameras} cameras")

            if min_cameras <= n_cameras <= max_cameras:
                sparse_regions[state.lower().replace(" ", "_")] = gdf
                print(f"  -> Valid sparse region!")
        except Exception as e:
            print(f"  Error querying {state}: {e}")
            continue

    return sparse_regions
