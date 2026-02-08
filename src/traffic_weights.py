"""
Traffic weight computation module for ALPR simulation.

This module provides functions to compute traffic weights from:
1. AADT (Annual Average Daily Traffic) data from state DOT shapefiles
2. OSM-based proxy estimates for edges without AADT coverage

The traffic weights are used for:
- Endpoint attractiveness (higher-traffic nodes attract more activity)
- Route utility (prefer routes on higher-traffic corridors)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)


# =============================================================================
# OSM HIGHWAY CLASS -> PROXY AADT MAPPING
# =============================================================================

# Literature-informed estimates of AADT by road class
# Sources: FHWA Highway Statistics, various state DOT reports
OSM_HIGHWAY_PROXY_AADT: dict[str, float] = {
    'motorway': 80000,
    'motorway_link': 40000,
    'trunk': 40000,
    'trunk_link': 20000,
    'primary': 20000,
    'primary_link': 10000,
    'secondary': 10000,
    'secondary_link': 5000,
    'tertiary': 5000,
    'tertiary_link': 2500,
    'residential': 2000,
    'unclassified': 1500,
    'living_street': 500,
    'service': 500,
}

# Default for unknown highway types
DEFAULT_PROXY_AADT = 1000


# =============================================================================
# AADT COLUMN DETECTION
# =============================================================================

# Common column names for AADT across different state DOT schemas
AADT_COLUMN_PATTERNS = [
    'AADT',
    'ADT',
    'FAADT',  # Maine DOT: Functional/Final AADT
    'CUR_AADT',  # Pennsylvania DOT
    'AADT_VN',
    'AADTVN',
    'AVG_AADT',
    'ANNUAL_ADT',
    'TRAFFIC',
    'VOLUME',
    'VOL',
]


def detect_aadt_column(gdf: gpd.GeoDataFrame) -> str | None:
    """
    Auto-detect the AADT column in a GeoDataFrame.

    Args:
        gdf: GeoDataFrame from state DOT shapefile

    Returns:
        Column name containing AADT, or None if not found
    """
    columns_upper = {c.upper(): c for c in gdf.columns}

    # First pass: exact matches (highest priority)
    for pattern in AADT_COLUMN_PATTERNS:
        if pattern in columns_upper:
            return columns_upper[pattern]

    # Second pass: partial matches, but exclude metadata columns
    # (e.g., 'aadt_type' describes AADT estimation method, not the value)
    excluded_suffixes = ('_TYPE', '_FLAG', '_SRC', '_SOURCE', '_CODE', '_DESC')
    for pattern in AADT_COLUMN_PATTERNS:
        for col_upper, col_orig in columns_upper.items():
            if pattern in col_upper:
                # Skip metadata columns that just describe AADT, not contain it
                if any(col_upper.endswith(suffix) for suffix in excluded_suffixes):
                    continue
                return col_orig

    return None


# =============================================================================
# AADT DATA LOADING
# =============================================================================

def load_aadt_shapefile(
    state: str,
    aadt_path: Path | str,
    aadt_column: str | None = None,
    clip_bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """
    Load AADT shapefile for a given state.

    Handles both LineString (road segments) and Point (traffic count stations) geometry.
    Also handles string AADT values with commas (e.g., "4,760").

    Args:
        state: State abbreviation (GA, TN, VA, NC, PA, ME)
        aadt_path: Path to shapefile
        aadt_column: Column name for AADT (auto-detected if None)
        clip_bbox: Optional (west, south, east, north) to clip data

    Returns:
        GeoDataFrame with 'geometry' and 'aadt' columns

    Raises:
        FileNotFoundError: If shapefile doesn't exist
        ValueError: If AADT column cannot be detected
    """
    aadt_path = Path(aadt_path)

    if not aadt_path.exists():
        raise FileNotFoundError(f"AADT shapefile not found: {aadt_path}")

    logger.info(f"Loading AADT data for {state} from {aadt_path}")
    gdf = gpd.read_file(aadt_path)

    # Auto-detect AADT column if not provided
    if aadt_column is None:
        aadt_column = detect_aadt_column(gdf)
        if aadt_column is None:
            raise ValueError(
                f"Could not auto-detect AADT column. "
                f"Available columns: {list(gdf.columns)}"
            )
        logger.info(f"Auto-detected AADT column: {aadt_column}")

    # Ensure CRS is WGS84 (EPSG:4326) for consistency with OSMnx
    if gdf.crs is None:
        logger.warning("No CRS in shapefile, assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # Clip to bounding box if provided
    if clip_bbox is not None:
        west, south, east, north = clip_bbox
        gdf = gdf.cx[west:east, south:north]
        logger.info(f"Clipped to bbox: {len(gdf)} features")

    # Extract AADT values and handle missing/invalid
    aadt_values = gdf[aadt_column].copy()

    # Handle string values with commas (e.g., "4,760" -> 4760)
    if aadt_values.dtype == 'object':
        aadt_values = aadt_values.astype(str).str.replace(',', '', regex=False)

    aadt_values = pd.to_numeric(aadt_values, errors='coerce')
    aadt_values = aadt_values.fillna(0).clip(lower=0)

    # Log geometry type
    geom_types = gdf.geometry.geom_type.unique().tolist()
    logger.info(f"Geometry types: {geom_types}")

    # Create clean output DataFrame
    result = gpd.GeoDataFrame({
        'geometry': gdf.geometry,
        'aadt': aadt_values,
        'state': state,
    }, crs=gdf.crs)

    # Filter to valid geometries with non-zero AADT
    result = result[result.geometry.is_valid & ~result.geometry.is_empty]
    result = result[result['aadt'] > 0]  # Filter out zero/missing AADT

    logger.info(f"Loaded {len(result)} AADT features for {state}")
    logger.info(f"AADT range: {result['aadt'].min():.0f} - {result['aadt'].max():.0f}")

    return result


# =============================================================================
# SPATIAL JOIN: AADT TO OSM EDGES
# =============================================================================

def _edges_to_geodataframe(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Convert OSMnx graph edges to GeoDataFrame with edge keys.

    Args:
        G: OSMnx road network graph

    Returns:
        GeoDataFrame with edge geometries and (u, v, key) index
    """
    edges = []

    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            geom = data['geometry']
        else:
            # Create straight line from node coordinates
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            geom = LineString([
                (u_data['x'], u_data['y']),
                (v_data['x'], v_data['y'])
            ])

        edges.append({
            'u': u,
            'v': v,
            'key': key,
            'geometry': geom,
            'highway': data.get('highway', 'unknown'),
            'length': data.get('length', 0),
        })

    return gpd.GeoDataFrame(edges, crs="EPSG:4326")


def spatial_join_aadt_to_edges(
    G: nx.MultiDiGraph,
    aadt_gdf: gpd.GeoDataFrame,
    buffer_m: float | None = None,
    line_buffer_m: float | None = None,
    point_buffer_m: float | None = None,
    crs_meters: str = "EPSG:3857",
) -> tuple[dict[tuple[int, int, int], float], float]:
    """
    Spatially join AADT data to OSM road edges.

    Handles both:
    - LineString geometry: Buffer and intersect with edges
    - Point geometry: Find nearest edge for each traffic count station

    Args:
        G: OSMnx road network graph
        aadt_gdf: GeoDataFrame with AADT data (Point or LineString)
        buffer_m: Legacy shared buffer distance in meters (applies to both line/point if set)
        line_buffer_m: Buffer distance for line-based AADT joins (default 50 m)
        point_buffer_m: Max distance for point-based nearest-edge assignment (default 100 m)
        crs_meters: CRS for meter-based operations

    Returns:
        Tuple of:
        - dict mapping (u, v, key) -> aadt value
        - coverage fraction (0-1) of edges with AADT match
    """
    if aadt_gdf.empty:
        logger.warning("Empty AADT GeoDataFrame, returning empty weights")
        return {}, 0.0

    # Backward compatibility:
    # - If legacy buffer_m is provided and explicit per-geometry buffers are not,
    #   use it for both line and point behavior.
    # - Otherwise use method-specific defaults consistent with manuscript text.
    if buffer_m is not None:
        if line_buffer_m is None:
            line_buffer_m = buffer_m
        if point_buffer_m is None:
            point_buffer_m = buffer_m
    if line_buffer_m is None:
        line_buffer_m = 50.0
    if point_buffer_m is None:
        point_buffer_m = 100.0

    # Convert edges to GeoDataFrame
    edges_gdf = _edges_to_geodataframe(G)
    n_edges = len(edges_gdf)

    if edges_gdf.empty:
        logger.warning("No edges in graph")
        return {}, 0.0

    # Detect geometry type
    geom_types = aadt_gdf.geometry.geom_type.unique()
    is_point_data = 'Point' in geom_types

    logger.info(f"Joining {len(aadt_gdf)} AADT {'points' if is_point_data else 'segments'} to {n_edges} road edges...")

    # Project to meters
    edges_proj = edges_gdf.to_crs(crs_meters)
    aadt_proj = aadt_gdf.to_crs(crs_meters)

    if is_point_data:
        # Point data: find nearest edge for each traffic count station
        edge_aadt = _join_points_to_edges(edges_proj, aadt_proj, point_buffer_m)
    else:
        # LineString data: buffer and intersect
        edge_aadt = _join_lines_to_edges(edges_proj, aadt_proj, line_buffer_m)

    # Calculate coverage
    n_matched = len(edge_aadt)
    coverage = n_matched / n_edges if n_edges > 0 else 0.0

    logger.info(f"AADT coverage: {n_matched}/{n_edges} edges ({coverage:.1%})")

    return edge_aadt, coverage


def _join_points_to_edges(
    edges_proj: gpd.GeoDataFrame,
    aadt_proj: gpd.GeoDataFrame,
    buffer_m: float,
) -> dict[tuple[int, int, int], float]:
    """
    Join Point AADT data (traffic count stations) to nearest edges.

    For each AADT point, finds the nearest edge within buffer_m distance
    and assigns the AADT value to that edge.
    """
    from scipy.spatial import cKDTree

    # Get edge midpoints for nearest-neighbor matching
    edge_midpoints = []
    edge_keys = []

    for idx, row in edges_proj.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            # Use centroid of edge as representative point
            midpoint = geom.centroid
            edge_midpoints.append([midpoint.x, midpoint.y])
            edge_keys.append((row['u'], row['v'], row['key']))

    if not edge_midpoints:
        return {}

    # Build KD-tree of edge midpoints
    edge_tree = cKDTree(edge_midpoints)

    # For each AADT point, find nearest edge
    edge_aadt_lists: dict[tuple, list[float]] = {}

    for idx, row in aadt_proj.iterrows():
        point = row.geometry
        aadt_val = row['aadt']

        if point is None or point.is_empty or aadt_val <= 0:
            continue

        # Query nearest edge
        dist, edge_idx = edge_tree.query([point.x, point.y], k=1)

        # Only assign if within buffer distance
        if dist <= buffer_m:
            edge_key = edge_keys[edge_idx]
            if edge_key not in edge_aadt_lists:
                edge_aadt_lists[edge_key] = []
            edge_aadt_lists[edge_key].append(aadt_val)

    # Average AADT values for edges with multiple nearby stations
    edge_aadt = {}
    for edge_key, values in edge_aadt_lists.items():
        edge_aadt[edge_key] = np.mean(values)

    return edge_aadt


def _join_lines_to_edges(
    edges_proj: gpd.GeoDataFrame,
    aadt_proj: gpd.GeoDataFrame,
    buffer_m: float,
) -> dict[tuple[int, int, int], float]:
    """
    Join LineString AADT data (road segments) to edges via buffer intersection.
    """
    # Buffer AADT segments
    aadt_buffered = aadt_proj.copy()
    aadt_buffered['geometry'] = aadt_buffered.geometry.buffer(buffer_m)

    # Spatial join: find which AADT buffers intersect each edge
    joined = gpd.sjoin(
        edges_proj,
        aadt_buffered[['geometry', 'aadt']],
        how='left',
        predicate='intersects'
    )

    # Group by edge and aggregate AADT (mean of overlapping segments)
    edge_aadt = {}
    for (u, v, key), group in joined.groupby(['u', 'v', 'key']):
        aadt_values = group['aadt'].dropna()
        if len(aadt_values) > 0:
            edge_aadt[(u, v, key)] = aadt_values.mean()

    return edge_aadt


# =============================================================================
# OSM-BASED PROXY TRAFFIC ESTIMATION
# =============================================================================

def proxy_traffic_from_osm(
    G: nx.MultiDiGraph,
    base_aadt: dict[str, float] | None = None,
    use_lanes: bool = True,
    use_maxspeed: bool = True,
) -> dict[tuple[int, int, int], float]:
    """
    Estimate traffic proxy from OSM edge attributes.

    Formula:
        proxy = base_aadt[highway] * lane_multiplier * speed_multiplier

    Where:
        lane_multiplier = lanes / 2 (normalize to 2-lane road)
        speed_multiplier = maxspeed / 50 (normalize to 50 km/h)

    Args:
        G: Road network with 'highway', 'lanes', 'maxspeed' attrs
        base_aadt: Custom base AADT by highway class (uses defaults if None)
        use_lanes: Adjust by lane count
        use_maxspeed: Adjust by speed limit

    Returns:
        dict mapping (u, v, key) -> proxy_aadt
    """
    if base_aadt is None:
        base_aadt = OSM_HIGHWAY_PROXY_AADT

    proxy_weights = {}

    for u, v, key, data in G.edges(keys=True, data=True):
        # Get highway class
        highway = data.get('highway', 'unknown')
        if isinstance(highway, list):
            highway = highway[0]

        # Base AADT for this road class
        base = base_aadt.get(highway, DEFAULT_PROXY_AADT)

        # Lane multiplier
        lane_mult = 1.0
        if use_lanes:
            lanes = data.get('lanes')
            if lanes is not None:
                try:
                    if isinstance(lanes, list):
                        lanes = lanes[0]
                    lanes = int(lanes)
                    lane_mult = min(lanes / 2.0, 2.0)  # Normalize to 2-lane road and cap
                except (ValueError, TypeError):
                    pass

        # Speed multiplier
        speed_mult = 1.0
        if use_maxspeed:
            maxspeed = data.get('maxspeed')
            if maxspeed is not None:
                try:
                    if isinstance(maxspeed, list):
                        maxspeed = maxspeed[0]
                    # Handle "50 mph" or "50" format
                    maxspeed_str = str(maxspeed).lower()
                    if 'mph' in maxspeed_str:
                        speed_val = float(maxspeed_str.replace('mph', '').strip()) * 1.609
                    else:
                        speed_val = float(maxspeed_str.split()[0])
                    speed_mult = speed_val / 50.0  # Normalize to 50 km/h
                except (ValueError, TypeError):
                    pass

        # Final proxy value
        proxy = base * lane_mult * speed_mult
        proxy_weights[(u, v, key)] = max(1.0, proxy)  # Floor at 1

    logger.info(f"Computed OSM proxy for {len(proxy_weights)} edges")

    return proxy_weights


# =============================================================================
# TRAFFIC WEIGHT COMBINATION AND NORMALIZATION
# =============================================================================

def combine_traffic_sources(
    aadt_weights: dict[tuple[int, int, int], float],
    proxy_weights: dict[tuple[int, int, int], float],
    aadt_coverage: float,
    coverage_threshold: float = 0.3,
) -> dict[tuple[int, int, int], float]:
    """
    Combine AADT and proxy weights intelligently.

    Strategy:
    - If coverage >= threshold: Use AADT where available, fill gaps with proxy
    - If coverage < threshold: Use proxy only (log warning)

    Args:
        aadt_weights: AADT-based weights (may be partial)
        proxy_weights: OSM proxy weights (should be complete)
        aadt_coverage: Fraction of edges with AADT
        coverage_threshold: Minimum AADT coverage to use

    Returns:
        Combined edge weights
    """
    if aadt_coverage < coverage_threshold:
        logger.warning(
            f"AADT coverage {aadt_coverage:.1%} below threshold "
            f"{coverage_threshold:.1%}, using OSM proxy only"
        )
        return proxy_weights.copy()

    # Start with proxy, overwrite with AADT where available
    combined = proxy_weights.copy()
    combined.update(aadt_weights)

    n_aadt = len(aadt_weights)
    n_proxy = len(combined) - n_aadt
    logger.info(f"Combined weights: {n_aadt} from AADT, {n_proxy} from proxy")

    return combined


def normalize_traffic_weights(
    weights: dict[tuple[int, int, int], float],
    method: str = 'log',
    clip_percentile: float = 99.0,
) -> dict[tuple[int, int, int], float]:
    """
    Normalize traffic weights for use in utility functions.

    Methods:
    - 'minmax': Scale to [0, 1]
    - 'zscore': (x - mean) / std
    - 'log': log(1 + x) then minmax
    - 'rank': Replace with percentile rank

    Args:
        weights: Raw traffic weights
        method: Normalization method
        clip_percentile: Clip outliers above this percentile

    Returns:
        Normalized weights
    """
    if not weights:
        return {}

    values = np.array(list(weights.values()))
    keys = list(weights.keys())

    # Clip outliers
    if clip_percentile < 100:
        clip_val = np.percentile(values, clip_percentile)
        values = np.clip(values, 0, clip_val)

    if method == 'minmax':
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            values = (values - min_val) / (max_val - min_val)
        else:
            values = np.ones_like(values) * 0.5

    elif method == 'zscore':
        mean_val = values.mean()
        std_val = values.std()
        if std_val > 0:
            values = (values - mean_val) / std_val
        else:
            values = np.zeros_like(values)

    elif method == 'log':
        values = np.log1p(values)
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            values = (values - min_val) / (max_val - min_val)
        else:
            values = np.ones_like(values) * 0.5

    elif method == 'rank':
        from scipy.stats import rankdata
        values = rankdata(values, method='average') / len(values)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Ensure minimum floor for all edges
    values = np.maximum(values, 0.01)

    return dict(zip(keys, values))


# =============================================================================
# NODE-LEVEL TRAFFIC SCORES
# =============================================================================

def compute_node_traffic_scores(
    G: nx.MultiDiGraph,
    edge_traffic: dict[tuple[int, int, int], float],
    aggregation: str = 'sum',
) -> dict[int, float]:
    """
    Compute traffic score for each node from incident edges.

    traffic_node = agg(traffic_weight for all incident edges)

    This is used for endpoint attractiveness computation.

    Args:
        G: Road network graph
        edge_traffic: Traffic weights per edge
        aggregation: How to combine incident edges ('sum', 'mean', 'max')

    Returns:
        dict mapping node_id -> traffic_score
    """
    node_scores: dict[int, list[float]] = {n: [] for n in G.nodes()}

    for (u, v, key), weight in edge_traffic.items():
        if u in node_scores:
            node_scores[u].append(weight)
        if v in node_scores:
            node_scores[v].append(weight)

    result = {}
    for node, weights in node_scores.items():
        if not weights:
            result[node] = 0.01  # Minimum floor
        elif aggregation == 'sum':
            result[node] = sum(weights)
        elif aggregation == 'mean':
            result[node] = np.mean(weights)
        elif aggregation == 'max':
            result[node] = max(weights)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    # Normalize to [0, 1] range
    max_score = max(result.values()) if result else 1.0
    if max_score > 0:
        result = {k: v / max_score for k, v in result.items()}

    logger.info(f"Computed traffic scores for {len(result)} nodes")

    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_traffic_weights_for_region(
    G: nx.MultiDiGraph,
    aadt_path: Path | str | None = None,
    state: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[tuple[int, int, int], float], dict[int, float], dict[str, Any]]:
    """
    Compute all traffic weights for a region in one call.

    Args:
        G: OSMnx road network graph
        aadt_path: Path to AADT shapefile (None to use proxy only)
        state: State abbreviation
        bbox: Bounding box (west, south, east, north)
        config: Configuration overrides

    Returns:
        Tuple of:
        - edge_traffic: dict mapping (u, v, key) -> normalized weight
        - node_traffic: dict mapping node_id -> normalized score
        - metadata: dict with coverage stats and parameters
    """
    config = config or {}

    line_buffer_m = config.get('aadt_line_buffer_m', config.get('aadt_buffer_m', 50.0))
    point_buffer_m = config.get('aadt_point_buffer_m', config.get('aadt_buffer_m', 100.0))
    coverage_threshold = config.get('aadt_coverage_threshold', 0.3)
    normalize_method = config.get('normalize_method', 'log')
    clip_percentile = config.get('clip_percentile', 99.0)

    # Compute OSM proxy (always available)
    proxy_weights = proxy_traffic_from_osm(G)

    # Try to load AADT if path provided
    aadt_weights = {}
    aadt_coverage = 0.0

    if aadt_path is not None and Path(aadt_path).exists():
        try:
            aadt_gdf = load_aadt_shapefile(
                state=state or 'UNK',
                aadt_path=aadt_path,
                clip_bbox=bbox,
            )
            aadt_weights, aadt_coverage = spatial_join_aadt_to_edges(
                G,
                aadt_gdf,
                line_buffer_m=line_buffer_m,
                point_buffer_m=point_buffer_m,
            )
        except Exception as e:
            logger.warning(f"Failed to load AADT: {e}")
    else:
        logger.info("No AADT path provided, using OSM proxy only")

    # Combine sources
    edge_traffic = combine_traffic_sources(
        aadt_weights, proxy_weights, aadt_coverage, coverage_threshold
    )

    # Normalize
    edge_traffic = normalize_traffic_weights(
        edge_traffic, method=normalize_method, clip_percentile=clip_percentile
    )

    # Compute node scores
    node_traffic = compute_node_traffic_scores(G, edge_traffic)

    # Metadata for reporting
    metadata = {
        'n_edges': len(edge_traffic),
        'n_nodes': len(node_traffic),
        'aadt_coverage': aadt_coverage,
        'used_aadt': aadt_coverage >= coverage_threshold,
        'normalize_method': normalize_method,
        'config': config,
    }

    return edge_traffic, node_traffic, metadata


# Import pandas for type coercion
import pandas as pd
