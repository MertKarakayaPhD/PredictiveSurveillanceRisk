#!/usr/bin/env python3
"""
Traffic-Weighted Road Network Simulation Pipeline

This script:
1. Loads AADT data for each region's state (if available)
2. Computes traffic weights (AADT + OSM proxy fallback)
3. Archives existing results
4. Runs simulation with traffic-aware attractiveness and routing
5. Computes validation metrics
6. Saves results and comparison report

Usage:
    python scripts/run_traffic_weighted_simulation.py --all-regions --n-vehicles 5000
    python scripts/run_traffic_weighted_simulation.py --region atlanta --n-vehicles 1000
    python scripts/run_traffic_weighted_simulation.py --validate  # Check AADT files only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

REGIONS = ['atlanta', 'memphis', 'richmond', 'charlotte', 'lehigh_valley', 'maine']

REGION_STATES = {
    'atlanta': 'GA',
    'memphis': 'TN',
    'richmond': 'VA',
    'charlotte': 'NC',
    'lehigh_valley': 'PA',
    'maine': 'ME',
}

# Traffic weighting configuration
TRAFFIC_CONFIG = {
    # Spatial join
    # - line-based AADT segments: 50 m buffer
    # - point-based AADT stations: nearest edge within 100 m
    'aadt_line_buffer_m': 50.0,
    'aadt_point_buffer_m': 100.0,
    'aadt_coverage_threshold': 0.005,  # 0.5% - low threshold for point-based data

    # Attractiveness blend
    'traffic_blend_factor': 0.7,

    # Route utility (STRONG traffic preference)
    'lambda_traffic': 0.5,

    # Trip distance constraints (US median commute ~7 miles)
    'min_trip_distance_m': 8047.0,   # 5 miles
    'max_trip_distance_m': 16093.0,  # 10 miles

    # Normalization
    'normalize_method': 'log',
    'clip_percentile': 99.0,
}


# =============================================================================
# PATH SETUP
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
AADT_DIR = DATA_RAW / 'aadt'
ARCHIVE_DIR = DATA_PROCESSED / 'archive'
RESULTS_DIR = BASE_DIR / 'results' / 'traffic_weighted'

CAMERA_PATHS = {
    'atlanta': DATA_RAW / 'atlanta_cameras.geojson',
    'memphis': DATA_RAW / 'memphis_cameras.geojson',
    'richmond': DATA_RAW / 'richmond_cameras.geojson',
    'charlotte': DATA_RAW / 'charlotte_cameras.geojson',
    'lehigh_valley': DATA_RAW / 'lehigh_valley_cameras.geojson',
    'maine': DATA_RAW / 'maine_cameras.geojson',
}

AADT_PATHS = {
    'GA': AADT_DIR / 'georgia_aadt.shp',
    'TN': AADT_DIR / 'tennessee_aadt.shp',
    'VA': AADT_DIR / 'virginia_aadt.shp',
    'NC': AADT_DIR / 'north_carolina_aadt.shp',
    'PA': AADT_DIR / 'pennsylvania_aadt.shp',
    'ME': AADT_DIR / 'maine_aadt.shp',
}


# =============================================================================
# ARCHIVE FUNCTIONS
# =============================================================================

def archive_current_results(archive_name: str | None = None) -> Path:
    """
    Archive existing results before running traffic-weighted simulation.

    Creates: data/processed/archive/{archive_name}/
    """
    if archive_name is None:
        archive_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_pre_traffic'

    archive_path = ARCHIVE_DIR / archive_name
    archive_path.mkdir(parents=True, exist_ok=True)

    # Copy existing trajectory files
    n_copied = 0
    for pkl_file in DATA_PROCESSED.glob('road_trajectories_*.pkl'):
        shutil.copy2(pkl_file, archive_path / pkl_file.name)
        n_copied += 1

    # Save metadata
    metadata = {
        'archived_at': datetime.now().isoformat(),
        'files_copied': n_copied,
        'purpose': 'Pre-traffic-weighting baseline',
    }
    with open(archive_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Archived {n_copied} files to {archive_path}")
    return archive_path


# =============================================================================
# TRAFFIC WEIGHT COMPUTATION
# =============================================================================

def load_road_network_for_region(region: str) -> tuple:
    """Load road network and compute bounding box from cameras."""
    camera_path = CAMERA_PATHS[region]

    with open(camera_path) as f:
        data = json.load(f)

    features = [f for f in data.get('features', []) if f['geometry']['type'] == 'Point']
    if not features:
        raise ValueError(f"No cameras found for {region}")

    lats = [f['geometry']['coordinates'][1] for f in features]
    lons = [f['geometry']['coordinates'][0] for f in features]

    north, south = max(lats) + 0.05, min(lats) - 0.05
    east, west = max(lons) + 0.05, min(lons) - 0.05
    bbox = (west, south, east, north)

    logger.info(f"Downloading road network for {region}...")
    G_road = ox.graph_from_bbox(bbox=bbox, network_type='drive', simplify=True)
    logger.info(f"Loaded {G_road.number_of_nodes()} nodes, {G_road.number_of_edges()} edges")

    return G_road, bbox


def compute_traffic_weights_for_region(
    G_road,
    region: str,
    bbox: tuple[float, float, float, float],
) -> tuple[dict, dict, dict]:
    """
    Compute traffic weights for a region.

    Returns:
        (edge_traffic, node_traffic, metadata)
    """
    from src.traffic_weights import (
        load_aadt_shapefile,
        spatial_join_aadt_to_edges,
        proxy_traffic_from_osm,
        combine_traffic_sources,
        normalize_traffic_weights,
        compute_node_traffic_scores,
    )

    state = REGION_STATES[region]
    aadt_path = AADT_PATHS.get(state)
    config = TRAFFIC_CONFIG

    # Compute OSM proxy (always available)
    logger.info(f"Computing OSM proxy traffic weights for {region}...")
    proxy_weights = proxy_traffic_from_osm(G_road)
    logger.info(f"Computed {len(proxy_weights)} proxy edge weights")

    # Try to load AADT if available
    aadt_weights = {}
    aadt_coverage = 0.0

    if aadt_path and aadt_path.exists():
        logger.info(f"Loading AADT data for {state} from {aadt_path}...")
        try:
            aadt_gdf = load_aadt_shapefile(state=state, aadt_path=aadt_path, clip_bbox=bbox)
            aadt_weights, aadt_coverage = spatial_join_aadt_to_edges(
                G_road,
                aadt_gdf,
                line_buffer_m=config['aadt_line_buffer_m'],
                point_buffer_m=config['aadt_point_buffer_m'],
            )
            logger.info(f"AADT coverage: {aadt_coverage:.1%}")
        except Exception as e:
            logger.warning(f"Failed to load AADT for {state}: {e}")
    else:
        logger.info(f"No AADT data available for {state}, using OSM proxy only")

    # Combine sources
    edge_traffic = combine_traffic_sources(
        aadt_weights,
        proxy_weights,
        aadt_coverage,
        coverage_threshold=config['aadt_coverage_threshold'],
    )

    # Normalize
    edge_traffic = normalize_traffic_weights(
        edge_traffic,
        method=config['normalize_method'],
        clip_percentile=config['clip_percentile'],
    )

    # Compute node scores
    node_traffic = compute_node_traffic_scores(G_road, edge_traffic)

    # Metadata
    metadata = {
        'region': region,
        'state': state,
        'n_edges': len(edge_traffic),
        'n_nodes': len(node_traffic),
        'aadt_available': aadt_path.exists() if aadt_path else False,
        'aadt_coverage': aadt_coverage,
        'used_aadt': aadt_coverage >= config['aadt_coverage_threshold'],
        'config': config,
    }

    return edge_traffic, node_traffic, metadata


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_traffic_weighted_simulation(
    region: str,
    n_vehicles: int = 5000,
    n_trips_per_vehicle: int = 10,
    seed: int = 42,
    n_workers: int = 1,
) -> dict:
    """
    Run traffic-weighted simulation for a single region.

    Returns:
        Dict with simulation results and metadata
    """
    from scripts.road_network_simulation import simulate_road_network_trips

    camera_path = CAMERA_PATHS[region]
    config = TRAFFIC_CONFIG

    # Load road network
    G_road, bbox = load_road_network_for_region(region)

    # Compute traffic weights
    edge_traffic, node_traffic, weight_metadata = compute_traffic_weights_for_region(
        G_road, region, bbox
    )

    # Save weights for reproducibility
    weights_dir = RESULTS_DIR / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    with open(weights_dir / f'{region}_edge_traffic.pkl', 'wb') as f:
        pickle.dump(edge_traffic, f)
    with open(weights_dir / f'{region}_node_traffic.pkl', 'wb') as f:
        pickle.dump(node_traffic, f)
    with open(weights_dir / f'{region}_metadata.json', 'w') as f:
        json.dump(weight_metadata, f, indent=2)

    logger.info(f"Saved traffic weights to {weights_dir}")

    # Run simulation
    logger.info(f"Running simulation for {region} with {n_vehicles} vehicles...")

    result = simulate_road_network_trips(
        region=region,
        camera_geojson_path=camera_path,
        n_vehicles=n_vehicles,
        n_trips_per_vehicle=n_trips_per_vehicle,
        seed=seed,
        verbose=True,
        # Traffic weighting
        traffic_weights=node_traffic,
        edge_traffic_weights=edge_traffic,
        traffic_blend_factor=config['traffic_blend_factor'],
        lambda_traffic=config['lambda_traffic'],
        # Trip distance constraints
        min_trip_distance_m=config['min_trip_distance_m'],
        max_trip_distance_m=config['max_trip_distance_m'],
        # Parallel processing
        n_workers=n_workers,
    )

    if result is None:
        logger.error(f"Simulation failed for {region}")
        return None

    # Add metadata
    result['traffic_weight_metadata'] = weight_metadata
    result['config'] = config

    # Save result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f'road_trajectories_{region}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)

    logger.info(f"Saved results to {result_path}")

    return result


# =============================================================================
# VALIDATION
# =============================================================================

def compute_validation_metrics(results: dict[str, dict]) -> dict:
    """
    Compute validation metrics across all regions.

    Returns:
        Dict with validation metrics per region
    """
    validation = {}

    for region, result in results.items():
        if result is None:
            continue

        stats = result.get('network_stats', {})
        weight_meta = result.get('traffic_weight_metadata', {})

        # Trip distance stats from metadata
        trip_meta = result.get('trip_metadata', [])
        if trip_meta:
            route_lengths = [t.get('route_length_m', 0) for t in trip_meta if t.get('route_length_m', 0) > 0]
            if route_lengths:
                route_lengths_miles = np.array(route_lengths) / 1609.34
                trip_stats = {
                    'min_miles': float(np.min(route_lengths_miles)),
                    'median_miles': float(np.median(route_lengths_miles)),
                    'max_miles': float(np.max(route_lengths_miles)),
                    'pct_in_range': float(np.mean((route_lengths_miles >= 5) & (route_lengths_miles <= 10))),
                }
            else:
                trip_stats = {}
        else:
            trip_stats = {}

        validation[region] = {
            'total_trips': stats.get('total_trips', 0),
            'aadt_coverage': weight_meta.get('aadt_coverage', 0),
            'used_aadt': weight_meta.get('used_aadt', False),
            'trip_distance_stats': trip_stats,
        }

    return validation


def generate_comparison_report(
    archive_path: Path | None,
    results: dict[str, dict],
    validation: dict,
) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# Traffic-Weighted Simulation Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        "",
        f"- Traffic blend factor: {TRAFFIC_CONFIG['traffic_blend_factor']}",
        f"- Lambda traffic (route utility): {TRAFFIC_CONFIG['lambda_traffic']}",
        f"- Trip distance range: {TRAFFIC_CONFIG['min_trip_distance_m']/1609.34:.1f} - {TRAFFIC_CONFIG['max_trip_distance_m']/1609.34:.1f} miles",
        "",
        "## Results by Region",
        "",
    ]

    for region in REGIONS:
        if region not in results or results[region] is None:
            lines.append(f"### {region.upper()}: FAILED")
            lines.append("")
            continue

        result = results[region]
        val = validation.get(region, {})

        lines.extend([
            f"### {region.upper()}",
            "",
            f"- Total trips: {val.get('total_trips', 'N/A')}",
            f"- AADT coverage: {val.get('aadt_coverage', 0):.1%}",
            f"- Used AADT: {val.get('used_aadt', False)}",
        ])

        trip_stats = val.get('trip_distance_stats', {})
        if trip_stats:
            lines.extend([
                f"- Trip distances: {trip_stats.get('min_miles', 0):.1f} - {trip_stats.get('max_miles', 0):.1f} miles",
                f"- Median trip distance: {trip_stats.get('median_miles', 0):.1f} miles",
                f"- % in target range (5-10 miles): {trip_stats.get('pct_in_range', 0):.1%}",
            ])

        lines.append("")

    if archive_path:
        lines.extend([
            "## Baseline Archive",
            "",
            f"Previous results archived to: `{archive_path}`",
            "",
        ])

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run traffic-weighted ALPR simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--region', '-r',
        type=str,
        choices=REGIONS + ['all'],
        default='all',
        help='Region to simulate (default: all)',
    )
    parser.add_argument(
        '--all-regions',
        action='store_true',
        help='Run all regions (same as --region all)',
    )
    parser.add_argument(
        '--n-vehicles', '-n',
        type=int,
        default=5000,
        help='Number of vehicles per region (default: 5000)',
    )
    parser.add_argument(
        '--n-trips',
        type=int,
        default=10,
        help='Trips per vehicle (default: 10)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--no-archive',
        action='store_true',
        help='Skip archiving existing results',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate AADT files, do not run simulation',
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of parallel worker processes (default: 1 = sequential)',
    )

    args = parser.parse_args()

    # Validate-only mode
    if args.validate:
        from scripts.download_aadt_data import validate_aadt_files, print_validation_report
        results = validate_aadt_files()
        print_validation_report(results)
        return

    # Determine regions to run
    if args.all_regions or args.region == 'all':
        regions = REGIONS
    else:
        regions = [args.region]

    logger.info(f"Running traffic-weighted simulation for: {regions}")
    logger.info(f"  n_vehicles: {args.n_vehicles}")
    logger.info(f"  n_trips: {args.n_trips}")
    logger.info(f"  seed: {args.seed}")
    logger.info(f"  workers: {args.workers}")

    # Archive existing results
    archive_path = None
    if not args.no_archive:
        archive_path = archive_current_results()

    # Run simulations
    all_results = {}
    for region in regions:
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {region.upper()}")
        logger.info(f"{'='*60}")

        try:
            result = run_traffic_weighted_simulation(
                region=region,
                n_vehicles=args.n_vehicles,
                n_trips_per_vehicle=args.n_trips,
                seed=args.seed,
                n_workers=args.workers,
            )
            all_results[region] = result
        except Exception as e:
            logger.error(f"Failed to process {region}: {e}")
            all_results[region] = None

    # Validation
    logger.info("\nComputing validation metrics...")
    validation = compute_validation_metrics(all_results)

    # Generate report
    report = generate_comparison_report(archive_path, all_results, validation)

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"\nReport saved to: {report_path}")
    print("\n" + report)


if __name__ == '__main__':
    main()
