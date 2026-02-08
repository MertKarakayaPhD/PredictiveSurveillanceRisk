#!/usr/bin/env python3
"""
AADT Data Acquisition Script

This script provides guidance for downloading AADT (Annual Average Daily Traffic)
data from state DOT portals. Since most state DOT portals require manual
navigation and download, this script:

1. Lists the download URLs and instructions for each state
2. Validates downloaded files
3. Reports on file schema and coverage

USAGE:
    python scripts/download_aadt_data.py --validate    # Check existing files
    python scripts/download_aadt_data.py --list        # Show download instructions

MANUAL DOWNLOAD REQUIRED:
    State DOT portals typically require manual download. After downloading,
    place files in: data/raw/aadt/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.traffic_weights import detect_aadt_column


# =============================================================================
# AADT DATA SOURCES
# =============================================================================

AADT_SOURCES = {
    'GA': {
        'region': 'Atlanta',
        'url': 'https://www.dot.ga.gov/GDOT/Pages/GISDataDownload.aspx',
        'direct_download': None,  # Requires navigation
        'steps': [
            '1. Go to: https://www.dot.ga.gov/GDOT/Pages/GISDataDownload.aspx',
            '2. Scroll to "Transportation" section',
            '3. Look for "Traffic Counts" or "AADT" shapefile',
            '4. Download the ZIP file',
            '5. Extract to: data/raw/aadt/georgia_aadt.shp',
        ],
        'alternative_url': 'https://gis-dp.dot.ga.gov/arcgis/rest/services',
        'expected_columns': ['AADT', 'AADTVN', 'AADT_YEAR', 'ROUTE_ID'],
        'expected_file': 'georgia_aadt.shp',
    },
    'TN': {
        'region': 'Memphis',
        'url': 'https://www.tn.gov/tdot/long-range-planning-home/longrange-highway-data-office/hwy-traffvolume.html',
        'direct_download': None,
        'steps': [
            '1. Go to: https://www.tn.gov/tdot/long-range-planning-home/longrange-highway-data-office/hwy-traffvolume.html',
            '2. Click "GIS Data" or "Download Traffic Volume Data"',
            '3. Download the statewide AADT shapefile',
            '4. Extract to: data/raw/aadt/tennessee_aadt.shp',
        ],
        'alternative_url': 'https://tn-tdot.maps.arcgis.com/home/index.html',
        'expected_columns': ['AADT', 'ADT', 'YEAR', 'ROUTE'],
        'expected_file': 'tennessee_aadt.shp',
    },
    'VA': {
        'region': 'Richmond',
        'url': 'https://vdot.maps.arcgis.com/home/index.html',
        'direct_download': None,
        'steps': [
            '1. Go to: https://vdot.maps.arcgis.com/home/index.html',
            '2. Search for "Traffic Volume" or "AADT"',
            '3. Find "Traffic Volume Estimates" layer',
            '4. Export/Download as shapefile',
            '5. Extract to: data/raw/aadt/virginia_aadt.shp',
        ],
        'alternative_url': 'https://www.virginiaroads.org/datasets',
        'expected_columns': ['AADT', 'ADT', 'AADT_VN', 'TRAFFIC'],
        'expected_file': 'virginia_aadt.shp',
    },
    'NC': {
        'region': 'Charlotte',
        'url': 'https://connect.ncdot.gov/resources/gis/Pages/GIS-Data-Layers.aspx',
        'direct_download': None,
        'steps': [
            '1. Go to: https://connect.ncdot.gov/resources/gis/Pages/GIS-Data-Layers.aspx',
            '2. Look for "Traffic Volume" or "AADT" under Transportation layers',
            '3. Download the statewide shapefile',
            '4. Extract to: data/raw/aadt/north_carolina_aadt.shp',
        ],
        'alternative_url': 'https://ncdot.maps.arcgis.com/home/index.html',
        'expected_columns': ['AADT', 'ADT', 'AADTVN', 'TRAFFIC_VO'],
        'expected_file': 'north_carolina_aadt.shp',
    },
    'PA': {
        'region': 'Lehigh Valley',
        'url': 'https://data-pennshare.opendata.arcgis.com/',
        'direct_download': None,
        'steps': [
            '1. Go to: https://data-pennshare.opendata.arcgis.com/',
            '2. Search for "Traffic Volume" or "AADT"',
            '3. Download "Traffic Volume" or "Annual Average Daily Traffic" layer',
            '4. Export as shapefile',
            '5. Extract to: data/raw/aadt/pennsylvania_aadt.shp',
        ],
        'alternative_url': 'https://gis.penndot.gov/arcgis/rest/services',
        'expected_columns': ['AADT', 'ADT', 'AVG_AADT', 'VOLUME'],
        'expected_file': 'pennsylvania_aadt.shp',
    },
    'ME': {
        'region': 'Maine',
        'url': 'https://www.maine.gov/mdot/mapviewer/',
        'direct_download': None,
        'steps': [
            '1. Go to: https://www.maine.gov/mdot/mapviewer/',
            '2. Use the map viewer to find Traffic Volume layer',
            '3. Or try: https://maine.hub.arcgis.com/ and search "traffic"',
            '4. Download as shapefile',
            '5. Extract to: data/raw/aadt/maine_aadt.shp',
        ],
        'alternative_url': 'https://maine.hub.arcgis.com/',
        'expected_columns': ['AADT', 'ADT', 'TRAFFIC', 'VOLUME'],
        'expected_file': 'maine_aadt.shp',
    },
}

# Where to put downloaded files
AADT_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'aadt'


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_aadt_files() -> dict[str, dict]:
    """
    Check that downloaded AADT files exist and have expected schema.

    Returns:
        Dict with validation results per state
    """
    results = {}

    AADT_DIR.mkdir(parents=True, exist_ok=True)

    for state, info in AADT_SOURCES.items():
        filepath = AADT_DIR / info['expected_file']
        result = {
            'state': state,
            'region': info['region'],
            'expected_file': str(filepath),
            'exists': filepath.exists(),
            'columns': [],
            'aadt_column': None,
            'row_count': 0,
            'aadt_range': None,
        }

        if filepath.exists():
            try:
                import geopandas as gpd
                gdf = gpd.read_file(filepath)
                result['columns'] = list(gdf.columns)
                result['row_count'] = len(gdf)

                # Use the same detector as runtime simulation pipeline.
                detected = detect_aadt_column(gdf)
                if detected:
                    result['aadt_column'] = detected

                if result['aadt_column']:
                    import pandas as pd
                    aadt_values = gdf[result['aadt_column']]
                    if aadt_values.dtype == 'object':
                        aadt_values = aadt_values.astype(str).str.replace(',', '', regex=False)
                    aadt_values = pd.to_numeric(aadt_values, errors='coerce').dropna()
                    if len(aadt_values) > 0:
                        result['aadt_range'] = (
                            float(aadt_values.min()),
                            float(aadt_values.max())
                        )

            except Exception as e:
                result['error'] = str(e)

        results[state] = result

    return results


def print_validation_report(results: dict[str, dict]) -> None:
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("AADT DATA VALIDATION REPORT")
    print("=" * 70)

    found = 0
    missing = 0

    for state, result in results.items():
        status = "FOUND" if result['exists'] else "MISSING"
        symbol = "[+]" if result['exists'] else "[-]"

        print(f"\n{symbol} {state} ({result['region']})")
        print(f"    File: {result['expected_file']}")
        print(f"    Status: {status}")

        if result['exists']:
            found += 1
            print(f"    Rows: {result['row_count']}")
            print(f"    AADT Column: {result['aadt_column'] or 'NOT FOUND'}")
            if result['aadt_range']:
                print(f"    AADT Range: {result['aadt_range'][0]:.0f} - {result['aadt_range'][1]:.0f}")
            if 'error' in result:
                print(f"    Error: {result['error']}")
        else:
            missing += 1
            print(f"    Download from: {AADT_SOURCES[state]['url']}")

    print("\n" + "-" * 70)
    print(f"SUMMARY: {found} found, {missing} missing")
    print("-" * 70 + "\n")


def print_download_instructions() -> None:
    """Print download instructions for all states."""
    print("\n" + "=" * 70)
    print("AADT DATA DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print(f"\nTarget directory: {AADT_DIR}\n")

    for state, info in AADT_SOURCES.items():
        print(f"\n{'='*50}")
        print(f"{state} - {info['region']}")
        print(f"{'='*50}")
        print(f"\nPrimary URL: {info['url']}")
        print(f"Alternative: {info['alternative_url']}")
        print(f"\nExpected file: {info['expected_file']}")
        print(f"Expected AADT columns: {info['expected_columns']}")
        print("\nDownload steps:")
        for step in info['steps']:
            print(f"    {step}")

    print("\n" + "=" * 70)
    print("After downloading, run: python scripts/download_aadt_data.py --validate")
    print("=" * 70 + "\n")


def create_aadt_directory() -> None:
    """Create the AADT data directory if it doesn't exist."""
    AADT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {AADT_DIR}")

    # Create a README in the directory
    readme_path = AADT_DIR / 'README.md'
    if not readme_path.exists():
        readme_content = """# AADT Data Directory

Place downloaded AADT shapefiles here.

## Expected Files

| State | File | Region |
|-------|------|--------|
| GA | georgia_aadt.shp | Atlanta |
| TN | tennessee_aadt.shp | Memphis |
| VA | virginia_aadt.shp | Richmond |
| NC | north_carolina_aadt.shp | Charlotte |
| PA | pennsylvania_aadt.shp | Lehigh Valley |
| ME | maine_aadt.shp | Maine |

## Download Instructions

Run `python scripts/download_aadt_data.py --list` for download instructions.

## Validation

Run `python scripts/download_aadt_data.py --validate` to check downloaded files.
"""
        readme_path.write_text(readme_content)
        print(f"Created: {readme_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AADT data acquisition and validation tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_aadt_data.py --list      # Show download instructions
    python scripts/download_aadt_data.py --validate  # Validate existing files
    python scripts/download_aadt_data.py --setup     # Create directory structure
        """
    )

    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate existing AADT files'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List download instructions for all states'
    )
    parser.add_argument(
        '--setup', '-s',
        action='store_true',
        help='Create AADT directory structure'
    )
    parser.add_argument(
        '--state',
        type=str,
        choices=list(AADT_SOURCES.keys()),
        help='Show info for specific state only'
    )

    args = parser.parse_args()

    # Default to showing list if no args
    if not any([args.validate, args.list, args.setup]):
        args.list = True

    if args.setup:
        create_aadt_directory()

    if args.list:
        if args.state:
            info = AADT_SOURCES[args.state]
            print(f"\n{args.state} - {info['region']}")
            print(f"URL: {info['url']}")
            for step in info['steps']:
                print(f"  {step}")
        else:
            print_download_instructions()

    if args.validate:
        results = validate_aadt_files()
        print_validation_report(results)


if __name__ == '__main__':
    main()
