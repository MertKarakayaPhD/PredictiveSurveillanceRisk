"""Tests for data loading module."""

import pytest
import geopandas as gpd
from shapely.geometry import Point

from src.data import _parse_osm_response, get_camera_stats


class TestParseOsmResponse:
    """Tests for OSM response parsing."""

    def test_parse_empty_response(self):
        """Empty response returns empty GeoDataFrame."""
        data = {"elements": []}
        result = _parse_osm_response(data)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_parse_single_camera(self):
        """Single camera node is parsed correctly."""
        data = {
            "elements": [
                {
                    "type": "node",
                    "id": 123456789,
                    "lat": 33.7490,
                    "lon": -84.3880,
                    "tags": {
                        "man_made": "surveillance",
                        "surveillance:type": "ALPR",
                        "operator": "Flock Safety",
                    },
                }
            ]
        }
        result = _parse_osm_response(data)

        assert len(result) == 1
        assert result.iloc[0]["osm_id"] == 123456789
        assert result.iloc[0]["lat"] == 33.7490
        assert result.iloc[0]["lon"] == -84.3880
        assert result.iloc[0]["operator"] == "Flock Safety"
        # Colons replaced with underscores
        assert "surveillance_type" in result.columns

    def test_parse_multiple_cameras(self):
        """Multiple camera nodes are parsed correctly."""
        data = {
            "elements": [
                {
                    "type": "node",
                    "id": 1,
                    "lat": 33.0,
                    "lon": -84.0,
                    "tags": {},
                },
                {
                    "type": "node",
                    "id": 2,
                    "lat": 34.0,
                    "lon": -85.0,
                    "tags": {},
                },
            ]
        }
        result = _parse_osm_response(data)

        assert len(result) == 2
        assert set(result["osm_id"]) == {1, 2}

    def test_parse_ignores_non_nodes(self):
        """Non-node elements are ignored."""
        data = {
            "elements": [
                {"type": "way", "id": 1},
                {"type": "node", "id": 2, "lat": 33.0, "lon": -84.0, "tags": {}},
            ]
        }
        result = _parse_osm_response(data)

        assert len(result) == 1
        assert result.iloc[0]["osm_id"] == 2


class TestCameraStats:
    """Tests for camera statistics computation."""

    def test_empty_geodataframe(self):
        """Empty GeoDataFrame returns zero stats."""
        gdf = gpd.GeoDataFrame(
            columns=["osm_id", "lat", "lon", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
        stats = get_camera_stats(gdf)

        assert stats["n_cameras"] == 0
        assert stats["coverage_area_km2"] == 0

    def test_basic_stats(self):
        """Basic statistics are computed correctly."""
        gdf = gpd.GeoDataFrame(
            {
                "osm_id": [1, 2, 3],
                "lat": [33.0, 33.1, 33.2],
                "lon": [-84.0, -84.1, -84.2],
                "operator": ["Flock", "Flock", "Other"],
                "geometry": [
                    Point(-84.0, 33.0),
                    Point(-84.1, 33.1),
                    Point(-84.2, 33.2),
                ],
            },
            crs="EPSG:4326",
        )
        stats = get_camera_stats(gdf)

        assert stats["n_cameras"] == 3
        assert stats["operators"]["Flock"] == 2
        assert stats["operators"]["Other"] == 1
        assert stats["coverage_area_km2"] > 0
