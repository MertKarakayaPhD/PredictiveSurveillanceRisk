"""
Unit tests for traffic_weights module.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.traffic_weights import (
    OSM_HIGHWAY_PROXY_AADT,
    detect_aadt_column,
    proxy_traffic_from_osm,
    normalize_traffic_weights,
    combine_traffic_sources,
    compute_node_traffic_scores,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def simple_road_graph():
    """Create a simple road network graph for testing."""
    G = nx.MultiDiGraph()

    # Add nodes with coordinates
    G.add_node(0, x=-70.0, y=43.0)
    G.add_node(1, x=-70.1, y=43.0)
    G.add_node(2, x=-70.1, y=43.1)
    G.add_node(3, x=-70.0, y=43.1)

    # Add edges with different highway types
    G.add_edge(0, 1, 0, highway='primary', length=1000, lanes=4, maxspeed='50')
    G.add_edge(1, 2, 0, highway='secondary', length=1000, lanes=2)
    G.add_edge(2, 3, 0, highway='residential', length=500)
    G.add_edge(3, 0, 0, highway='tertiary', length=500, lanes=2)

    # Bidirectional edges
    G.add_edge(1, 0, 0, highway='primary', length=1000, lanes=4, maxspeed='50')
    G.add_edge(2, 1, 0, highway='secondary', length=1000, lanes=2)

    return G


@pytest.fixture
def mock_gdf():
    """Create a mock GeoDataFrame with AADT column."""
    import geopandas as gpd
    from shapely.geometry import LineString

    return gpd.GeoDataFrame({
        'AADT': [10000, 20000, 5000],
        'geometry': [
            LineString([(-70.0, 43.0), (-70.1, 43.0)]),
            LineString([(-70.1, 43.0), (-70.1, 43.1)]),
            LineString([(-70.0, 43.0), (-70.0, 43.1)]),
        ],
    }, crs="EPSG:4326")


# =============================================================================
# TEST: OSM PROXY TRAFFIC
# =============================================================================

def test_proxy_traffic_from_osm_basic(simple_road_graph):
    """Test that OSM proxy computation produces weights."""
    weights = proxy_traffic_from_osm(simple_road_graph)

    assert len(weights) == 6  # 6 edges in the graph
    assert all(v > 0 for v in weights.values())


def test_proxy_traffic_highway_ordering(simple_road_graph):
    """Test that primary > secondary > residential in proxy weights."""
    weights = proxy_traffic_from_osm(simple_road_graph)

    # Get weights by highway type
    primary_weight = weights[(0, 1, 0)]
    secondary_weight = weights[(1, 2, 0)]
    residential_weight = weights[(2, 3, 0)]

    assert primary_weight > secondary_weight
    assert secondary_weight > residential_weight


def test_proxy_traffic_lane_multiplier(simple_road_graph):
    """Test that lane count affects proxy weight."""
    # Without lanes
    weights_no_lanes = proxy_traffic_from_osm(simple_road_graph, use_lanes=False)

    # With lanes
    weights_with_lanes = proxy_traffic_from_osm(simple_road_graph, use_lanes=True)

    # Primary edge has 4 lanes, so should be higher with lane multiplier
    assert weights_with_lanes[(0, 1, 0)] >= weights_no_lanes[(0, 1, 0)]


def test_osm_highway_proxy_values():
    """Test that OSM proxy values are reasonable."""
    assert OSM_HIGHWAY_PROXY_AADT['motorway'] > OSM_HIGHWAY_PROXY_AADT['primary']
    assert OSM_HIGHWAY_PROXY_AADT['primary'] > OSM_HIGHWAY_PROXY_AADT['secondary']
    assert OSM_HIGHWAY_PROXY_AADT['residential'] < OSM_HIGHWAY_PROXY_AADT['tertiary']


# =============================================================================
# TEST: NORMALIZATION
# =============================================================================

def test_normalize_traffic_weights_minmax():
    """Test minmax normalization produces [0, 1] range."""
    weights = {(0, 1, 0): 100, (1, 2, 0): 1000, (2, 3, 0): 50000}

    normalized = normalize_traffic_weights(weights, method='minmax')

    assert all(0 <= v <= 1 for v in normalized.values())
    assert max(normalized.values()) == 1.0


def test_normalize_traffic_weights_log():
    """Test log normalization produces [0, 1] range."""
    weights = {(0, 1, 0): 100, (1, 2, 0): 1000, (2, 3, 0): 50000}

    normalized = normalize_traffic_weights(weights, method='log')

    assert all(0 <= v <= 1 for v in normalized.values())


def test_normalize_traffic_weights_preserves_order():
    """Test normalization preserves relative ordering."""
    weights = {(0, 1, 0): 100, (1, 2, 0): 1000, (2, 3, 0): 50000}

    for method in ['minmax', 'log', 'rank']:
        normalized = normalize_traffic_weights(weights, method=method)

        assert normalized[(0, 1, 0)] < normalized[(1, 2, 0)]
        assert normalized[(1, 2, 0)] < normalized[(2, 3, 0)]


def test_normalize_empty_weights():
    """Test normalization handles empty input."""
    weights = {}
    normalized = normalize_traffic_weights(weights)
    assert normalized == {}


def test_normalize_single_value():
    """Test normalization handles single value."""
    weights = {(0, 1, 0): 1000}
    normalized = normalize_traffic_weights(weights, method='minmax')
    assert (0, 1, 0) in normalized


# =============================================================================
# TEST: COMBINE SOURCES
# =============================================================================

def test_combine_traffic_sources_aadt_preferred():
    """Test that AADT overwrites proxy when available."""
    proxy_weights = {(0, 1, 0): 100, (1, 2, 0): 200, (2, 3, 0): 300}
    aadt_weights = {(0, 1, 0): 50000}  # Only one edge has AADT

    combined = combine_traffic_sources(
        aadt_weights, proxy_weights, aadt_coverage=0.5, coverage_threshold=0.3
    )

    # AADT should overwrite proxy for edge (0, 1, 0)
    assert combined[(0, 1, 0)] == 50000
    # Other edges should keep proxy values
    assert combined[(1, 2, 0)] == 200
    assert combined[(2, 3, 0)] == 300


def test_combine_traffic_sources_low_coverage_uses_proxy():
    """Test that low AADT coverage falls back to proxy only."""
    proxy_weights = {(0, 1, 0): 100, (1, 2, 0): 200}
    aadt_weights = {(0, 1, 0): 50000}

    combined = combine_traffic_sources(
        aadt_weights, proxy_weights, aadt_coverage=0.1, coverage_threshold=0.3
    )

    # Should use proxy only (not overwrite with AADT)
    assert combined == proxy_weights


# =============================================================================
# TEST: NODE TRAFFIC SCORES
# =============================================================================

def test_compute_node_traffic_scores_sum(simple_road_graph):
    """Test node score computation with sum aggregation."""
    edge_traffic = {
        (0, 1, 0): 1.0,
        (1, 0, 0): 1.0,
        (1, 2, 0): 0.5,
        (2, 1, 0): 0.5,
        (2, 3, 0): 0.2,
        (3, 0, 0): 0.3,
    }

    scores = compute_node_traffic_scores(simple_road_graph, edge_traffic, aggregation='sum')

    # Node 1 should have highest score (connected to primary edges)
    assert scores[1] >= scores[2]
    # All nodes should have scores in [0, 1] after normalization
    assert all(0 <= v <= 1 for v in scores.values())


def test_compute_node_traffic_scores_mean(simple_road_graph):
    """Test node score computation with mean aggregation."""
    edge_traffic = {
        (0, 1, 0): 1.0,
        (1, 0, 0): 1.0,
        (1, 2, 0): 0.5,
        (2, 1, 0): 0.5,
        (2, 3, 0): 0.2,
        (3, 0, 0): 0.3,
    }

    scores = compute_node_traffic_scores(simple_road_graph, edge_traffic, aggregation='mean')

    assert all(0 <= v <= 1 for v in scores.values())


def test_compute_node_traffic_scores_empty():
    """Test node scores with no edge traffic data."""
    G = nx.MultiDiGraph()
    G.add_node(0)
    G.add_node(1)

    scores = compute_node_traffic_scores(G, {})

    # Should return floor values
    assert all(v >= 0.01 for v in scores.values())


# =============================================================================
# TEST: AADT COLUMN DETECTION
# =============================================================================

def test_detect_aadt_column_exact_match():
    """Test exact AADT column detection."""
    import pandas as pd

    df = pd.DataFrame({'AADT': [1, 2, 3], 'other_col': [4, 5, 6]})
    assert detect_aadt_column(df) == 'AADT'


def test_detect_aadt_column_case_insensitive():
    """Test case-insensitive column detection."""
    import pandas as pd

    df = pd.DataFrame({'aadt': [1, 2, 3], 'other_col': [4, 5, 6]})
    assert detect_aadt_column(df) == 'aadt'


def test_detect_aadt_column_partial_match():
    """Test partial AADT column detection."""
    import pandas as pd

    df = pd.DataFrame({'AADT_2023': [1, 2, 3], 'other_col': [4, 5, 6]})
    result = detect_aadt_column(df)
    assert result == 'AADT_2023'


def test_detect_aadt_column_not_found():
    """Test AADT column detection when not present."""
    import pandas as pd

    df = pd.DataFrame({'traffic': [1, 2, 3], 'volume': [4, 5, 6]})
    # Should still find via partial match
    result = detect_aadt_column(df)
    assert result is not None  # 'traffic' or 'volume' should match


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

def test_full_traffic_weight_pipeline(simple_road_graph):
    """Test full pipeline from OSM proxy to node scores."""
    # Step 1: Compute proxy weights
    proxy_weights = proxy_traffic_from_osm(simple_road_graph)
    assert len(proxy_weights) > 0

    # Step 2: Combine (no AADT in this test)
    combined = combine_traffic_sources({}, proxy_weights, aadt_coverage=0.0)
    assert combined == proxy_weights

    # Step 3: Normalize
    normalized = normalize_traffic_weights(combined, method='log')
    assert all(0 <= v <= 1 for v in normalized.values())

    # Step 4: Compute node scores
    node_scores = compute_node_traffic_scores(simple_road_graph, normalized)
    assert all(0 <= v <= 1 for v in node_scores.values())


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
