"""Tests for network construction module."""

import pytest
import networkx as nx
import numpy as np

from src.network import (
    compute_topology_metrics,
    get_edge_distance_stats,
    subsample_network,
)


class TestTopologyMetrics:
    """Tests for network topology metrics."""

    def test_empty_graph(self):
        """Empty graph returns zero metrics."""
        G = nx.DiGraph()
        metrics = compute_topology_metrics(G)

        assert metrics["n_cameras"] == 0
        assert metrics["n_edges"] == 0
        assert metrics["density"] == 0.0

    def test_simple_graph(self):
        """Simple graph metrics are computed correctly."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        G.add_edge(1, 2, road_distance=100)
        G.add_edge(2, 3, road_distance=200)
        G.add_edge(1, 3, road_distance=300)

        metrics = compute_topology_metrics(G)

        assert metrics["n_cameras"] == 3
        assert metrics["n_edges"] == 3
        assert metrics["density"] > 0

    def test_betweenness_centrality(self):
        """Betweenness centrality identifies hub nodes."""
        # Star graph: center node should have high betweenness
        G = nx.DiGraph()
        G.add_edges_from([
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (2, 0), (3, 0), (4, 0),
        ])

        metrics = compute_topology_metrics(G)

        # Center node (0) should be in choke points
        assert 0 in metrics.get("choke_points", []) or metrics["max_betweenness"] > 0


class TestEdgeDistanceStats:
    """Tests for edge distance statistics."""

    def test_empty_graph(self):
        """Empty graph returns zero stats."""
        G = nx.DiGraph()
        stats = get_edge_distance_stats(G)

        assert stats["min_distance_m"] == 0
        assert stats["mean_distance_m"] == 0

    def test_distance_stats(self):
        """Distance statistics are computed correctly."""
        G = nx.DiGraph()
        G.add_edge(1, 2, road_distance=100)
        G.add_edge(2, 3, road_distance=200)
        G.add_edge(3, 4, road_distance=300)

        stats = get_edge_distance_stats(G)

        assert stats["min_distance_m"] == 100
        assert stats["max_distance_m"] == 300
        assert stats["mean_distance_m"] == 200


class TestSubsampleNetwork:
    """Tests for network subsampling."""

    def test_subsample_fraction(self):
        """Subsampling respects the fraction parameter."""
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        for i in range(100):
            G.add_edge(i, (i + 1) % 100)

        G_sub = subsample_network(G, fraction=0.5, seed=42)

        assert G_sub.number_of_nodes() == 50

    def test_subsample_full(self):
        """Fraction 1.0 returns all nodes."""
        G = nx.DiGraph()
        G.add_nodes_from(range(10))

        G_sub = subsample_network(G, fraction=1.0)

        assert G_sub.number_of_nodes() == 10

    def test_subsample_reproducible(self):
        """Same seed produces same result."""
        G = nx.DiGraph()
        G.add_nodes_from(range(100))

        G_sub1 = subsample_network(G, fraction=0.3, seed=42)
        G_sub2 = subsample_network(G, fraction=0.3, seed=42)

        assert set(G_sub1.nodes()) == set(G_sub2.nodes())

    def test_invalid_fraction(self):
        """Invalid fraction raises error."""
        G = nx.DiGraph()
        G.add_nodes_from(range(10))

        with pytest.raises(ValueError):
            subsample_network(G, fraction=0)

        with pytest.raises(ValueError):
            subsample_network(G, fraction=1.5)
