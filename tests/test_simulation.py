"""Tests for trajectory simulation module."""

import pytest
import networkx as nx
import numpy as np

from src.simulation import (
    Trajectory,
    generate_trajectories,
    split_trajectories,
    get_trajectory_stats,
)


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_trajectory_creation(self):
        """Trajectory can be created with basic attributes."""
        traj = Trajectory(
            vehicle_id="v001",
            camera_sequence=[1, 2, 3],
            timestamps=[0.0, 60.0, 120.0],
        )

        assert traj.vehicle_id == "v001"
        assert len(traj) == 3
        assert traj.camera_sequence == [1, 2, 3]

    def test_observations(self):
        """Observations returns camera-timestamp pairs."""
        traj = Trajectory(
            vehicle_id="v001",
            camera_sequence=[1, 2],
            timestamps=[0.0, 60.0],
        )

        obs = traj.observations()
        assert obs == [(1, 0.0), (2, 60.0)]


class TestGenerateTrajectories:
    """Tests for trajectory generation."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple camera network for testing."""
        G = nx.DiGraph()
        # Linear chain: 1 -> 2 -> 3 -> 4 -> 5
        for i in range(1, 5):
            G.add_edge(i, i + 1, road_distance=100)
        # Add some back edges for bidirectional travel
        for i in range(2, 6):
            G.add_edge(i, i - 1, road_distance=100)
        return G

    def test_generates_trajectories(self, simple_graph):
        """Trajectories are generated successfully."""
        trajs = generate_trajectories(
            simple_graph,
            n_vehicles=10,
            n_trips_per_vehicle=2,
            model="random_walk",
            seed=42,
        )

        assert len(trajs) > 0
        assert all(isinstance(t, Trajectory) for t in trajs)

    def test_trajectory_lengths(self, simple_graph):
        """Trajectories respect min/max length constraints."""
        trajs = generate_trajectories(
            simple_graph,
            n_vehicles=10,
            n_trips_per_vehicle=5,
            min_path_length=2,
            max_path_length=4,
            seed=42,
        )

        for traj in trajs:
            assert 2 <= len(traj) <= 4

    def test_reproducibility(self, simple_graph):
        """Same seed produces same trajectories."""
        trajs1 = generate_trajectories(
            simple_graph, n_vehicles=5, n_trips_per_vehicle=2, seed=42
        )
        trajs2 = generate_trajectories(
            simple_graph, n_vehicles=5, n_trips_per_vehicle=2, seed=42
        )

        for t1, t2 in zip(trajs1, trajs2):
            assert t1.camera_sequence == t2.camera_sequence

    def test_different_models(self, simple_graph):
        """All mobility models produce valid trajectories."""
        for model in ["random_walk", "gravity", "exploration_return"]:
            trajs = generate_trajectories(
                simple_graph,
                n_vehicles=5,
                n_trips_per_vehicle=2,
                model=model,
                seed=42,
            )
            assert len(trajs) > 0

    def test_invalid_model(self, simple_graph):
        """Invalid model raises error."""
        with pytest.raises(ValueError):
            generate_trajectories(
                simple_graph,
                n_vehicles=1,
                n_trips_per_vehicle=1,
                model="invalid_model",
            )


class TestSplitTrajectories:
    """Tests for train/test splitting."""

    def test_split_ratio(self):
        """Split respects the train fraction."""
        trajs = [
            Trajectory(f"v{i}", [1, 2, 3], [0, 1, 2])
            for i in range(100)
        ]

        train, test = split_trajectories(trajs, train_fraction=0.8, seed=42)

        # Split is by vehicle, so approximately 80/20
        assert len(train) + len(test) == 100
        assert 70 <= len(train) <= 90

    def test_no_vehicle_overlap(self):
        """No vehicle appears in both train and test."""
        trajs = [
            Trajectory(f"v{i % 10}", [1, 2], [0, 1])
            for i in range(100)  # 10 vehicles, 10 trips each
        ]

        train, test = split_trajectories(trajs, train_fraction=0.7, seed=42)

        train_vehicles = set(t.vehicle_id for t in train)
        test_vehicles = set(t.vehicle_id for t in test)

        assert train_vehicles.isdisjoint(test_vehicles)


class TestTrajectoryStats:
    """Tests for trajectory statistics."""

    def test_empty_list(self):
        """Empty list returns zero stats."""
        stats = get_trajectory_stats([])

        assert stats["n_trajectories"] == 0
        assert stats["n_vehicles"] == 0

    def test_basic_stats(self):
        """Basic statistics are computed correctly."""
        trajs = [
            Trajectory("v1", [1, 2, 3], [0, 1, 2]),
            Trajectory("v1", [2, 3, 4, 5], [0, 1, 2, 3]),
            Trajectory("v2", [1, 3], [0, 1]),
        ]

        stats = get_trajectory_stats(trajs)

        assert stats["n_trajectories"] == 3
        assert stats["n_vehicles"] == 2
        assert stats["n_observations"] == 9  # 3 + 4 + 2
        assert stats["n_unique_cameras"] == 5  # 1, 2, 3, 4, 5
        assert stats["avg_length"] == 3.0
        assert stats["min_length"] == 2
        assert stats["max_length"] == 4
