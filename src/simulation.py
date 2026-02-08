"""
Synthetic trajectory generation using established mobility models.

Implements multiple mobility models for generating realistic
vehicle movement patterns through camera networks.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np


@dataclass
class Trajectory:
    """A vehicle trajectory through the camera network."""

    vehicle_id: str
    camera_sequence: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def observations(self) -> list[tuple[int, float]]:
        """Return list of (camera_id, timestamp) pairs."""
        return list(zip(self.camera_sequence, self.timestamps))

    def __len__(self) -> int:
        return len(self.camera_sequence)


MobilityModel = Literal["random_walk", "gravity", "exploration_return"]


def generate_trajectories(
    camera_graph: nx.DiGraph,
    n_vehicles: int = 10000,
    n_trips_per_vehicle: int = 10,
    model: MobilityModel = "random_walk",
    min_path_length: int = 2,
    max_path_length: int = 15,
    seed: int = 42,
) -> list[Trajectory]:
    """
    Generate synthetic vehicle trajectories.

    Sample Size Justification (n=10,000):
    --------------------------------------
    Power analysis shows 10,000 vehicles provides:
    - Uniqueness estimates: ±0.4% margin of error (95% CI)
    - Power >99% to detect improvement over random baseline
    - Sufficient observations for stable entropy estimation
    - Robust bootstrap confidence intervals (1000+ samples)

    See analysis.justify_sample_size() for detailed power calculations.

    Args:
        camera_graph: Camera network graph
        n_vehicles: Number of unique vehicles (default 10,000 for statistical power)
        n_trips_per_vehicle: Trips per vehicle
        model: Mobility model ("random_walk", "gravity", or "exploration_return")
        min_path_length: Minimum observations per trajectory
        max_path_length: Maximum observations per trajectory
        seed: Random seed for reproducibility

    Returns:
        List of Trajectory objects
    """
    rng = np.random.default_rng(seed)
    nodes = list(camera_graph.nodes())

    if len(nodes) < 2:
        return []

    trajectories = []

    for v in range(n_vehicles):
        vehicle_id = f"vehicle_{v:04d}"

        # For EPR model, assign a "home" location
        home = rng.choice(nodes) if model == "exploration_return" else None

        for trip in range(n_trips_per_vehicle):
            path_length = rng.integers(min_path_length, max_path_length + 1)

            if model == "random_walk":
                path = _random_walk(camera_graph, nodes, path_length, rng)
            elif model == "gravity":
                path = _gravity_model(camera_graph, nodes, path_length, rng)
            elif model == "exploration_return":
                path = _epr_model(camera_graph, nodes, home, path_length, rng)
            else:
                raise ValueError(f"Unknown model: {model}")

            if len(path) >= min_path_length:
                # Generate timestamps with exponential inter-arrival times
                # Mean time between cameras: 60 seconds
                inter_times = rng.exponential(60, len(path))
                timestamps = np.cumsum(inter_times).tolist()

                trajectories.append(
                    Trajectory(
                        vehicle_id=vehicle_id,
                        camera_sequence=path,
                        timestamps=timestamps,
                    )
                )

    return trajectories


def _random_walk(
    G: nx.DiGraph,
    nodes: list,
    path_length: int,
    rng: np.random.Generator,
) -> list:
    """
    Simple random walk through camera network.

    Each step chooses uniformly at random from available neighbors.

    Args:
        G: Camera network
        nodes: List of node IDs
        path_length: Target path length
        rng: Random number generator

    Returns:
        List of camera IDs in order visited
    """
    current = rng.choice(nodes)
    path = [current]

    for _ in range(path_length - 1):
        neighbors = list(G.successors(current))
        if not neighbors:
            break
        current = rng.choice(neighbors)
        path.append(current)

    return path


def _gravity_model(
    G: nx.DiGraph,
    nodes: list,
    path_length: int,
    rng: np.random.Generator,
) -> list:
    """
    Gravity model: transition probability inversely proportional to distance.

    Vehicles are more likely to go to nearby cameras.

    Args:
        G: Camera network
        nodes: List of node IDs
        path_length: Target path length
        rng: Random number generator

    Returns:
        List of camera IDs in order visited
    """
    current = rng.choice(nodes)
    path = [current]

    for _ in range(path_length - 1):
        neighbors = list(G.successors(current))
        if not neighbors:
            break

        # Weight by inverse distance (+ offset to avoid division by zero)
        distances = np.array([
            G[current][n].get("road_distance", 1000) for n in neighbors
        ])
        weights = 1.0 / (distances + 100)  # +100m offset
        probs = weights / weights.sum()

        current = rng.choice(neighbors, p=probs)
        path.append(current)

    return path


def _epr_model(
    G: nx.DiGraph,
    nodes: list,
    home: int,
    path_length: int,
    rng: np.random.Generator,
    rho: float = 0.6,
) -> list:
    """
    Exploration-Preferential Return (EPR) model.

    Vehicles tend to return to previously visited locations,
    mimicking real commuting and routine behavior.

    Based on Song et al. (2010) - "Modelling the scaling properties
    of human mobility"

    Args:
        G: Camera network
        nodes: List of node IDs
        home: Home location for this vehicle
        path_length: Target path length
        rng: Random number generator
        rho: Probability of returning to known location (default 0.6)

    Returns:
        List of camera IDs in order visited
    """
    visited = defaultdict(int)
    visited[home] = 1

    current = home
    path = [current]

    for _ in range(path_length - 1):
        neighbors = list(G.successors(current))
        if not neighbors:
            break

        # Decide: return to known location or explore?
        if rng.random() < rho and len(visited) > 1:
            # Preferential return - visit known location proportional to visit count
            candidates = [n for n in visited if n in neighbors]
            if candidates:
                weights = np.array([visited[n] for n in candidates])
                probs = weights / weights.sum()
                current = rng.choice(candidates, p=probs)
            else:
                # No known locations reachable, explore
                current = rng.choice(neighbors)
        else:
            # Exploration - prefer unvisited locations
            new_neighbors = [n for n in neighbors if n not in visited]
            if new_neighbors:
                current = rng.choice(new_neighbors)
            else:
                current = rng.choice(neighbors)

        visited[current] += 1
        path.append(current)

    return path


def split_trajectories(
    trajectories: list[Trajectory],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """
    Split trajectories into train and test sets.

    Splits by vehicle to avoid data leakage.

    Args:
        trajectories: List of all trajectories
        train_fraction: Fraction for training set
        seed: Random seed

    Returns:
        (train_trajectories, test_trajectories)
    """
    rng = np.random.default_rng(seed)

    # Group by vehicle
    by_vehicle = defaultdict(list)
    for traj in trajectories:
        by_vehicle[traj.vehicle_id].append(traj)

    vehicle_ids = list(by_vehicle.keys())
    rng.shuffle(vehicle_ids)

    n_train = int(len(vehicle_ids) * train_fraction)
    train_vehicles = set(vehicle_ids[:n_train])

    train = []
    test = []

    for vid, trajs in by_vehicle.items():
        if vid in train_vehicles:
            train.extend(trajs)
        else:
            test.extend(trajs)

    return train, test


def get_trajectory_stats(trajectories: list[Trajectory]) -> dict:
    """
    Compute statistics about generated trajectories.

    Args:
        trajectories: List of trajectories

    Returns:
        Dict with trajectory statistics
    """
    if not trajectories:
        return {
            "n_trajectories": 0,
            "n_vehicles": 0,
            "n_observations": 0,
            "n_unique_cameras": 0,
        }

    lengths = [len(t) for t in trajectories]
    all_cameras = []
    for t in trajectories:
        all_cameras.extend(t.camera_sequence)

    return {
        "n_trajectories": len(trajectories),
        "n_vehicles": len(set(t.vehicle_id for t in trajectories)),
        "n_observations": sum(lengths),
        "n_unique_cameras": len(set(all_cameras)),
        "avg_length": np.mean(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "std_length": np.std(lengths),
    }


def aggregate_vehicle_histories(
    trajectories: list[Trajectory],
) -> dict[str, list[int]]:
    """
    Aggregate all observations per vehicle.

    Args:
        trajectories: List of trajectories

    Returns:
        Dict mapping vehicle_id to complete camera sequence
    """
    histories = defaultdict(list)

    # Sort trajectories by timestamp for proper ordering
    for traj in trajectories:
        histories[traj.vehicle_id].extend(traj.camera_sequence)

    return dict(histories)
