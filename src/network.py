"""
Camera network graph construction and analysis.

Builds directed graphs representing camera-to-camera transitions
based on road network connectivity.
"""

from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree


def build_camera_network(
    gdf_cameras: gpd.GeoDataFrame,
    road_graph: nx.MultiDiGraph,
    snap_tolerance_m: float = 50.0,
    max_edge_distance_m: float = 10000.0,
) -> nx.DiGraph:
    """
    Build a directed graph of camera-to-camera transitions.

    Cameras are snapped to nearest road node, then connected based on
    road network connectivity.

    Args:
        gdf_cameras: Camera locations as GeoDataFrame
        road_graph: Road network graph from OSMnx
        snap_tolerance_m: Max distance to snap camera to road (meters)
        max_edge_distance_m: Max road distance to create edge (meters)

    Returns:
        DiGraph where nodes are cameras, edges are possible transitions
    """
    if gdf_cameras.empty:
        return nx.DiGraph()

    # Project to UTM for distance calculations
    utm_crs = gdf_cameras.estimate_utm_crs()
    gdf_proj = gdf_cameras.to_crs(utm_crs)

    # Get road node coordinates
    road_nodes = ox.graph_to_gdfs(road_graph, edges=False)
    road_nodes_proj = road_nodes.to_crs(utm_crs)

    # Build KD-tree for fast nearest neighbor search
    road_coords = np.array([(p.x, p.y) for p in road_nodes_proj.geometry])
    tree = cKDTree(road_coords)

    camera_coords = np.array([(p.x, p.y) for p in gdf_proj.geometry])
    distances, indices = tree.query(camera_coords)

    # Build camera graph
    G = nx.DiGraph()

    # Add camera nodes that snap to roads within tolerance
    camera_to_road = {}  # Maps camera osm_id to road node id

    for idx, row in gdf_cameras.iterrows():
        cam_idx = gdf_cameras.index.get_loc(idx)
        if distances[cam_idx] <= snap_tolerance_m:
            road_node_id = road_nodes.index[indices[cam_idx]]
            osm_id = row["osm_id"]

            G.add_node(
                osm_id,
                lat=row["lat"],
                lon=row["lon"],
                road_node=road_node_id,
                operator=row.get("operator", "unknown"),
                direction=row.get("camera_direction", None),
            )
            camera_to_road[osm_id] = road_node_id

    # Add edges based on road connectivity
    node_list = list(G.nodes())
    n_nodes = len(node_list)

    print(f"Computing shortest paths for {n_nodes} cameras...")

    for i, cam1 in enumerate(node_list):
        road_node1 = G.nodes[cam1]["road_node"]

        for cam2 in node_list:
            if cam1 == cam2:
                continue

            road_node2 = G.nodes[cam2]["road_node"]

            try:
                path_length = nx.shortest_path_length(
                    road_graph,
                    road_node1,
                    road_node2,
                    weight="length",
                )

                if path_length <= max_edge_distance_m:
                    G.add_edge(cam1, cam2, road_distance=path_length)
            except nx.NetworkXNoPath:
                pass

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_nodes} cameras")

    return G


def build_camera_network_fast(
    gdf_cameras: gpd.GeoDataFrame,
    road_graph: nx.MultiDiGraph,
    snap_tolerance_m: float = 50.0,
    k_nearest: int = 10,
) -> nx.DiGraph:
    """
    Build camera network using only k-nearest neighbors.

    Faster alternative for large networks that only connects
    cameras to their k nearest neighbors.

    Args:
        gdf_cameras: Camera locations
        road_graph: Road network graph
        snap_tolerance_m: Max distance to snap camera to road
        k_nearest: Number of nearest cameras to connect

    Returns:
        DiGraph of camera network
    """
    if gdf_cameras.empty:
        return nx.DiGraph()

    # Project to UTM for distance calculations
    utm_crs = gdf_cameras.estimate_utm_crs()
    gdf_proj = gdf_cameras.to_crs(utm_crs)

    # Get road nodes
    road_nodes = ox.graph_to_gdfs(road_graph, edges=False)
    road_nodes_proj = road_nodes.to_crs(utm_crs)

    # Snap cameras to roads
    road_coords = np.array([(p.x, p.y) for p in road_nodes_proj.geometry])
    road_tree = cKDTree(road_coords)

    camera_coords = np.array([(p.x, p.y) for p in gdf_proj.geometry])
    distances, indices = road_tree.query(camera_coords)

    # Build camera graph with nodes
    G = nx.DiGraph()
    valid_cameras = []

    for idx, row in gdf_cameras.iterrows():
        cam_idx = gdf_cameras.index.get_loc(idx)
        if distances[cam_idx] <= snap_tolerance_m:
            road_node_id = road_nodes.index[indices[cam_idx]]
            osm_id = row["osm_id"]

            G.add_node(
                osm_id,
                lat=row["lat"],
                lon=row["lon"],
                road_node=road_node_id,
                operator=row.get("operator", "unknown"),
                direction=row.get("camera_direction", None),
            )
            valid_cameras.append((osm_id, camera_coords[cam_idx]))

    # Build KD-tree of camera locations for k-nearest
    if len(valid_cameras) < 2:
        return G

    cam_ids = [c[0] for c in valid_cameras]
    cam_coords = np.array([c[1] for c in valid_cameras])
    cam_tree = cKDTree(cam_coords)

    # Connect each camera to k-nearest neighbors
    k = min(k_nearest + 1, len(valid_cameras))  # +1 because query includes self
    distances, indices = cam_tree.query(cam_coords, k=k)

    print(f"Connecting {len(valid_cameras)} cameras to {k - 1} nearest neighbors...")

    for i, cam1 in enumerate(cam_ids):
        road_node1 = G.nodes[cam1]["road_node"]

        for j in range(1, k):  # Skip self (j=0)
            cam2 = cam_ids[indices[i, j]]
            road_node2 = G.nodes[cam2]["road_node"]

            try:
                path_length = nx.shortest_path_length(
                    road_graph,
                    road_node1,
                    road_node2,
                    weight="length",
                )
                G.add_edge(cam1, cam2, road_distance=path_length)
            except nx.NetworkXNoPath:
                pass

    return G


def compute_topology_metrics(G: nx.DiGraph) -> dict:
    """
    Compute network topology metrics.

    Args:
        G: Camera network graph

    Returns:
        Dict with topology metrics
    """
    if G.number_of_nodes() == 0:
        return {
            "n_cameras": 0,
            "n_edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "avg_clustering": 0.0,
            "max_betweenness": 0.0,
            "choke_points": [],
        }

    metrics = {
        "n_cameras": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for n, d in G.degree()]),
    }

    # Clustering coefficient (convert to undirected)
    G_undirected = G.to_undirected()
    if G_undirected.number_of_edges() > 0:
        metrics["avg_clustering"] = nx.average_clustering(G_undirected)
    else:
        metrics["avg_clustering"] = 0.0

    # Betweenness centrality (identifies choke points)
    if G.number_of_nodes() > 1:
        betweenness = nx.betweenness_centrality(G, weight="road_distance")
        metrics["betweenness"] = betweenness
        metrics["max_betweenness"] = max(betweenness.values()) if betweenness else 0.0

        # Identify choke points (top 10% by betweenness)
        if betweenness:
            threshold = np.percentile(list(betweenness.values()), 90)
            metrics["choke_points"] = [
                n for n, b in betweenness.items() if b >= threshold
            ]
        else:
            metrics["choke_points"] = []
    else:
        metrics["betweenness"] = {}
        metrics["max_betweenness"] = 0.0
        metrics["choke_points"] = []

    # Connected components
    if G.number_of_nodes() > 0:
        # Weakly connected (ignoring edge direction)
        n_weakly_connected = nx.number_weakly_connected_components(G)
        metrics["n_weakly_connected_components"] = n_weakly_connected

        # Largest component size
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        metrics["largest_component_size"] = len(largest_wcc)
        metrics["largest_component_fraction"] = len(largest_wcc) / G.number_of_nodes()

    return metrics


def get_edge_distance_stats(G: nx.DiGraph) -> dict:
    """
    Compute statistics about edge distances in the network.

    Args:
        G: Camera network graph

    Returns:
        Dict with distance statistics
    """
    if G.number_of_edges() == 0:
        return {
            "min_distance_m": 0,
            "max_distance_m": 0,
            "mean_distance_m": 0,
            "median_distance_m": 0,
        }

    distances = [
        data.get("road_distance", 0) for _, _, data in G.edges(data=True)
    ]

    return {
        "min_distance_m": np.min(distances),
        "max_distance_m": np.max(distances),
        "mean_distance_m": np.mean(distances),
        "median_distance_m": np.median(distances),
        "std_distance_m": np.std(distances),
    }


def subsample_network(
    G: nx.DiGraph,
    fraction: float,
    seed: int = 42,
) -> nx.DiGraph:
    """
    Randomly subsample cameras from the network.

    Useful for studying effect of camera density on predictability.

    Args:
        G: Original camera network
        fraction: Fraction of cameras to keep (0-1)
        seed: Random seed

    Returns:
        Subsampled camera network
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n_keep = max(1, int(len(nodes) * fraction))

    keep_nodes = set(rng.choice(nodes, size=n_keep, replace=False))

    return G.subgraph(keep_nodes).copy()


def remove_high_betweenness_cameras(
    G: nx.DiGraph,
    percentile: float = 90,
) -> nx.DiGraph:
    """
    Remove cameras with high betweenness centrality.

    Used to study the importance of choke points.

    Args:
        G: Camera network
        percentile: Remove cameras above this percentile of betweenness

    Returns:
        Network with high-betweenness cameras removed
    """
    if G.number_of_nodes() <= 1:
        return G.copy()

    betweenness = nx.betweenness_centrality(G, weight="road_distance")
    threshold = np.percentile(list(betweenness.values()), percentile)

    keep_nodes = [n for n, b in betweenness.items() if b < threshold]

    return G.subgraph(keep_nodes).copy()


def get_camera_importance_ranking(G: nx.DiGraph) -> list[tuple[int, float]]:
    """
    Rank cameras by importance (betweenness centrality).

    Args:
        G: Camera network

    Returns:
        List of (camera_id, betweenness_score) sorted by importance
    """
    if G.number_of_nodes() == 0:
        return []

    betweenness = nx.betweenness_centrality(G, weight="road_distance")
    ranked = sorted(betweenness.items(), key=lambda x: -x[1])

    return ranked
