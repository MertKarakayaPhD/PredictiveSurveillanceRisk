#!/usr/bin/env python3
"""
Road-Network-Based Trajectory Simulation

This script implements the methodologically correct approach to ALPR trajectory simulation:
- Vehicles follow ROAD NETWORK paths, not camera-to-camera jumps
- Endpoints (origins/destinations) are sampled from road nodes based on attractiveness weights
- Cameras are PASSIVE OBSERVERS that record vehicles passing within detection range
- Uniqueness is computed on random observation points, not home-anchored prefixes

This addresses Reviewer 2's Issue #5: "Trajectories are generated on the camera graph,
not on the road network with camera hits... This is not a minor modeling choice;
it changes the meaning of the entire predictability experiment."

Key Design Principles:
1. DECOUPLE endpoints from camera placement - endpoints sampled from road nodes, NOT cameras
2. DO NOT let "home" be "nearest camera" - homes are sampled from residential attractiveness
3. DO NOT use prefix uniqueness - use random observation points
4. Hybrid OD model: EPR-style activity set + gravity-based destination choice

Author: ALPR Research Team
Date: January 2026
"""

import argparse
import json
import pickle
import hashlib
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None


# =============================================================================
# MULTIPROCESSING GLOBALS (initialized per worker)
# =============================================================================

_worker_data = {}

CHECKPOINT_VERSION = 1


def stable_u64(text: str) -> int:
    """
    Stable 64-bit hash for deterministic per-vehicle seeds across processes/runs.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _cache_get(cache: OrderedDict | None, key):
    if cache is None:
        return None
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _cache_put(cache: OrderedDict | None, key, value, max_size: int):
    if cache is None or max_size <= 0:
        return
    if key in cache:
        cache.pop(key, None)
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)


def _checkpoint_manifest_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "manifest.json"


def _checkpoint_shards_dir(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "shards"


def _checkpoint_signature(
    *,
    region: str,
    n_vehicles: int,
    n_trips_per_vehicle: int,
    seed: int,
    k_shortest: int,
    p_return: float,
    detection_radius_m: float,
    min_trip_distance_m: float,
    max_trip_distance_m: float,
    camera_query_backend: str,
    destination_candidate_pool_size: int,
    store_trip_metadata: bool,
) -> dict:
    return {
        "region": str(region),
        "n_vehicles": int(n_vehicles),
        "n_trips_per_vehicle": int(n_trips_per_vehicle),
        "seed": int(seed),
        "k_shortest": int(k_shortest),
        "p_return": float(p_return),
        "detection_radius_m": float(detection_radius_m),
        "min_trip_distance_m": float(min_trip_distance_m),
        "max_trip_distance_m": float(max_trip_distance_m),
        "camera_query_backend": str(camera_query_backend),
        "destination_candidate_pool_size": int(destination_candidate_pool_size),
        "store_trip_metadata": bool(store_trip_metadata),
    }


def _load_checkpoint_state(
    checkpoint_dir: Path,
    signature: dict,
    n_vehicles: int,
    store_trip_metadata: bool,
    verbose: bool,
) -> tuple[dict[int, dict], dict[int, list], set[int], dict, int]:
    manifest_path = _checkpoint_manifest_path(checkpoint_dir)
    shards_dir = _checkpoint_shards_dir(checkpoint_dir)

    if not manifest_path.exists():
        return {}, {}, set(), {}, 0

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to read checkpoint manifest ({manifest_path}): {exc}")
        return {}, {}, set(), {}, 0

    if int(manifest.get("version", -1)) != CHECKPOINT_VERSION:
        if verbose:
            print("[WARN] Checkpoint version mismatch; ignoring previous checkpoint.")
        return {}, {}, set(), {}, 0

    if manifest.get("signature") != signature:
        if verbose:
            print("[WARN] Checkpoint signature mismatch; ignoring previous checkpoint.")
        return {}, {}, set(), {}, 0

    completed: dict[int, dict] = {}
    trip_meta: dict[int, list] = {}
    completed_idx: set[int] = set()

    shard_files = manifest.get("shards", [])
    for shard_name in shard_files:
        shard_path = shards_dir / shard_name
        if not shard_path.exists():
            continue
        with shard_path.open("rb") as f:
            payload = pickle.load(f)
        records = payload.get("records", [])
        for vehicle_idx, trajectory, trip_meta_list in records:
            if vehicle_idx < 0 or vehicle_idx >= n_vehicles:
                continue
            completed[vehicle_idx] = trajectory
            if store_trip_metadata:
                trip_meta[vehicle_idx] = trip_meta_list or []
            completed_idx.add(vehicle_idx)

    stats = manifest.get("stats", {}) if isinstance(manifest.get("stats"), dict) else {}
    shard_counter = int(manifest.get("shard_counter", len(shard_files)))
    return completed, trip_meta, completed_idx, stats, shard_counter


def _flush_checkpoint_records(
    *,
    checkpoint_dir: Path,
    signature: dict,
    pending_records: list,
    completed_indices: set[int],
    stats: dict,
    shard_counter: int,
    verbose: bool,
) -> int:
    if not pending_records:
        return shard_counter

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = _checkpoint_shards_dir(checkpoint_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _checkpoint_manifest_path(checkpoint_dir)

    shard_counter += 1
    shard_name = f"part_{shard_counter:05d}.pkl"
    shard_path = shards_dir / shard_name

    with shard_path.open("wb") as f:
        pickle.dump({"records": pending_records}, f, protocol=pickle.HIGHEST_PROTOCOL)

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    existing_shards = manifest.get("shards", [])
    if not isinstance(existing_shards, list):
        existing_shards = []
    existing_shards.append(shard_name)

    manifest = {
        "version": CHECKPOINT_VERSION,
        "signature": signature,
        "updated_at": int(time.time()),
        "completed_vehicle_indices": sorted(int(x) for x in completed_indices),
        "n_completed": int(len(completed_indices)),
        "stats": {
            "total_trips": int(stats.get("total_trips", 0)),
            "trips_with_hits": int(stats.get("trips_with_hits", 0)),
            "trips_with_2plus_hits": int(stats.get("trips_with_2plus_hits", 0)),
            "all_hit_counts": [int(x) for x in stats.get("all_hit_counts", [])],
        },
        "shards": existing_shards,
        "shard_counter": int(shard_counter),
    }

    tmp_manifest = manifest_path.with_suffix(".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_manifest.replace(manifest_path)

    if verbose:
        print(
            f"[INFO] Checkpoint flushed: {len(completed_indices)} vehicles complete "
            f"({shard_name})"
        )
    return shard_counter


def resolve_camera_query_backend(requested_backend: str, n_workers: int) -> str:
    """
    Resolve requested camera query backend.

    Options:
    - scipy-kdtree
    - torch-cuda
    - torch-cuda-service
    - auto
    """
    mode = (requested_backend or "scipy-kdtree").strip().lower()
    if mode not in {"scipy-kdtree", "torch-cuda", "torch-cuda-service", "auto"}:
        raise ValueError(f"Unsupported camera query backend: {requested_backend}")

    if mode == "auto":
        if torch is not None and torch.cuda.is_available():
            if n_workers == 1:
                return "torch-cuda"
            return "torch-cuda-service"
        return "scipy-kdtree"

    if mode == "torch-cuda":
        if torch is None:
            print("[WARN] torch-cuda requested but torch is not installed. Falling back to scipy-kdtree.")
            return "scipy-kdtree"
        if not torch.cuda.is_available():
            print("[WARN] torch-cuda requested but CUDA is unavailable. Falling back to scipy-kdtree.")
            return "scipy-kdtree"
        if n_workers > 1:
            print(
                "[WARN] torch-cuda requested with n_workers > 1. "
                "Switching to torch-cuda-service backend."
            )
            return "torch-cuda-service"
        return "torch-cuda"

    if mode == "torch-cuda-service":
        if torch is None:
            print("[WARN] torch-cuda-service requested but torch is not installed. Falling back to scipy-kdtree.")
            return "scipy-kdtree"
        if not torch.cuda.is_available():
            print("[WARN] torch-cuda-service requested but CUDA is unavailable. Falling back to scipy-kdtree.")
            return "scipy-kdtree"
        if n_workers <= 1:
            return "torch-cuda"
        return "torch-cuda-service"

    return "scipy-kdtree"


def build_camera_query_payload(
    camera_positions: dict[int, tuple[float, float]],
    camera_ids: list[int],
    camera_query_backend: str,
    cuda_batch_size: int = 256,
    cuda_min_work: int = 2_000_000,
) -> dict | None:
    """
    Build backend-specific payload for camera proximity queries.
    """
    if camera_query_backend != "torch-cuda":
        return None
    if torch is None or not torch.cuda.is_available():
        return None

    xs_m: list[float] = []
    ys_m: list[float] = []
    for cid in camera_ids:
        lat = float(camera_positions[cid][0])
        lon = float(camera_positions[cid][1])
        xs_m.append(lon * 111000.0 * np.cos(np.radians(lat)))
        ys_m.append(lat * 111000.0)
    if not xs_m:
        return None

    device = torch.device("cuda")
    return {
        "camera_xs_m_t": torch.tensor(xs_m, dtype=torch.float32, device=device),
        "camera_ys_m_t": torch.tensor(ys_m, dtype=torch.float32, device=device),
        "camera_ids": list(camera_ids),
        "cuda_batch_size": max(1, int(cuda_batch_size)),
        "cuda_min_work": max(1, int(cuda_min_work)),
    }


def _query_nearby_cameras_kdtree(
    lat: float,
    lon: float,
    camera_tree: cKDTree,
    camera_ids: list[int],
    detection_radius_m: float,
) -> list[int]:
    lat_center = lat
    x_m = lon * 111000 * np.cos(np.radians(lat_center))
    y_m = lat * 111000
    nearby_indices = camera_tree.query_ball_point([x_m, y_m], detection_radius_m)
    return [camera_ids[idx] for idx in nearby_indices]


def _collect_camera_hits_torch_cuda(
    route: list[int],
    node_coords: dict[int, tuple[float, float]],
    payload: dict,
    detection_radius_m: float,
) -> list[int]:
    """
    Batched CUDA camera-hit detection for all unique nodes in route.
    """
    camera_xs_m_t = payload["camera_xs_m_t"]
    camera_ys_m_t = payload["camera_ys_m_t"]
    camera_ids = payload["camera_ids"]
    batch_size = int(payload.get("cuda_batch_size", 256))

    route_nodes: list[int] = []
    seen_nodes = set()
    for node in route:
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        if node in node_coords:
            route_nodes.append(node)

    if not route_nodes:
        return []

    node_lats = np.array([node_coords[n][0] for n in route_nodes], dtype=np.float32)
    node_lons = np.array([node_coords[n][1] for n in route_nodes], dtype=np.float32)
    node_lats_t = torch.as_tensor(node_lats, device=camera_xs_m_t.device)
    node_lons_t = torch.as_tensor(node_lons, device=camera_xs_m_t.device)

    r2 = float(detection_radius_m) * float(detection_radius_m)
    seen_cameras = set()
    camera_hits: list[int] = []

    for start in range(0, node_lats_t.shape[0], batch_size):
        end = min(start + batch_size, node_lats_t.shape[0])
        lats = node_lats_t[start:end]  # [B]
        lons = node_lons_t[start:end]  # [B]

        x_nodes_m = lons * 111000.0 * torch.cos(torch.deg2rad(lats))
        y_nodes_m = lats * 111000.0
        dx = x_nodes_m[:, None] - camera_xs_m_t[None, :]
        dy = y_nodes_m[:, None] - camera_ys_m_t[None, :]
        dist2 = dx * dx + dy * dy
        hit_pairs = torch.nonzero(dist2 <= r2, as_tuple=False)
        if hit_pairs.numel() == 0:
            continue

        for _, col in hit_pairs.detach().cpu().tolist():
            cam_id = camera_ids[int(col)]
            if cam_id not in seen_cameras:
                seen_cameras.add(cam_id)
                camera_hits.append(cam_id)

    return camera_hits


def _gpu_camera_query_server(
    request_queue,
    response_queues: list,
    node_coords: dict[int, tuple[float, float]],
    camera_positions: dict[int, tuple[float, float]],
    camera_ids: list[int],
    cuda_batch_size: int,
    route_cache_size: int,
):
    """
    Dedicated GPU service process for multi-worker camera proximity queries.
    """
    payload = build_camera_query_payload(
        camera_positions=camera_positions,
        camera_ids=camera_ids,
        camera_query_backend="torch-cuda",
        cuda_batch_size=cuda_batch_size,
        cuda_min_work=1,
    )
    if payload is None:
        while True:
            msg = request_queue.get()
            if msg is None:
                break
            slot, req_id, _route_nodes, _radius_m = msg
            response_queues[slot].put((req_id, tuple(), "cuda_payload_unavailable"))
        return

    route_cache = OrderedDict()
    while True:
        msg = request_queue.get()
        if msg is None:
            break

        slot, req_id, route_nodes, detection_radius_m = msg
        try:
            key = (tuple(route_nodes), float(detection_radius_m))
            cached = _cache_get(route_cache, key)
            if cached is not None:
                response_queues[slot].put((req_id, cached, None))
                continue

            hits = _collect_camera_hits_torch_cuda(
                route=list(route_nodes),
                node_coords=node_coords,
                payload=payload,
                detection_radius_m=float(detection_radius_m),
            )
            hits_tuple = tuple(hits)
            _cache_put(route_cache, key, hits_tuple, route_cache_size)
            response_queues[slot].put((req_id, hits_tuple, None))
        except Exception as exc:
            response_queues[slot].put((req_id, tuple(), str(exc)))


def _collect_camera_hits_torch_cuda_service(
    route: list[int],
    service: dict,
    detection_radius_m: float,
) -> list[int]:
    """
    Query camera hits via dedicated GPU service process.
    """
    req_id = int(service["next_req_id"])
    service["next_req_id"] = req_id + 1

    service["request_queue"].put(
        (service["worker_slot"], req_id, tuple(route), float(detection_radius_m))
    )
    while True:
        resp_id, hits_tuple, error = service["response_queue"].get()
        if int(resp_id) != req_id:
            continue
        if error:
            raise RuntimeError(f"GPU camera query service error: {error}")
        return list(hits_tuple)


def _init_worker(G_simple_data, G_road_data, node_coords, camera_positions,
                 camera_tree_data, camera_ids, A_work, A_home, A_other,
                 edge_traffic_weights, k_shortest, p_return, detection_radius_m,
                 lambda_traffic, min_trip_distance_m, max_trip_distance_m,
                 n_trips_per_vehicle, base_seed,
                 camera_query_backend, camera_query_payload,
                 camera_query_service_init,
                 destination_candidate_pool_size, path_cache_size, route_cache_size,
                 store_trip_metadata):
    """Initialize worker process with shared data."""
    global _worker_data
    camera_query_service = None
    if camera_query_service_init is not None:
        with camera_query_service_init["worker_counter_lock"]:
            slot = int(camera_query_service_init["worker_counter"].value)
            camera_query_service_init["worker_counter"].value = slot + 1
        response_queues = camera_query_service_init["response_queues"]
        if response_queues:
            slot = slot % len(response_queues)
            camera_query_service = {
                "worker_slot": int(slot),
                "request_queue": camera_query_service_init["request_queue"],
                "response_queue": response_queues[slot],
                "next_req_id": 0,
            }

    _worker_data = {
        'G_simple': G_simple_data,
        'G_road': G_road_data,
        'node_coords': node_coords,
        'camera_positions': camera_positions,
        'camera_tree': camera_tree_data,
        'camera_ids': camera_ids,
        'A_work': A_work,
        'A_home': A_home,
        'A_other': A_other,
        'edge_traffic_weights': edge_traffic_weights,
        'k_shortest': k_shortest,
        'p_return': p_return,
        'detection_radius_m': detection_radius_m,
        'lambda_traffic': lambda_traffic,
        'min_trip_distance_m': min_trip_distance_m,
        'max_trip_distance_m': max_trip_distance_m,
        'n_trips_per_vehicle': n_trips_per_vehicle,
        'base_seed': base_seed,
        'camera_query_backend': camera_query_backend,
        'camera_query_payload': camera_query_payload,
        'camera_query_service': camera_query_service,
        'destination_candidate_pool_size': destination_candidate_pool_size,
        'path_cache_size': int(path_cache_size),
        'route_cache_size': int(route_cache_size),
        'store_trip_metadata': bool(store_trip_metadata),
        'ksp_cache': OrderedDict(),
        'route_obs_cache': OrderedDict(),
    }


def _simulate_single_vehicle(args):
    """
    Worker function to simulate a single vehicle.

    Args:
        args: Tuple of (vehicle_id, home_node, vehicle_index)

    Returns:
        Tuple of (vehicle_index, trajectory_dict, list of trip_metadata dicts)
    """
    vid, home_node, vehicle_idx = args

    # Get worker data
    G_simple = _worker_data['G_simple']
    G_road = _worker_data['G_road']
    node_coords = _worker_data['node_coords']
    camera_positions = _worker_data['camera_positions']
    camera_tree = _worker_data['camera_tree']
    camera_ids = _worker_data['camera_ids']
    A_work = _worker_data['A_work']
    A_home = _worker_data['A_home']
    A_other = _worker_data['A_other']
    edge_traffic_weights = _worker_data['edge_traffic_weights']
    k_shortest = _worker_data['k_shortest']
    p_return = _worker_data['p_return']
    detection_radius_m = _worker_data['detection_radius_m']
    lambda_traffic = _worker_data['lambda_traffic']
    min_trip_distance_m = _worker_data['min_trip_distance_m']
    max_trip_distance_m = _worker_data['max_trip_distance_m']
    n_trips_per_vehicle = _worker_data['n_trips_per_vehicle']
    base_seed = _worker_data['base_seed']
    camera_query_backend = _worker_data['camera_query_backend']
    camera_query_payload = _worker_data['camera_query_payload']
    camera_query_service = _worker_data['camera_query_service']
    destination_candidate_pool_size = _worker_data['destination_candidate_pool_size']
    path_cache_size = _worker_data['path_cache_size']
    route_cache_size = _worker_data['route_cache_size']
    store_trip_metadata = _worker_data['store_trip_metadata']
    ksp_cache = _worker_data['ksp_cache']
    route_obs_cache = _worker_data['route_obs_cache']

    # Create deterministic RNG for this vehicle
    rng = np.random.default_rng(base_seed + vehicle_idx)

    A_by_type = {
        'work': A_work,
        'shop': A_other,
        'social': A_other,
        'other': A_other,
        'home': A_home,
    }

    activity_set = set()
    vehicle_camera_hits = []
    trip_metadata_list = []

    current_node = home_node

    for trip_num in range(n_trips_per_vehicle):
        # Generate tour structure
        tour = generate_tour_structure(
            n_stops=rng.integers(1, 5),
            seed=(base_seed + stable_u64(f"{vid}:{trip_num}")) % (2**31)
        )

        for activity in tour:
            # Choose destination
            A_dest = A_by_type.get(activity, A_other)

            destination = choose_destination(
                current_node=current_node,
                activity_type=activity,
                home_node=home_node,
                activity_set=activity_set,
                A_dest=A_dest,
                G_road=G_road,
                node_coords=node_coords,
                p_return=p_return,
                min_distance_m=min_trip_distance_m,
                max_distance_m=max_trip_distance_m,
                candidate_pool_size=destination_candidate_pool_size,
                rng=rng,
            )

            # Route and observe cameras
            route, camera_hits, route_length = route_and_observe(
                origin=current_node,
                destination=destination,
                G_simple=G_simple,
                G_road=G_road,
                camera_positions=camera_positions,
                camera_tree=camera_tree,
                camera_ids=camera_ids,
                node_coords=node_coords,
                k_shortest=k_shortest,
                detection_radius_m=detection_radius_m,
                edge_traffic_weights=edge_traffic_weights,
                lambda_traffic=lambda_traffic,
                camera_query_backend=camera_query_backend,
                camera_query_payload=camera_query_payload,
                camera_query_service=camera_query_service,
                ksp_cache=ksp_cache,
                path_cache_size=path_cache_size,
                route_obs_cache=route_obs_cache,
                route_cache_size=route_cache_size,
                rng=rng,
            )

            # Update activity set
            if activity != 'home':
                activity_set.add(destination)

            # Record trip
            if store_trip_metadata:
                trip_metadata_list.append({
                    'vehicle_id': vid,
                    'origin': current_node,
                    'destination': destination,
                    'activity': activity,
                    'route_length_m': route_length,
                    'n_camera_hits': len(camera_hits),
                })
            else:
                # Keep minimal per-trip hit counts for aggregate observation stats.
                trip_metadata_list.append({
                    'n_camera_hits': len(camera_hits),
                })

            # Extend vehicle's camera observations
            vehicle_camera_hits.extend(camera_hits)

            # Move to destination
            current_node = destination

    trajectory = {
        'vehicle_id': vid,
        'camera_hits': vehicle_camera_hits,
        'home_node': home_node,
    }

    return vehicle_idx, trajectory, trip_metadata_list


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RoadTrip:
    """A single trip from origin to destination on the road network."""
    vehicle_id: str
    origin_node: int  # Road network node ID
    destination_node: int  # Road network node ID
    route: list[int] = field(default_factory=list)  # Road edge sequence
    route_length_m: float = 0.0
    camera_hits: list[int] = field(default_factory=list)  # Camera IDs observed
    camera_hit_distances: list[float] = field(default_factory=list)  # Distance along route when hit


@dataclass
class VehicleHistory:
    """Complete travel history for a vehicle."""
    vehicle_id: str
    home_node: int  # Road network node (NOT a camera)
    activity_set: set[int] = field(default_factory=set)  # Known destinations (road nodes)
    trips: list[RoadTrip] = field(default_factory=list)

    def all_camera_hits(self) -> list[int]:
        """Get all camera observations in order."""
        hits = []
        for trip in self.trips:
            hits.extend(trip.camera_hits)
        return hits


# =============================================================================
# STEP 0: BUILD PLACE UNIVERSE (H3 HEXAGONS)
# =============================================================================

def build_place_universe(
    G_road: nx.MultiDiGraph,
    method: str = 'h3',
    h3_resolution: int = 9,
    sample_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """
    Create candidate endpoint set V* ⊆ V_road.

    CRITICAL: These are potential origins/destinations for trips.
    They are sampled from ROAD NODES, not cameras.

    Args:
        G_road: Road network graph from OSMnx
        method: 'h3' for hexagonal zones or 'sample' for random node sampling
        h3_resolution: H3 resolution (9 = ~0.1 km² hexagons)
        sample_fraction: Fraction of nodes to keep if method='sample'
        seed: Random seed

    Returns:
        Dict mapping place_id -> road_node_id
    """
    rng = np.random.default_rng(seed)
    nodes_gdf = ox.graph_to_gdfs(G_road, edges=False)

    if method == 'h3':
        # Group road nodes into H3 hexagons, use centroid-nearest node per hex
        hex_to_nodes = defaultdict(list)

        for node_id, row in nodes_gdf.iterrows():
            hex_id = h3.latlng_to_cell(row['y'], row['x'], h3_resolution)
            hex_to_nodes[hex_id].append((node_id, row['y'], row['x']))

        place_universe = {}
        for hex_id, nodes in hex_to_nodes.items():
            # Get hex centroid
            hex_center = h3.cell_to_latlng(hex_id)

            # Find nearest node to centroid
            min_dist = float('inf')
            best_node = nodes[0][0]
            for node_id, lat, lon in nodes:
                dist = ((lat - hex_center[0])**2 + (lon - hex_center[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_node = node_id

            place_universe[hex_id] = best_node

        print(f"Built place universe: {len(place_universe)} H3 hexagons (res={h3_resolution})")

    elif method == 'sample':
        # Randomly sample a fraction of road nodes
        all_nodes = list(nodes_gdf.index)
        n_sample = max(100, int(len(all_nodes) * sample_fraction))
        sampled = rng.choice(all_nodes, size=min(n_sample, len(all_nodes)), replace=False)

        place_universe = {f"node_{i}": node_id for i, node_id in enumerate(sampled)}
        print(f"Built place universe: {len(place_universe)} sampled road nodes")

    else:
        raise ValueError(f"Unknown method: {method}")

    return place_universe


# =============================================================================
# STEP 1: COMPUTE ATTRACTIVENESS WEIGHTS
# =============================================================================

def compute_attractiveness(
    G_road: nx.MultiDiGraph,
    place_universe: dict[str, int],
    use_osm_pois: bool = True,
    fallback_to_centrality: bool = True,
    traffic_weights: dict[int, float] | None = None,
    traffic_blend_factor: float = 0.7,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """
    Compute attractiveness weights for endpoint sampling.

    Priority order:
    1. Traffic weights (if provided) - blended with degree-based weights
    2. POI density from road graph tags (commercial, residential, etc.)
    3. Road centrality (betweenness) as proxy for activity
    4. Uniform weights as fallback

    Args:
        G_road: Road network graph
        place_universe: Dict mapping place_id -> road_node_id
        use_osm_pois: Try to use OSM POI data if available
        fallback_to_centrality: Use centrality if POI data unavailable
        traffic_weights: Dict mapping node_id -> traffic score (normalized 0-1)
        traffic_blend_factor: How much to weight traffic vs degree (0-1, default 0.7)

    Returns:
        Tuple of (A_home, A_work, A_other) weight dicts
    """
    road_nodes = set(place_universe.values())

    # Initialize with uniform weights
    A_home = {n: 1.0 for n in road_nodes}
    A_work = {n: 1.0 for n in road_nodes}
    A_other = {n: 1.0 for n in road_nodes}

    # Compute degree-based weights
    degree_work = {}
    degree_home = {}
    for node in road_nodes:
        if node in G_road:
            degree = G_road.degree(node)
            # Scale by degree - intersections are more active
            degree_work[node] = max(1.0, degree / 2)
            # Residential areas: prefer lower-degree nodes (quieter streets)
            degree_home[node] = max(0.5, 4.0 - degree / 2)
        else:
            degree_work[node] = 1.0
            degree_home[node] = 1.0

    # Blend with traffic weights if provided
    if traffic_weights is not None:
        print(f"Blending traffic weights (factor={traffic_blend_factor})")
        for node in road_nodes:
            traffic_score = traffic_weights.get(node, 0.5)  # Default to mid-range
            degree_w = degree_work.get(node, 1.0)
            degree_h = degree_home.get(node, 1.0)

            # Normalize degree weights locally
            max_degree_w = max(degree_work.values()) if degree_work else 1.0
            max_degree_h = max(degree_home.values()) if degree_home else 1.0
            norm_degree_w = degree_w / max_degree_w
            norm_degree_h = degree_h / max_degree_h

            # Blend: work/other prefer high-traffic nodes
            # blend(a, b) = (1 - factor) * a + factor * b
            A_work[node] = (1 - traffic_blend_factor) * norm_degree_w + traffic_blend_factor * traffic_score
            A_other[node] = (1 - traffic_blend_factor) * norm_degree_w + traffic_blend_factor * traffic_score

            # Home prefers low-traffic, low-degree nodes
            inv_traffic = 1.0 - traffic_score
            A_home[node] = (1 - traffic_blend_factor) * norm_degree_h + traffic_blend_factor * inv_traffic
    else:
        # Use degree-based weights directly
        for node in road_nodes:
            A_work[node] = degree_work.get(node, 1.0)
            A_other[node] = degree_work.get(node, 1.0)
            A_home[node] = degree_home.get(node, 1.0)

    # If we want centrality-based weights (only if no traffic weights)
    if fallback_to_centrality and traffic_weights is None:
        try:
            # Use a subgraph of just the place universe nodes for speed
            subgraph = G_road.subgraph(road_nodes)
            if subgraph.number_of_edges() > 0:
                # Betweenness centrality (slow for large graphs, sample)
                n_nodes = len(road_nodes)
                k_sample = min(100, n_nodes)
                betweenness = nx.betweenness_centrality(
                    subgraph, k=k_sample, weight='length'
                )

                # Higher betweenness = more traffic = better for work/commercial
                max_b = max(betweenness.values()) if betweenness else 1.0
                for node, b in betweenness.items():
                    A_work[node] *= (1 + b / max_b)
                    A_other[node] *= (1 + 0.5 * b / max_b)
                    # Lower betweenness for residential
                    A_home[node] *= (1.5 - b / max_b)
        except Exception as e:
            print(f"Warning: Centrality calculation failed: {e}")

    # Normalize weights to sum to 1
    def normalize(weights):
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()} if total > 0 else weights

    A_home = normalize(A_home)
    A_work = normalize(A_work)
    A_other = normalize(A_other)

    print(f"Computed attractiveness weights for {len(road_nodes)} places")
    if traffic_weights is not None:
        print(f"  Using traffic weights with blend factor {traffic_blend_factor}")

    return A_home, A_work, A_other


# =============================================================================
# STEP 2: ASSIGN VEHICLE HOMES
# =============================================================================

def assign_vehicle_homes(
    n_vehicles: int,
    A_home: dict[int, float],
    seed: int = 42,
) -> dict[str, int]:
    """
    Sample home location for each vehicle from residential attractiveness.

    CRITICAL: Homes are sampled from ROAD NODES based on attractiveness,
    NOT from camera locations. This decouples endpoints from camera placement.

    Args:
        n_vehicles: Number of vehicles to create
        A_home: Residential attractiveness weights
        seed: Random seed

    Returns:
        Dict mapping vehicle_id -> home_road_node
    """
    rng = np.random.default_rng(seed)

    nodes = list(A_home.keys())
    weights = np.array([A_home[n] for n in nodes])
    probs = weights / weights.sum()

    homes = {}
    for i in range(n_vehicles):
        vehicle_id = f"vehicle_{i:05d}"
        home_node = rng.choice(nodes, p=probs)
        homes[vehicle_id] = home_node

    print(f"Assigned homes for {n_vehicles} vehicles")

    return homes


# =============================================================================
# STEP 3: TRIP CHAINING (TOUR STRUCTURE)
# =============================================================================

def generate_tour_structure(
    n_stops: int = 3,
    seed: int = 42,
) -> list[str]:
    """
    Generate a tour structure for a day's travel.

    Structure: home -> activity_1 -> activity_2 -> ... -> home

    Activity types: 'work', 'shop', 'social', 'other'

    Args:
        n_stops: Number of intermediate stops
        seed: Random seed

    Returns:
        List of activity types (e.g., ['work', 'shop', 'home'])
    """
    rng = np.random.default_rng(seed)

    activity_types = ['work', 'shop', 'social', 'other']
    activity_weights = [0.5, 0.25, 0.15, 0.1]  # Work is most common

    activities = rng.choice(
        activity_types,
        size=n_stops,
        p=activity_weights
    ).tolist()

    # Tour always ends at home
    activities.append('home')

    return activities


# =============================================================================
# STEP 4: EPR + GRAVITY DESTINATION CHOICE
# =============================================================================

def choose_destination(
    current_node: int,
    activity_type: str,
    home_node: int,
    activity_set: set[int],
    A_dest: dict[int, float],
    G_road: nx.MultiDiGraph,
    node_coords: dict[int, tuple[float, float]],
    p_return: float = 0.6,
    distance_decay_beta: float = 0.001,  # Decay per meter
    min_distance_m: float = 8047.0,  # 5 miles in meters
    max_distance_m: float = 16093.0,  # 10 miles in meters
    candidate_pool_size: int = 0,
    rng: np.random.Generator = None,
) -> int:
    """
    EPR + Gravity destination choice on road network with distance constraints.

    With probability p_return:
        Choose uniformly from previously visited activity locations
    Else (explore):
        P(d=j | o=i) ∝ A(j) * exp(-β * dist(i,j))
        Subject to: min_distance_m <= dist(i,j) <= max_distance_m

    Args:
        current_node: Current road node
        activity_type: Type of activity ('work', 'shop', 'social', 'other', 'home')
        home_node: Vehicle's home node
        activity_set: Previously visited destinations
        A_dest: Attractiveness weights for destination type
        G_road: Road network graph
        node_coords: Dict mapping node -> (lat, lon)
        p_return: Probability of returning to known location
        distance_decay_beta: Distance decay parameter
        min_distance_m: Minimum trip distance (default 5 miles = 8047m)
        max_distance_m: Maximum trip distance (default 10 miles = 16093m)
        rng: Random number generator

    Returns:
        Destination road node ID
    """
    if rng is None:
        rng = np.random.default_rng()

    # If going home, return home node (no distance constraint for home trips)
    if activity_type == 'home':
        return home_node

    candidate_nodes = list(A_dest.keys())
    if not candidate_nodes:
        return current_node

    # EPR decision: return to known location or explore?
    if rng.random() < p_return and len(activity_set) > 0:
        # Uniform return to known locations
        known = list(activity_set)
        if known:
            # Uniform over known activity locations
            weights = np.ones(len(known))
            probs = weights / weights.sum()
            return rng.choice(known, p=probs)

    # Exploration: Gravity model with distance constraints
    # P(j) ∝ A(j) * exp(-β * d(i,j)) for d(i,j) in [min_dist, max_dist]

    if current_node not in node_coords:
        # Fallback to random weighted by attractiveness
        weights = np.array([A_dest[n] for n in candidate_nodes])
        probs = weights / weights.sum()
        return rng.choice(candidate_nodes, p=probs)

    current_lat, current_lon = node_coords[current_node]

    # Optional speedup: sample a weighted candidate pool before distance scoring.
    if candidate_pool_size and candidate_pool_size > 0 and candidate_pool_size < len(candidate_nodes):
        attraction = np.array([max(float(A_dest.get(n, 0.0)), 0.0) for n in candidate_nodes], dtype=float)
        if attraction.sum() <= 0:
            attraction = np.ones(len(candidate_nodes), dtype=float)
        probs = attraction / attraction.sum()
        sampled_idx = rng.choice(
            len(candidate_nodes),
            size=int(candidate_pool_size),
            replace=False,
            p=probs,
        )
        candidate_nodes = [candidate_nodes[int(i)] for i in sampled_idx]

    filtered_nodes = [n for n in candidate_nodes if n != current_node and n in node_coords]
    if not filtered_nodes:
        return current_node

    node_lats = np.array([node_coords[n][0] for n in filtered_nodes], dtype=float)
    node_lons = np.array([node_coords[n][1] for n in filtered_nodes], dtype=float)
    dlat_m = (node_lats - current_lat) * 111000.0
    dlon_m = (node_lons - current_lon) * 111000.0 * np.cos(np.radians(current_lat))
    all_distances = np.sqrt(dlat_m * dlat_m + dlon_m * dlon_m)
    attraction = np.array([max(float(A_dest.get(n, 1e-6)), 1e-6) for n in filtered_nodes], dtype=float)
    all_weights = attraction * np.exp(-distance_decay_beta * all_distances)

    # Apply distance constraints
    # Priority: prefer nodes in [min_dist, max_dist] range
    valid_mask = (all_distances >= min_distance_m) & (all_distances <= max_distance_m)

    if valid_mask.sum() == 0:
        # Fallback 1: accept nodes above min distance
        valid_mask = all_distances >= min_distance_m

    if valid_mask.sum() == 0:
        # Fallback 2: accept all nodes (region may be too small)
        valid_mask = np.ones(len(all_distances), dtype=bool)

    # Filter to valid candidates
    dist_valid_nodes = [n for n, v in zip(filtered_nodes, valid_mask) if v]
    filtered_weights = all_weights[valid_mask]

    if len(dist_valid_nodes) == 0:
        return current_node

    if filtered_weights.sum() <= 0:
        filtered_weights = np.ones(len(filtered_weights))

    probs = filtered_weights / filtered_weights.sum()
    return rng.choice(dist_valid_nodes, p=probs)


# =============================================================================
# STEP 5: K-SHORTEST PATHS ROUTING WITH CAMERA DETECTION
# =============================================================================

def convert_to_simple_digraph(G_road: nx.MultiDiGraph, weight: str = 'length') -> nx.DiGraph:
    """
    Convert MultiDiGraph to simple DiGraph by keeping shortest edge per node pair.

    Args:
        G_road: MultiDiGraph from OSMnx
        weight: Edge weight attribute to compare

    Returns:
        Simple DiGraph
    """
    G_simple = nx.DiGraph()

    # Add all nodes first
    for node, data in G_road.nodes(data=True):
        G_simple.add_node(node, **data)

    # Add edges, keeping shortest for each (u,v) pair
    for u, v, data in G_road.edges(data=True):
        if G_simple.has_edge(u, v):
            if data.get(weight, 1) < G_simple[u][v].get(weight, float('inf')):
                G_simple[u][v].update(data)
        else:
            G_simple.add_edge(u, v, **data)

    return G_simple


def find_k_shortest_paths(
    G_simple: nx.DiGraph,
    origin: int,
    destination: int,
    k: int = 3,
    weight: str = 'length',
) -> list[list[int]]:
    """
    Find k-shortest paths between origin and destination.

    Args:
        G_simple: Simple DiGraph (NOT MultiDiGraph)
        origin: Origin node ID
        destination: Destination node ID
        k: Number of paths to find
        weight: Edge weight attribute

    Returns:
        List of paths (each path is list of node IDs)
    """
    try:
        paths = []
        gen = nx.shortest_simple_paths(G_simple, origin, destination, weight=weight)
        for i, path in enumerate(gen):
            if i >= k:
                break
            paths.append(path)
        return paths if paths else []
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
        return []


def route_and_observe(
    origin: int,
    destination: int,
    G_simple: nx.DiGraph,  # Pre-converted simple DiGraph for path finding
    G_road: nx.MultiDiGraph,  # Original graph for edge data
    camera_positions: dict[int, tuple[float, float]],  # camera_id -> (lat, lon)
    camera_tree: cKDTree,  # Spatial index for cameras
    camera_ids: list[int],  # Ordered list of camera IDs matching tree
    node_coords: dict[int, tuple[float, float]],
    k_shortest: int = 3,
    detection_radius_m: float = 100.0,
    edge_traffic_weights: dict[tuple[int, int, int], float] | None = None,
    lambda_traffic: float = 0.5,
    camera_query_backend: str = "scipy-kdtree",
    camera_query_payload: dict | None = None,
    camera_query_service: dict | None = None,
    ksp_cache: OrderedDict | None = None,
    path_cache_size: int = 0,
    route_obs_cache: OrderedDict | None = None,
    route_cache_size: int = 0,
    rng: np.random.Generator = None,
) -> tuple[list[int], list[int], float]:
    """
    Route from origin to destination and record camera observations.

    1. Generate k-shortest paths
    2. Select one with Gumbel noise on travel time (+ traffic term if provided)
    3. Walk along route, record cameras within detection radius

    Args:
        origin: Origin road node
        destination: Destination road node
        G_road: Road network graph
        camera_positions: Dict of camera_id -> (lat, lon)
        camera_tree: KD-tree for camera spatial queries
        camera_ids: Camera IDs in same order as tree
        node_coords: Road node coordinates
        k_shortest: Number of shortest paths to consider
        detection_radius_m: Camera detection radius
        edge_traffic_weights: Dict mapping (u, v, key) -> normalized traffic weight
        lambda_traffic: Weight for traffic preference in utility (default 0.5)
        rng: Random number generator

    Returns:
        Tuple of (route_nodes, camera_hits, route_length_m)
    """
    if rng is None:
        rng = np.random.default_rng()

    if origin == destination:
        return [origin], [], 0.0

    cache_key = (int(origin), int(destination), int(k_shortest))
    path_candidates = _cache_get(ksp_cache, cache_key)
    if path_candidates is None:
        # Find k-shortest paths using pre-converted simple graph
        paths = find_k_shortest_paths(G_simple, origin, destination, k=k_shortest)
        if not paths:
            _cache_put(ksp_cache, cache_key, [], path_cache_size)
            return [], [], 0.0

        # Precompute per-path stats once and cache for repeated OD lookups.
        path_candidates = []
        for path in paths:
            length = 0.0
            traffic_sum = 0.0
            n_edges = 0

            for i in range(len(path) - 1):
                edge_data = G_road.get_edge_data(path[i], path[i+1])
                if edge_data:
                    first_edge = list(edge_data.values())[0]
                    length += first_edge.get('length', 100)
                    if edge_traffic_weights is not None:
                        edge_key = (path[i], path[i+1], 0)
                        traffic_weight = edge_traffic_weights.get(edge_key, 0.5)
                        traffic_sum += traffic_weight
                        n_edges += 1

            traffic_mean = (traffic_sum / n_edges) if n_edges > 0 else 0.5
            path_candidates.append({
                "path": path,
                "length": float(length),
                "traffic_mean": float(traffic_mean),
            })
        _cache_put(ksp_cache, cache_key, path_candidates, path_cache_size)

    if not path_candidates:
        # No path found
        return [], [], 0.0

    paths = [item["path"] for item in path_candidates]
    path_lengths = np.array([item["length"] for item in path_candidates], dtype=float)
    path_traffic_means = np.array([item["traffic_mean"] for item in path_candidates], dtype=float)

    # Compute utility: U = -length + lambda_traffic * traffic_mean * scale + Gumbel
    # Higher traffic_mean = prefer routes on higher-traffic corridors
    if edge_traffic_weights is not None:
        # Scale traffic term relative to path lengths
        traffic_scale = np.mean(path_lengths) if len(path_lengths) > 0 else 1000.0
        utilities = (
            -path_lengths
            + lambda_traffic * path_traffic_means * traffic_scale
            + rng.gumbel(size=len(path_candidates)) * 500
        )
    else:
        # Original utility: -length + Gumbel noise
        utilities = -path_lengths + rng.gumbel(size=len(path_candidates)) * 500

    selected_idx = np.argmax(utilities)
    route = paths[selected_idx]
    route_length = path_lengths[selected_idx]

    route_key = tuple(route)
    route_obs_key = (
        route_key,
        float(detection_radius_m),
        str(camera_query_backend),
    )
    cached_hits = _cache_get(route_obs_cache, route_obs_key)
    if cached_hits is not None:
        return route, list(cached_hits), route_length

    use_cuda_query = False
    use_cuda_service_query = False
    if camera_query_backend == "torch-cuda" and camera_query_payload is not None:
        n_route_nodes = len({n for n in route if n in node_coords})
        n_cameras = len(camera_query_payload.get("camera_ids", []))
        work = n_route_nodes * n_cameras
        min_work = int(camera_query_payload.get("cuda_min_work", 1))
        use_cuda_query = work >= min_work
    elif camera_query_backend == "torch-cuda-service" and camera_query_service is not None:
        use_cuda_service_query = True

    if use_cuda_query:
        camera_hits = _collect_camera_hits_torch_cuda(
            route=route,
            node_coords=node_coords,
            payload=camera_query_payload,
            detection_radius_m=detection_radius_m,
        )
    elif use_cuda_service_query:
        camera_hits = _collect_camera_hits_torch_cuda_service(
            route=route,
            service=camera_query_service,
            detection_radius_m=detection_radius_m,
        )
    else:
        # Walk along route and detect cameras (CPU KD-tree path).
        camera_hits = []
        seen_cameras = set()

        for node in route:
            if node not in node_coords:
                continue
            lat, lon = node_coords[node]
            nearby_cam_ids = _query_nearby_cameras_kdtree(
                lat=lat,
                lon=lon,
                camera_tree=camera_tree,
                camera_ids=camera_ids,
                detection_radius_m=detection_radius_m,
            )
            for cam_id in nearby_cam_ids:
                if cam_id not in seen_cameras:
                    camera_hits.append(cam_id)
                    seen_cameras.add(cam_id)

    _cache_put(route_obs_cache, route_obs_key, tuple(camera_hits), route_cache_size)

    return route, camera_hits, route_length


# =============================================================================
# CAMERA DATA LOADING
# =============================================================================

def load_cameras(geojson_path: Path) -> tuple[dict[int, tuple[float, float]], cKDTree, list[int]]:
    """
    Load camera locations and build spatial index.

    Args:
        geojson_path: Path to camera GeoJSON file

    Returns:
        Tuple of (positions dict, KD-tree, camera_id list)
    """
    with open(geojson_path) as f:
        data = json.load(f)

    positions = {}
    coords_m = []
    camera_ids = []

    for i, feat in enumerate(data.get('features', [])):
        geom = feat.get('geometry', {})
        if geom.get('type') == 'Point':
            lon, lat = geom['coordinates'][:2]
            cam_id = i
            positions[cam_id] = (lat, lon)
            camera_ids.append(cam_id)

            # Convert to approximate meters for spatial indexing
            x_m = lon * 111000 * np.cos(np.radians(lat))
            y_m = lat * 111000
            coords_m.append([x_m, y_m])

    if coords_m:
        tree = cKDTree(np.array(coords_m))
    else:
        tree = None

    print(f"Loaded {len(positions)} cameras")

    return positions, tree, camera_ids


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def simulate_road_network_trips(
    region: str,
    camera_geojson_path: Path,
    n_vehicles: int = 2000,
    n_trips_per_vehicle: int = 10,
    k_shortest: int = 3,
    p_return: float = 0.6,
    detection_radius_m: float = 100.0,
    h3_resolution: int = 9,
    seed: int = 42,
    verbose: bool = True,
    # Traffic weighting parameters
    traffic_weights: dict[int, float] | None = None,
    edge_traffic_weights: dict[tuple[int, int, int], float] | None = None,
    # Parallel processing
    n_workers: int = 1,  # Set > 1 to enable multiprocessing
    mp_chunksize: int = 1,
    camera_query_backend: str = "scipy-kdtree",
    camera_query_cuda_batch_size: int = 256,
    camera_query_cuda_min_work: int = 2_000_000,
    destination_candidate_pool_size: int = 512,
    path_cache_size: int = 10000,
    route_cache_size: int = 20000,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval_vehicles: int = 250,
    resume_from_checkpoint: bool = False,
    store_trip_metadata: bool = True,
    traffic_blend_factor: float = 0.7,
    lambda_traffic: float = 0.5,
    # Trip distance constraints
    min_trip_distance_m: float = 8047.0,  # 5 miles
    max_trip_distance_m: float = 16093.0,  # 10 miles
) -> dict:
    """
    Full road-network trajectory simulation pipeline.

    Pipeline:
    1. Download/load road network (OSMnx)
    2. Build place universe (H3 hexagons)
    3. Compute attractiveness weights (with optional traffic weighting)
    4. Assign vehicle homes on road nodes
    5. For each vehicle:
       - Generate tour structure
       - Choose destinations via EPR+gravity (with distance constraints)
       - Route via k-shortest paths (with traffic preference)
       - Record camera hits along routes

    Args:
        region: Region name (e.g., 'atlanta', 'memphis')
        camera_geojson_path: Path to camera locations
        n_vehicles: Number of vehicles to simulate
        n_trips_per_vehicle: Trips per vehicle
        k_shortest: Number of shortest paths for route choice
        p_return: EPR return probability
        detection_radius_m: Camera detection radius
        h3_resolution: H3 hexagon resolution
        seed: Random seed
        verbose: Print progress
        traffic_weights: Node-level traffic scores for attractiveness (normalized 0-1)
        edge_traffic_weights: Edge-level traffic weights for route choice
        traffic_blend_factor: How much to weight traffic vs degree (0-1)
        lambda_traffic: Weight for traffic preference in route utility
        camera_query_backend: scipy-kdtree | torch-cuda | auto
        camera_query_cuda_batch_size: Batch size for CUDA route-node processing
        camera_query_cuda_min_work: Minimum (route_nodes * cameras) before CUDA path activates
        destination_candidate_pool_size: Exploration candidate pool size; 0 disables sampling
        path_cache_size: Max OD-route cache entries per worker/process
        route_cache_size: Max route->camera-hit cache entries per worker/process
        checkpoint_dir: Optional directory for incremental checkpoint shards
        checkpoint_interval_vehicles: Flush checkpoint every N completed vehicles
        resume_from_checkpoint: Resume from checkpoint shards if available and signature matches
        store_trip_metadata: Keep per-trip metadata in-memory/results (set False to reduce memory)
        min_trip_distance_m: Minimum trip distance (default 5 miles)
        max_trip_distance_m: Maximum trip distance (default 10 miles)

    Returns:
        Dict containing:
        - trajectories: List of camera-hit sequences per vehicle
        - trip_metadata: OD pairs, route lengths, hit counts
        - network_stats: Road network size, camera coverage, P(observation)
    """
    active_camera_query_backend = resolve_camera_query_backend(camera_query_backend, n_workers=n_workers)
    if camera_query_cuda_batch_size <= 0:
        raise ValueError("camera_query_cuda_batch_size must be >= 1")
    if camera_query_cuda_min_work <= 0:
        raise ValueError("camera_query_cuda_min_work must be >= 1")
    if destination_candidate_pool_size < 0:
        raise ValueError("destination_candidate_pool_size must be >= 0")
    if path_cache_size < 0:
        raise ValueError("path_cache_size must be >= 0")
    if route_cache_size < 0:
        raise ValueError("route_cache_size must be >= 0")
    if checkpoint_interval_vehicles <= 0:
        raise ValueError("checkpoint_interval_vehicles must be >= 1")

    if verbose:
        print(f"\n{'='*60}")
        print(f"ROAD NETWORK SIMULATION: {region.upper()}")
        print(f"{'='*60}")
        print(
            "Camera query backend: "
            f"requested={camera_query_backend} active={active_camera_query_backend}"
        )
        print(
            "Performance knobs: "
            f"candidate_pool={destination_candidate_pool_size}, "
            f"path_cache={path_cache_size}, route_cache={route_cache_size}"
        )

    # Load cameras
    camera_positions, camera_tree, camera_ids = load_cameras(camera_geojson_path)
    camera_query_payload = build_camera_query_payload(
        camera_positions=camera_positions,
        camera_ids=camera_ids,
        camera_query_backend=active_camera_query_backend,
        cuda_batch_size=camera_query_cuda_batch_size,
        cuda_min_work=camera_query_cuda_min_work,
    )

    if not camera_positions:
        print(f"No cameras found for {region}")
        return None

    # Get bounding box from cameras
    lats = [pos[0] for pos in camera_positions.values()]
    lons = [pos[1] for pos in camera_positions.values()]

    north, south = max(lats) + 0.05, min(lats) - 0.05
    east, west = max(lons) + 0.05, min(lons) - 0.05

    if verbose:
        print(f"Bounding box: ({south:.3f}, {west:.3f}) to ({north:.3f}, {east:.3f})")

    # Download road network
    if verbose:
        print("Downloading road network from OSM...")

    try:
        # OSMnx 2.x API: bbox = (left, bottom, right, top) = (west, south, east, north)
        bbox = (west, south, east, north)
        G_road = ox.graph_from_bbox(
            bbox=bbox,
            network_type='drive',
            simplify=True
        )
    except Exception as e:
        print(f"Error downloading road network: {e}")
        return None

    if verbose:
        print(f"Road network: {G_road.number_of_nodes()} nodes, {G_road.number_of_edges()} edges")

    # Convert to simple DiGraph for efficient path finding (done once)
    if verbose:
        print("Converting to simple DiGraph for path finding...")
    G_simple = convert_to_simple_digraph(G_road)
    if verbose:
        print(f"Simple graph: {G_simple.number_of_nodes()} nodes, {G_simple.number_of_edges()} edges")

    # Get node coordinates
    nodes_gdf = ox.graph_to_gdfs(G_road, edges=False)
    node_coords = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

    # Build place universe
    if verbose:
        print("Building place universe (H3 hexagons)...")
    place_universe = build_place_universe(G_road, method='h3', h3_resolution=h3_resolution, seed=seed)

    # Compute attractiveness (with optional traffic weighting)
    if verbose:
        print("Computing attractiveness weights...")
        if traffic_weights is not None:
            print(f"  Using traffic weights with blend factor {traffic_blend_factor}")
    A_home, A_work, A_other = compute_attractiveness(
        G_road,
        place_universe,
        traffic_weights=traffic_weights,
        traffic_blend_factor=traffic_blend_factor,
    )

    # Assign vehicle homes
    if verbose:
        print("Assigning vehicle homes...")
    vehicle_homes = assign_vehicle_homes(n_vehicles, A_home, seed=seed)

    # Prepare attractiveness lookup by activity type
    A_by_type = {
        'work': A_work,
        'shop': A_other,
        'social': A_other,
        'other': A_other,
        'home': A_home,
    }

    # Simulate trips
    if verbose:
        print(f"Simulating {n_vehicles} vehicles x {n_trips_per_vehicle} trips...")
        if n_workers > 1:
            print(f"  Using {n_workers} worker processes")

    # Track P(observation) stats
    trips_with_hits = 0
    trips_with_2plus_hits = 0
    total_trips = 0
    all_hit_counts = []

    vehicle_ids = list(vehicle_homes.keys())
    trajectories_by_idx: list[dict | None] = [None] * len(vehicle_ids)
    trip_metadata_by_idx: list[list | None] = [None] * len(vehicle_ids) if store_trip_metadata else []
    completed_vehicle_indices: set[int] = set()
    checkpoint_pending_records: list[tuple[int, dict, list]] = []
    checkpoint_shard_counter = 0
    checkpoint_root: Path | None = None

    checkpoint_signature = _checkpoint_signature(
        region=region,
        n_vehicles=n_vehicles,
        n_trips_per_vehicle=n_trips_per_vehicle,
        seed=seed,
        k_shortest=k_shortest,
        p_return=p_return,
        detection_radius_m=detection_radius_m,
        min_trip_distance_m=min_trip_distance_m,
        max_trip_distance_m=max_trip_distance_m,
        camera_query_backend=active_camera_query_backend,
        destination_candidate_pool_size=destination_candidate_pool_size,
        store_trip_metadata=store_trip_metadata,
    )

    if checkpoint_dir:
        checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        if resume_from_checkpoint:
            (
                loaded_trajectories,
                loaded_trip_metadata,
                loaded_completed_indices,
                loaded_stats,
                checkpoint_shard_counter,
            ) = _load_checkpoint_state(
                checkpoint_dir=checkpoint_root,
                signature=checkpoint_signature,
                n_vehicles=n_vehicles,
                store_trip_metadata=store_trip_metadata,
                verbose=verbose,
            )
            for idx, trajectory in loaded_trajectories.items():
                trajectories_by_idx[idx] = trajectory
            if store_trip_metadata:
                for idx, meta in loaded_trip_metadata.items():
                    trip_metadata_by_idx[idx] = meta
            completed_vehicle_indices = set(loaded_completed_indices)

            if loaded_stats:
                total_trips = int(loaded_stats.get("total_trips", 0))
                trips_with_hits = int(loaded_stats.get("trips_with_hits", 0))
                trips_with_2plus_hits = int(loaded_stats.get("trips_with_2plus_hits", 0))
                all_hit_counts = [int(x) for x in loaded_stats.get("all_hit_counts", [])]
            elif store_trip_metadata and loaded_trip_metadata:
                for meta_list in loaded_trip_metadata.values():
                    for trip in meta_list:
                        total_trips += 1
                        n_hits = int(trip.get("n_camera_hits", 0))
                        all_hit_counts.append(n_hits)
                        if n_hits >= 1:
                            trips_with_hits += 1
                        if n_hits >= 2:
                            trips_with_2plus_hits += 1

            if verbose and completed_vehicle_indices:
                print(
                    f"[INFO] Resumed checkpoint: {len(completed_vehicle_indices)}/{n_vehicles} "
                    "vehicles already complete."
                )

    def _update_trip_stats(trip_meta_list: list[dict]) -> None:
        nonlocal total_trips, trips_with_hits, trips_with_2plus_hits, all_hit_counts
        for trip in trip_meta_list:
            total_trips += 1
            n_hits = int(trip.get("n_camera_hits", 0))
            all_hit_counts.append(n_hits)
            if n_hits >= 1:
                trips_with_hits += 1
            if n_hits >= 2:
                trips_with_2plus_hits += 1

    def _flush_checkpoint_if_needed(force: bool = False) -> None:
        nonlocal checkpoint_shard_counter
        if checkpoint_root is None:
            return
        if (not force) and (len(checkpoint_pending_records) < checkpoint_interval_vehicles):
            return
        stats_payload = {
            "total_trips": total_trips,
            "trips_with_hits": trips_with_hits,
            "trips_with_2plus_hits": trips_with_2plus_hits,
            "all_hit_counts": all_hit_counts,
        }
        checkpoint_shard_counter = _flush_checkpoint_records(
            checkpoint_dir=checkpoint_root,
            signature=checkpoint_signature,
            pending_records=checkpoint_pending_records,
            completed_indices=completed_vehicle_indices,
            stats=stats_payload,
            shard_counter=checkpoint_shard_counter,
            verbose=verbose,
        )
        checkpoint_pending_records.clear()

    pending_worker_args = [
        (vid, vehicle_homes[vid], idx)
        for idx, vid in enumerate(vehicle_ids)
        if idx not in completed_vehicle_indices
    ]

    if n_workers > 1:
        # Parallel execution using multiprocessing
        from multiprocessing import Lock, Pool, Process, Queue, Value

        camera_query_service_init = None
        gpu_service_proc = None
        gpu_request_queue = None
        gpu_response_queues = None

        if active_camera_query_backend == "torch-cuda-service":
            gpu_request_queue = Queue(maxsize=max(64, n_workers * 8))
            gpu_response_queues = [Queue(maxsize=max(32, mp_chunksize * 8)) for _ in range(n_workers)]
            worker_counter = Value("i", 0)
            worker_counter_lock = Lock()
            camera_query_service_init = {
                "request_queue": gpu_request_queue,
                "response_queues": gpu_response_queues,
                "worker_counter": worker_counter,
                "worker_counter_lock": worker_counter_lock,
            }
            gpu_service_proc = Process(
                target=_gpu_camera_query_server,
                args=(
                    gpu_request_queue,
                    gpu_response_queues,
                    node_coords,
                    camera_positions,
                    camera_ids,
                    int(camera_query_cuda_batch_size),
                    int(route_cache_size),
                ),
            )
            gpu_service_proc.start()
            if verbose:
                print("[INFO] Started dedicated CUDA camera-query service process.")

        # Initialize pool with shared data
        init_args = (
            G_simple, G_road, node_coords, camera_positions,
            camera_tree, camera_ids, A_work, A_home, A_other,
            edge_traffic_weights, k_shortest, p_return, detection_radius_m,
            lambda_traffic, min_trip_distance_m, max_trip_distance_m,
            n_trips_per_vehicle, seed, active_camera_query_backend, camera_query_payload,
            camera_query_service_init,
            destination_candidate_pool_size, path_cache_size, route_cache_size, store_trip_metadata
        )

        if mp_chunksize <= 0:
            raise ValueError("mp_chunksize must be >= 1")

        try:
            with Pool(processes=n_workers, initializer=_init_worker, initargs=init_args) as pool:
                # Use unordered completion for smoother progress updates.
                for vehicle_idx, trajectory, trip_meta_list in tqdm(
                    pool.imap_unordered(_simulate_single_vehicle, pending_worker_args, chunksize=mp_chunksize),
                    total=len(pending_worker_args),
                    disable=not verbose,
                    desc="Simulating vehicles (parallel)",
                ):
                    trajectories_by_idx[vehicle_idx] = trajectory
                    if store_trip_metadata:
                        trip_metadata_by_idx[vehicle_idx] = trip_meta_list

                    _update_trip_stats(trip_meta_list)
                    completed_vehicle_indices.add(vehicle_idx)

                    if checkpoint_root is not None:
                        checkpoint_pending_records.append(
                            (vehicle_idx, trajectory, trip_meta_list if store_trip_metadata else [])
                        )
                        _flush_checkpoint_if_needed(force=False)
        finally:
            if gpu_service_proc is not None:
                try:
                    gpu_request_queue.put(None)
                except Exception:
                    pass
                gpu_service_proc.join(timeout=15)
                if gpu_service_proc.is_alive():
                    gpu_service_proc.terminate()
                    gpu_service_proc.join(timeout=5)
                if verbose:
                    print("[INFO] CUDA camera-query service process stopped.")

    else:
        # Sequential execution
        ksp_cache = OrderedDict()
        route_obs_cache = OrderedDict()

        for vid, home_node, vehicle_idx in tqdm(
            pending_worker_args,
            disable=not verbose,
            desc="Simulating vehicles",
        ):
            vehicle_rng = np.random.default_rng(seed + vehicle_idx)
            activity_set = set()
            vehicle_camera_hits = []
            current_node = home_node
            trip_meta_list = [] if store_trip_metadata else []

            for trip_num in range(n_trips_per_vehicle):
                # Generate tour structure
                tour = generate_tour_structure(
                    n_stops=vehicle_rng.integers(1, 5),
                    seed=(seed + stable_u64(f"{vid}:{trip_num}")) % (2**31)
                )

                for activity in tour:
                    # Choose destination
                    A_dest = A_by_type.get(activity, A_other)

                    destination = choose_destination(
                        current_node=current_node,
                        activity_type=activity,
                        home_node=home_node,
                        activity_set=activity_set,
                        A_dest=A_dest,
                        G_road=G_road,
                        node_coords=node_coords,
                        p_return=p_return,
                        min_distance_m=min_trip_distance_m,
                        max_distance_m=max_trip_distance_m,
                        candidate_pool_size=destination_candidate_pool_size,
                        rng=vehicle_rng,
                    )

                    # Route and observe cameras (with optional traffic preference)
                    route, camera_hits, route_length = route_and_observe(
                        origin=current_node,
                        destination=destination,
                        G_simple=G_simple,
                        G_road=G_road,
                        camera_positions=camera_positions,
                        camera_tree=camera_tree,
                        camera_ids=camera_ids,
                        node_coords=node_coords,
                        k_shortest=k_shortest,
                        detection_radius_m=detection_radius_m,
                        edge_traffic_weights=edge_traffic_weights,
                        lambda_traffic=lambda_traffic,
                        camera_query_backend=active_camera_query_backend,
                        camera_query_payload=camera_query_payload,
                        camera_query_service=None,
                        ksp_cache=ksp_cache,
                        path_cache_size=path_cache_size,
                        route_obs_cache=route_obs_cache,
                        route_cache_size=route_cache_size,
                        rng=vehicle_rng,
                    )

                    # Update activity set
                    if activity != 'home':
                        activity_set.add(destination)

                    # Record trip
                    trip_row = {
                        'vehicle_id': vid,
                        'origin': current_node,
                        'destination': destination,
                        'activity': activity,
                        'route_length_m': route_length,
                        'n_camera_hits': len(camera_hits),
                    }
                    if store_trip_metadata:
                        trip_meta_list.append(trip_row)

                    # Track P(observation)
                    total_trips += 1
                    n_hits = len(camera_hits)
                    all_hit_counts.append(n_hits)
                    if n_hits >= 1:
                        trips_with_hits += 1
                    if n_hits >= 2:
                        trips_with_2plus_hits += 1

                    # Extend vehicle's camera observations
                    vehicle_camera_hits.extend(camera_hits)

                    # Move to destination
                    current_node = destination

            trajectory = {
                'vehicle_id': vid,
                'camera_hits': vehicle_camera_hits,
                'home_node': home_node,
            }
            trajectories_by_idx[vehicle_idx] = trajectory
            if store_trip_metadata:
                trip_metadata_by_idx[vehicle_idx] = trip_meta_list
            completed_vehicle_indices.add(vehicle_idx)

            if checkpoint_root is not None:
                checkpoint_pending_records.append(
                    (vehicle_idx, trajectory, trip_meta_list if store_trip_metadata else [])
                )
                _flush_checkpoint_if_needed(force=False)

    _flush_checkpoint_if_needed(force=True)

    # Aggregate results in deterministic vehicle index order.
    all_trajectories = [x for x in trajectories_by_idx if x is not None]
    if store_trip_metadata:
        trip_metadata: list[dict] = []
        for meta_list in trip_metadata_by_idx:
            if not meta_list:
                continue
            trip_metadata.extend(meta_list)
    else:
        trip_metadata = []

    # Compute P(observation) metrics
    p_obs_1 = trips_with_hits / total_trips if total_trips > 0 else 0
    p_obs_2 = trips_with_2plus_hits / total_trips if total_trips > 0 else 0
    avg_hits = np.mean(all_hit_counts) if all_hit_counts else 0

    if verbose:
        print(f"\nP(observation) sanity-check metrics:")
        print(f"  P(>=1 camera hit) = {p_obs_1:.1%}")
        print(f"  P(>=2 camera hits) = {p_obs_2:.1%}")
        print(f"  Avg hits per trip = {avg_hits:.2f}")

    # Compile results
    results = {
        'region': region,
        'trajectories': all_trajectories,
        'trip_metadata': trip_metadata,
        'network_stats': {
            'n_road_nodes': G_road.number_of_nodes(),
            'n_road_edges': G_road.number_of_edges(),
            'n_cameras': len(camera_positions),
            'n_places': len(place_universe),
            'n_vehicles': n_vehicles,
            'n_trips_per_vehicle': n_trips_per_vehicle,
            'total_trips': total_trips,
        },
        'observation_stats': {
            'p_at_least_1': p_obs_1,
            'p_at_least_2': p_obs_2,
            'avg_hits_per_trip': avg_hits,
            'hit_distribution': np.histogram(all_hit_counts, bins=range(0, 12))[0].tolist(),
        },
        'parameters': {
            'k_shortest': k_shortest,
            'p_return': p_return,
            'detection_radius_m': detection_radius_m,
            'h3_resolution': h3_resolution,
            'seed': seed,
            'mp_chunksize': mp_chunksize,
            'camera_query_backend_requested': camera_query_backend,
            'camera_query_backend': active_camera_query_backend,
            'camera_query_cuda_batch_size': int(camera_query_cuda_batch_size),
            'camera_query_cuda_min_work': int(camera_query_cuda_min_work),
            'destination_candidate_pool_size': int(destination_candidate_pool_size),
            'path_cache_size': int(path_cache_size),
            'route_cache_size': int(route_cache_size),
            'checkpoint_dir': str(checkpoint_root) if checkpoint_root is not None else None,
            'checkpoint_interval_vehicles': int(checkpoint_interval_vehicles),
            'resume_from_checkpoint': bool(resume_from_checkpoint),
            'store_trip_metadata': bool(store_trip_metadata),
        },
    }

    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_random_point_uniqueness(
    trajectories: list[dict],
    k: int = 2,
    n_samples: int = 1000,
    ordered: bool = False,
    seed: int = 42,
) -> dict:
    """
    Compute uniqueness using k random observation points.

    This aligns with de Montjoye's methodology: sample k random points
    from a vehicle's observation history and check uniqueness.

    NOT prefix-based (first-n cameras).

    Args:
        trajectories: List of trajectory dicts with 'camera_hits'
        k: Number of random points to sample
        n_samples: Number of bootstrap samples for CI
        ordered: If True, treat as ordered tuple; if False, as unordered set
        seed: Random seed

    Returns:
        Dict with uniqueness rate and CI
    """
    # Filter to vehicles with at least k observations
    valid_vehicles = [
        t for t in trajectories
        if len(t.get('camera_hits', [])) >= k
    ]

    if not valid_vehicles:
        return {'uniqueness': 0.0, 'n_valid': 0, 'ci_lower': 0.0, 'ci_upper': 0.0}

    def compute_uniqueness_once(vehicles_sample, rng_inner):
        """Compute uniqueness for one bootstrap sample."""
        signature_counts = defaultdict(int)

        for v in vehicles_sample:
            hits = v.get('camera_hits', [])
            if len(hits) < k:
                continue

            # Sample k random points
            indices = rng_inner.choice(len(hits), size=k, replace=False)
            indices.sort()  # Keep temporal order if ordered
            sampled = tuple(hits[i] for i in indices)

            if not ordered:
                sampled = tuple(sorted(sampled))

            signature_counts[sampled] += 1

        unique_count = sum(1 for count in signature_counts.values() if count == 1)
        total = len(vehicles_sample)

        return unique_count / total if total > 0 else 0

    # Bootstrap: resample vehicles AND random observation points.
    # Point estimate = mean of bootstrap distribution so CI contains estimate.
    bootstrap_estimates = []
    for i in range(n_samples):
        boot_rng = np.random.default_rng(seed + i + 1)
        sample_indices = boot_rng.choice(len(valid_vehicles), size=len(valid_vehicles), replace=True)
        vehicle_sample = [valid_vehicles[j] for j in sample_indices]

        inner_rng = np.random.default_rng(seed + i + n_samples + 1)
        est = compute_uniqueness_once(vehicle_sample, inner_rng)
        bootstrap_estimates.append(est)

    bootstrap_estimates = np.array(bootstrap_estimates, dtype=float)
    point_estimate = float(np.mean(bootstrap_estimates))
    ci_lower = float(np.percentile(bootstrap_estimates, 2.5))
    ci_upper = float(np.percentile(bootstrap_estimates, 97.5))

    return {
        'uniqueness': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_valid': len(valid_vehicles),
        'method': 'random_points' if not ordered else 'random_points_ordered',
        'k': k,
    }


def compute_topology_baseline_accuracy(
    trajectories: list[dict],
    camera_graph: nx.DiGraph,
    k: int = 5,
) -> dict:
    """
    Compute topology-only baseline: Acc@K ≈ min(K / out_degree, 1)

    This is the expected accuracy if we just guessed uniformly
    among all outgoing neighbors.

    Args:
        trajectories: List of trajectory dicts
        camera_graph: Camera network graph
        k: Top-k for accuracy

    Returns:
        Dict with topology baseline stats
    """
    degrees = [camera_graph.out_degree(n) for n in camera_graph.nodes()]

    if not degrees:
        return {'topology_baseline': 0.0, 'avg_out_degree': 0.0}

    avg_degree = np.mean(degrees)

    # Expected Acc@K if uniform over neighbors
    expected_acc_k = []
    for d in degrees:
        if d > 0:
            expected_acc_k.append(min(k / d, 1.0))
        else:
            expected_acc_k.append(0.0)

    topology_baseline = np.mean(expected_acc_k)

    return {
        'topology_baseline_acc_k': topology_baseline,
        'avg_out_degree': avg_degree,
        'median_out_degree': np.median(degrees),
        'degree_distribution': np.histogram(degrees, bins=range(0, 25))[0].tolist(),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Road-network-based ALPR trajectory simulation'
    )
    parser.add_argument(
        '--region', type=str, default='atlanta',
        choices=['atlanta', 'memphis', 'richmond', 'charlotte', 'lehigh_valley', 'maine', 'all'],
        help='Region to simulate'
    )
    parser.add_argument('--n-vehicles', type=int, default=2000, help='Number of vehicles')
    parser.add_argument('--n-trips', type=int, default=10, help='Trips per vehicle')
    parser.add_argument('--k-shortest', type=int, default=3, help='Number of shortest paths')
    parser.add_argument('--p-return', type=float, default=0.6, help='EPR return probability')
    parser.add_argument('--detection-radius', type=float, default=100.0, help='Camera detection radius (m)')
    parser.add_argument('--h3-resolution', type=int, default=9, help='H3 hexagon resolution')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Region camera files
    region_files = {
        'atlanta': base_dir / 'data/raw/atlanta_cameras.geojson',
        'memphis': base_dir / 'data/raw/memphis_cameras.geojson',
        'richmond': base_dir / 'data/raw/richmond_cameras.geojson',
        'charlotte': base_dir / 'data/raw/charlotte_cameras.geojson',
        'lehigh_valley': base_dir / 'data/raw/lehigh_valley_cameras.geojson',
        'maine': base_dir / 'data/raw/maine_cameras.geojson',
    }

    regions_to_run = list(region_files.keys()) if args.region == 'all' else [args.region]

    all_results = {}

    for region in regions_to_run:
        camera_path = region_files[region]

        if not camera_path.exists():
            print(f"Warning: Camera file not found for {region}: {camera_path}")
            continue

        results = simulate_road_network_trips(
            region=region,
            camera_geojson_path=camera_path,
            n_vehicles=args.n_vehicles,
            n_trips_per_vehicle=args.n_trips,
            k_shortest=args.k_shortest,
            p_return=args.p_return,
            detection_radius_m=args.detection_radius,
            h3_resolution=args.h3_resolution,
            seed=args.seed,
            verbose=True,
        )

        if results is None:
            continue

        # Compute random-point uniqueness
        print("\nComputing random-point uniqueness...")
        u2_random = compute_random_point_uniqueness(
            results['trajectories'], k=2, ordered=False, seed=args.seed
        )
        u2_ordered = compute_random_point_uniqueness(
            results['trajectories'], k=2, ordered=True, seed=args.seed
        )

        results['uniqueness'] = {
            'u2_random': u2_random,
            'u2_ordered': u2_ordered,
        }

        print(f"\nUniqueness Results:")
        print(f"  U(2) random points (unordered): {u2_random['uniqueness']:.1%} "
              f"(95% CI: [{u2_random['ci_lower']:.1%}, {u2_random['ci_upper']:.1%}])")
        print(f"  U(2) random points (ordered):   {u2_ordered['uniqueness']:.1%} "
              f"(95% CI: [{u2_ordered['ci_lower']:.1%}, {u2_ordered['ci_upper']:.1%}])")

        # Save results
        output_file = output_dir / f'road_trajectories_{region}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)

        print(f"\nSaved results to {output_file}")

        all_results[region] = results

    # Summary across all regions
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL REGIONS")
        print(f"{'='*70}")
        print(f"{'Region':<15} {'Cameras':>8} {'P(>=1 hit)':>12} {'P(>=2 hits)':>12} {'U(2)':>10}")
        print("-" * 60)

        for region, results in all_results.items():
            n_cam = results['network_stats']['n_cameras']
            p1 = results['observation_stats']['p_at_least_1']
            p2 = results['observation_stats']['p_at_least_2']
            u2 = results['uniqueness']['u2_random']['uniqueness']

            print(f"{region:<15} {n_cam:>8} {p1:>11.1%} {p2:>11.1%} {u2:>9.1%}")


if __name__ == '__main__':
    main()
