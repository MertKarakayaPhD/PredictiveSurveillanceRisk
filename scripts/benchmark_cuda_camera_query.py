#!/usr/bin/env python3
"""
Synthetic benchmark for CUDA camera-query kernel vs CPU KD-tree path.

No metro simulation is run. This isolates the camera-hit query hotspot.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial import cKDTree

from scripts.road_network_simulation import (
    _collect_camera_hits_torch_cuda,
    _query_nearby_cameras_kdtree,
    build_camera_query_payload,
    torch,
)


def parse_int_csv(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_cameras(n_cameras: int, seed: int) -> tuple[dict[int, tuple[float, float]], cKDTree, list[int]]:
    rng = np.random.default_rng(seed)
    lats = 39.5 + 0.9 * rng.random(n_cameras)
    lons = -76.5 + 0.9 * rng.random(n_cameras)

    camera_positions: dict[int, tuple[float, float]] = {}
    camera_ids: list[int] = []
    coords_m: list[list[float]] = []
    for i in range(n_cameras):
        lat = float(lats[i])
        lon = float(lons[i])
        camera_positions[i] = (lat, lon)
        camera_ids.append(i)
        x_m = lon * 111000.0 * np.cos(np.radians(lat))
        y_m = lat * 111000.0
        coords_m.append([x_m, y_m])
    tree = cKDTree(np.asarray(coords_m, dtype=np.float64))
    return camera_positions, tree, camera_ids


def cpu_route_hits(
    route: list[int],
    node_coords: dict[int, tuple[float, float]],
    camera_tree: cKDTree,
    camera_ids: list[int],
    radius_m: float,
) -> list[int]:
    seen = set()
    hits: list[int] = []
    for node in route:
        if node not in node_coords:
            continue
        lat, lon = node_coords[node]
        nearby = _query_nearby_cameras_kdtree(
            lat=lat, lon=lon, camera_tree=camera_tree, camera_ids=camera_ids, detection_radius_m=radius_m
        )
        for cam in nearby:
            if cam not in seen:
                seen.add(cam)
                hits.append(cam)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CUDA camera-query kernel against CPU KD-tree.")
    parser.add_argument("--camera-counts", type=str, default="2000,5000")
    parser.add_argument("--route-node-counts", type=str, default="200,800,2000")
    parser.add_argument("--radius-m", type=float, default=100.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-batch-size", type=int, default=256)
    args = parser.parse_args()

    if torch is None or not torch.cuda.is_available():
        print("[SKIP] CUDA unavailable; benchmark skipped.")
        return 0

    camera_counts = parse_int_csv(args.camera_counts)
    route_counts = parse_int_csv(args.route_node_counts)

    print("camera_count,route_nodes,cpu_sec,gpu_sec,gpu_vs_cpu,match")
    for n_cameras in camera_counts:
        camera_positions, camera_tree, camera_ids = build_cameras(n_cameras, seed=args.seed + n_cameras)
        payload = build_camera_query_payload(
            camera_positions=camera_positions,
            camera_ids=camera_ids,
            camera_query_backend="torch-cuda",
            cuda_batch_size=args.cuda_batch_size,
            cuda_min_work=1,  # force CUDA in kernel benchmark
        )
        if payload is None:
            print(f"{n_cameras},0,0,0,0,False")
            continue

        rng = np.random.default_rng(args.seed + 100 + n_cameras)
        for n_route_nodes in route_counts:
            node_lats = 39.5 + 0.9 * rng.random(n_route_nodes)
            node_lons = -76.5 + 0.9 * rng.random(n_route_nodes)
            node_coords = {i: (float(node_lats[i]), float(node_lons[i])) for i in range(n_route_nodes)}
            route = list(range(n_route_nodes))

            cpu_times = []
            gpu_times = []
            cpu_hits_ref: list[int] | None = None
            gpu_hits_ref: list[int] | None = None
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                cpu_hits = cpu_route_hits(route, node_coords, camera_tree, camera_ids, args.radius_m)
                cpu_times.append(time.perf_counter() - t0)

                t1 = time.perf_counter()
                gpu_hits = _collect_camera_hits_torch_cuda(
                    route=route,
                    node_coords=node_coords,
                    payload=payload,
                    detection_radius_m=args.radius_m,
                )
                gpu_times.append(time.perf_counter() - t1)

                cpu_hits_ref = cpu_hits
                gpu_hits_ref = gpu_hits

            cpu_sec = min(cpu_times)
            gpu_sec = min(gpu_times)
            ratio = (gpu_sec / cpu_sec) if cpu_sec > 0 else float("inf")
            match = (set(cpu_hits_ref or []) == set(gpu_hits_ref or []))
            print(f"{n_cameras},{n_route_nodes},{cpu_sec:.6f},{gpu_sec:.6f},{ratio:.3f},{match}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

