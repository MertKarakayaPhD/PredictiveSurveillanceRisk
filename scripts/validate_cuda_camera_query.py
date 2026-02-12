#!/usr/bin/env python3
"""
Validate CUDA camera-query prototype against current KD-tree behavior.

This is a lightweight synthetic test; it does not run metro simulations.
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


def build_synthetic_cameras(n_cameras: int, seed: int) -> tuple[dict[int, tuple[float, float]], cKDTree, list[int]]:
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

    tree = cKDTree(np.array(coords_m, dtype=np.float64))
    return camera_positions, tree, camera_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CUDA camera-query against KD-tree output.")
    parser.add_argument("--n-cameras", type=int, default=3000)
    parser.add_argument("--n-nodes", type=int, default=500)
    parser.add_argument("--radius-m", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if torch is None or not torch.cuda.is_available():
        print("[SKIP] CUDA unavailable; torch-cuda validation skipped.")
        return 0

    camera_positions, camera_tree, camera_ids = build_synthetic_cameras(
        n_cameras=args.n_cameras,
        seed=args.seed,
    )
    payload = build_camera_query_payload(
        camera_positions=camera_positions,
        camera_ids=camera_ids,
        camera_query_backend="torch-cuda",
        cuda_batch_size=256,
    )
    if payload is None:
        print("[SKIP] Could not build CUDA payload.")
        return 0

    rng = np.random.default_rng(args.seed + 1)
    node_lats = 39.5 + 0.9 * rng.random(args.n_nodes)
    node_lons = -76.5 + 0.9 * rng.random(args.n_nodes)
    node_coords = {i: (float(node_lats[i]), float(node_lons[i])) for i in range(args.n_nodes)}

    mismatches = 0
    t_cpu = 0.0
    t_gpu = 0.0

    for node_id in range(args.n_nodes):
        lat, lon = node_coords[node_id]

        t0 = time.perf_counter()
        cpu_hits = _query_nearby_cameras_kdtree(
            lat=lat,
            lon=lon,
            camera_tree=camera_tree,
            camera_ids=camera_ids,
            detection_radius_m=args.radius_m,
        )
        t_cpu += time.perf_counter() - t0

        t1 = time.perf_counter()
        gpu_hits = _collect_camera_hits_torch_cuda(
            route=[node_id],
            node_coords=node_coords,
            payload=payload,
            detection_radius_m=args.radius_m,
        )
        t_gpu += time.perf_counter() - t1

        if set(cpu_hits) != set(gpu_hits):
            mismatches += 1

    print(f"[INFO] nodes={args.n_nodes} cameras={args.n_cameras} radius_m={args.radius_m}")
    print(f"[INFO] CPU total={t_cpu:.4f}s | GPU total={t_gpu:.4f}s")
    if t_cpu > 0:
        print(f"[INFO] relative GPU/CPU = {t_gpu / t_cpu:.3f}x")

    if mismatches > 0:
        print(f"[FAIL] mismatches={mismatches}/{args.n_nodes}")
        return 1

    print("[OK] CUDA camera-query outputs match KD-tree set results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

