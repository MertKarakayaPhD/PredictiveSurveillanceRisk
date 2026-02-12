#!/usr/bin/env python3
"""
Lightweight validation for experimental compute backends.

This does NOT run full metro simulations. It validates:
- backend resolution logic
- destination-choice function correctness invariants
- optional microbenchmark timings on synthetic data
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from scripts.road_network_simulation import (
    build_destination_payload,
    choose_destination_fast,
    resolve_compute_backend,
    torch,
)


def synthetic_payload(n_nodes: int, backend: str) -> tuple[dict, dict[int, tuple[float, float]], int]:
    rng = np.random.default_rng(123)
    node_ids = np.arange(1000, 1000 + n_nodes, dtype=int)
    lats = 39.5 + 0.8 * rng.random(n_nodes)
    lons = -76.0 + 0.8 * rng.random(n_nodes)
    weights = 0.1 + rng.random(n_nodes)

    node_coords = {int(n): (float(lat), float(lon)) for n, lat, lon in zip(node_ids, lats, lons)}
    a_dest = {int(n): float(w) for n, w in zip(node_ids, weights)}
    payload = build_destination_payload(a_dest, node_coords=node_coords, compute_backend=backend)
    home_node = int(node_ids[0])
    return payload, node_coords, home_node


def run_invariant_checks(backend: str, iterations: int) -> None:
    payload, node_coords, home_node = synthetic_payload(n_nodes=2000, backend=backend)
    rng = np.random.default_rng(42)
    known = set(list(payload["nodes_np"][:20]))
    current_node = int(payload["nodes_np"][100])
    node_set = set(int(x) for x in payload["nodes_np"])

    # home activity must return home
    out = choose_destination_fast(
        current_node=current_node,
        activity_type="home",
        home_node=home_node,
        activity_set=known,
        dest_payload=payload,
        node_coords=node_coords,
        rng=rng,
    )
    assert out == home_node, "home activity must return home_node"

    # general activity should always return a valid candidate node id
    for _ in range(iterations):
        out = choose_destination_fast(
            current_node=current_node,
            activity_type="work",
            home_node=home_node,
            activity_set=known,
            dest_payload=payload,
            node_coords=node_coords,
            rng=rng,
        )
        assert int(out) in node_set, "destination must be from payload candidate nodes"


def microbench(backend: str, iterations: int) -> float:
    payload, node_coords, home_node = synthetic_payload(n_nodes=4000, backend=backend)
    rng = np.random.default_rng(7)
    known = set(list(payload["nodes_np"][:50]))
    node_list = [int(x) for x in payload["nodes_np"]]

    t0 = time.perf_counter()
    for i in range(iterations):
        current_node = node_list[i % len(node_list)]
        _ = choose_destination_fast(
            current_node=current_node,
            activity_type="work",
            home_node=home_node,
            activity_set=known,
            dest_payload=payload,
            node_coords=node_coords,
            rng=rng,
        )
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and microbenchmark experimental backend code paths.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--microbench-iters", type=int, default=10000)
    args = parser.parse_args()

    backends = ["cpu", "torch-cpu"]
    if torch is not None and torch.cuda.is_available():
        backends.append("torch-cuda")

    print("[CHECK] Backend resolution")
    assert resolve_compute_backend("cpu", n_workers=8) == "cpu"
    assert resolve_compute_backend("auto", n_workers=8) == "cpu"
    if torch is not None and torch.cuda.is_available():
        assert resolve_compute_backend("auto", n_workers=1) == "torch-cuda"
    print("[OK] backend resolution")

    timings: list[tuple[str, float]] = []
    for backend in backends:
        print(f"[CHECK] invariants backend={backend}")
        run_invariant_checks(backend=backend, iterations=args.iterations)
        print(f"[OK] invariants backend={backend}")

        print(f"[BENCH] backend={backend} iterations={args.microbench_iters}")
        elapsed = microbench(backend=backend, iterations=args.microbench_iters)
        timings.append((backend, elapsed))
        print(f"[DONE] backend={backend} elapsed={elapsed:.4f}s")

    if timings:
        base = timings[0][1]
        print("\n[SUMMARY] relative speed vs cpu (lower is better):")
        for backend, elapsed in timings:
            rel = elapsed / base if base > 0 else float("inf")
            print(f"  {backend:10s} elapsed={elapsed:.4f}s  rel={rel:.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

