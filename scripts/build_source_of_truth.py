#!/usr/bin/env python3
"""
Build data/source_of_truth.json from simulation outputs.

This script is the reproducible "source of truth" pipeline for the paper:
- Loads camera GeoJSON files to compute convex-hull area and density
- Loads road-network simulation pickles (data/processed/road_trajectories_*.pkl)
  to extract observation probabilities and trip counts
- Recomputes uniqueness U(k) with bootstrap-mean point estimates
- Computes predictability (Acc@1/Acc@5) from Markov (order=1) on camera-hit sequences
- Optionally ingests Chicago validation metrics
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np

# Ensure project root is importable when running from `scripts/`
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction import MarkovPredictor, evaluate_predictor
from src.simulation import Trajectory


def convex_hull_area_km2(camera_geojson_path: Path) -> float:
    gdf = gpd.read_file(camera_geojson_path)
    if len(gdf) == 0:
        return 0.0
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    return float(gdf_proj.union_all().convex_hull.area / 1e6)


def compute_random_point_uniqueness_bootstrap_mean(
    trajectories: list[dict],
    k: int,
    n_bootstrap: int,
    seed: int,
    ordered: bool = False,
) -> dict:
    valid = [t for t in trajectories if len(t.get("camera_hits", [])) >= k]
    if not valid:
        return {"uniqueness": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_valid": 0}

    def compute_once(vehicle_sample: list[dict], rng_inner: np.random.Generator) -> float:
        signature_counts: dict[tuple[int, ...], int] = defaultdict(int)

        for v in vehicle_sample:
            hits = v.get("camera_hits", [])
            if len(hits) < k:
                continue

            indices = rng_inner.choice(len(hits), size=k, replace=False)
            indices.sort()
            sampled = tuple(hits[i] for i in indices)
            if not ordered:
                sampled = tuple(sorted(sampled))
            signature_counts[sampled] += 1

        unique_count = sum(1 for c in signature_counts.values() if c == 1)
        return unique_count / len(vehicle_sample)

    estimates = []
    for i in range(n_bootstrap):
        boot_rng = np.random.default_rng(seed + i + 1)
        sample_idx = boot_rng.choice(len(valid), size=len(valid), replace=True)
        vehicle_sample = [valid[j] for j in sample_idx]

        inner_rng = np.random.default_rng(seed + i + n_bootstrap + 1)
        estimates.append(compute_once(vehicle_sample, inner_rng))

    estimates = np.array(estimates, dtype=float)
    return {
        "uniqueness": float(np.mean(estimates)),
        "ci_lower": float(np.percentile(estimates, 2.5)),
        "ci_upper": float(np.percentile(estimates, 97.5)),
        "n_valid": len(valid),
    }


def compute_markov_accuracy(
    trajectories: list[dict],
    order: int,
    seed: int,
) -> dict:
    trajs = [
        Trajectory(vehicle_id=t["vehicle_id"], camera_sequence=t.get("camera_hits", []))
        for t in trajectories
    ]

    rng = np.random.default_rng(seed)
    idx = np.arange(len(trajs))
    rng.shuffle(idx)
    split = int(0.8 * len(trajs))
    train = [trajs[i] for i in idx[:split]]
    test = [trajs[i] for i in idx[split:]]

    predictor = MarkovPredictor(order=order)
    predictor.fit(train)
    metrics = evaluate_predictor(predictor, test, k_values=[1, 5])

    return {
        "acc1": float(metrics["accuracy@1"]),
        "acc5": float(metrics["accuracy@5"]),
        "pred_coverage": float(metrics["coverage"]),
        "n_predictions": int(metrics["n_predictions"]),
        "n_no_prediction": int(metrics["n_no_prediction"]),
        "markov_order": order,
    }


def round_region_values(region: dict) -> dict:
    """Round floats for stable diffs/printing."""
    out = dict(region)
    for key in [
        "area_km2",
        "density_km2",
        "density_km2_total",
        "p_obs_1",
        "p_obs_2",
        "avg_hits_per_trip",
        "acc1",
        "acc5",
        "pred_coverage",
        "u2",
        "u2_ci_lower",
        "u2_ci_upper",
        "u3",
        "u3_ci_lower",
        "u3_ci_upper",
        "u4",
        "u4_ci_lower",
        "u4_ci_upper",
        "u5",
        "u5_ci_lower",
        "u5_ci_upper",
    ]:
        if key in out and isinstance(out[key], (float, int)):
            if key == "area_km2":
                out[key] = round(float(out[key]), 1)
            elif key.endswith("_km2"):
                out[key] = round(float(out[key]), 3)
            else:
                out[key] = round(float(out[key]), 3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data/source_of_truth.json")
    parser.add_argument(
        "--output",
        type=str,
        default="data/source_of_truth.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Bootstrap resamples for uniqueness CIs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--markov-order",
        type=int,
        default=1,
        help="Markov order for predictability (Acc@1/Acc@5)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_raw = base_dir / "data" / "raw"
    data_processed = base_dir / "data" / "processed"

    sim_regions = ["atlanta", "memphis", "richmond", "charlotte", "lehigh_valley", "maine"]
    categories = {
        "atlanta": "Dense",
        "memphis": "Moderate",
        "richmond": "Moderate",
        "charlotte": "Moderate",
        "lehigh_valley": "Sparse-Moderate",
        "maine": "Sparse",
    }

    output = {
        "_comment": "Source of truth for all region statistics - computed from road_network_simulation.py",
        "_methodology": {
            "trajectory_model": "road_network_OD",
            "place_universe": "H3_hexagons_resolution_9",
            "area_method": "convex_hull_area_utm",
            "destination_choice": "EPR_inspired_return_or_gravity_explore",
            "routing": "OSMnx_k_shortest_paths_gumbel_choice",
            "camera_detection": "100m_radius_kdtree_query",
            "uniqueness_method": "random_k_point_sampling_bootstrap_mean",
            "predictability_method": f"Markov_order_{args.markov_order}_vehicle_split_acc@1_acc@5",
            "n_vehicles": 5000,
            "n_trips_per_vehicle": 10,
            "p_return": 0.6,
            "computed_date": datetime.now().strftime("%Y-%m-%d"),
        },
        "regions": {},
        "_psr": {
            "_note": (
                "PSR can be recomputed from per-region fields. "
                "Formula: PSR = w_D*D + w_N*N + w_Pi*Pi where D is sigmoid-normalized "
                "density (rho0=0.10, k=1.5), N = sqrt(R*C) with R=U(2), "
                "C=min(1, sqrt(n)/50), and Pi=Acc@5 (Markov, vehicle-split)."
            ),
            "weights": {"w_N": 0.58, "w_D": 0.31, "w_Pi": 0.11},
            "density_reference_km2": 0.10,
            "density_sigmoid_k": 1.5,
            "_recalculation_needed": False,
        },
    }

    for region in sim_regions:
        camera_path = data_raw / f"{region}_cameras.geojson"
        sim_path = data_processed / f"road_trajectories_{region}.pkl"

        if not camera_path.exists():
            raise FileNotFoundError(camera_path)
        if not sim_path.exists():
            raise FileNotFoundError(sim_path)

        # Cameras and area
        camera_gdf = gpd.read_file(camera_path)
        n_cameras = int(len(camera_gdf))
        area_km2 = convex_hull_area_km2(camera_path)
        density_km2 = (n_cameras / area_km2) if area_km2 > 0 else 0.0

        # Simulation outputs
        data = pickle.loads(sim_path.read_bytes())
        trajectories = data.get("trajectories", [])
        obs = data.get("observation_stats", {})
        total_trips = len(data.get("trip_metadata", []))
        network_stats = data.get("network_stats", {})

        # Uniqueness U(2..5)
        u = {}
        for k in [2, 3, 4, 5]:
            uk = compute_random_point_uniqueness_bootstrap_mean(
                trajectories=trajectories, k=k, n_bootstrap=args.bootstrap, seed=args.seed
            )
            u[f"u{k}"] = uk["uniqueness"]
            u[f"u{k}_ci_lower"] = uk["ci_lower"]
            u[f"u{k}_ci_upper"] = uk["ci_upper"]
            u[f"n_valid_u{k}"] = uk["n_valid"]

        # Predictability (Markov)
        pred = compute_markov_accuracy(trajectories=trajectories, order=args.markov_order, seed=args.seed)

        output["regions"][region] = round_region_values(
            {
                "cameras": n_cameras,
                "area_km2": area_km2,
                "density_km2": density_km2,
                "p_obs_1": float(obs.get("p_at_least_1", 0.0)),
                "p_obs_2": float(obs.get("p_at_least_2", 0.0)),
                "avg_hits_per_trip": float(obs.get("avg_hits_per_trip", 0.0)),
                "acc1": pred["acc1"],
                "acc5": pred["acc5"],
                "pred_coverage": pred["pred_coverage"],
                "n_predictions": pred["n_predictions"],
                "u2": u["u2"],
                "u2_ci_lower": u["u2_ci_lower"],
                "u2_ci_upper": u["u2_ci_upper"],
                "u3": u["u3"],
                "u3_ci_lower": u["u3_ci_lower"],
                "u3_ci_upper": u["u3_ci_upper"],
                "u4": u["u4"],
                "u4_ci_lower": u["u4_ci_lower"],
                "u4_ci_upper": u["u4_ci_upper"],
                "u5": u["u5"],
                "u5_ci_lower": u["u5_ci_lower"],
                "u5_ci_upper": u["u5_ci_upper"],
                "n_valid_u2": u["n_valid_u2"],
                "n_valid_u3": u["n_valid_u3"],
                "n_valid_u4": u["n_valid_u4"],
                "n_valid_u5": u["n_valid_u5"],
                "total_trips": total_trips,
                "n_road_nodes": int(network_stats.get("n_road_nodes", 0)),
                "category": categories.get(region, "Simulated"),
                "data_type": "Road-network simulated",
            }
        )

    # Chicago validation (optional, but included if present)
    chicago_path = base_dir / "results" / "chicago_validation" / "chicago_validation_results.json"
    chicago_cameras_path = data_raw / "chicago_cameras.geojson"
    if chicago_path.exists() and chicago_cameras_path.exists():
        chicago = json.loads(chicago_path.read_text(encoding="utf-8"))
        n_cameras_total = int(chicago.get("cameras", {}).get("n_cameras", 0))
        n_connected = int(chicago.get("cameras", {}).get("topology", {}).get("n_cameras", 0))
        area_km2 = convex_hull_area_km2(chicago_cameras_path)

        pred = chicago.get("real_predictability", {})
        output["regions"]["chicago"] = round_region_values(
            {
                # Chicago is used for real-world demonstration and is conditional on observation.
                # Use the road-network-connected camera subset for comparability with camera-hit sequences.
                "cameras": n_connected,
                "cameras_total": n_cameras_total,
                "cameras_connected": n_connected,
                "area_km2": area_km2,
                "density_km2": (n_connected / area_km2) if area_km2 > 0 else 0.0,
                "density_km2_total": (n_cameras_total / area_km2) if area_km2 > 0 else 0.0,
                "acc1": float(pred.get("accuracy_at_1", 0.0)),
                "acc5": float(pred.get("accuracy_at_5", 0.0)),
                "skill": float(pred.get("skill_score", 0.0)),
                "category": "Validation",
                "data_type": "Real taxi",
                "_note": "Chicago used for real-world demonstration only. Values are conditional on observation.",
            }
        )

    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
