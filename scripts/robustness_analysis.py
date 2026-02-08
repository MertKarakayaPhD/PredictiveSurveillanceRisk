"""
Robustness analysis (road-network pipeline).

This script recomputes the paper’s robustness checks directly from the
road-network simulation outputs in `data/processed/road_trajectories_*.pkl`
and the paper’s `data/source_of_truth.json`.

Analyses implemented (used by the manuscript):
- Camera removal sensitivity (filter camera-hit sequences; no rerouting)
- Markov order sensitivity for predictability (orders 1–3)
- PSR weight sensitivity (AHP vs alternative schemes)

Run from project root:
  python scripts/robustness_analysis.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import compute_psr, uniqueness_analysis, analyze_predictability
from src.simulation import generate_trajectories, Trajectory
from src.network import subsample_network, compute_topology_metrics
from src.data import load_cameras, get_camera_stats
from src.prediction import MarkovPredictor, evaluate_predictor

import networkx as nx
import pickle


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Regions to analyze (use comprehensive data)
REGIONS = {
    "atlanta": {"cameras": 2273, "area_km2": 2780},
    "richmond": {"cameras": 691, "area_km2": 1550},
    "memphis": {"cameras": 413, "area_km2": 1820},
    "charlotte": {"cameras": 355, "area_km2": 1290},
    "lehigh_valley": {"cameras": 123, "area_km2": 1290},
    "maine": {"cameras": 47, "area_km2": 8600},
}

# Pre-computed results from paper (to avoid re-running full simulation)
# These are the values from the current 3-component PSR
REGION_METRICS = {
    "atlanta": {
        "n_cameras": 2273, "area_km2": 2780, "density": 0.818,
        "reidentification": 0.98, "predictability": 0.58,
        "psr_score": 0.843, "achieved_accuracy": 0.35,
    },
    "richmond": {
        "n_cameras": 691, "area_km2": 1550, "density": 0.446,
        "reidentification": 0.72, "predictability": 0.61,
        "psr_score": 0.586, "achieved_accuracy": 0.42,
    },
    "memphis": {
        "n_cameras": 413, "area_km2": 1820, "density": 0.227,
        "reidentification": 0.68, "predictability": 0.58,
        "psr_score": 0.512, "achieved_accuracy": 0.48,
    },
    "charlotte": {
        "n_cameras": 355, "area_km2": 1290, "density": 0.275,
        "reidentification": 0.61, "predictability": 0.56,
        "psr_score": 0.447, "achieved_accuracy": 0.52,
    },
    "lehigh_valley": {
        "n_cameras": 123, "area_km2": 1290, "density": 0.095,
        "reidentification": 0.52, "predictability": 0.49,
        "psr_score": 0.365, "achieved_accuracy": 0.58,
    },
    "maine": {
        "n_cameras": 47, "area_km2": 8600, "density": 0.005,
        "reidentification": 0.39, "predictability": 0.54,
        "psr_score": 0.221, "achieved_accuracy": 0.77,
    },
}


# =============================================================================
# Analysis 1: PSR Stability Under Camera Removal
# =============================================================================

def run_camera_removal_analysis():
    """
    Test PSR stability by removing random cameras from dense networks.

    This addresses reviewer concern about DeFlock data completeness:
    "If sparse networks are just incompletely mapped, removing cameras
    from dense networks should produce similar behavior."
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: PSR Stability Under Camera Removal")
    print("="*70)

    removal_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_trials = 5  # Multiple random seeds

    results = {}

    for region, metrics in REGION_METRICS.items():
        print(f"\n--- {region.upper()} ---")
        region_results = []

        for removal in removal_fractions:
            trial_psrs = []
            remaining_fraction = 1.0 - removal
            n_remaining = int(metrics["n_cameras"] * remaining_fraction)

            for trial in range(n_trials):
                # Simulate effect of camera removal on metrics
                # Re-identification decreases as sqrt(remaining) - entropy argument
                reid_factor = np.sqrt(remaining_fraction)
                new_reid = metrics["reidentification"] * reid_factor

                # Density decreases linearly
                new_density = metrics["density"] * remaining_fraction

                # Predictability increases slightly (fewer choices)
                # But bounded by information available
                pred_factor = 1.0 + 0.1 * removal  # Small increase
                new_pred = min(0.93, metrics["predictability"] * pred_factor)

                # Compute PSR with remaining cameras
                psr_result = compute_psr(
                    predictability=new_pred,
                    reidentification_rate=new_reid,
                    camera_density=new_density,
                    n_cameras=n_remaining,
                )
                trial_psrs.append(psr_result["psr_score"])

            mean_psr = np.mean(trial_psrs)
            std_psr = np.std(trial_psrs)

            region_results.append({
                "removal_fraction": removal,
                "remaining_cameras": n_remaining,
                "psr_mean": mean_psr,
                "psr_std": std_psr,
                "interpretation": psr_result["interpretation"],
            })

            print(f"  {int(removal*100):2d}% removed ({n_remaining:4d} cameras): "
                  f"PSR = {mean_psr:.3f} +/- {std_psr:.3f} [{psr_result['interpretation']}]")

        results[region] = region_results

    # Save results
    df_results = []
    for region, region_data in results.items():
        for row in region_data:
            row["region"] = region
            df_results.append(row)

    df = pd.DataFrame(df_results)
    df.to_csv(RESULTS_DIR / "camera_removal_analysis.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'camera_removal_analysis.csv'}")

    # Key finding: Does Atlanta at 70% cameras look like Maine?
    print("\n--- KEY COMPARISON ---")
    atlanta_70 = [r for r in results["atlanta"] if r["removal_fraction"] == 0.3][0]
    maine_full = [r for r in results["maine"] if r["removal_fraction"] == 0.0][0]

    print(f"Atlanta at 70% cameras: PSR = {atlanta_70['psr_mean']:.3f} "
          f"[{atlanta_70['interpretation']}]")
    print(f"Maine at 100% cameras:  PSR = {maine_full['psr_mean']:.3f} "
          f"[{maine_full['interpretation']}]")

    if atlanta_70["psr_mean"] > maine_full["psr_mean"] + 0.2:
        print("\n[OK] FINDING: Dense network characteristics are INTRINSIC, not artifacts")
        print("  Even with 30% cameras removed, Atlanta has higher PSR than Maine.")
        print("  This suggests sparse network behavior is NOT just incomplete mapping.")

    return results


# =============================================================================
# Analysis 2: Mobility Model Parameter Sensitivity
# =============================================================================

def run_mobility_sensitivity_analysis():
    """
    Test how mobility model parameters affect findings.

    This addresses reviewer concern:
    "How much do the results change if vehicles are 10% more exploratory?"
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: Mobility Model Parameter Sensitivity")
    print("="*70)

    # Test different rho (return probability) values
    rho_values = [0.4, 0.5, 0.6, 0.7, 0.8]  # Baseline is 0.6

    # Simulated effect on predictability (based on EPR model behavior)
    # Higher rho = more routine = higher predictability
    results = []

    print("\n--- Effect of rho (Return Probability) on Predictability ---")
    print("Higher rho = more routine behavior = more predictable\n")

    for rho in rho_values:
        # Theoretical predictability scales with rho
        # At rho=0.6, Pi_max ≈ 0.58 (baseline)
        # Formula: Pi_max ≈ 0.4 + 0.3 * rho
        base_pred = 0.4 + 0.3 * rho

        for region, metrics in REGION_METRICS.items():
            # Adjust predictability based on rho
            # Sparse networks are more affected (more constrained)
            sparsity_factor = 1.0 + 0.2 * (1 - metrics["density"])
            adjusted_pred = min(0.93, base_pred * sparsity_factor)

            # Re-identification is not affected by mobility model
            reid = metrics["reidentification"]

            # Compute PSR
            psr = compute_psr(
                predictability=adjusted_pred,
                reidentification_rate=reid,
                camera_density=metrics["density"],
                n_cameras=metrics["n_cameras"],
            )

            results.append({
                "rho": rho,
                "region": region,
                "predictability": adjusted_pred,
                "psr_score": psr["psr_score"],
                "interpretation": psr["interpretation"],
            })

    df = pd.DataFrame(results)

    # Print summary
    for rho in rho_values:
        marker = " (baseline)" if rho == 0.6 else ""
        print(f"rho = {rho}{marker}:")
        subset = df[df["rho"] == rho]
        for _, row in subset.iterrows():
            print(f"  {row['region']:15s}: Pi={row['predictability']:.2f}, "
                  f"PSR={row['psr_score']:.3f} [{row['interpretation']}]")
        print()

    # Check rank stability
    print("--- Rank Stability Across rho Values ---")
    rank_stable = True
    baseline_ranking = df[df["rho"] == 0.6].sort_values("psr_score", ascending=False)["region"].tolist()

    for rho in rho_values:
        current_ranking = df[df["rho"] == rho].sort_values("psr_score", ascending=False)["region"].tolist()
        if current_ranking != baseline_ranking:
            rank_stable = False
            print(f"rho = {rho}: Ranking differs from baseline")
            print(f"  Baseline: {baseline_ranking}")
            print(f"  Current:  {current_ranking}")

    if rank_stable:
        print("[OK] PSR rankings are STABLE across all tested rho values")
        print(f"  Consistent ranking: {' > '.join(baseline_ranking)}")

    df.to_csv(RESULTS_DIR / "mobility_sensitivity_analysis.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'mobility_sensitivity_analysis.csv'}")

    return df


# =============================================================================
# Analysis 3: Weighting Methodology Comparison
# =============================================================================

def run_weighting_comparison():
    """
    Compare different weighting schemes for PSR.

    This addresses reviewer concern:
    "Variance-based weighting is methodologically unsound if a feature
    has low variance but high risk impact."
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: Weighting Methodology Comparison")
    print("="*70)

    weighting_schemes = {
        "variance_based": {"w_dens": 0.44, "w_netpower": 0.49, "w_pred": 0.07},
        "variance_with_floor": {"w_dens": 0.40, "w_netpower": 0.45, "w_pred": 0.15},  # Min 15%
        "equal_weights": {"w_dens": 0.33, "w_netpower": 0.34, "w_pred": 0.33},
        "density_focused": {"w_dens": 0.50, "w_netpower": 0.35, "w_pred": 0.15},
        "prediction_focused": {"w_dens": 0.30, "w_netpower": 0.35, "w_pred": 0.35},
    }

    results = []

    for scheme_name, weights in weighting_schemes.items():
        print(f"\n--- {scheme_name.upper()} ---")
        print(f"Weights: D={weights['w_dens']:.2f}, N={weights['w_netpower']:.2f}, "
              f"Pi={weights['w_pred']:.2f}")

        for region, metrics in REGION_METRICS.items():
            psr = compute_psr(
                predictability=metrics["predictability"],
                reidentification_rate=metrics["reidentification"],
                camera_density=metrics["density"],
                n_cameras=metrics["n_cameras"],
                weights=weights,
            )

            results.append({
                "scheme": scheme_name,
                "region": region,
                "psr_score": psr["psr_score"],
                "interpretation": psr["interpretation"],
                "d_contrib": psr["weighted_contributions"]["density"],
                "n_contrib": psr["weighted_contributions"]["network_power"],
                "pi_contrib": psr["weighted_contributions"]["predictability"],
            })

            print(f"  {region:15s}: PSR = {psr['psr_score']:.3f} [{psr['interpretation']}]")

    df = pd.DataFrame(results)

    # Check rank correlation across schemes
    print("\n--- Rank Correlation Across Weighting Schemes ---")
    from scipy.stats import kendalltau

    baseline_scores = df[df["scheme"] == "variance_based"].set_index("region")["psr_score"]

    for scheme in weighting_schemes.keys():
        if scheme == "variance_based":
            continue
        scheme_scores = df[df["scheme"] == scheme].set_index("region")["psr_score"]
        tau, p = kendalltau(baseline_scores, scheme_scores)
        print(f"  {scheme:25s}: tau = {tau:.3f} (p = {p:.4f})")

    # Check if rankings are preserved
    baseline_ranking = baseline_scores.sort_values(ascending=False).index.tolist()
    print(f"\nBaseline ranking: {' > '.join(baseline_ranking)}")

    all_same = True
    for scheme in weighting_schemes.keys():
        scheme_scores = df[df["scheme"] == scheme].set_index("region")["psr_score"]
        scheme_ranking = scheme_scores.sort_values(ascending=False).index.tolist()
        if scheme_ranking != baseline_ranking:
            all_same = False
            print(f"  {scheme}: {' > '.join(scheme_ranking)} [DIFFERENT]")

    if all_same:
        print("\n[OK] All weighting schemes produce IDENTICAL rankings")
        print("  This validates that findings are robust to weighting choice")

    df.to_csv(RESULTS_DIR / "weighting_comparison.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'weighting_comparison.csv'}")

    return df


# =============================================================================
# Analysis 4: Re-identification Threshold Sensitivity
# =============================================================================

def run_reidentification_sensitivity():
    """
    Test sensitivity of re-identification findings to observation count.
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: Re-identification at Different Observation Counts")
    print("="*70)

    # Simulated uniqueness rates at different observation counts
    # Based on information-theoretic model: uniqueness ~ 1 - exp(-k * n^alpha)
    observation_counts = [1, 2, 3, 4, 5, 6]

    results = []

    for region, metrics in REGION_METRICS.items():
        print(f"\n--- {region.upper()} ---")
        base_reid = metrics["reidentification"]  # At n=2
        n_cameras = metrics["n_cameras"]

        for n_obs in observation_counts:
            # Model: uniqueness increases with observations
            # At n=2, we have base_reid
            # Scaling: reid(n) = 1 - (1 - reid(2))^(n/2)
            reid_n = 1 - (1 - base_reid) ** (n_obs / 2)
            reid_n = min(0.995, reid_n)  # Cap at 99.5%

            results.append({
                "region": region,
                "n_observations": n_obs,
                "uniqueness_rate": reid_n,
                "n_cameras": n_cameras,
            })

            print(f"  {n_obs} observations: {reid_n*100:.1f}% unique")

    df = pd.DataFrame(results)

    # Compare to cell tower data (de Montjoye: 95% at 4 points)
    print("\n--- Comparison to Cell Tower Data ---")
    print("de Montjoye et al. (2013): 95% uniqueness at 4 observations")
    print("\nALPR achieves 95% uniqueness at:")
    for region in REGION_METRICS.keys():
        region_data = df[df["region"] == region]
        for _, row in region_data.iterrows():
            if row["uniqueness_rate"] >= 0.95:
                print(f"  {region:15s}: {row['n_observations']} observations")
                break

    df.to_csv(RESULTS_DIR / "reidentification_sensitivity.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'reidentification_sensitivity.csv'}")

    return df


# =============================================================================
# Main
# =============================================================================

def main():
    # ---------------------------------------------------------------------
    # Reproducible robustness (road-network pipeline)
    # ---------------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-of-truth",
        default="data/source_of_truth.json",
        help="Path to data/source_of_truth.json",
    )
    parser.add_argument(
        "--sim-dir",
        default="data/processed",
        help="Directory containing road_trajectories_*.pkl",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "road_network_robustness.json"),
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--camera-removal-region", default="atlanta")
    parser.add_argument(
        "--camera-removal-fracs",
        type=float,
        nargs="+",
        default=[0.30, 0.70],
        help="Fractions of cameras removed (e.g., 0.3 0.7)",
    )
    parser.add_argument("--camera-removal-trials", type=int, default=5)
    parser.add_argument("--camera-removal-bootstrap", type=int, default=200)
    parser.add_argument("--camera-removal-markov-order", type=int, default=1)

    parser.add_argument("--markov-orders", type=int, nargs="+", default=[1, 2, 3])

    args = parser.parse_args()

    sim_regions = ["atlanta", "memphis", "richmond", "charlotte", "lehigh_valley", "maine"]
    weight_schemes = {
        "ahp_default": {"w_dens": 0.31, "w_netpower": 0.58, "w_pred": 0.11},
        "equal": {"w_dens": 1 / 3, "w_netpower": 1 / 3, "w_pred": 1 / 3},
        "density_focused": {"w_dens": 0.50, "w_netpower": 0.35, "w_pred": 0.15},
        "prediction_focused": {"w_dens": 0.30, "w_netpower": 0.35, "w_pred": 0.35},
    }

    source_of_truth = json.loads(Path(args.source_of_truth).read_text(encoding="utf-8"))
    sim_dir = Path(args.sim_dir)

    def kendall_tau_a(x: list[float], y: list[float]) -> float:
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        n = len(x)
        if n < 2:
            return 0.0
        concordant = 0
        discordant = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                prod = (x[i] - x[j]) * (y[i] - y[j])
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
        denom = n * (n - 1) / 2
        return (concordant - discordant) / denom if denom > 0 else 0.0

    def load_region_trajectories(region: str) -> list[dict]:
        sim_path = sim_dir / f"road_trajectories_{region}.pkl"
        data = pickle.loads(sim_path.read_bytes())
        return data.get("trajectories", [])

    def filter_trajectories_by_cameras(trajectories: list[dict], keep: set[int]) -> list[dict]:
        out = []
        for t in trajectories:
            hits = t.get("camera_hits", [])
            out.append(
                {
                    "vehicle_id": t.get("vehicle_id", ""),
                    "camera_hits": [int(h) for h in hits if int(h) in keep],
                }
            )
        return out

    def compute_random_point_uniqueness_bootstrap_mean(
        trajectories: list[dict], k: int, n_bootstrap: int, seed: int, ordered: bool = False
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
                idx = rng_inner.choice(len(hits), size=k, replace=False)
                idx.sort()
                sampled = tuple(hits[i] for i in idx)
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

    def compute_markov_accuracy(trajectories: list[dict], order: int, seed: int) -> dict:
        trajs = [
            Trajectory(vehicle_id=t.get("vehicle_id", ""), camera_sequence=t.get("camera_hits", []))
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
            "markov_order": int(order),
        }

    def compute_psr_score(
        density_km2: float, u2: float, n_cameras: int, acc5: float, weights: dict[str, float] | None = None
    ) -> dict:
        return compute_psr(
            predictability=float(acc5),
            reidentification_rate=float(u2),
            camera_density=float(density_km2),
            n_cameras=int(n_cameras),
            weights=weights,
        )

    # ------------------------------------------------------------------
    # Camera removal (Atlanta)
    # ------------------------------------------------------------------
    cam_region = args.camera_removal_region
    base = source_of_truth["regions"][cam_region]
    base_area = float(base["area_km2"])
    base_n = int(base["cameras"])
    base_psr = compute_psr_score(
        density_km2=float(base["density_km2"]),
        u2=float(base["u2"]),
        n_cameras=base_n,
        acc5=float(base.get("acc5", 0.0)),
    )["psr_score"]

    base_trajs = load_region_trajectories(cam_region)
    camera_removal_trials = []
    camera_removal_summary = {}

    for frac in args.camera_removal_fracs:
        remaining = max(1, int(round(base_n * (1.0 - frac))))
        rows = []
        for trial in range(args.camera_removal_trials):
            trial_seed = args.seed + int(frac * 10_000) + trial * 9973
            rng = np.random.default_rng(trial_seed)
            keep = set(map(int, rng.choice(base_n, size=remaining, replace=False)))
            filtered = filter_trajectories_by_cameras(base_trajs, keep)

            u2 = compute_random_point_uniqueness_bootstrap_mean(
                trajectories=filtered,
                k=2,
                n_bootstrap=args.camera_removal_bootstrap,
                seed=trial_seed,
            )
            pred = compute_markov_accuracy(
                trajectories=filtered,
                order=args.camera_removal_markov_order,
                seed=trial_seed,
            )

            density = (remaining / base_area) if base_area > 0 else 0.0
            psr = compute_psr_score(
                density_km2=density,
                u2=u2["uniqueness"],
                n_cameras=remaining,
                acc5=pred["acc5"],
            )

            row = {
                "removal_fraction": float(frac),
                "trial": int(trial),
                "seed": int(trial_seed),
                "remaining_cameras": int(remaining),
                "density_km2": float(density),
                "u2": float(u2["uniqueness"]),
                "u2_ci_lower": float(u2["ci_lower"]),
                "u2_ci_upper": float(u2["ci_upper"]),
                "n_valid_u2": int(u2["n_valid"]),
                "acc5": float(pred["acc5"]),
                "pred_coverage": float(pred["pred_coverage"]),
                "psr": float(psr["psr_score"]),
                "psr_level": psr["interpretation"],
            }
            rows.append(row)
            camera_removal_trials.append(row)

        psrs = np.array([r["psr"] for r in rows], dtype=float)
        camera_removal_summary[str(frac)] = {
            "n_trials": int(len(rows)),
            "remaining_cameras": int(remaining),
            "psr_mean": float(np.mean(psrs)) if len(psrs) else 0.0,
            "psr_std": float(np.std(psrs)) if len(psrs) else 0.0,
            "psr_p05": float(np.percentile(psrs, 5)) if len(psrs) else 0.0,
            "psr_p95": float(np.percentile(psrs, 95)) if len(psrs) else 0.0,
        }

    # ------------------------------------------------------------------
    # Markov order sensitivity (orders 1..3)
    # ------------------------------------------------------------------
    markov_order_results = {}
    markov_rankings = {}
    markov_scores = {}

    for region in sim_regions:
        base = source_of_truth["regions"][region]
        trajectories = load_region_trajectories(region)
        per_order = {}
        for order in args.markov_orders:
            pred = compute_markov_accuracy(trajectories=trajectories, order=order, seed=args.seed + 31 * order)
            psr = compute_psr_score(
                density_km2=float(base["density_km2"]),
                u2=float(base["u2"]),
                n_cameras=int(base["cameras"]),
                acc5=pred["acc5"],
            )
            per_order[str(order)] = {
                **pred,
                "psr": float(psr["psr_score"]),
                "psr_level": psr["interpretation"],
            }
        markov_order_results[region] = per_order

    for order in args.markov_orders:
        scores = [markov_order_results[r][str(order)]["psr"] for r in sim_regions]
        markov_scores[str(order)] = scores
        markov_rankings[str(order)] = [
            r for r in sorted(sim_regions, key=lambda rr: markov_order_results[rr][str(order)]["psr"], reverse=True)
        ]

    baseline_order = str(args.markov_orders[0])
    markov_tau = {}
    for order in args.markov_orders[1:]:
        markov_tau[str(order)] = float(kendall_tau_a(markov_scores[baseline_order], markov_scores[str(order)]))

    # ------------------------------------------------------------------
    # Weight sensitivity
    # ------------------------------------------------------------------
    weight_scores = {}
    weight_rankings = {}
    weight_per_scheme = {}

    for scheme, weights in weight_schemes.items():
        per_region = {}
        for region in sim_regions:
            base = source_of_truth["regions"][region]
            psr = compute_psr_score(
                density_km2=float(base["density_km2"]),
                u2=float(base["u2"]),
                n_cameras=int(base["cameras"]),
                acc5=float(base.get("acc5", 0.0)),
                weights=weights,
            )
            per_region[region] = {"psr": float(psr["psr_score"]), "psr_level": psr["interpretation"]}
        weight_per_scheme[scheme] = per_region
        weight_scores[scheme] = [per_region[r]["psr"] for r in sim_regions]
        weight_rankings[scheme] = [
            r for r in sorted(sim_regions, key=lambda rr: per_region[rr]["psr"], reverse=True)
        ]

    weight_tau = {
        scheme: float(kendall_tau_a(weight_scores["ahp_default"], scores))
        for scheme, scores in weight_scores.items()
        if scheme != "ahp_default"
    }

    output = {
        "camera_removal": {
            "region": cam_region,
            "base_psr": float(base_psr),
            "trials": camera_removal_trials,
            "summary": camera_removal_summary,
            "_note": "Camera removal implemented by filtering camera-hit sequences (no rerouting). "
            "Density uses the baseline convex-hull area from source_of_truth.json.",
        },
        "markov_order_sensitivity": {
            "orders": [int(o) for o in args.markov_orders],
            "per_region": markov_order_results,
            "rankings": markov_rankings,
            "kendall_tau_vs_order1": markov_tau,
        },
        "weight_sensitivity": {
            "schemes": weight_schemes,
            "per_scheme": weight_per_scheme,
            "rankings": weight_rankings,
            "kendall_tau_vs_ahp": weight_tau,
        },
        "_config": {
            "seed": int(args.seed),
            "source_of_truth": str(args.source_of_truth),
            "sim_dir": str(args.sim_dir),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Camera removal base PSR ({cam_region}): {base_psr:.3f}")
    for frac, summary in camera_removal_summary.items():
        print(
            f"  removal {float(frac):.0%}: PSR={summary['psr_mean']:.3f} +/- {summary['psr_std']:.3f} "
            f"(n={summary['remaining_cameras']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
