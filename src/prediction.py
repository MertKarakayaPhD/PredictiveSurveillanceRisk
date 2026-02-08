"""
Mobility prediction models.

Implements Markov chain predictors and baseline models for
next-location prediction.
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from .simulation import Trajectory


class BasePredictor(ABC):
    """Abstract base class for location predictors."""

    @abstractmethod
    def fit(self, trajectories: list[Trajectory]) -> None:
        """Train the predictor on trajectory data."""
        pass

    @abstractmethod
    def predict(self, history: list[int]) -> dict[int, float]:
        """
        Predict probability distribution over next locations.

        Args:
            history: Recent camera observations

        Returns:
            Dict mapping camera_id -> probability
        """
        pass

    def predict_top_k(
        self, history: list[int], k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Return top-k most likely next locations.

        Args:
            history: Recent camera observations
            k: Number of predictions to return

        Returns:
            List of (camera_id, probability) tuples sorted by probability
        """
        probs = self.predict(history)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return sorted_probs[:k]


class MarkovPredictor(BasePredictor):
    """
    Higher-order Markov chain for next-location prediction.

    Given a sequence of k camera observations, predicts the next camera
    based on learned transition probabilities.
    """

    def __init__(self, order: int = 2):
        """
        Initialize Markov predictor.

        Args:
            order: Order of the Markov chain (history length to consider)
        """
        self.order = order
        self.transitions: dict[tuple, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.state_counts: dict[tuple, int] = defaultdict(int)

    def fit(self, trajectories: list[Trajectory]) -> None:
        """
        Learn transition probabilities from trajectories.

        Args:
            trajectories: Training trajectories
        """
        self.transitions.clear()
        self.state_counts.clear()

        for traj in trajectories:
            seq = traj.camera_sequence
            for i in range(len(seq) - self.order):
                state = tuple(seq[i : i + self.order])
                next_loc = seq[i + self.order]
                self.transitions[state][next_loc] += 1
                self.state_counts[state] += 1

    def predict(self, history: list[int]) -> dict[int, float]:
        """
        Predict probability distribution over next locations.

        Args:
            history: Recent camera observations

        Returns:
            Dict mapping camera_id -> probability
        """
        if len(history) < self.order:
            return {}

        state = tuple(history[-self.order :])
        total = self.state_counts[state]

        if total == 0:
            return {}

        return {
            loc: count / total for loc, count in self.transitions[state].items()
        }


class FrequencyPredictor(BasePredictor):
    """
    Baseline predictor using overall visit frequency.

    Ignores history and predicts based on how often each
    camera appears in the training data.
    """

    def __init__(self):
        self.frequencies: dict[int, float] = {}

    def fit(self, trajectories: list[Trajectory]) -> None:
        """Learn visit frequencies from trajectories."""
        all_cameras = []
        for traj in trajectories:
            all_cameras.extend(traj.camera_sequence)

        total = len(all_cameras)
        if total == 0:
            self.frequencies = {}
            return

        counts = Counter(all_cameras)
        self.frequencies = {cam: count / total for cam, count in counts.items()}

    def predict(self, history: list[int]) -> dict[int, float]:
        """Return visit frequency distribution (ignores history)."""
        return self.frequencies.copy()


class UniformPredictor(BasePredictor):
    """
    Random baseline predictor.

    Predicts uniformly over all cameras seen in training.
    """

    def __init__(self):
        self.cameras: set[int] = set()

    def fit(self, trajectories: list[Trajectory]) -> None:
        """Learn set of cameras from trajectories."""
        self.cameras = set()
        for traj in trajectories:
            self.cameras.update(traj.camera_sequence)

    def predict(self, history: list[int]) -> dict[int, float]:
        """Return uniform distribution over all cameras."""
        if not self.cameras:
            return {}
        prob = 1.0 / len(self.cameras)
        return {cam: prob for cam in self.cameras}


class NeighborPredictor(BasePredictor):
    """
    Predictor using network structure.

    Given the last camera, predicts uniformly over neighbors
    in the camera network.
    """

    def __init__(self, camera_graph):
        """
        Initialize with camera network.

        Args:
            camera_graph: NetworkX DiGraph of camera network
        """
        self.graph = camera_graph
        self.cameras: set[int] = set()

    def fit(self, trajectories: list[Trajectory]) -> None:
        """Learn set of cameras from trajectories."""
        self.cameras = set()
        for traj in trajectories:
            self.cameras.update(traj.camera_sequence)

    def predict(self, history: list[int]) -> dict[int, float]:
        """Predict uniformly over neighbors of last camera."""
        if not history:
            return {}

        last_cam = history[-1]
        if last_cam not in self.graph:
            return {}

        neighbors = list(self.graph.successors(last_cam))
        if not neighbors:
            return {}

        prob = 1.0 / len(neighbors)
        return {n: prob for n in neighbors}


def evaluate_predictor(
    predictor: BasePredictor,
    test_trajectories: list[Trajectory],
    k_values: list[int] | None = None,
    min_history: int | None = None,
    camera_graph=None,
) -> dict:
    """
    Evaluate prediction accuracy.

    Args:
        predictor: Trained predictor
        test_trajectories: Test trajectories
        k_values: Values of k for accuracy@k (default: [1, 3, 5])
        min_history: Minimum history length required for prediction
                    (default: predictor.order if available, else 1)
        camera_graph: Optional NetworkX graph for computing topology metrics

    Returns:
        Dict with accuracy@k metrics and statistics
    """
    if k_values is None:
        k_values = [1, 3, 5]

    if min_history is None:
        min_history = getattr(predictor, "order", 1)

    results = {f"accuracy@{k}": [] for k in k_values}
    n_predictions = 0
    n_no_prediction = 0

    # Track neighbor counts for topology-normalized metrics
    neighbor_counts = []

    for traj in test_trajectories:
        seq = traj.camera_sequence

        for i in range(min_history, len(seq)):
            history = seq[:i]
            actual_next = seq[i]

            predictions = predictor.predict_top_k(history, max(k_values))

            if not predictions:
                n_no_prediction += 1
                continue

            n_predictions += 1
            predicted_locs = [loc for loc, prob in predictions]

            # Track number of possible next locations for skill computation
            n_possible = len(predictions)
            if camera_graph is not None and len(history) > 0:
                last_cam = history[-1]
                if last_cam in camera_graph:
                    n_possible = max(1, camera_graph.out_degree(last_cam))
            neighbor_counts.append(n_possible)

            for k in k_values:
                hit = actual_next in predicted_locs[:k]
                results[f"accuracy@{k}"].append(int(hit))

    # Compute means
    final_results = {}
    for key, vals in results.items():
        if vals:
            final_results[key] = np.mean(vals)
        else:
            final_results[key] = 0.0

    final_results["n_predictions"] = n_predictions
    final_results["n_no_prediction"] = n_no_prediction
    final_results["coverage"] = (
        n_predictions / (n_predictions + n_no_prediction)
        if (n_predictions + n_no_prediction) > 0
        else 0.0
    )

    # Compute topology metrics if we have neighbor data
    if neighbor_counts:
        avg_neighbors = np.mean(neighbor_counts)
        final_results["avg_neighbors"] = avg_neighbors

        # Compute skill scores for each k
        # Skill = (Acc - Random) / (Perfect - Random)
        # Random baseline for Acc@k is min(k, avg_neighbors) / avg_neighbors
        for k in k_values:
            acc = final_results[f"accuracy@{k}"]
            random_baseline = min(k, avg_neighbors) / avg_neighbors if avg_neighbors > 0 else 0
            perfect = 1.0

            if perfect > random_baseline:
                skill = (acc - random_baseline) / (perfect - random_baseline)
            else:
                skill = 0.0

            final_results[f"skill@{k}"] = skill
            final_results[f"random_baseline@{k}"] = random_baseline

    return final_results


def compute_skill_scores(
    accuracy: float,
    avg_neighbors: float,
    k: int = 5,
) -> dict:
    """
    Compute topology-normalized accuracy (skill score).

    The skill score measures how much better the predictor performs
    compared to a random baseline that accounts for topology constraints.

    Formula:
        Skill = (Acc - Random) / (Perfect - Random)

    where:
        - Random = min(k, avg_neighbors) / avg_neighbors
        - Perfect = 1.0

    This addresses Reviewer 2's concern that sparse networks with few
    neighbors per camera will have high raw accuracy simply because
    there are few options to choose from.

    Args:
        accuracy: Raw prediction accuracy (0-1)
        avg_neighbors: Average number of outgoing edges per camera
        k: Value of k for Acc@k metric

    Returns:
        Dict with skill score and components
    """
    # Random baseline: probability of hitting correct answer by chance
    # For Acc@k, this is min(k, n_options) / n_options
    random_baseline = min(k, avg_neighbors) / avg_neighbors if avg_neighbors > 0 else 0
    perfect = 1.0

    # Skill score
    if perfect > random_baseline:
        skill = (accuracy - random_baseline) / (perfect - random_baseline)
    else:
        skill = 0.0

    # Topology penalty: log(avg_neighbors) penalizes networks with few neighbors
    # where prediction is trivially easy
    topology_penalty = np.log(max(avg_neighbors, 1))

    # Topology-normalized accuracy
    topology_normalized = skill * topology_penalty

    return {
        "accuracy": accuracy,
        "avg_neighbors": avg_neighbors,
        "k": k,
        "random_baseline": random_baseline,
        "skill_score": skill,
        "topology_penalty": topology_penalty,
        "topology_normalized_accuracy": topology_normalized,
        "interpretation": _interpret_skill(skill),
    }


def _interpret_skill(skill: float) -> str:
    """Interpret skill score value."""
    if skill >= 0.8:
        return "Excellent: Predictor greatly exceeds random baseline"
    elif skill >= 0.6:
        return "Good: Predictor substantially exceeds random baseline"
    elif skill >= 0.4:
        return "Moderate: Predictor meaningfully exceeds random baseline"
    elif skill >= 0.2:
        return "Weak: Predictor slightly exceeds random baseline"
    elif skill > 0:
        return "Marginal: Predictor barely exceeds random baseline"
    else:
        return "None: Predictor at or below random baseline"


def compare_predictors(
    predictors: dict[str, BasePredictor],
    train_trajectories: list[Trajectory],
    test_trajectories: list[Trajectory],
    k_values: list[int] | None = None,
) -> dict[str, dict]:
    """
    Compare multiple predictors on the same data.

    Args:
        predictors: Dict mapping predictor name to predictor instance
        train_trajectories: Training data
        test_trajectories: Test data
        k_values: Values of k for accuracy@k

    Returns:
        Dict mapping predictor name to evaluation results
    """
    results = {}

    for name, predictor in predictors.items():
        print(f"Evaluating {name}...")
        predictor.fit(train_trajectories)
        results[name] = evaluate_predictor(predictor, test_trajectories, k_values)

    return results


def cross_validate(
    predictor_class: type,
    trajectories: list[Trajectory],
    n_folds: int = 5,
    k_values: list[int] | None = None,
    seed: int = 42,
    **predictor_kwargs,
) -> dict:
    """
    Perform k-fold cross-validation.

    Args:
        predictor_class: Class of predictor to evaluate
        trajectories: All trajectories
        n_folds: Number of folds
        k_values: Values of k for accuracy@k
        seed: Random seed
        **predictor_kwargs: Arguments to pass to predictor constructor

    Returns:
        Dict with mean and std of metrics across folds
    """
    if k_values is None:
        k_values = [1, 3, 5]

    rng = np.random.default_rng(seed)

    # Group by vehicle and shuffle
    by_vehicle = defaultdict(list)
    for traj in trajectories:
        by_vehicle[traj.vehicle_id].append(traj)

    vehicle_ids = list(by_vehicle.keys())
    rng.shuffle(vehicle_ids)

    # Create folds
    fold_size = len(vehicle_ids) // n_folds
    fold_results = {f"accuracy@{k}": [] for k in k_values}

    for fold in range(n_folds):
        # Split vehicles into train/test
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(vehicle_ids)
        test_vehicles = set(vehicle_ids[test_start:test_end])

        train = []
        test = []
        for vid, trajs in by_vehicle.items():
            if vid in test_vehicles:
                test.extend(trajs)
            else:
                train.extend(trajs)

        # Train and evaluate
        predictor = predictor_class(**predictor_kwargs)
        predictor.fit(train)
        results = evaluate_predictor(predictor, test, k_values)

        for k in k_values:
            fold_results[f"accuracy@{k}"].append(results[f"accuracy@{k}"])

    # Aggregate results
    final = {}
    for key, vals in fold_results.items():
        final[f"{key}_mean"] = np.mean(vals)
        final[f"{key}_std"] = np.std(vals)

    return final


def evaluate_with_baselines(
    main_predictor: BasePredictor,
    train_trajectories: list[Trajectory],
    test_trajectories: list[Trajectory],
    camera_graph=None,
    k_values: list[int] | None = None,
) -> dict:
    """
    Evaluate a predictor against frequency and uniform baselines.

    This addresses Reviewer 2's concern that Markov prediction accuracy
    should be compared against proper baselines, not just random chance.

    Computes:
    1. Main predictor accuracy (e.g., Markov)
    2. Frequency baseline: predicts most common next cameras
    3. Uniform baseline: predicts uniformly at random
    4. Skill scores relative to each baseline

    Args:
        main_predictor: The predictor to evaluate (e.g., MarkovPredictor)
        train_trajectories: Training data
        test_trajectories: Test data
        camera_graph: Optional graph for topology metrics
        k_values: Values of k for accuracy@k

    Returns:
        Dict with comprehensive comparison metrics
    """
    if k_values is None:
        k_values = [1, 3, 5]

    # Train all predictors
    main_predictor.fit(train_trajectories)

    frequency_predictor = FrequencyPredictor()
    frequency_predictor.fit(train_trajectories)

    uniform_predictor = UniformPredictor()
    uniform_predictor.fit(train_trajectories)

    # Evaluate all
    main_results = evaluate_predictor(
        main_predictor, test_trajectories, k_values, camera_graph=camera_graph
    )
    freq_results = evaluate_predictor(
        frequency_predictor, test_trajectories, k_values, camera_graph=camera_graph
    )
    uniform_results = evaluate_predictor(
        uniform_predictor, test_trajectories, k_values, camera_graph=camera_graph
    )

    # Compute skill scores relative to baselines
    results = {
        "main": main_results,
        "frequency_baseline": freq_results,
        "uniform_baseline": uniform_results,
        "skill_vs_frequency": {},
        "skill_vs_uniform": {},
        "improvement_over_frequency": {},
        "improvement_over_uniform": {},
    }

    for k in k_values:
        main_acc = main_results[f"accuracy@{k}"]
        freq_acc = freq_results[f"accuracy@{k}"]
        uniform_acc = uniform_results[f"accuracy@{k}"]

        # Skill vs frequency baseline
        if 1.0 > freq_acc:
            skill_freq = (main_acc - freq_acc) / (1.0 - freq_acc)
        else:
            skill_freq = 0.0
        results["skill_vs_frequency"][f"@{k}"] = skill_freq

        # Skill vs uniform baseline
        if 1.0 > uniform_acc:
            skill_uniform = (main_acc - uniform_acc) / (1.0 - uniform_acc)
        else:
            skill_uniform = 0.0
        results["skill_vs_uniform"][f"@{k}"] = skill_uniform

        # Absolute improvements
        results["improvement_over_frequency"][f"@{k}"] = main_acc - freq_acc
        results["improvement_over_uniform"][f"@{k}"] = main_acc - uniform_acc

    # Summary interpretation
    acc_5 = main_results.get("accuracy@5", 0)
    freq_5 = freq_results.get("accuracy@5", 0)
    uniform_5 = uniform_results.get("accuracy@5", 0)

    results["summary"] = {
        "main_accuracy@5": acc_5,
        "frequency_accuracy@5": freq_5,
        "uniform_accuracy@5": uniform_5,
        "lift_over_frequency": (acc_5 / freq_5 - 1) * 100 if freq_5 > 0 else float('inf'),
        "lift_over_uniform": (acc_5 / uniform_5 - 1) * 100 if uniform_5 > 0 else float('inf'),
        "interpretation": _interpret_baseline_comparison(acc_5, freq_5, uniform_5),
    }

    return results


def _interpret_baseline_comparison(main: float, freq: float, uniform: float) -> str:
    """Interpret predictor performance vs baselines."""
    if main <= uniform:
        return "Poor: Main predictor at or below random baseline"
    elif main <= freq:
        return "Weak: Main predictor below frequency baseline"
    elif main - freq < 0.05:
        return "Marginal: Main predictor barely exceeds frequency baseline"
    elif main - freq < 0.15:
        return "Moderate: Main predictor meaningfully exceeds frequency baseline"
    elif main - freq < 0.30:
        return "Good: Main predictor substantially exceeds frequency baseline"
    else:
        return "Excellent: Main predictor greatly exceeds frequency baseline"
