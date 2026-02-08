"""Tests for prediction module."""

import pytest
import numpy as np

from src.simulation import Trajectory
from src.prediction import (
    MarkovPredictor,
    FrequencyPredictor,
    UniformPredictor,
    evaluate_predictor,
)


class TestMarkovPredictor:
    """Tests for Markov chain predictor."""

    def test_fit_and_predict(self):
        """Predictor learns and predicts transitions."""
        trajs = [
            Trajectory("v1", [1, 2, 3], [0, 1, 2]),
            Trajectory("v1", [1, 2, 3], [0, 1, 2]),
            Trajectory("v2", [1, 2, 4], [0, 1, 2]),
        ]

        predictor = MarkovPredictor(order=1)
        predictor.fit(trajs)

        # After seeing [2], should predict 3 (2/3) and 4 (1/3)
        probs = predictor.predict([2])

        assert 3 in probs
        assert 4 in probs
        assert abs(probs[3] - 2/3) < 0.01
        assert abs(probs[4] - 1/3) < 0.01

    def test_order_2(self):
        """Order-2 Markov uses 2-step history."""
        trajs = [
            Trajectory("v1", [1, 2, 3], [0, 1, 2]),
            Trajectory("v2", [1, 2, 4], [0, 1, 2]),
        ]

        predictor = MarkovPredictor(order=2)
        predictor.fit(trajs)

        # After seeing [1, 2], should predict 3 or 4
        probs = predictor.predict([1, 2])

        assert 3 in probs or 4 in probs

    def test_insufficient_history(self):
        """Returns empty dict if history too short."""
        trajs = [Trajectory("v1", [1, 2, 3], [0, 1, 2])]

        predictor = MarkovPredictor(order=2)
        predictor.fit(trajs)

        probs = predictor.predict([1])  # Only 1 item, need 2

        assert probs == {}

    def test_unknown_state(self):
        """Returns empty dict for unseen state."""
        trajs = [Trajectory("v1", [1, 2, 3], [0, 1, 2])]

        predictor = MarkovPredictor(order=1)
        predictor.fit(trajs)

        probs = predictor.predict([99])  # Never seen

        assert probs == {}

    def test_predict_top_k(self):
        """Top-k predictions are sorted by probability."""
        trajs = [
            Trajectory("v1", [1, 2], [0, 1]),
            Trajectory("v2", [1, 2], [0, 1]),
            Trajectory("v3", [1, 3], [0, 1]),
        ]

        predictor = MarkovPredictor(order=1)
        predictor.fit(trajs)

        top_k = predictor.predict_top_k([1], k=2)

        assert len(top_k) == 2
        assert top_k[0][0] == 2  # Most frequent
        assert top_k[0][1] > top_k[1][1]  # Sorted by probability


class TestFrequencyPredictor:
    """Tests for frequency baseline predictor."""

    def test_predict_by_frequency(self):
        """Predictions match visit frequency."""
        trajs = [
            Trajectory("v1", [1, 1, 1, 2], [0, 1, 2, 3]),  # 1 appears 3 times, 2 once
        ]

        predictor = FrequencyPredictor()
        predictor.fit(trajs)

        probs = predictor.predict([99])  # History doesn't matter

        assert abs(probs[1] - 0.75) < 0.01
        assert abs(probs[2] - 0.25) < 0.01


class TestUniformPredictor:
    """Tests for uniform baseline predictor."""

    def test_uniform_distribution(self):
        """All cameras have equal probability."""
        trajs = [
            Trajectory("v1", [1, 2, 3], [0, 1, 2]),
        ]

        predictor = UniformPredictor()
        predictor.fit(trajs)

        probs = predictor.predict([99])  # History doesn't matter

        assert len(probs) == 3
        assert all(abs(p - 1/3) < 0.01 for p in probs.values())


class TestEvaluatePredictor:
    """Tests for predictor evaluation."""

    def test_accuracy_calculation(self):
        """Accuracy is computed correctly."""
        # Perfect predictor scenario
        trajs = [
            Trajectory("v1", [1, 2, 3, 4], [0, 1, 2, 3]),
            Trajectory("v1", [1, 2, 3, 4], [0, 1, 2, 3]),
        ]

        predictor = MarkovPredictor(order=1)
        predictor.fit(trajs)

        # Test on same data (should have high accuracy)
        results = evaluate_predictor(predictor, trajs, k_values=[1, 5])

        assert "accuracy@1" in results
        assert "accuracy@5" in results
        assert 0 <= results["accuracy@1"] <= 1
        assert results["accuracy@5"] >= results["accuracy@1"]

    def test_coverage_metric(self):
        """Coverage tracks prediction availability."""
        trajs = [
            Trajectory("v1", [1, 2, 3], [0, 1, 2]),
        ]

        predictor = MarkovPredictor(order=1)
        predictor.fit(trajs)

        results = evaluate_predictor(predictor, trajs)

        assert "coverage" in results
        assert 0 <= results["coverage"] <= 1
