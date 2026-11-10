"""
test_extended_metrics.py
------------------------
Unit tests for extended statistical metrics and bootstrap confidence intervals
in frontend/model_evaluation.py, covering:
- F2-score (beta=2 recall-prioritized metric)
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Negative Predictive Value (NPV)
- Brier Score & Probability Calibration
- Log Loss
- 95% Stratified Bootstrap Confidence Intervals
- Intelligent Decision Threshold Optimization
"""

import pytest
import numpy as np
import pandas as pd
from frontend.model_evaluation import (
    get_evaluator,
    get_dataset_intelligence,
    get_threshold_recommendations,
    ModelEvaluator,
)


class TestExtendedEvaluationMetrics:
    """Tests the Phase 2 mathematical metric expansions on held-out test data."""

    @classmethod
    def setup_class(cls):
        cls.evaluator = get_evaluator()
        cls.metrics_50 = cls.evaluator.compute_metrics(threshold=0.50)

    def test_01_f2_score_calculation(self):
        """F2 score places 4x more weight on recall than precision: (5*P*R)/(4*P + R)."""
        m = self.metrics_50
        p = m["precision"]  # 0.80
        r = m["recall"]     # 0.50
        expected_f2 = (1 + 2**2) * (p * r) / ((2**2 * p) + r)
        assert pytest.approx(m["f2_score"], rel=1e-3) == expected_f2
        assert 0.0 <= m["f2_score"] <= 1.0

    def test_02_balanced_accuracy_calculation(self):
        """Balanced accuracy is (Sensitivity + Specificity) / 2."""
        m = self.metrics_50
        sensitivity = m["recall"]
        specificity = m["specificity"]
        expected_ba = (sensitivity + specificity) / 2.0
        assert pytest.approx(m["balanced_accuracy"], rel=1e-3) == expected_ba
        assert 0.5 <= m["balanced_accuracy"] <= 1.0

    def test_03_matthews_correlation_coefficient(self):
        """MCC measures quality of binary classification in [-1, 1]."""
        m = self.metrics_50
        assert -1.0 <= m["mcc"] <= 1.0
        # Given TP=4, FP=1, FN=4, TN=999, MCC should be positive and substantial
        assert m["mcc"] > 0.5

    def test_04_negative_predictive_value(self):
        """NPV is TN / (TN + FN). Given 999 TN and 4 FN, NPV ~ 999/1003 ~ 0.9960."""
        m = self.metrics_50
        expected_npv = m["tn"] / (m["tn"] + m["fn"])
        assert pytest.approx(m["npv"], rel=1e-3) == expected_npv
        assert m["npv"] > 0.99

    def test_05_brier_score_and_log_loss(self):
        """Brier score is mean squared error of probability predictions."""
        ev = self.evaluator
        assert 0.0 <= ev.brier_score <= 0.01  # Should be extremely low due to high specificity
        assert ev.log_loss >= 0.0

    def test_06_calibration_curve_output(self):
        """Calibration curve returns empirical probabilities in valid [0, 1] range."""
        ev = self.evaluator
        prob_true = ev.calibration_prob_true
        prob_pred = ev.calibration_prob_pred
        assert len(prob_true) == len(prob_pred)
        assert len(prob_true) > 0
        assert np.all(prob_true >= 0.0) and np.all(prob_true <= 1.0)
        assert np.all(prob_pred >= 0.0) and np.all(prob_pred <= 1.0)


class TestStratifiedBootstrapConfidenceIntervals:
    """Tests the 95% Stratified Bootstrap CI engine."""

    @classmethod
    def setup_class(cls):
        cls.evaluator = get_evaluator()

    def test_07_bootstrap_ci_structure_and_bounds(self):
        """Bootstrap CIs must contain valid 95% confidence intervals for all primary metrics."""
        cis = self.evaluator.compute_bootstrap_confidence_intervals(n_bootstraps=100)
        required_metrics = ["accuracy", "precision", "recall", "f1_score", "f2_score", "roc_auc", "pr_auc"]
        for m in required_metrics:
            assert m in cis, f"Metric {m} missing from confidence intervals"
            ci = cis[m]
            assert "mean" in ci
            assert "ci_lower" in ci
            assert "ci_upper" in ci
            assert ci["ci_lower"] <= ci["ci_upper"], f"Lower bound {ci['ci_lower']} > Upper bound {ci['ci_upper']}"
            assert 0.0 <= ci["ci_lower"] <= 1.0
            assert 0.0 <= ci["ci_upper"] <= 1.0

    def test_08_bootstrap_ci_reproducibility(self):
        """Custom bootstrap computation with fixed seed must be deterministic."""
        ci_run1 = self.evaluator.compute_bootstrap_confidence_intervals(n_bootstraps=50, random_seed=42)
        ci_run2 = self.evaluator.compute_bootstrap_confidence_intervals(n_bootstraps=50, random_seed=42)
        for metric in ["recall", "f1_score", "roc_auc"]:
            assert pytest.approx(ci_run1[metric]["ci_lower"], abs=1e-4) == ci_run2[metric]["ci_lower"]
            assert pytest.approx(ci_run1[metric]["ci_upper"], abs=1e-4) == ci_run2[metric]["ci_upper"]


class TestDecisionThresholdOptimization:
    """Tests Phase 4 decision threshold sweep and recommendation engine."""

    def test_09_threshold_recommendations(self):
        recs = get_threshold_recommendations()
        assert "f1_optimal" in recs
        assert "f2_optimal" in recs
        assert "balanced" in recs

        f1_rec = recs["f1_optimal"]
        assert 0.05 <= f1_rec["threshold"] <= 0.95
        assert 0.0 <= f1_rec["f1_score"] <= 1.0

        f2_rec = recs["f2_optimal"]
        assert 0.05 <= f2_rec["threshold"] <= 0.95
        # Recall at F2 optimal threshold should be equal to or higher than at F1 optimal threshold
        assert f2_rec["recall"] >= f1_rec["recall"] or pytest.approx(f2_rec["recall"], abs=0.05) == f1_rec["recall"]

    def test_10_threshold_sweep_data(self):
        ev = get_evaluator()
        sweep_df = ev.get_multi_threshold_table([0.10, 0.30, 0.50, 0.70, 0.90])
        assert isinstance(sweep_df, pd.DataFrame)
        expected_cols = {"Threshold (θ)", "Precision", "Recall", "F1-Score", "Accuracy"}
        assert expected_cols.issubset(set(sweep_df.columns))
        assert len(sweep_df) == 5
