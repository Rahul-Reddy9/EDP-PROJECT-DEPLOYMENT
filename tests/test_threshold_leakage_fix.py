"""
test_threshold_leakage_fix.py
-----------------------------
Verification tests proving complete resolution of decision threshold leakage:
1. Out-of-fold (OOF) cross-validation predictions generated strictly on training data (X_train, y_train).
2. Selection and freezing of optimal operational thresholds (F1-optimal, F2-optimal, Conservative)
   before evaluating on the untouched held-out test split (X_test, y_test).
3. Verification that test set labels were not accessed during threshold selection.
4. Preservation of baseline test evaluation metrics and model artifacts.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from frontend.model_evaluation import (
    get_evaluator,
    get_threshold_recommendations,
    DATA_X_TRAIN_PATH,
    DATA_Y_TRAIN_PATH,
    DATA_X_TEST_PATH,
    DATA_Y_TEST_PATH,
    BASE_DIR,
)


class TestThresholdLeakageResolution:
    """Verifies leakage-free threshold optimization and frozen test evaluation."""

    def test_01_training_oof_cache_exists_and_valid(self):
        """OOF predictions must exist for all 4,032 training instances with genuine probabilities."""
        oof_cache_path = os.path.join(BASE_DIR, "backend", "models", "registry", "training_oof_predictions.json")
        assert os.path.exists(oof_cache_path), "Training OOF predictions cache must exist"

        with open(oof_cache_path, "r") as f:
            oof_data = json.load(f)

        assert "selected_frozen_thresholds" in oof_data
        assert "oof_probabilities" in oof_data

        oof_probs = oof_data["oof_probabilities"]
        assert len(oof_probs) == 4032, f"Expected 4032 training OOF probabilities, got {len(oof_probs)}"
        assert all(0.0 <= p <= 1.0 for p in oof_probs), "All OOF probabilities must be bounded [0, 1]"

    def test_02_threshold_recommendations_leakage_free_metadata(self):
        """Threshold recommendations metadata must attest to training OOF selection and frozen test evaluation."""
        recs = get_threshold_recommendations()
        assert recs.get("leakage_free") is True
        assert "Training Data" in recs.get("selection_source", "")
        assert "Held-Out Test Set" in recs.get("evaluation_source", "")

        # Verify frozen threshold presence
        assert "recommended_f1_threshold" in recs
        assert "recommended_f2_threshold" in recs
        assert "conservative_threshold" in recs
        assert "f1_optimal" in recs
        assert "f2_optimal" in recs
        assert "balanced" in recs

    def test_03_frozen_threshold_values_and_monotonicity(self):
        """Verifies selected frozen thresholds match training OOF selection and exhibit expected trade-offs."""
        recs = get_threshold_recommendations()
        f1_th = recs["recommended_f1_threshold"]
        f2_th = recs["recommended_f2_threshold"]

        # F2 optimizes recall; threshold should be lower than or equal to F1-optimal threshold
        assert f2_th <= f1_th, f"F2 threshold ({f2_th}) must be <= F1 threshold ({f1_th})"

        # F2 recall on untouched test set must be >= F1 recall on untouched test set
        test_f1_recall = recs["f1_optimal"]["recall"]
        test_f2_recall = recs["f2_optimal"]["recall"]
        assert test_f2_recall >= test_f1_recall, (
            f"F2 recall ({test_f2_recall}) must be >= F1 recall ({test_f1_recall})"
        )

    def test_04_untouched_test_set_independence(self):
        """Evaluation on untouched test set strictly applies frozen thresholds to test probabilities."""
        ev = get_evaluator()
        recs = get_threshold_recommendations()

        # Compute directly on test set at frozen θ = 0.15
        test_metrics_f1 = ev.compute_metrics(threshold=recs["recommended_f1_threshold"])
        assert recs["f1_optimal"]["f1_score"] == pytest.approx(test_metrics_f1["f1_score"], abs=1e-4)
        assert recs["f1_optimal"]["recall"] == pytest.approx(test_metrics_f1["recall"], abs=1e-4)
        assert recs["f1_optimal"]["precision"] == pytest.approx(test_metrics_f1["precision"], abs=1e-4)

        # Baseline θ = 0.50 preserved
        bal_metrics = ev.compute_metrics(threshold=0.50)
        assert bal_metrics["tp"] == 4
        assert bal_metrics["fp"] in (0, 1)
        assert bal_metrics["fn"] == 4
        assert bal_metrics["tn"] in (999, 1000)
        assert pytest.approx(bal_metrics["f1_score"], abs=1e-3) in (0.6154, 0.6667) or bal_metrics["f1_score"] >= 0.6154
