"""
test_pipeline_optimization.py
-----------------------------
Comprehensive automated test suite verifying the v2.0 optimized pipeline:
1. Complete elimination of data leakage in feature scaling (strictly fitted on training split).
2. Elimination of double-scaling anomalies via smart input detection.
3. Quantifiable superiority of v2.0 model over baseline (PR-AUC, ROC-AUC, Precision, Recall).
4. Out-of-fold calibrated threshold performance and trade-off guarantees.
5. High-throughput inference latency SLA verification (<15ms single, <100ms batch).
6. Deterministic reproducibility across multiple prediction runs.
"""

import os
import time
import pytest
import numpy as np
import pandas as pd
import joblib

from backend.src.predict import FraudPredictor, FEATURE_ORDER, get_sample_transaction
from frontend.model_evaluation import get_evaluator, get_dataset_intelligence, get_threshold_recommendations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "creditcard.csv")
KNOWN_TEST_PATH = os.path.join(PROJECT_ROOT, "fraudguard_known_answer_test.csv")


class TestPipelineOptimization:
    """Verifies end-to-end data preprocessing, leakage resolution, and model optimization."""

    @pytest.fixture(scope="class")
    @classmethod
    def predictor(cls):
        return FraudPredictor()

    @pytest.fixture(scope="class")
    @classmethod
    def evaluator(cls):
        return get_evaluator()

    def test_01_zero_data_leakage_in_scaler(self):
        """Scaler mean and scale must be derived strictly from training split, NOT full raw data."""
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        raw_df = pd.read_csv(RAW_DATA_PATH)

        # Full raw dataset statistics (leaked stats)
        raw_mean_time = raw_df["Time"].mean()
        raw_std_time = raw_df["Time"].std(ddof=0)
        raw_mean_amount = raw_df["Amount"].mean()
        raw_std_amount = raw_df["Amount"].std(ddof=0)

        # Scaler must NOT match full raw stats
        assert scaler.mean_[0] != pytest.approx(raw_mean_time, abs=1.0), "Time mean was leaked from full dataset"
        assert scaler.mean_[1] != pytest.approx(raw_mean_amount, abs=0.1), "Amount mean was leaked from full dataset"

        # Scaler must match training partition statistics (approx 85631.6 and 163.08)
        assert pytest.approx(scaler.mean_[0], rel=1e-2) == 85631.60
        assert pytest.approx(scaler.mean_[1], rel=1e-2) == 163.08

    def test_02_double_scaling_idempotence_and_known_answer_test(self, predictor):
        """Predictor must auto-detect pre-scaled test files and avoid double-scaling collapse."""
        if os.path.exists(KNOWN_TEST_PATH):
            df_known = pd.read_csv(KNOWN_TEST_PATH)
        else:
            x_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "X_test.csv"))
            y_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "y_test.csv")).squeeze()
            fraud_idx = np.where(y_test == 1)[0]
            legit_idx = np.where(y_test == 0)[0][:len(fraud_idx)]
            df_known = pd.concat([x_test.iloc[fraud_idx], x_test.iloc[legit_idx]], ignore_index=True)

        # The first 8 transactions are confirmed fraud
        probas = predictor.predict_proba(df_known)
        assert len(probas) == len(df_known)

        fraud_probas = probas[:8]
        # Under double-scaling bug, recall was 0/8 at th=0.50. Under fix, it must catch at least 4/8:
        detected_50 = (fraud_probas >= 0.50).sum()
        assert detected_50 >= 4, f"Expected >= 4 frauds detected at th=0.50, got {detected_50}"

        # At strategic operational threshold th=0.10, it must detect >= 6 of 8 frauds:
        detected_10 = (fraud_probas >= 0.10).sum()
        assert detected_10 >= 6, f"Expected >= 6 frauds detected at th=0.10, got {detected_10}"

    def test_03_model_v2_metrics_superiority(self, evaluator):
        """Optimized v2.0 model must demonstrably beat baseline v1.0 PR-AUC and ROC-AUC."""
        # Baseline: PR-AUC = 0.7492, ROC-AUC = 0.9910
        assert evaluator.pr_auc > 0.8000, f"Expected PR-AUC > 0.8000, got {evaluator.pr_auc:.4f}"
        assert evaluator.roc_auc > 0.9940, f"Expected ROC-AUC > 0.9940, got {evaluator.roc_auc:.4f}"
        assert evaluator.brier_score < 0.0035, f"Expected Brier score < 0.0035, got {evaluator.brier_score:.6f}"

        # At canonical th=0.50: precision must be 100% (zero false positives)
        m_50 = evaluator.compute_metrics(threshold=0.50)
        assert m_50["fp"] == 0, f"Expected 0 false positives at θ=0.50, got {m_50['fp']}"
        assert m_50["precision"] == 1.0, "Expected 100% precision at θ=0.50"
        assert m_50["f1_score"] > 0.65, f"Expected F1 > 0.65, got {m_50['f1_score']:.4f}"

    def test_04_out_of_fold_threshold_trade_offs(self, evaluator):
        """Threshold recommendations must achieve balanced fraud recall with low false alarm rate."""
        recs = get_threshold_recommendations()
        assert recs["leakage_free"] is True

        # At θ=0.10 (balanced/F1-optimal)
        m_10 = evaluator.compute_metrics(threshold=0.10)
        assert m_10["tp"] >= 6, f"Expected >= 6 true positives at θ=0.10, got {m_10['tp']}"
        assert m_10["recall"] >= 0.75, f"Expected recall >= 75% at θ=0.10, got {m_10['recall']:.2%}"
        assert m_10["specificity"] >= 0.995, f"Expected specificity >= 99.5%, got {m_10['specificity']:.2%}"
        assert m_10["fp"] <= 5, f"Expected <= 5 false alarms, got {m_10['fp']}"

    def test_05_prediction_latency_sla(self, predictor):
        """Single transaction inference must run in <15ms; batch of 100 in <50ms."""
        sample = get_sample_transaction(fraud=False)

        # Warm-up run
        predictor.predict(sample, threshold=0.50, explain=False)

        # Measure 20 consecutive single predictions
        single_latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            res = predictor.predict(sample, threshold=0.50, explain=False)
            single_latencies.append((time.perf_counter() - t0) * 1000.0)

        median_latency = sorted(single_latencies)[len(single_latencies) // 2]
        assert median_latency < 15.0, f"Median single latency was {median_latency:.2f}ms, expected < 15ms"

        # Batch of 100 records
        batch = [sample] * 100
        t0 = time.perf_counter()
        batch_res = predictor.predict(batch, threshold=0.50, explain=False)
        batch_latency = (time.perf_counter() - t0) * 1000.0

        assert len(batch_res) == 100
        assert batch_latency < 50.0, f"Batch 100 latency was {batch_latency:.2f}ms, expected < 50ms"

    def test_06_deterministic_reproducibility(self, predictor):
        """Identical inputs must produce identical probabilities across repeated invocations."""
        sample = get_sample_transaction(fraud=True)
        prob1 = predictor.predict_proba(sample)[0]
        prob2 = predictor.predict_proba(sample)[0]
        assert prob1 == pytest.approx(prob2, abs=1e-9)
        assert prob1 > 0.99, f"Fraud sample probability must be > 0.99, got {prob1}"
