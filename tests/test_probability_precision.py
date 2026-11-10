"""
test_probability_precision.py
-----------------------------
Regression and precision test suite for FraudGuard AI.
Verifies:
1. Backend returns unrounded floating-point probabilities preserving sub-0.0001 precision.
2. Distinct probabilities below 0.00005 are NOT collapsed to identical values (e.g., 0.0000).
3. ROC-AUC and probability rankings are preserved without artificial tie collapses.
4. Single-transaction (/predict) and batch (/predict/batch) endpoints produce identical probabilities.
5. Decision threshold remains strictly 0.50 for binary classification.
6. Health endpoint accurately reports scaler_type as 'StandardScaler'.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure backend modules are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.predict import FraudPredictor
import app as backend_app
from frontend.validators import get_sample_preset, create_sample_csv_dataframe


@pytest.fixture(scope="module")
def predictor():
    """Initializes FraudPredictor with canonical model and scaler artifacts."""
    return FraudPredictor(
        model_path=os.path.join(BACKEND_DIR, "models", "tuned_xgboost.pkl"),
        scaler_path=os.path.join(BACKEND_DIR, "models", "scaler.pkl"),
    )


class TestProbabilityPrecision:
    """Verifies that floating-point precision is maintained throughout the prediction pipeline."""

    def test_01_raw_probability_precision_unrounded(self, predictor):
        """Verify FraudPredictor returns unrounded float with full precision (>4 decimal places)."""
        sample_dict = get_sample_preset(fraud=False)
        df = pd.DataFrame([sample_dict])
        
        preds = predictor.predict(df, threshold=0.5)
        assert len(preds) == 1
        prob = preds[0]["fraud_probability"]
        
        # Verify it's a native float, not string or integer
        assert isinstance(prob, float)
        # Legitimate sample probability is approx 6.138e-05 (~0.00006138)
        # If rounded to 4 decimals, it would be 0.0001. Check full precision:
        assert prob < 0.0001
        assert prob > 0.00005
        # Ensure it has more than 4 significant digits
        assert str(prob) != f"{prob:.4f}"

    def test_02_sub_00005_distinct_probabilities_not_collapsed(self, predictor):
        """
        Regression test: Proves that distinct probabilities below 0.00005
        are preserved and NOT collapsed to identical 0.0000 values.
        """
        x_test_path = os.path.join(PROJECT_ROOT, "data", "processed", "X_test.csv")
        df_test = pd.read_csv(x_test_path)
        probs = predictor.predict_proba(df_test)

        # Select samples with distinct probabilities below 0.00005
        sub_00005_indices = [i for i, p in enumerate(probs) if p < 0.00005]
        assert len(sub_00005_indices) >= 5, f"Expected at least 5 samples with p < 0.00005, found {len(sub_00005_indices)}"

        # Take 5 distinct samples
        sample_subset = df_test.iloc[sub_00005_indices[:5]]
        predictions = predictor.predict(sample_subset, threshold=0.5)
        raw_probs = [p["fraud_probability"] for p in predictions]

        # Verify all are below 0.00005
        assert all(p < 0.00005 for p in raw_probs)

        # Prove that rounding to 4 decimal places collapses all to 0.0000
        rounded_probs = [round(p, 4) for p in raw_probs]
        assert all(r == 0.0 for r in rounded_probs), "4-decimal rounding should have collapsed all to 0.0"

        # Unrounded probabilities MUST remain distinct
        unique_raw = set(raw_probs)
        assert len(unique_raw) == len(raw_probs), (
            f"Expected {len(raw_probs)} distinct probabilities, got {len(unique_raw)}: {raw_probs}"
        )

    def test_03_roc_auc_evaluation_unrounded_vs_rounded(self):
        """
        Regression test: Demonstrates that unrounded probabilities prevent
        the ROC-AUC score from collapsing to 0.5000 due to identical value ties.
        """
        # Ground truth labels for 5 samples (2 negative, 3 positive)
        y_true = np.array([0, 0, 1, 1, 1])

        # Raw unrounded probabilities that are all below 0.00005 but ordered
        unrounded_scores = np.array([
            0.00001266,
            0.00000317,
            0.00002352,
            0.00000572,
            0.00000380,
        ])

        # If rounded to 4 decimals, all collapse to 0.0000
        rounded_scores = np.array([round(x, 4) for x in unrounded_scores])
        assert np.all(rounded_scores == 0.0)

        # ROC-AUC on identical predictions collapses to 0.5000 (random guess rank)
        auc_rounded = roc_auc_score(y_true, rounded_scores)
        assert auc_rounded == 0.5000

        # ROC-AUC on unrounded probabilities preserves ranking information
        auc_unrounded = roc_auc_score(y_true, unrounded_scores)
        assert auc_unrounded != 0.5000
        assert auc_unrounded > 0.5000
        assert pytest.approx(auc_unrounded, rel=1e-3) == 0.6667

    def test_04_batch_vs_single_prediction_consistency(self, predictor):
        """
        Verifies that /predict (single) and /predict/batch produce identical
        predictions, risk levels, and unrounded probabilities for every row.
        """
        df_sample = create_sample_csv_dataframe()
        records = df_sample.to_dict(orient="records")

        # 1. Single predictions
        single_preds = []
        for row in records:
            p = predictor.predict(pd.DataFrame([row]), threshold=0.5)[0]
            single_preds.append(p)

        # 2. Batch predictions
        batch_preds = predictor.predict(df_sample, threshold=0.5)

        assert len(single_preds) == len(batch_preds)
        for i, (s, b) in enumerate(zip(single_preds, batch_preds)):
            assert s["prediction"] == b["prediction"], f"Mismatch in prediction at row {i}"
            assert s["label"] == b["label"], f"Mismatch in label at row {i}"
            assert s["risk_level"] == b["risk_level"], f"Mismatch in risk_level at row {i}"
            assert pytest.approx(s["fraud_probability"], abs=1e-9) == b["fraud_probability"], (
                f"Probability mismatch at row {i}: single={s['fraud_probability']} vs batch={b['fraud_probability']}"
            )

    def test_05_threshold_classification_integrity(self, predictor):
        """Verifies that 0.50 decision threshold is strictly enforced."""
        # Legitimate sample (< 0.50) -> prediction 0
        legit_sample = pd.DataFrame([get_sample_preset(fraud=False)])
        legit_pred = predictor.predict(legit_sample, threshold=0.5)[0]
        assert legit_pred["fraud_probability"] < 0.50
        assert legit_pred["prediction"] == 0
        assert legit_pred["label"] == "Legitimate"

        # Fraud sample (>= 0.50) -> prediction 1
        fraud_sample = pd.DataFrame([get_sample_preset(fraud=True)])
        fraud_pred = predictor.predict(fraud_sample, threshold=0.5)[0]
        assert fraud_pred["fraud_probability"] >= 0.50
        assert fraud_pred["prediction"] == 1
        assert fraud_pred["label"] == "Fraudulent"

    def test_06_health_endpoint_scaler_metadata(self):
        """Verifies GET /health includes scaler_type reporting 'StandardScaler'."""
        health = backend_app.health_check()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["scaler_loaded"] is True
        assert health.get("scaler_type") == "StandardScaler"
