"""
test_api_extended.py
--------------------
Integration tests for the expanded FastAPI microservice endpoints:
- GET /model/info
- GET /model/metrics
- GET /model/threshold
- GET /model/compare
- POST /predict with Tree SHAP explainability (?explain=true)
- Response schema validation (legitimate_probability, model_version, explanation)
"""

import os
import sys
import time
import pytest

WEEK8_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if WEEK8_DIR not in sys.path:
    sys.path.insert(0, WEEK8_DIR)

from frontend.validators import get_sample_preset
import src.predict as predict_module
import app as backend_app
from app import app


# Ensure predictor is initialized with absolute paths
if backend_app.predictor is None or getattr(backend_app.predictor, "model", None) is None:
    backend_app.predictor = predict_module.FraudPredictor(
        model_path=os.path.join(WEEK8_DIR, "models", "tuned_xgboost.pkl"),
        scaler_path=os.path.join(WEEK8_DIR, "models", "scaler.pkl"),
    )


class TestExtendedFastAPIEndpoints:
    """Verifies Phase 6 expanded API endpoints and Tree SHAP explanation response."""

    def test_01_model_info_endpoint(self):
        """GET /model/info returns model architecture, hyperparameters, and feature metadata."""
        data = backend_app.get_model_info()
        assert data["status"] == "active"
        assert data["model_version"] == "1.0-baseline"
        assert "XGBClassifier" in data["model_type"]
        assert data["expected_features_count"] == 30
        assert "StandardScaler" in data["scaler_type"]
        assert len(data["feature_order"]) == 30
        assert "hyperparameters" in data
        assert isinstance(data["hyperparameters"], dict)

    def test_02_model_metrics_endpoint(self):
        """GET /model/metrics returns benchmark test performance and bootstrap CIs."""
        data = backend_app.get_model_metrics(threshold=0.50)
        assert data["status"] == "success"
        assert data["split"] == "held_out_test_set"
        assert data["test_records"] == 1008
        assert data["test_fraud_cases"] == 8

        # Primary benchmark metrics
        metrics = data["metrics"]
        assert pytest.approx(metrics["accuracy"], rel=1e-3) == 0.9950
        assert pytest.approx(metrics["precision"], rel=1e-3) == 0.8000
        assert pytest.approx(metrics["recall"], rel=1e-3) == 0.5000
        assert pytest.approx(metrics["f1_score"], rel=1e-3) == 0.6154
        assert pytest.approx(metrics["roc_auc"], rel=1e-3) == 0.9910
        assert pytest.approx(metrics["pr_auc"], rel=1e-3) == 0.7492

        # Confusion matrix
        cm = data["confusion_matrix"]
        assert cm["true_positives"] == 4
        assert cm["false_positives"] == 1
        assert cm["false_negatives"] == 4
        assert cm["true_negatives"] == 999

        # Bootstrap CIs present
        assert "confidence_intervals_95" in data
        assert "recall" in data["confidence_intervals_95"]

    def test_03_model_threshold_endpoint(self):
        """GET /model/threshold returns strategic operating threshold recommendations."""
        data = backend_app.get_model_threshold_recommendations()
        assert data["status"] == "success"
        assert "recommendations" in data
        recs = data["recommendations"]
        assert "f1_optimal" in recs
        assert "f2_optimal" in recs
        assert "balanced" in recs
        assert recs["f1_optimal"]["threshold"] > 0

    def test_04_model_compare_endpoint(self):
        """GET /model/compare returns comparison manifest for all 5 candidate models."""
        data = backend_app.get_model_comparison()
        assert data["status"] == "success"
        manifest = data["manifest"]
        assert "models" in manifest
        assert len(manifest["models"]) == 5

    def test_05_predict_with_tree_shap_explainability(self):
        """POST /predict with explain=True returns feature contributions and base value."""
        sample = get_sample_preset(fraud=True)
        payload = backend_app.TransactionPayload(**sample)

        # Warm up once to prime tree structure cache
        backend_app.predict_transaction(payload=payload, threshold=0.5, explain=True)

        t0 = time.perf_counter()
        resp = backend_app.predict_transaction(payload=payload, threshold=0.5, explain=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.prediction == 1
        assert resp.label == "Fraudulent"
        assert resp.fraud_probability > 0.50
        assert pytest.approx(resp.fraud_probability + resp.legitimate_probability, rel=1e-4) == 1.0
        assert resp.model_version == "1.0-baseline"

        # Explainability validation
        assert resp.explanation is not None
        exp = resp.explanation
        assert "Tree SHAP" in exp["method"]
        assert "bias" in exp
        assert "top_positive" in exp
        assert "top_negative" in exp
        assert "top_features" in exp
        assert len(exp["top_features"]) == 30

        # Features should have feature name, contribution, and scaled_value
        top_driver = exp["top_features"][0]
        assert "feature" in top_driver
        assert "contribution" in top_driver
        assert "scaled_value" in top_driver
        assert isinstance(top_driver["contribution"], float)

        # Performance: Tree SHAP prediction contribs must run in < 50ms
        assert latency_ms < 50.0, f"SHAP calculation took {latency_ms:.2f}ms, expected < 50ms"

    def test_06_predict_without_explain(self):
        """POST /predict with explain=False omits explanation field for minimal payload."""
        sample = get_sample_preset(fraud=False)
        payload = backend_app.TransactionPayload(**sample)
        resp = backend_app.predict_transaction(payload=payload, threshold=0.5, explain=False)

        assert resp.prediction == 0
        assert resp.label == "Legitimate"
        assert resp.explanation is None
        assert resp.legitimate_probability > 0.99
        assert resp.model_version == "1.0-baseline"
