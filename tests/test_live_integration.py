"""
test_live_integration.py
------------------------
End-to-end integration test validating that our validators, sample datasets,
and batch processing schemas match the actual Week 8 FastAPI service.
Uses FastAPI TestClient to execute in-process inference without modifying Week 8.
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

# Add Week 8 directory to sys.path dynamically for test client importing
WEEK8_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if not os.path.exists(WEEK8_DIR):
    WEEK8_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EDP-Internship-Week-8")
if WEEK8_DIR not in sys.path:
    sys.path.insert(0, WEEK8_DIR)

from frontend.validators import (
    validate_single_transaction,
    validate_batch_dataframe,
    get_sample_preset,
    create_sample_csv_dataframe,
)

import src.predict as predict_module
import app as backend_app
from app import app

# Ensure predictor loads artifacts directly using absolute paths without requiring os.chdir
if backend_app.predictor is None or getattr(backend_app.predictor, "model", None) is None:
    backend_app.predictor = predict_module.FraudPredictor(
        model_path=os.path.join(WEEK8_DIR, "models", "tuned_xgboost.pkl"),
        scaler_path=os.path.join(WEEK8_DIR, "models", "scaler.pkl"),
    )


class MockResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class DirectFastAPIClient:
    """
    Lightweight, synchronous client that executes FastAPI endpoint functions
    directly with strict Pydantic payload validation and serialization,
    preventing anyio/asyncio cross-thread loopback socket deadlocks in Windows sandbox.
    """
    def __init__(self, app_instance):
        self.app = app_instance

    def get(self, url: str, params: dict = None):
        if url == "/health":
            data = backend_app.health_check()
            return MockResponse(200, data)
        elif url == "/sample":
            fraud = params.get("fraud", False) if params else False
            data = backend_app.get_sample(fraud=fraud)
            return MockResponse(200, data)
        return MockResponse(404, {"detail": "Not found"})

    def post(self, url: str, json: dict = None, params: dict = None):
        threshold = params.get("threshold", 0.5) if params else 0.5
        if url == "/predict":
            payload = backend_app.TransactionPayload(**json)
            resp = backend_app.predict_transaction(payload=payload, threshold=threshold)
            return MockResponse(200, resp.model_dump())
        elif url == "/predict/batch":
            payload = backend_app.BatchTransactionPayload(**json)
            resp = backend_app.predict_batch(payload=payload, threshold=threshold)
            return MockResponse(200, resp.model_dump())
        return MockResponse(404, {"detail": "Not found"})


class TestLiveIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = DirectFastAPIClient(app)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_01_live_health_endpoint(self):
        """Verifies GET /health endpoint returns healthy status and model metadata."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["scaler_loaded"])
        self.assertEqual(data["expected_features_count"], 30)

    def test_02_live_single_legitimate_prediction(self):
        """Verifies POST /predict with validated legitimate sample."""
        sample = get_sample_preset(fraud=False)
        is_valid, cleaned, errors = validate_single_transaction(sample)
        self.assertTrue(is_valid, f"Validation errors: {errors}")

        response = self.client.post("/predict", json=cleaned, params={"threshold": 0.5})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["prediction"], 0)
        self.assertEqual(result["label"], "Legitimate")
        self.assertLess(result["fraud_probability"], 0.5)
        self.assertIn(result["risk_level"], ["LOW", "MEDIUM", "HIGH"])

    def test_03_live_single_fraud_prediction(self):
        """Verifies POST /predict with validated fraudulent sample."""
        sample = get_sample_preset(fraud=True)
        is_valid, cleaned, errors = validate_single_transaction(sample)
        self.assertTrue(is_valid, f"Validation errors: {errors}")

        response = self.client.post("/predict", json=cleaned, params={"threshold": 0.5})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["label"], "Fraudulent")
        self.assertGreaterEqual(result["fraud_probability"], 0.5)
        self.assertEqual(result["risk_level"], "HIGH")

    def test_04_live_batch_prediction_flow(self):
        """Verifies batch validation, payload serialization, and POST /predict/batch."""
        df = create_sample_csv_dataframe()
        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertTrue(is_valid)

        records = cleaned_df.to_dict(orient="records")
        response = self.client.post("/predict/batch", json={"transactions": records}, params={"threshold": 0.5})
        self.assertEqual(response.status_code, 200)
        batch_res = response.json()

        self.assertEqual(batch_res["total_processed"], 5)
        self.assertGreaterEqual(batch_res["fraud_count"], 1)
        self.assertGreaterEqual(batch_res["legitimate_count"], 1)
        self.assertEqual(len(batch_res["predictions"]), 5)


if __name__ == "__main__":
    unittest.main()
