"""
test_api_client.py
------------------
Unit tests for APIClient using unittest.mock to simulate FastAPI server responses,
edge-proxy transient errors (HTTP 520, 502, 503, 504, 429), timeouts, bounded retries
with exponential backoff, checkpoint recovery, and payload canonicalization.
"""

import unittest
from unittest.mock import patch, MagicMock
from requests.exceptions import ConnectionError, Timeout

from frontend.api_client import APIClient, TRANSIENT_STATUS_CODES
from frontend.validators import get_sample_preset, FEATURE_ORDER


class TestAPIClient(unittest.TestCase):
    def setUp(self):
        self.client = APIClient(base_url="http://127.0.0.1:8000", timeout=2.0)
        self.sample = get_sample_preset(fraud=False)

    @patch("frontend.api_client.requests.get")
    def test_01_health_check_healthy(self, mock_get):
        """Mock successful health check response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "model_loaded": True,
            "scaler_loaded": True,
            "model_type": "XGBClassifier",
            "expected_features_count": 30,
        }
        mock_get.return_value = mock_resp

        is_healthy, data, err = self.client.check_health()
        self.assertTrue(is_healthy)
        self.assertIsNone(err)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("latency_ms", data)

    @patch("frontend.api_client.requests.get")
    def test_02_health_check_unhealthy(self, mock_get):
        """Mock 503 Service Unavailable health check response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service unavailable: model artifacts could not be loaded"
        mock_get.return_value = mock_resp

        is_healthy, data, err = self.client.check_health()
        self.assertFalse(is_healthy)
        self.assertIn("503", err)

    @patch("frontend.api_client.requests.get")
    def test_03_health_check_connection_error(self, mock_get):
        """Mock connection error when FastAPI is offline."""
        mock_get.side_effect = ConnectionError("Connection refused")

        is_healthy, data, err = self.client.check_health()
        self.assertFalse(is_healthy)
        self.assertIn("Unable to connect", err)

    @patch("frontend.api_client.requests.get")
    def test_04_get_sample_success(self, mock_get):
        """Mock retrieving sample transaction from API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "type": "Legitimate Example",
            "payload": self.sample,
        }
        mock_get.return_value = mock_resp

        sample, err = self.client.get_sample(fraud=False)
        self.assertIsNotNone(sample)
        self.assertIsNone(err)
        self.assertEqual(sample["Amount"], self.sample["Amount"])

    @patch("frontend.api_client.requests.post")
    def test_05_predict_single_success(self, mock_post):
        """Mock successful single transaction prediction."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "prediction": 0,
            "label": "Legitimate",
            "fraud_probability": 0.0125,
            "threshold_used": 0.5,
            "risk_level": "LOW",
        }
        mock_post.return_value = mock_resp

        result, err = self.client.predict_single(self.sample, threshold=0.5)
        self.assertIsNotNone(result)
        self.assertIsNone(err)
        self.assertEqual(result["prediction"], 0)
        self.assertEqual(result["label"], "Legitimate")
        self.assertEqual(result["risk_level"], "LOW")

    @patch("frontend.api_client.requests.post")
    def test_06_predict_single_validation_error(self, mock_post):
        """Mock 422 Unprocessable Entity error from backend schema validator."""
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"detail": "Field 'Time' required"}
        mock_post.return_value = mock_resp

        result, err = self.client.predict_single(self.sample)
        self.assertIsNone(result)
        self.assertIn("API Validation Error", err)

    @patch("frontend.api_client.requests.post")
    def test_07_predict_single_timeout(self, mock_post):
        """Mock request timeout during prediction."""
        mock_post.side_effect = Timeout("Timed out")

        result, err = self.client.predict_single(self.sample)
        self.assertIsNone(result)
        self.assertIn("timed out", err)

    @patch("frontend.api_client.requests.post")
    def test_08_predict_batch_success(self, mock_post):
        """Mock successful batch prediction."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"x-request-id": "req-test-123"}
        mock_resp.json.return_value = {
            "total_processed": 2,
            "fraud_count": 1,
            "legitimate_count": 1,
            "predictions": [
                {
                    "prediction": 0,
                    "label": "Legitimate",
                    "fraud_probability": 0.05,
                    "threshold_used": 0.5,
                    "risk_level": "LOW",
                },
                {
                    "prediction": 1,
                    "label": "Fraudulent",
                    "fraud_probability": 0.95,
                    "threshold_used": 0.5,
                    "risk_level": "HIGH",
                },
            ],
        }
        mock_post.return_value = mock_resp

        batch_result, err = self.client.predict_batch([self.sample, self.sample], threshold=0.5)
        self.assertIsNotNone(batch_result)
        self.assertIsNone(err)
        self.assertEqual(batch_result["total_processed"], 2)
        self.assertEqual(batch_result["fraud_count"], 1)
        self.assertEqual(batch_result["legitimate_count"], 1)
        self.assertAlmostEqual(batch_result["fraud_rate"], 0.5)

    @patch("frontend.api_client.requests.post")
    def test_09_predict_batch_chunking(self, mock_post):
        """Mock batch processing chunking logic for large payloads."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "total_processed": 2,
            "fraud_count": 0,
            "legitimate_count": 2,
            "predictions": [
                {"prediction": 0, "label": "Legitimate", "fraud_probability": 0.01, "threshold_used": 0.5, "risk_level": "LOW"},
                {"prediction": 0, "label": "Legitimate", "fraud_probability": 0.02, "threshold_used": 0.5, "risk_level": "LOW"},
            ],
        }
        mock_post.return_value = mock_resp

        # Send 4 items with chunk_size=2 -> should result in 2 post calls
        items = [self.sample] * 4
        batch_result, err = self.client.predict_batch(items, threshold=0.5, chunk_size=2)
        self.assertIsNotNone(batch_result)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(batch_result["total_processed"], 4)

    def test_10_predict_batch_empty_list(self):
        """Empty transaction list should return an error immediately."""
        result, err = self.client.predict_batch([])
        self.assertIsNone(result)
        self.assertIn("empty", err)

    @patch("frontend.api_client.time.sleep")
    @patch("frontend.api_client.requests.post")
    def test_11_predict_batch_transient_520_retry_success(self, mock_post, mock_sleep):
        """Transient HTTP 520 (e.g. Cloudflare / Render proxy error) should trigger bounded retry and recover."""
        # Attempt 1: HTTP 520 Cloudflare edge error
        resp_520 = MagicMock()
        resp_520.status_code = 520
        resp_520.text = "520: Web server is returning an unknown error"
        resp_520.headers = {"cf-ray": "ray-520-abc"}

        # Attempt 2: HTTP 200 Success
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.headers = {"x-request-id": "req-recovery-success"}
        resp_200.json.return_value = {
            "total_processed": 1,
            "fraud_count": 0,
            "legitimate_count": 1,
            "predictions": [
                {"prediction": 0, "label": "Legitimate", "fraud_probability": 0.02, "threshold_used": 0.5, "risk_level": "LOW"}
            ],
        }

        mock_post.side_effect = [resp_520, resp_200]

        result, err = self.client.predict_batch([self.sample], threshold=0.5, chunk_size=100, max_retries=2, backoff_base=0.01)
        self.assertIsNotNone(result)
        self.assertIsNone(err)
        self.assertEqual(result["total_processed"], 1)
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()  # Exponential backoff invoked

    @patch("frontend.api_client.time.sleep")
    @patch("frontend.api_client.requests.post")
    def test_12_predict_batch_transient_codes_502_503_504_429(self, mock_post, mock_sleep):
        """HTTP 502, 503, 504, and 429 are all recognized as transient and retried."""
        for code in [429, 502, 503, 504]:
            mock_post.reset_mock()
            mock_sleep.reset_mock()

            resp_err = MagicMock()
            resp_err.status_code = code
            resp_err.text = f"Gateway Error {code}"
            resp_err.headers = {}

            resp_ok = MagicMock()
            resp_ok.status_code = 200
            resp_ok.headers = {}
            resp_ok.json.return_value = {
                "total_processed": 1,
                "fraud_count": 0,
                "legitimate_count": 1,
                "predictions": [{"prediction": 0, "label": "Legitimate", "fraud_probability": 0.01, "threshold_used": 0.5, "risk_level": "LOW"}],
            }

            mock_post.side_effect = [resp_err, resp_ok]
            res, err = self.client.predict_batch([self.sample], threshold=0.5, max_retries=1, backoff_base=0.01)
            self.assertIsNotNone(res, f"Failed for transient code {code}")
            self.assertIsNone(err)
            self.assertEqual(mock_post.call_count, 2)

    @patch("frontend.api_client.time.sleep")
    @patch("frontend.api_client.requests.post")
    def test_13_predict_batch_exhaust_retries_checkpoint_recovery(self, mock_post, mock_sleep):
        """Exhausting retries on a chunk preserves completed chunks and returns partial checkpoint."""
        # Chunk 1 (row 1): Success
        resp_c1 = MagicMock()
        resp_c1.status_code = 200
        resp_c1.headers = {}
        resp_c1.json.return_value = {
            "total_processed": 1,
            "fraud_count": 0,
            "legitimate_count": 1,
            "predictions": [{"prediction": 0, "label": "Legitimate", "fraud_probability": 0.01, "threshold_used": 0.5, "risk_level": "LOW"}],
        }

        # Chunk 2 (row 2): Persistent 520 error across all 3 attempts
        resp_c2_fail = MagicMock()
        resp_c2_fail.status_code = 520
        resp_c2_fail.text = "Origin Handshake Failure"
        resp_c2_fail.headers = {}

        mock_post.side_effect = [resp_c1, resp_c2_fail, resp_c2_fail, resp_c2_fail]

        # 2 records with chunk_size=1 -> 2 chunks total
        items = [self.sample, self.sample]
        result, err = self.client.predict_batch(items, threshold=0.5, chunk_size=1, max_retries=2, backoff_base=0.01)

        # Checkpoint validation
        self.assertIsNotNone(result)
        self.assertIsNotNone(err)
        self.assertTrue(result.get("is_partial"))
        self.assertEqual(result.get("completed_chunks"), 1)
        self.assertEqual(result.get("total_processed"), 1)  # Chunk 1 was saved!
        self.assertEqual(result.get("failed_chunk_number"), 2)
        self.assertIn("Batch chunk 2/2", err)
        self.assertIn("520", err)

    @patch("frontend.api_client.time.sleep")
    @patch("frontend.api_client.requests.post")
    def test_14_predict_batch_timeout_retry_success(self, mock_post, mock_sleep):
        """Network timeout should trigger retry and successfully resume."""
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.headers = {}
        resp_ok.json.return_value = {
            "total_processed": 1,
            "fraud_count": 1,
            "legitimate_count": 0,
            "predictions": [{"prediction": 1, "label": "Fraudulent", "fraud_probability": 0.92, "threshold_used": 0.5, "risk_level": "HIGH"}],
        }

        mock_post.side_effect = [Timeout("Socket read timed out"), resp_ok]

        result, err = self.client.predict_batch([self.sample], threshold=0.5, max_retries=1, backoff_base=0.01)
        self.assertIsNotNone(result)
        self.assertIsNone(err)
        self.assertEqual(result["fraud_count"], 1)
        self.assertEqual(mock_post.call_count, 2)

    @patch("frontend.api_client.requests.post")
    def test_15_predict_batch_excludes_class_column_from_payload(self, mock_post):
        """Ground truth 'Class' and arbitrary metadata must be stripped from model input payloads."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "total_processed": 1,
            "fraud_count": 0,
            "legitimate_count": 1,
            "predictions": [{"prediction": 0, "label": "Legitimate", "fraud_probability": 0.05, "threshold_used": 0.5, "risk_level": "LOW"}],
        }
        mock_post.return_value = mock_resp

        # Transaction with extra 'Class' and metadata columns
        sample_with_class = self.sample.copy()
        sample_with_class["Class"] = 0
        sample_with_class["Customer_ID"] = "CUST-999"
        sample_with_class["Random_Meta"] = 12345

        result, err = self.client.predict_batch([sample_with_class], threshold=0.5)
        self.assertIsNotNone(result)
        self.assertIsNone(err)

        # Inspect the JSON sent to requests.post
        call_kwargs = mock_post.call_args[1]
        sent_payload = call_kwargs["json"]["transactions"][0]

        # Ensure Class and other extra keys are NOT present in model inputs
        self.assertNotIn("Class", sent_payload)
        self.assertNotIn("Customer_ID", sent_payload)
        self.assertNotIn("Random_Meta", sent_payload)
        # Ensure exact 30 canonical features are present
        self.assertEqual(sorted(list(sent_payload.keys())), sorted(FEATURE_ORDER))

    @patch("frontend.api_client.requests.post")
    def test_16_predict_batch_telemetry_and_callback(self, mock_post):
        """Per-chunk telemetry callback should receive timing, status, and request IDs."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"x-request-id": "server-req-abc"}
        mock_resp.json.return_value = {
            "total_processed": 1,
            "fraud_count": 0,
            "legitimate_count": 1,
            "predictions": [{"prediction": 0, "label": "Legitimate", "fraud_probability": 0.01, "threshold_used": 0.5, "risk_level": "LOW"}],
        }
        mock_post.return_value = mock_resp

        callback_events = []
        def on_chunk(info):
            callback_events.append(info)

        result, err = self.client.predict_batch([self.sample], threshold=0.5, progress_callback=on_chunk)
        self.assertIsNotNone(result)
        self.assertEqual(len(callback_events), 1)
        event = callback_events[0]
        self.assertEqual(event["chunk_number"], 1)
        self.assertEqual(event["status_code"], 200)
        self.assertEqual(event["status"], "SUCCESS")
        self.assertIn("latency_ms", event)
        self.assertIn("request_id", event)

    @patch("frontend.api_client.requests.post")
    def test_17_predict_batch_configurable_chunk_size(self, mock_post):
        """Chunk size is configurable and defaults to 100 rows."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "total_processed": 10,
            "fraud_count": 0,
            "legitimate_count": 10,
            "predictions": [{"prediction": 0, "label": "Legitimate", "fraud_probability": 0.01, "threshold_used": 0.5, "risk_level": "LOW"}] * 10,
        }
        mock_post.return_value = mock_resp

        # 250 records with chunk_size=100 -> ceil(250/100) = 3 chunks
        items = [self.sample] * 250
        result, err = self.client.predict_batch(items, threshold=0.5, chunk_size=100)
        self.assertIsNotNone(result)
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(result["total_chunks"], 3)


if __name__ == "__main__":
    unittest.main()
