"""
api_client.py
-------------
HTTP client for communicating with the FastAPI Fraud Detection Service.
Provides robust connection management, timeout handling, chunked batch
processing, and structured error reporting.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import time
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError, RequestException


class APIClient:
    """Client for the Week 8 FastAPI inference service."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        base_url = base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Checks the health and readiness of the FastAPI inference service.
        
        Returns:
            Tuple of (is_healthy, health_metadata, error_message)
        """
        url = f"{self.base_url}/health"
        start_time = time.time()
        try:
            response = requests.get(url, timeout=self.timeout)
            latency_ms = round((time.time() - start_time) * 1000, 2)

            if response.status_code == 200:
                data = response.json()
                data["latency_ms"] = latency_ms
                return True, data, None
            else:
                return False, {}, (
                    f"Health check failed with status code {response.status_code}: {response.text}"
                )
        except ConnectionError:
            return False, {}, (
                f"Unable to connect to FastAPI service at {self.base_url}. "
                "Ensure the server is running on the specified host and port."
            )
        except Timeout:
            return False, {}, f"Connection timed out while contacting {self.base_url}."
        except RequestException as exc:
            return False, {}, f"Request error during health check: {str(exc)}"

    def get_sample(self, fraud: bool = False) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """Fetches a pre-configured sample transaction from the API.
        
        Args:
            fraud: True for fraudulent transaction, False for legitimate.
            
        Returns:
            Tuple of (payload_dict, error_message)
        """
        url = f"{self.base_url}/sample"
        params = {"fraud": str(fraud).lower()}
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get("payload"), None
            return None, f"Failed to get sample (HTTP {response.status_code}): {response.text}"
        except ConnectionError:
            return None, f"Cannot connect to API at {self.base_url}."
        except Timeout:
            return None, "Request timed out."
        except RequestException as exc:
            return None, f"Request error: {str(exc)}"

    def predict_single(
        self, transaction: Dict[str, float], threshold: float = 0.5
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Sends a single transaction to /predict.
        
        Args:
            transaction: Dictionary of 30 feature values.
            threshold: Decision threshold [0.0 - 1.0].
            
        Returns:
            Tuple of (prediction_result, error_message)
        """
        url = f"{self.base_url}/predict"
        params = {"threshold": float(threshold)}
        try:
            response = requests.post(
                url,
                params=params,
                json=transaction,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 422:
                # Validation error from FastAPI / Pydantic
                detail = response.json().get("detail", response.text)
                return None, f"API Validation Error: {detail}"
            else:
                return None, f"Inference failed (HTTP {response.status_code}): {response.text}"
        except ConnectionError:
            return None, f"Connection to {self.base_url} failed. Is the FastAPI service online?"
        except Timeout:
            return None, f"Prediction request timed out after {self.timeout}s."
        except RequestException as exc:
            return None, f"Prediction error: {str(exc)}"

    def predict_batch(
        self,
        transactions: List[Dict[str, float]],
        threshold: float = 0.5,
        chunk_size: int = 500,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Sends multiple transactions to /predict/batch, automatically chunking
        large payloads to prevent request size timeouts.
        
        Args:
            transactions: List of 30-feature dictionaries.
            threshold: Decision threshold [0.0 - 1.0].
            chunk_size: Maximum items per network request.
            
        Returns:
            Tuple of (aggregated_batch_result, error_message)
        """
        if not transactions:
            return None, "Transaction list is empty."

        url = f"{self.base_url}/predict/batch"
        params = {"threshold": float(threshold)}

        all_predictions: List[Dict[str, Any]] = []
        total_fraud = 0
        total_legit = 0

        # Process in chunks
        for i in range(0, len(transactions), chunk_size):
            chunk = transactions[i : i + chunk_size]
            payload = {"transactions": chunk}

            try:
                response = requests.post(
                    url,
                    params=params,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=max(self.timeout, 30.0),
                )
                if response.status_code != 200:
                    return None, (
                        f"Batch chunk {i // chunk_size + 1} failed "
                        f"(HTTP {response.status_code}): {response.text}"
                    )
                chunk_result = response.json()
                total_fraud += chunk_result.get("fraud_count", 0)
                total_legit += chunk_result.get("legitimate_count", 0)
                all_predictions.extend(chunk_result.get("predictions", []))
            except ConnectionError:
                return None, f"Connection lost to {self.base_url} during batch inference."
            except Timeout:
                return None, f"Batch request timed out on chunk {i // chunk_size + 1}."
            except RequestException as exc:
                return None, f"Batch request error: {str(exc)}"

        aggregated = {
            "total_processed": len(all_predictions),
            "fraud_count": total_fraud,
            "legitimate_count": total_legit,
            "fraud_rate": round(total_fraud / len(all_predictions), 4) if all_predictions else 0.0,
            "predictions": all_predictions,
        }
        return aggregated, None
