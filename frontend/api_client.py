"""
api_client.py
-------------
HTTP client for communicating with the FastAPI Fraud Detection Service.
Provides robust connection management, bounded retries with exponential
backoff for transient gateway/proxy errors (520, 502, 503, 504, 429),
configurable chunking, per-chunk telemetry logging, checkpoint preservation,
and canonical feature extraction (ensuring 'Class' is excluded from model inputs).
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
import os
import time
import uuid
import random
import logging
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError, RequestException

from frontend.validators import FEATURE_ORDER

# Configure module-level logger
logger = logging.getLogger("fraudguard.api_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [APIClient] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# HTTP status codes that indicate transient edge/proxy/gateway failures eligible for retry
TRANSIENT_STATUS_CODES = {429, 502, 503, 504, 520}


def _sanitize_error_snippet(response_text: str) -> str:
    """Extracts sanitized error detail from response, preventing payload leakage into logs."""
    if not response_text:
        return ""
    try:
        import json
        data = json.loads(response_text)
        if isinstance(data, dict) and "detail" in data:
            detail = data["detail"]
            if isinstance(detail, list):
                err_msgs = [f"{e.get('loc', [])}: {e.get('msg', '')}" for e in detail if isinstance(e, dict)]
                return "; ".join(err_msgs[:3])
            return str(detail)[:200]
    except Exception:
        pass
    return response_text[:120].strip()


DEFAULT_RENDER_BACKEND_URL = "https://edp-project-deployment.onrender.com"
DEFAULT_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"


def resolve_default_api_url() -> str:
    """
    Resolves the active default backend API URL:
    1. Checks 'API_BASE_URL' environment variable if explicitly configured.
    2. Checks Streamlit secrets (st.secrets['API_BASE_URL']) if available in cloud runtime.
    3. Checks if running in an explicit local development mode via
       ENV in ('local', 'dev', 'development') or LOCAL_DEV in ('true', '1') -> returns DEFAULT_LOCAL_BACKEND_URL.
    4. Production / deployed fallback: returns DEFAULT_RENDER_BACKEND_URL.
    """
    env_url = os.getenv("API_BASE_URL")
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "API_BASE_URL" in st.secrets:
            sec_url = str(st.secrets["API_BASE_URL"]).strip()
            if sec_url:
                return sec_url.rstrip("/")
    except Exception:
        pass

    env_mode = (os.getenv("ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()
    if env_mode in ("local", "dev", "development") or os.getenv("LOCAL_DEV", "").strip().lower() in ("true", "1"):
        return DEFAULT_LOCAL_BACKEND_URL

    return DEFAULT_RENDER_BACKEND_URL


class APIClient:
    """Client for the Week 8 FastAPI inference service with enterprise-grade resilience."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0, token: Optional[str] = None):
        base_url = base_url or resolve_default_api_url()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token: Optional[str] = token
        self.role: Optional[str] = None
        self.username: Optional[str] = None
        self.display_name: Optional[str] = None

    def set_token(
        self,
        token: Optional[str],
        role: Optional[str] = None,
        username: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """Sets authorization Bearer token and user identity for subsequent requests."""
        self.token = token
        self.role = role
        self.username = username
        self.display_name = display_name

    def _get_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Returns standard headers including Authorization Bearer token when present."""
        headers: Dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def login(self, username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Authenticates with /auth/login and stores the access token."""
        url = f"{self.base_url}/auth/login"
        try:
            resp = requests.post(
                url, json={"username": username, "password": password}, timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                self.set_token(
                    token=data.get("access_token"),
                    role=data.get("role"),
                    username=data.get("username"),
                    display_name=data.get("display_name"),
                )
                return True, data, None
            elif resp.status_code == 401:
                return False, None, "Invalid username or password."
            else:
                return False, None, f"Login failed (HTTP {resp.status_code}): {resp.text}"
        except Exception as exc:
            return False, None, f"Login connection error: {str(exc)}"

    def logout(self) -> Tuple[bool, Optional[str]]:
        """Terminates session on server and clears local token."""
        url = f"{self.base_url}/auth/logout"
        try:
            resp = requests.post(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                self.set_token(None, None, None, None)
                return True, None
            elif resp.status_code == 503:
                return False, f"Logout persistence failure (HTTP 503): {resp.text}"
            return False, f"Logout failed (HTTP {resp.status_code}): {resp.text}"
        except Exception as exc:
            self.set_token(None, None, None, None)
            return False, f"Logout connection error: {str(exc)}"

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
        self, transaction: Dict[str, float], threshold: float = 0.5, explain: bool = False
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Sends a single transaction to /predict.
        
        Ensures any auxiliary non-canonical columns (such as 'Class') are excluded
        from the request payload sent to the backend.
        
        Args:
            transaction: Dictionary of feature values.
            threshold: Decision threshold [0.0 - 1.0].
            explain: Whether to request Tree SHAP feature contribution breakdown.
            
        Returns:
            Tuple of (prediction_result, error_message)
        """
        url = f"{self.base_url}/predict"
        params = {"threshold": float(threshold), "explain": str(explain).lower()}

        # Filter model inputs: strictly preserve canonical features, exclude Class
        clean_tx = {k: float(transaction[k]) for k in FEATURE_ORDER if k in transaction}

        try:
            response = requests.post(
                url,
                params=params,
                json=clean_tx,
                headers=self._get_headers({"Content-Type": "application/json"}),
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
        chunk_size: int = 100,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Sends multiple transactions to /predict/batch using sequential, configurable chunking
        with bounded retries and exponential backoff for transient gateway/proxy errors (520, 502, etc.).

        Key Resilience Characteristics:
        1. Configurable chunk size (default 100) to keep per-request edge payload light.
        2. Strictly sequential processing (concurrency = 1) to avoid socket/edge proxy starvation.
        3. Retries ONLY the failed chunk upon transient HTTP status codes (520, 502, 503, 504, 429)
           or network timeouts/connection resets with jittered exponential backoff.
        4. Excludes auxiliary columns (e.g. 'Class') from payload sent to the FastAPI microservice.
        5. Preserves completed chunks and original input row ordering even if a downstream chunk aborts.
        6. Logs per-chunk telemetry (request ID, timing, status code, response snippet).

        Args:
            transactions: List of feature dictionaries.
            threshold: Decision threshold [0.0 - 1.0].
            chunk_size: Maximum items per network request (default: 100).
            max_retries: Maximum retry attempts per failed chunk (default: 3).
            backoff_base: Base seconds for exponential backoff (default: 1.0).
            progress_callback: Optional callback receiving per-chunk telemetry dictionaries.

        Returns:
            Tuple of (batch_result_dict, error_message). On partial failure, batch_result_dict
            contains all successfully scored chunks up to the failure point.
        """
        if not transactions:
            return None, "Transaction list is empty."

        url = f"{self.base_url}/predict/batch"
        params = {"threshold": float(threshold)}

        total_records = len(transactions)
        total_chunks = (total_records + chunk_size - 1) // chunk_size

        all_predictions: List[Dict[str, Any]] = []
        chunk_telemetry_logs: List[Dict[str, Any]] = []
        total_fraud = 0
        total_legit = 0
        completed_chunks_count = 0

        logger.info(
            "Starting batch scoring: %d records, chunk_size=%d (%d total chunks), threshold=%.2f, max_retries=%d",
            total_records, chunk_size, total_chunks, threshold, max_retries
        )

        for chunk_idx, i in enumerate(range(0, total_records, chunk_size)):
            chunk_num = chunk_idx + 1
            raw_chunk = transactions[i : i + chunk_size]
            row_start = i
            row_end = min(i + chunk_size, total_records)

            # Sanitize payload: strictly retain canonical 30 features, exclude 'Class' or metadata
            clean_chunk = [
                {k: float(row[k]) for k in FEATURE_ORDER if k in row}
                for row in raw_chunk
            ]
            payload = {"transactions": clean_chunk}

            chunk_success = False
            last_chunk_error = ""

            # Bounded retry loop with exponential backoff for this chunk only
            for attempt in range(max_retries + 1):
                attempt_num = attempt + 1
                req_id = f"fg_c{chunk_num}_a{attempt_num}_{uuid.uuid4().hex[:6]}"
                headers = self._get_headers({
                    "Content-Type": "application/json",
                    "X-Request-ID": req_id,
                })

                t_start = time.time()
                status_code: Optional[int] = None
                response_text = ""
                error_category = None

                try:
                    response = requests.post(
                        url,
                        params=params,
                        json=payload,
                        headers=headers,
                        timeout=max(self.timeout, 30.0),
                    )
                    latency_ms = round((time.time() - t_start) * 1000, 2)
                    status_code = response.status_code
                    response_text = response.text or ""

                    # Check for server-provided request ID in headers
                    server_req_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or req_id

                    if response.status_code == 200:
                        chunk_result = response.json()
                        chunk_preds = chunk_result.get("predictions", [])
                        f_count = chunk_result.get("fraud_count", 0)
                        l_count = chunk_result.get("legitimate_count", 0)

                        total_fraud += f_count
                        total_legit += l_count
                        all_predictions.extend(chunk_preds)
                        completed_chunks_count += 1
                        chunk_success = True

                        log_entry = {
                            "chunk_number": chunk_num,
                            "total_chunks": total_chunks,
                            "row_start": row_start,
                            "row_end": row_end,
                            "attempt": attempt_num,
                            "max_attempts": max_retries + 1,
                            "status_code": 200,
                            "status": "SUCCESS",
                            "latency_ms": latency_ms,
                            "request_id": server_req_id,
                            "items_processed": len(chunk_preds),
                        }
                        chunk_telemetry_logs.append(log_entry)
                        logger.info(
                            "Chunk %d/%d (rows %d-%d) attempt %d succeeded: HTTP 200 in %.1fms (req_id=%s)",
                            chunk_num, total_chunks, row_start, row_end, attempt_num, latency_ms, server_req_id
                        )

                        if progress_callback:
                            try:
                                progress_callback(log_entry)
                            except Exception:
                                pass
                        break  # Chunk completed successfully, move to next chunk

                    elif response.status_code in TRANSIENT_STATUS_CODES:
                        error_category = f"Transient HTTP {response.status_code}"
                        snippet = _sanitize_error_snippet(response_text)
                        last_chunk_error = (
                            f"Batch chunk {chunk_num}/{total_chunks} (rows {row_start}:{row_end}) failed "
                            f"(HTTP {response.status_code}): {snippet}"
                        )
                        logger.warning(
                            "Chunk %d/%d attempt %d/%d hit transient HTTP %d in %.1fms (req_id=%s): %s",
                            chunk_num, total_chunks, attempt_num, max_retries + 1, response.status_code,
                            latency_ms, server_req_id, snippet
                        )
                    else:
                        # Non-transient client/server error (e.g., 400, 422 validation failure)
                        snippet = _sanitize_error_snippet(response_text)
                        last_chunk_error = (
                            f"Batch chunk {chunk_num}/{total_chunks} (rows {row_start}:{row_end}) failed "
                            f"with fatal error (HTTP {response.status_code}): {snippet}"
                        )
                        logger.error(
                            "Chunk %d/%d attempt %d encountered fatal error HTTP %d: %s",
                            chunk_num, total_chunks, attempt_num, response.status_code, snippet
                        )
                        log_entry = {
                            "chunk_number": chunk_num,
                            "total_chunks": total_chunks,
                            "row_start": row_start,
                            "row_end": row_end,
                            "attempt": attempt_num,
                            "status_code": response.status_code,
                            "status": "FATAL_ERROR",
                            "latency_ms": latency_ms,
                            "request_id": server_req_id,
                            "response_snippet": snippet,
                        }
                        chunk_telemetry_logs.append(log_entry)
                        if progress_callback:
                            try:
                                progress_callback(log_entry)
                            except Exception:
                                pass
                        break  # Fatal error, do not retry

                except (ConnectionError, Timeout) as exc:
                    latency_ms = round((time.time() - t_start) * 1000, 2)
                    err_type = "Timeout" if isinstance(exc, Timeout) else "ConnectionError"
                    error_category = f"Network {err_type}"
                    last_chunk_error = (
                        f"Batch chunk {chunk_num}/{total_chunks} (rows {row_start}:{row_end}) "
                        f"{err_type.lower()} after {latency_ms}ms: {str(exc)}"
                    )
                    logger.warning(
                        "Chunk %d/%d attempt %d/%d encountered %s in %.1fms (req_id=%s): %s",
                        chunk_num, total_chunks, attempt_num, max_retries + 1, err_type,
                        latency_ms, req_id, str(exc)
                    )
                except RequestException as exc:
                    latency_ms = round((time.time() - t_start) * 1000, 2)
                    last_chunk_error = (
                        f"Batch chunk {chunk_num}/{total_chunks} unexpected network exception: {str(exc)}"
                    )
                    logger.error("Chunk %d/%d attempt %d request exception: %s", chunk_num, total_chunks, attempt_num, str(exc))
                    break

                # Apply exponential backoff with jitter if attempts remain
                if attempt < max_retries and error_category is not None:
                    backoff_delay = backoff_base * (2 ** attempt) + random.uniform(0.05, 0.25)
                    log_entry = {
                        "chunk_number": chunk_num,
                        "total_chunks": total_chunks,
                        "row_start": row_start,
                        "row_end": row_end,
                        "attempt": attempt_num,
                        "max_attempts": max_retries + 1,
                        "status_code": status_code,
                        "status": f"RETRYING_IN_{backoff_delay:.2f}s",
                        "latency_ms": latency_ms,
                        "request_id": req_id,
                        "error_category": error_category,
                        "response_snippet": response_text[:200] if response_text else None,
                    }
                    chunk_telemetry_logs.append(log_entry)
                    if progress_callback:
                        try:
                            progress_callback(log_entry)
                        except Exception:
                            pass

                    logger.info("Retrying chunk %d (attempt %d/%d) in %.2fs...", chunk_num, attempt_num + 1, max_retries + 1, backoff_delay)
                    time.sleep(backoff_delay)

            # If chunk failed after exhausting all attempts, stop and return partial result
            if not chunk_success:
                logger.error(
                    "Batch processing aborted at chunk %d/%d. Total successfully scored: %d/%d records.",
                    chunk_num, total_chunks, len(all_predictions), total_records
                )
                partial_result = {
                    "total_processed": len(all_predictions),
                    "total_requested": total_records,
                    "completed_chunks": completed_chunks_count,
                    "total_chunks": total_chunks,
                    "failed_chunk_number": chunk_num,
                    "failed_chunk_row_start": row_start,
                    "failed_chunk_row_end": row_end,
                    "fraud_count": total_fraud,
                    "legitimate_count": total_legit,
                    "fraud_rate": round(total_fraud / len(all_predictions), 4) if all_predictions else 0.0,
                    "predictions": all_predictions,
                    "is_partial": True,
                    "telemetry_logs": chunk_telemetry_logs,
                }
                return partial_result, last_chunk_error

        # All chunks completed successfully
        logger.info(
            "Batch scoring finished successfully: %d/%d records processed in %d chunks. Flagged fraud: %d (%.2f%%)",
            len(all_predictions), total_records, total_chunks, total_fraud,
            (total_fraud / len(all_predictions) * 100.0) if all_predictions else 0.0
        )
        aggregated = {
            "total_processed": len(all_predictions),
            "total_requested": total_records,
            "completed_chunks": total_chunks,
            "total_chunks": total_chunks,
            "fraud_count": total_fraud,
            "legitimate_count": total_legit,
            "fraud_rate": round(total_fraud / len(all_predictions), 4) if all_predictions else 0.0,
            "predictions": all_predictions,
            "is_partial": False,
            "telemetry_logs": chunk_telemetry_logs,
        }
        return aggregated, None

    def get_model_info(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetches active model metadata from /model/info."""
        url = f"{self.base_url}/model/info"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json(), None
            elif resp.status_code == 401:
                return None, "Unauthorized: Authentication required to access model info."
            return None, f"Failed to get model info (HTTP {resp.status_code}): {resp.text}"
        except Exception as exc:
            return None, f"Error contacting /model/info: {str(exc)}"

    def get_model_metrics(self, threshold: float = 0.50) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetches ground-truth evaluation metrics and bootstrap 95% CIs from /model/metrics."""
        url = f"{self.base_url}/model/metrics"
        try:
            resp = requests.get(
                url, params={"threshold": threshold}, headers=self._get_headers(), timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json(), None
            elif resp.status_code == 401:
                return None, "Unauthorized: Authentication required to access model metrics."
            elif resp.status_code == 403:
                return None, "Forbidden: MLOps Administrator role required to access model metrics."
            return None, f"Failed to get model metrics (HTTP {resp.status_code}): {resp.text}"
        except Exception as exc:
            return None, f"Error contacting /model/metrics: {str(exc)}"

    def get_threshold_recommendations(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetches threshold optimization recommendations from /model/threshold."""
        url = f"{self.base_url}/model/threshold"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json(), None
            elif resp.status_code == 401:
                return None, "Unauthorized: Authentication required to access threshold recommendations."
            elif resp.status_code == 403:
                return None, "Forbidden: MLOps Administrator role required to access threshold recommendations."
            return None, f"Failed to get threshold recommendations (HTTP {resp.status_code}): {resp.text}"
        except Exception as exc:
            return None, f"Error contacting /model/threshold: {str(exc)}"

    def get_model_comparison(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetches candidate model benchmark manifest from /model/compare."""
        url = f"{self.base_url}/model/compare"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json(), None
            elif resp.status_code == 401:
                return None, "Unauthorized: Authentication required to access model comparison."
            elif resp.status_code == 403:
                return None, "Forbidden: MLOps Administrator role required to access model comparison."
            return None, f"Failed to get model comparison (HTTP {resp.status_code}): {resp.text}"
        except Exception as exc:
            return None, f"Error contacting /model/compare: {str(exc)}"

    def predict_csv_upload(
        self,
        file_bytes: bytes,
        filename: str = "transactions.csv",
        threshold: float = 0.50,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Sends an uploaded CSV file directly to the protected /predict/csv endpoint."""
        url = f"{self.base_url}/predict/csv"
        params = {"threshold": float(threshold)}
        headers = self._get_headers()
        files = {"file": (filename, file_bytes, "text/csv")}
        try:
            resp = requests.post(
                url, files=files, params=params, headers=headers, timeout=max(self.timeout, 60.0)
            )
            if resp.status_code == 200:
                return resp.json(), None
            elif resp.status_code == 401:
                return None, "Unauthorized: Valid authentication token required for CSV scoring."
            elif resp.status_code == 403:
                return None, "Forbidden: Insufficient privileges for CSV scoring."
            elif resp.status_code == 413:
                return None, f"Payload Too Large: {resp.json().get('detail', resp.text)}"
            else:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
                return None, f"CSV prediction failed (HTTP {resp.status_code}): {detail}"
        except Exception as exc:
            return None, f"CSV upload error: {str(exc)}"
