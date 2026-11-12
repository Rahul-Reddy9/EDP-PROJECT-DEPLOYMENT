"""
verify_live_e2e.py
------------------
Performs a genuine live end-to-end integration verification:
1. Boots real FastAPI backend via Uvicorn in a background thread on 127.0.0.1:8009
2. Probes /health endpoint over live HTTP loopback
3. Authenticates with credentials to obtain a cryptographic JWT token
4. Dispatches live POST /predict transaction with explicit scaling
5. Dispatches live POST /predict/batch multi-record transaction
6. Tests JWT token revocation and fail-closed security
7. Safely shuts down the Uvicorn server
"""

import os
import sys
import time
import threading
import requests
import uvicorn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Set environment before importing app
os.environ["ENV"] = "development"
os.environ["JWT_SECRET_KEY"] = "fraudguard_development_secret_key_minimum_32_bytes_long"
os.environ["ADMIN_PASSWORD"] = "Admin_Production_Secure_2026!"
os.environ["ANALYST_PASSWORD"] = "Analyst_Production_Secure_2026!"

from backend.app import app
from backend.src.predict import get_sample_transaction

def run_server(server: uvicorn.Server):
    try:
        import asyncio
        asyncio.run(server.serve())
    except Exception as e:
        print(f"[E2E Server Exception]: {e}", flush=True)

def main():
    port = 8009
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    server_thread = threading.Thread(target=server.run, daemon=True)

    print(f"[E2E] Starting live Uvicorn microservice in background thread on {base_url}...")
    server_thread.start()

    try:
        # Step 1: Wait for server to become responsive
        max_retries = 40
        is_up = False
        health_resp = None
        for i in range(max_retries):
            try:
                r = requests.get(f"{base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    health_resp = r.json()
                    is_up = True
                    break
            except Exception:
                time.sleep(0.25)

        if not is_up:
            raise RuntimeError(f"Server failed to become responsive on {base_url} within 10 seconds.")

        print(f"[E2E] Server is HEALTHY: status={health_resp.get('status')}, version={health_resp.get('model_version')}")

        # Step 2: Authenticate to obtain real JWT token
        print("[E2E] Authenticating via POST /auth/login with analyst credentials...")
        login_resp = requests.post(
            f"{base_url}/auth/login",
            data={"username": "analyst", "password": "Analyst_Production_Secure_2026!"},
            timeout=5.0,
        )
        assert login_resp.status_code == 200, f"Login failed: {login_resp.status_code} {login_resp.text}"
        login_data = login_resp.json()
        token = login_data["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}
        print(f"[E2E] Authentication successful! JWT Token acquired (prefix: {token[:20]}..., role: {login_data.get('role')})")

        # Step 3: Fetch legitimate and fraudulent sample presets
        sample_legit = get_sample_transaction(fraud=False)
        sample_fraud = get_sample_transaction(fraud=True)

        # Step 4: Execute Genuine Live Prediction: Legitimate Transaction
        print("[E2E] Sending live single prediction request (Clean Merchant)...")
        t0 = time.time()
        pred_legit_resp = requests.post(
            f"{base_url}/predict?threshold=0.50&explain=true&is_preprocessed=false",
            json=sample_legit,
            headers=auth_headers,
            timeout=5.0,
        )
        lat_legit_ms = (time.time() - t0) * 1000.0
        assert pred_legit_resp.status_code == 200, f"Prediction failed: {pred_legit_resp.status_code} {pred_legit_resp.text}"
        legit_data = pred_legit_resp.json()
        print(f"[E2E] Single Legitimate Prediction: HTTP {pred_legit_resp.status_code} in {lat_legit_ms:.2f}ms")
        print(f"      Label: {legit_data['label']} | Prob: {legit_data['fraud_probability']:.6f} | Risk: {legit_data['risk_level']}")
        assert legit_data["prediction"] == 0
        assert legit_data["label"] == "Legitimate"
        assert legit_data["risk_level"] == "LOW"

        # Step 5: Execute Genuine Live Prediction: Fraudulent Transaction
        print("[E2E] Sending live single prediction request (Syndicate Attack)...")
        t0 = time.time()
        pred_fraud_resp = requests.post(
            f"{base_url}/predict?threshold=0.50&explain=true&is_preprocessed=false",
            json=sample_fraud,
            headers=auth_headers,
            timeout=5.0,
        )
        lat_fraud_ms = (time.time() - t0) * 1000.0
        assert pred_fraud_resp.status_code == 200, f"Prediction failed: {pred_fraud_resp.status_code} {pred_fraud_resp.text}"
        fraud_data = pred_fraud_resp.json()
        print(f"[E2E] Single Fraud Prediction: HTTP {pred_fraud_resp.status_code} in {lat_fraud_ms:.2f}ms")
        print(f"      Label: {fraud_data['label']} | Prob: {fraud_data['fraud_probability']:.6f} | Risk: {fraud_data['risk_level']}")
        assert fraud_data["prediction"] == 1
        assert fraud_data["label"] == "Fraudulent"
        assert fraud_data["risk_level"] == "HIGH"

        # Step 6: Execute Genuine Live Batch Prediction
        print("[E2E] Sending live batch prediction request (2 transactions)...")
        batch_payload = {"transactions": [sample_legit, sample_fraud]}
        t0 = time.time()
        batch_resp = requests.post(
            f"{base_url}/predict/batch?threshold=0.50&is_preprocessed=false",
            json=batch_payload,
            headers=auth_headers,
            timeout=5.0,
        )
        lat_batch_ms = (time.time() - t0) * 1000.0
        assert batch_resp.status_code == 200, f"Batch prediction failed: {batch_resp.status_code} {batch_resp.text}"
        batch_data = batch_resp.json()
        print(f"[E2E] Batch Prediction: HTTP {batch_resp.status_code} in {lat_batch_ms:.2f}ms")
        print(f"      Processed: {batch_data['total_processed']} | Fraud: {batch_data['fraud_count']} | Legit: {batch_data['legitimate_count']}")
        assert batch_data["total_processed"] == 2
        assert batch_data["fraud_count"] == 1
        assert batch_data["legitimate_count"] == 1

        # Step 7: Test Real Logout & Persistent Revocation
        print("[E2E] Testing token revocation via POST /auth/logout...")
        logout_resp = requests.post(f"{base_url}/auth/logout", headers=auth_headers, timeout=5.0)
        assert logout_resp.status_code == 200, f"Logout failed: {logout_resp.status_code}"
        print(f"[E2E] Logout successful: {logout_resp.json().get('message')}")

        # Step 8: Verify Revoked Token Rejection
        print("[E2E] Verifying revoked token rejection on subsequent request...")
        reused_resp = requests.post(
            f"{base_url}/predict?threshold=0.50",
            json=sample_legit,
            headers=auth_headers,
            timeout=5.0,
        )
        print(f"[E2E] Reused Token Response: HTTP {reused_resp.status_code} (detail: {reused_resp.json().get('detail')})")
        assert reused_resp.status_code == 401, f"Expected 401 Unauthorized for revoked token, got {reused_resp.status_code}"

        print("\n=======================================================")
        print("GENUINE LIVE END-TO-END HTTP VERIFICATION: 100% PASSED!")
        print("=======================================================")

    finally:
        print("[E2E] Shutting down live Uvicorn microservice...")
        server.should_exit = True
        server_thread.join(timeout=3)
        print("[E2E] Uvicorn microservice stopped cleanly.")

if __name__ == "__main__":
    main()
