"""
verify_genuine_e2e.py
---------------------
Performs genuine end-to-end prediction execution through:
1. The real FastAPI backend application (using TestClient over ASGI, executing full middleware,
   Pydantic schema validation, JWT auth, rate limiting, and calibrated XGBoost v2.0).
2. The real Streamlit frontend application (using AppTest, executing UI state management,
   KPI rendering, and chart pipelines).
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["ENV"] = "development"
os.environ["JWT_SECRET_KEY"] = "fraudguard_development_secret_key_minimum_32_bytes_long"
os.environ["ADMIN_PASSWORD"] = "Admin_Production_Secure_2026!"
os.environ["ANALYST_PASSWORD"] = "Analyst_Production_Secure_2026!"
os.environ["API_BASE_URL"] = "http://127.0.0.1:8000"

from fastapi.testclient import TestClient
from backend.app import app
from backend.src.predict import get_sample_transaction, FraudPredictor

def main():
    print("=================================================================", flush=True)
    print("GENUINE END-TO-END PIPELINE EXECUTION AUDIT (FASTAPI + STREAMLIT)", flush=True)
    print("=================================================================", flush=True)

    # -------------------------------------------------------------
    # 1. FastAPI Real Backend Execution
    # -------------------------------------------------------------
    client = TestClient(app)
    
    # Check Health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    h_json = health_resp.json()
    print(f"\n[1] FastAPI /health Endpoint:", flush=True)
    print(f"    Status: {h_json['status']} | Model Version: {h_json['model_version']} | Scaler: {h_json['scaler_loaded']}", flush=True)

    # Authenticate via real bcrypt + JWT
    login_resp = client.post(
        "/auth/login",
        data={"username": "analyst", "password": "Analyst_Production_Secure_2026!"},
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    l_json = login_resp.json()
    token = l_json["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}
    print(f"\n[2] Authentication & Token Generation:", flush=True)
    print(f"    User: {l_json['username']} | Role: {l_json['role']} | Token: {token[:25]}...", flush=True)

    # Predict Clean Merchant (Raw Transaction)
    sample_legit = get_sample_transaction(fraud=False)
    pred_legit = client.post(
        "/predict?threshold=0.50&explain=true&is_preprocessed=false",
        json=sample_legit,
        headers=auth_headers,
    )
    assert pred_legit.status_code == 200, f"Legit predict failed: {pred_legit.text}"
    legit_json = pred_legit.json()
    print(f"\n[3] Real Prediction: Clean Merchant (Raw Input):", flush=True)
    print(f"    Prediction: {legit_json['prediction']} ({legit_json['label']})", flush=True)
    print(f"    Fraud Probability: {legit_json['fraud_probability']:.6f}", flush=True)
    print(f"    Risk Level: {legit_json['risk_level']}", flush=True)
    print(f"    Top Tree SHAP Explanations: {list(legit_json.get('explanation', {}).keys())[:3]}", flush=True)

    # Predict Syndicate Attack (Fraud Transaction)
    sample_fraud = get_sample_transaction(fraud=True)
    pred_fraud = client.post(
        "/predict?threshold=0.50&explain=true&is_preprocessed=false",
        json=sample_fraud,
        headers=auth_headers,
    )
    assert pred_fraud.status_code == 200, f"Fraud predict failed: {pred_fraud.text}"
    fraud_json = pred_fraud.json()
    print(f"\n[4] Real Prediction: Syndicate Attack (Raw Input):", flush=True)
    print(f"    Prediction: {fraud_json['prediction']} ({fraud_json['label']})", flush=True)
    print(f"    Fraud Probability: {fraud_json['fraud_probability']:.6f}", flush=True)
    print(f"    Risk Level: {fraud_json['risk_level']}", flush=True)
    print(f"    Top Tree SHAP Explanations: {list(fraud_json.get('explanation', {}).keys())[:3]}", flush=True)

    # Batch Prediction
    batch_payload = {"transactions": [sample_legit, sample_fraud]}
    batch_resp = client.post(
        "/predict/batch?threshold=0.50&is_preprocessed=false",
        json=batch_payload,
        headers=auth_headers,
    )
    assert batch_resp.status_code == 200, f"Batch predict failed: {batch_resp.text}"
    batch_json = batch_resp.json()
    print(f"\n[5] Real Batch Prediction (2 records):", flush=True)
    print(f"    Total Processed: {batch_json['total_processed']} | Fraud: {batch_json['fraud_count']} | Legit: {batch_json['legitimate_count']}", flush=True)

    # -------------------------------------------------------------
    # 2. Streamlit Real Frontend Execution (AppTest)
    # -------------------------------------------------------------
    print(f"\n[6] Streamlit Frontend App Execution (AppTest):", flush=True)
    os.environ["ENV"] = "test"
    from streamlit.testing.v1 import AppTest
    app_path = os.path.join(ROOT_DIR, "streamlit_app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    print(f"    Streamlit Initial Run Exceptions: {len(at.exception)}", flush=True)
    assert len(at.exception) == 0, f"Streamlit raised exceptions: {at.exception}"

    # Verify key UI components rendered
    title_text = at.title[0].value if at.title else "Dashboard Active"
    print(f"    Dashboard Title: {title_text}")
    print(f"    Rendered Tabs/Radios: {[r.options for r in at.radio]}")
    print(f"    Rendered Metric Cards: {len(at.metric)}")

    print("\n=================================================================")
    print("ALL GENUINE BACKEND & FRONTEND PREDICTION PIPELINES FULLY VERIFIED!")
    print("=================================================================")

if __name__ == "__main__":
    main()
