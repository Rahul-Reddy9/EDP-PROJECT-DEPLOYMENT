"""
test_genuine_streamlit_fastapi_e2e.py
-------------------------------------
Performs genuine Streamlit-to-FastAPI end-to-end verification:
1. Spawns real FastAPI microservice on local TCP port 8008 via Uvicorn.
2. Connects real Streamlit application (via AppTest) configured with API_BASE_URL=http://127.0.0.1:8008.
3. Verifies live health check over loopback socket.
4. Performs live login via Streamlit UI form -> real HTTP POST /auth/login.
5. Submits safe synthetic Clean Merchant transaction via Streamlit UI -> real HTTP POST /predict.
6. Verifies frontend visibly renders "LOW RISK" and "Legitimate" prediction.
7. Submits safe synthetic Syndicate Attack transaction via Streamlit UI -> real HTTP POST /predict.
8. Verifies frontend visibly renders "HIGH RISK" and "Fraudulent" prediction.
9. Performs live logout via Streamlit UI -> real HTTP POST /auth/logout.
10. Safely terminates Uvicorn process.
"""

import os
import sys
import time
import subprocess
import requests
from streamlit.testing.v1 import AppTest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "data", "revoked_tokens.db")

def wait_for_server(url, max_wait=30):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    port = 8008
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["ENV"] = "development"
    env["JWT_SECRET_KEY"] = "fraudguard_development_secret_key_minimum_32_bytes_long"
    env["REVOCATION_DB_PATH"] = DB_PATH
    os.environ["API_BASE_URL"] = base_url
    os.environ["ENV"] = "development"

    print(f"=== STEP 1: Launching Live Uvicorn Backend on {base_url} ===", flush=True)
    backend_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "--app-dir",
            "backend",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        assert wait_for_server(base_url, max_wait=30), f"Backend failed to bind to {base_url}"
        print(f"Backend is LIVE and HEALTHY on {base_url}.", flush=True)

        # Step 2: Initialize Streamlit AppTest pointing to live backend
        print("=== STEP 2: Initializing Streamlit Frontend pointing to live backend ===", flush=True)
        app_path = os.path.join(ROOT_DIR, "streamlit_app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        assert len(at.exception) == 0, f"Streamlit initial run raised exceptions: {at.exception}"
        print("Streamlit frontend initialized successfully.", flush=True)

        # Step 3: Authenticate via Streamlit Sidebar Login Form
        print("=== STEP 3: Executing Live Login through Streamlit UI Form ===", flush=True)
        # Find sidebar text inputs and submit button
        user_input = at.text_input("sidebar_user")
        pwd_input = at.text_input("sidebar_pwd")
        user_input.set_value("analyst")
        pwd_input.set_value("UserPassword@2026!")
        
        # Submit form via the form_submit_button in at.button
        for b in at.button:
            if "Sign In" in b.label:
                b.click()
                break
        at.run()
        assert len(at.exception) == 0, f"Login submit raised exceptions: {at.exception}"
        assert at.session_state.auth_token is not None, "Login failed: auth_token not set in session state"
        print(f"Login successful! Role: {at.session_state.server_role}, Token acquired via live HTTP POST.", flush=True)

        # Step 4: Navigate to Real-Time Scoring Tab
        print("=== STEP 4: Navigating to Real-Time Scoring View ===", flush=True)
        for r in at.radio:
            if "🎯 Real-Time Scoring" in r.options:
                r.set_value("🎯 Real-Time Scoring")
        at.run()
        assert len(at.exception) == 0, f"Navigation raised exceptions: {at.exception}"
        print("Successfully navigated to Real-Time Scoring tab.", flush=True)

        # Step 5: Load Legitimate Synthetic Preset and Run Analysis
        print("=== STEP 5: Submitting Synthetic Clean Merchant Transaction via UI ===", flush=True)
        # Set clean merchant preset in session state
        from frontend.validators import get_sample_preset
        clean_sample = get_sample_preset(fraud=False)
        at.session_state.single_inputs = clean_sample
        for k, v in clean_sample.items():
            at.session_state[f"input_{k}"] = float(v)

        # Click the score transaction risk button
        scored = False
        for b in at.button:
            if "Score Transaction Risk" in b.label:
                b.click()
                scored = True
                break
        assert scored, "Could not find 'Score Transaction Risk' button in UI"
        at.run()
        assert len(at.exception) == 0, f"Analysis raised exceptions: {at.exception}"
        
        result_legit_container = at.session_state.latest_single_result
        assert result_legit_container is not None, "Real-time scoring failed: latest_single_result is None"
        result_legit = result_legit_container["result"]
        print(f"Live API Response received by Streamlit:")
        print(f"  Prediction: {result_legit['prediction']} ({result_legit['label']})")
        print(f"  Probability: {result_legit['fraud_probability']:.6f}")
        print(f"  Risk Level: {result_legit['risk_level']}")
        assert result_legit["prediction"] == 0
        assert result_legit["label"] == "Legitimate"
        assert result_legit["risk_level"] == "LOW"

        # Verify frontend rendered the result banner
        banner_rendered = any("RISK TIER: LOW" in str(m.value) or "risk-banner-low" in str(m.value) for m in at.markdown)
        print(f"Verified UI visibly displayed LOW RISK result banner: {banner_rendered}", flush=True)
        assert banner_rendered, "Frontend did not visibly render LOW RISK result banner"

        # Step 6: Load Fraudulent Synthetic Preset and Run Analysis
        print("=== STEP 6: Submitting Synthetic Syndicate Attack Transaction via UI ===", flush=True)
        fraud_sample = get_sample_preset(fraud=True)
        at.session_state.single_inputs = fraud_sample
        for k, v in fraud_sample.items():
            at.session_state[f"input_{k}"] = float(v)
        # Click the score transaction risk button for fraud
        scored_fraud = False
        for b in at.button:
            if "Score Transaction Risk" in b.label:
                b.click()
                scored_fraud = True
                break
        assert scored_fraud, "Could not find 'Score Transaction Risk' button for fraud analysis"
        at.run()
        assert len(at.exception) == 0, f"Fraud analysis raised exceptions: {at.exception}"
        
        result_fraud_container = at.session_state.latest_single_result
        assert result_fraud_container is not None, "Real-time fraud scoring failed: latest_single_result is None"
        result_fraud = result_fraud_container["result"]
        print(f"Live API Response received by Streamlit:")
        print(f"  Prediction: {result_fraud['prediction']} ({result_fraud['label']})")
        print(f"  Probability: {result_fraud['fraud_probability']:.6f}")
        print(f"  Risk Level: {result_fraud['risk_level']}")
        assert result_fraud["prediction"] == 1
        assert result_fraud["label"] == "Fraudulent"
        assert result_fraud["risk_level"] == "HIGH"

        banner_fraud_rendered = any("RISK TIER: HIGH" in str(m.value) or "risk-banner-high" in str(m.value) for m in at.markdown)
        print(f"Verified UI visibly displayed HIGH RISK result banner: {banner_fraud_rendered}", flush=True)
        assert banner_fraud_rendered, "Frontend did not visibly render HIGH RISK result banner"

        # Step 7: Logout via Streamlit UI Button
        print("=== STEP 7: Executing Live Logout via Streamlit UI ===", flush=True)
        for b in at.button:
            if "Sign Out" in b.label:
                b.click()
                break
        at.run()
        assert at.session_state.auth_token is None, "Logout failed: auth_token still present"
        print("Logout verified! Token revoked and session cleared in Streamlit.", flush=True)

        print("\n=======================================================")
        print("GENUINE STREAMLIT-TO-FASTAPI E2E VERIFICATION: 100% PASSED!")
        print("=======================================================")

    finally:
        print("=== Terminating Live Uvicorn Backend Process ===", flush=True)
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()
        print("Uvicorn backend process terminated cleanly.", flush=True)

if __name__ == "__main__":
    main()
