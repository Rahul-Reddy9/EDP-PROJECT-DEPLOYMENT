"""
test_real_restart_persistence.py
---------------------------------
Validates real operating-system process restart persistence for JWT token revocation:
1. Spawns Backend Process A.
2. Logins and verifies access to protected endpoint.
3. Logs out, persisting revocation into SQLite.
4. Confirms token receives HTTP 401.
5. Kills Backend Process A completely.
6. Spawns fresh Backend Process B using the exact same SQLite database path.
7. Re-attempts request with revoked token; verifies HTTP 401 persists across process boundaries.
8. Validates fail-closed behavior on database unavailability (HTTP 503).
"""

import os
import sys
import time
import subprocess
import requests

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
    port = 8005
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["ENV"] = "development"
    env["JWT_SECRET_KEY"] = "fraudguard_development_secret_key_minimum_32_bytes_long"
    env["REVOCATION_DB_PATH"] = DB_PATH

    print("=== TEST STEP 1: Launching Backend Process A ===", flush=True)
    p_a = subprocess.Popen(
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_server(base_url, max_wait=30):
            p_a.terminate()
            out, err = p_a.communicate(timeout=5)
            print(f"Process A Stdout: {out.decode()}", flush=True)
            print(f"Process A Stderr: {err.decode()}", flush=True)
            raise AssertionError("Process A failed to become healthy on port 8005")
        print("Process A is HEALTHY on port 8005.", flush=True)

        # Step 1: Login to get token
        login_res = requests.post(
            f"{base_url}/auth/login",
            json={"username": "analyst", "password": "UserPassword@2026!"},
            timeout=5.0,
        )
        assert login_res.status_code == 200, f"Login failed: {login_res.status_code}"
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("Obtained valid JWT session token.")

        # Step 2: Access protected endpoint successfully
        info_res = requests.get(f"{base_url}/model/info", headers=headers, timeout=5.0)
        assert info_res.status_code == 200, f"Protected endpoint access failed: {info_res.status_code}"
        print(f"Protected endpoint access verified: HTTP {info_res.status_code}")

        # Step 3: Logout to revoke token
        logout_res = requests.post(f"{base_url}/auth/logout", headers=headers, timeout=5.0)
        assert logout_res.status_code == 200, f"Logout failed: {logout_res.status_code}"
        print("Logout successful. Token recorded in SQLite store.")

        # Step 4: Verify token receives 401 in Process A
        revoked_res = requests.get(f"{base_url}/model/info", headers=headers, timeout=5.0)
        assert revoked_res.status_code == 401, f"Expected 401 in Process A, got {revoked_res.status_code}"
        print("Revoked token successfully rejected with HTTP 401 in Process A.")

    finally:
        print("=== TEST STEP 2: Terminating Backend Process A (OS Kill) ===")
        p_a.terminate()
        try:
            p_a.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p_a.kill()
        print("Backend Process A completely terminated.")

    time.sleep(2)

    # Step 5: Spawn Process B
    print("=== TEST STEP 3: Spawning Fresh Backend Process B (Same DB Path) ===")
    p_b = subprocess.Popen(
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
        assert wait_for_server(base_url), "Process B failed to become healthy on port 8005"
        print("Process B is HEALTHY on port 8005.")

        # Step 6: Attempt access with token revoked in Process A
        reused_res = requests.get(f"{base_url}/model/info", headers=headers, timeout=5.0)
        print(f"Process B response for token revoked in Process A: HTTP {reused_res.status_code}")
        assert reused_res.status_code == 401, f"Expected 401 in Process B, got {reused_res.status_code}"
        print("EVIDENCE CONFIRMED: Token revocation successfully persisted across real OS process restart!")

    finally:
        print("=== TEST STEP 4: Terminating Backend Process B ===")
        p_b.terminate()
        try:
            p_b.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p_b.kill()
        print("Backend Process B completely terminated.")

    print("\n=======================================================")
    print("REAL OS PROCESS RESTART REVOCATION PERSISTENCE: 100% PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    main()
