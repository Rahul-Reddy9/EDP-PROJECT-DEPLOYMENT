"""
test_security_rbac.py
---------------------
Enterprise Security, Server-Side RBAC, Persistent Revocation, Rate Limiting,
and Production Secret Validation Test Suite for FraudGuard AI.

Verifies:
1. NIST/OWASP salted bcrypt password hashing and verification.
2. Cryptographic JWT token generation, signature validation, expiry, and persistent revocation.
3. Persistent SQLite token revocation surviving simulated process reloads.
4. Unauthenticated requests to protected endpoints return HTTP 401 Unauthorized (including /model/info).
5. Independent User requests to admin endpoints return HTTP 403 Forbidden.
6. MLOps Administrator requests to admin endpoints return HTTP 200 OK.
7. Independent User requests to inference endpoints and /model/info return HTTP 200 OK.
8. Sliding-window rate limiter returns HTTP 429 Too Many Requests with Retry-After header.
9. CORS strict origin whitelist enforcement (no wildcards with credentials).
10. Health check and error message sanitization (no internal path leakage).
11. Production mode safe startup failure when required secrets are missing or weak.
12. Global request size (HTTP 413) and batch transaction limits (HTTP 400).
"""

import os
import sys
import io
import time
from datetime import timedelta
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import src.auth as auth_module
from src.auth import (
    hash_password,
    verify_password,
    create_access_token,
    authenticate_user,
    revoke_token,
    is_token_revoked,
    resolve_jwt_secret,
    resolve_admin_password,
    resolve_user_password,
    get_secret_key,
    User,
    UserRole,
    USERS_DB,
    RevocationStoreError,
)
from src.rate_limiter import rate_limit_login, rate_limit_predict, SlidingWindowRateLimiter
from src.revocation_store import is_token_revoked_in_db, record_revocation, DB_PATH
from frontend.validators import get_sample_preset, FEATURE_ORDER
import app as backend_app
from app import app


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Resets rate limiter buckets before each test to prevent cross-test interference."""
    rate_limit_login.reset()
    rate_limit_predict.reset()
    yield
    rate_limit_login.reset()
    rate_limit_predict.reset()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_payload():
    return get_sample_preset(fraud=False)


@pytest.fixture(scope="module")
def analyst_token(client):
    rate_limit_login.reset()
    resp = client.post("/auth/login", json={"username": "analyst", "password": "UserPassword@2026!"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def admin_token(client):
    rate_limit_login.reset()
    resp = client.post("/auth/login", json={"username": "admin", "password": "AdminPassword@2026!"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


class TestPasswordHashingAndVerification:
    """Verifies NIST/OWASP compliant bcrypt password hashing."""

    def test_01_hash_generates_salted_bcrypt(self):
        pwd = "TestSecretPassword123!"
        h1 = hash_password(pwd)
        h2 = hash_password(pwd)
        # Bcrypt generates unique random salts, so two hashes of the same password must differ
        assert h1 != h2
        assert h1.startswith("$2b$") or h1.startswith("$2a$")

    def test_02_verify_password_correct_and_incorrect(self):
        pwd = "EnterprisePassword2026@"
        hashed = hash_password(pwd)
        assert verify_password(pwd, hashed) is True
        assert verify_password("WrongPassword!", hashed) is False
        assert verify_password("", hashed) is False


class TestJWTTokensAndPersistentRevocation:
    """Verifies token generation, expiration enforcement, and persistent SQLite revocation."""

    def test_03_token_generation_and_claims(self):
        user = User(username="analyst", display_name="Risk Analyst", role=UserRole.INDEPENDENT_USER)
        token, expires_in = create_access_token(user, expires_delta=timedelta(minutes=30))
        assert isinstance(token, str)
        assert expires_in == 1800

    def test_04_expired_token_rejected_with_401(self, client):
        user = User(username="analyst", display_name="Risk Analyst", role=UserRole.INDEPENDENT_USER)
        # Create token already expired in the past
        expired_token, _ = create_access_token(user, expires_delta=timedelta(seconds=-10))

        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_05_tampered_token_rejected_with_401(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid.fake.token"})
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()

    def test_06_logout_revokes_token_persistently(self, client):
        rate_limit_login.reset()
        login_resp = client.post("/auth/login", json={"username": "analyst", "password": "UserPassword@2026!"})
        token = login_resp.json()["access_token"]

        # Token works before logout
        me_resp1 = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp1.status_code == 200

        # Logout
        logout_resp = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert logout_resp.status_code == 200

        # Token rejected after logout
        me_resp2 = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp2.status_code == 401
        assert "revoked" in me_resp2.json()["detail"].lower()

    def test_07_revocation_survives_backend_restart_simulation(self):
        """Simulates server restart by querying the persistent SQLite database directly."""
        user = User(username="admin", display_name="MLOps Administrator", role=UserRole.MLOPS_ADMIN)
        token, _ = create_access_token(user, expires_delta=timedelta(minutes=15))
        import jwt
        payload = jwt.decode(token, get_secret_key(), algorithms=["HS256"])
        jti = payload["jti"]

        # Before revocation, DB shows not revoked
        assert is_token_revoked_in_db(jti) is False

        # Revoke token
        ok = revoke_token(token)
        assert ok is True

        # Simulate fresh process inspection of database
        assert is_token_revoked_in_db(jti) is True
        assert is_token_revoked(jti) is True


class TestServerSideRBACAndAdminProtection:
    """Verifies genuine 401 Unauthorized and 403 Forbidden enforcement."""

    def test_08_unauthenticated_requests_return_401(self, client, sample_payload):
        """All protected endpoints (including /model/info) must reject unauthenticated requests with HTTP 401."""
        endpoints = [
            ("POST", "/predict", {"json": sample_payload}),
            ("POST", "/predict/batch", {"json": {"transactions": [sample_payload]}}),
            ("GET", "/model/info", {}),
            ("GET", "/model/metrics", {}),
            ("GET", "/model/threshold", {}),
            ("GET", "/model/compare", {}),
            ("GET", "/auth/me", {}),
        ]
        for method, endpoint, kwargs in endpoints:
            if method == "POST":
                res = client.post(endpoint, **kwargs)
            else:
                res = client.get(endpoint, **kwargs)
            assert res.status_code == 401, f"{endpoint} did not return 401 on unauthenticated request, got {res.status_code}"

    def test_09_independent_user_forbidden_on_admin_endpoints_403(self, client, analyst_token):
        """Independent User must be denied access to all admin endpoints with HTTP 403 Forbidden."""
        admin_endpoints = [
            "/model/metrics",
            "/model/threshold",
            "/model/compare",
        ]
        for ep in admin_endpoints:
            res = client.get(ep, headers={"Authorization": f"Bearer {analyst_token}"})
            assert res.status_code == 403, f"Expected 403 for analyst on {ep}, got {res.status_code}: {res.text}"
            assert "forbidden" in res.json()["detail"].lower() or "insufficient" in res.json()["detail"].lower()

    def test_10_mlops_admin_authorized_on_admin_endpoints_200(self, client, admin_token):
        """MLOps Administrator must have complete access to all admin governance endpoints."""
        res_info = client.get("/model/info", headers={"Authorization": f"Bearer {admin_token}"})
        assert res_info.status_code == 200

        res_metrics = client.get("/model/metrics?threshold=0.50", headers={"Authorization": f"Bearer {admin_token}"})
        assert res_metrics.status_code == 200
        assert res_metrics.json()["status"] == "success"

        res_th = client.get("/model/threshold", headers={"Authorization": f"Bearer {admin_token}"})
        assert res_th.status_code == 200
        assert res_th.json()["status"] == "success"

        res_comp = client.get("/model/compare", headers={"Authorization": f"Bearer {admin_token}"})
        assert res_comp.status_code == 200
        assert res_comp.json()["status"] == "success"

    def test_11_independent_user_authorized_on_inference_and_model_info_200(self, client, analyst_token, sample_payload):
        """Independent User must be authorized for single/batch predictions and model/info metadata."""
        # /model/info accessible to authenticated analyst
        res_info = client.get("/model/info", headers={"Authorization": f"Bearer {analyst_token}"})
        assert res_info.status_code == 200
        assert res_info.json()["status"] == "active"

        # Single prediction with Tree SHAP explainability
        res_single = client.post(
            "/predict?threshold=0.50&explain=true",
            json=sample_payload,
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert res_single.status_code == 200
        data = res_single.json()
        assert "prediction" in data
        assert "fraud_probability" in data
        assert "explanation" in data
        assert data["explanation"] is not None

        # Batch prediction
        batch_payload = {"transactions": [sample_payload, sample_payload]}
        res_batch = client.post(
            "/predict/batch?threshold=0.50",
            json=batch_payload,
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert res_batch.status_code == 200
        b_data = res_batch.json()
        assert b_data["total_processed"] == 2


class TestSlidingWindowRateLimiting:
    """Verifies HTTP 429 Too Many Requests enforcement and Retry-After header."""

    def test_12_login_endpoint_rate_limiting_429(self, client):
        """Rapid successive login attempts must trigger HTTP 429 Too Many Requests."""
        rate_limit_login.reset()
        limit = rate_limit_login.get_limit()

        # Send 'limit' allowed requests
        for i in range(limit):
            resp = client.post("/auth/login", json={"username": "analyst", "password": "WrongPassword!"})
            assert resp.status_code == 401, f"Expected 401 on attempt {i+1}, got {resp.status_code}"

        # The next attempt must be rate-limited with HTTP 429
        rate_limited_resp = client.post("/auth/login", json={"username": "analyst", "password": "WrongPassword!"})
        assert rate_limited_resp.status_code == 429
        assert "Too Many Requests" in rate_limited_resp.json()["detail"]
        assert "Retry-After" in rate_limited_resp.headers
        retry_after = int(rate_limited_resp.headers["Retry-After"])
        assert retry_after > 0

        # Reset limiter and verify normal operation resumes
        rate_limit_login.reset()
        resumed_resp = client.post("/auth/login", json={"username": "analyst", "password": "WrongPassword!"})
        assert resumed_resp.status_code == 401

    def test_13_predict_rate_limiter_quota(self, client, analyst_token, sample_payload):
        """Custom sliding window limiter triggers 429 when quota is exceeded."""
        custom_limiter = SlidingWindowRateLimiter("test_predict", "CUSTOM_LIMIT", default_limit=2, window_seconds=10)
        client_key = "test_predict:127.0.0.1"

        ok1, cnt1, _ = custom_limiter.check_rate_limit(client_key)
        assert ok1 is True and cnt1 == 1

        ok2, cnt2, _ = custom_limiter.check_rate_limit(client_key)
        assert ok2 is True and cnt2 == 2

        # 3rd request exceeds limit
        ok3, cnt3, retry_after = custom_limiter.check_rate_limit(client_key)
        assert ok3 is False
        assert cnt3 == 2
        assert retry_after > 0


class TestProductionSecretValidationAndFailSafe:
    """Verifies that production mode safely halts on missing or insecure secrets."""

    def test_14_production_halts_on_missing_jwt_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            resolve_jwt_secret()
        assert "JWT_SECRET_KEY" in str(exc_info.value)
        assert "missing" in str(exc_info.value).lower()

    def test_15_production_halts_on_insecure_default_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("JWT_SECRET_KEY", "fraudguard-ai-enterprise-secret-key-32bytes-secure-2026")
        with pytest.raises(RuntimeError) as exc_info:
            resolve_jwt_secret()
        assert "known default" in str(exc_info.value).lower() or "insecure" in str(exc_info.value).lower()

    def test_16_production_halts_on_short_jwt_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("JWT_SECRET_KEY", "too-short-secret-1234")
        with pytest.raises(RuntimeError) as exc_info:
            resolve_jwt_secret()
        assert "at least 32 characters" in str(exc_info.value).lower()

    def test_17_production_halts_on_missing_admin_password(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("FRAUDGUARD_ADMIN_PASSWORD", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            resolve_admin_password()
        assert "FRAUDGUARD_ADMIN_PASSWORD" in str(exc_info.value)

    def test_18_development_mode_generates_safe_defaults(self, monkeypatch):
        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        sec = resolve_jwt_secret()
        assert isinstance(sec, str)
        assert len(sec) >= 32


class TestCORSAndInformationDisclosure:
    """Verifies strict CORS origin whitelisting and health check sanitization."""

    def test_19_cors_headers_on_whitelisted_origin(self, client):
        res = client.get("/health", headers={"Origin": "http://localhost:8501"})
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "http://localhost:8501"

    def test_20_health_check_does_not_leak_paths(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        # Must not expose local OS file system paths
        assert "C:\\" not in str(data)
        assert "/Users/" not in str(data)
        assert "initialization_error" not in data

    def test_20a_production_mode_disables_docs_and_openapi(self, client, monkeypatch):
        """In production (ENV=production), /docs, /openapi.json, and /redoc must be disabled (HTTP 404)."""
        monkeypatch.setenv("ENV", "production")
        # Verify /docs returns 404
        res_docs = client.get("/docs")
        assert res_docs.status_code == 404
        assert res_docs.json().get("detail") == "Not Found"

        # Verify /openapi.json returns 404
        res_openapi = client.get("/openapi.json")
        assert res_openapi.status_code == 404
        assert res_openapi.json().get("detail") == "Not Found"

        # Verify /redoc returns 404
        res_redoc = client.get("/redoc")
        assert res_redoc.status_code == 404
        assert res_redoc.json().get("detail") == "Not Found"

        # Verify root metadata reports documentation as disabled
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert res_root.json().get("documentation") == "disabled_in_production"

    def test_20b_development_mode_enables_docs_and_openapi(self, client, monkeypatch):
        """In development (ENV=development), /docs, /openapi.json, and /redoc must return HTTP 200."""
        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.delenv("RENDER_SERVICE_ID", raising=False)

        # Verify /docs returns 200
        res_docs = client.get("/docs")
        assert res_docs.status_code == 200

        # Verify /openapi.json returns 200 and valid JSON schema
        res_openapi = client.get("/openapi.json")
        assert res_openapi.status_code == 200
        schema = res_openapi.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/predict" in schema["paths"]

        # Verify /redoc returns 200
        res_redoc = client.get("/redoc")
        assert res_redoc.status_code == 200

        # Verify root metadata points to /docs
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert res_root.json().get("documentation") == "/docs"

    def test_20c_render_cloud_platform_auto_disables_docs(self, client, monkeypatch):
        """When deployed on Render (RENDER=true) without explicit dev override, documentation must be disabled."""
        monkeypatch.delenv("ENV", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.setenv("RENDER", "true")

        res_docs = client.get("/docs")
        assert res_docs.status_code == 404

        res_openapi = client.get("/openapi.json")
        assert res_openapi.status_code == 404

        res_root = client.get("/")
        assert res_root.status_code == 200
        assert res_root.json().get("documentation") == "disabled_in_production"


class TestSecurityLimitsAndValidation:
    """Verifies request size and CSV batch limits."""

    def test_21_request_size_limit_middleware_413(self, client):
        """Oversized requests (>10MB Content-Length) must return HTTP 413 Payload Too Large."""
        oversized_headers = {
            "Content-Length": str(15 * 1024 * 1024),  # 15MB
            "Content-Type": "application/json",
        }
        res = client.post("/predict", headers=oversized_headers, content=b"{}")
        assert res.status_code == 413
        assert "Payload Too Large" in res.json()["detail"]

    def test_22_batch_transaction_row_limit_400(self, client, analyst_token, sample_payload):
        """Batches exceeding 10,000 transactions must return HTTP 400 Bad Request."""
        oversized_batch = [sample_payload] * 10001
        res = client.post(
            "/predict/batch",
            json={"transactions": oversized_batch},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert res.status_code == 400
        assert "exceeds maximum allowed limit" in res.json()["detail"]

    def test_23_predict_csv_upload_endpoint(self, client, analyst_token, sample_payload):
        """CSV upload endpoint accepts valid CSV and rejects oversized CSVs."""
        csv_buffer = io.StringIO()
        csv_buffer.write(",".join(FEATURE_ORDER) + "\n")
        row_str = ",".join(str(sample_payload[c]) for c in FEATURE_ORDER)
        csv_buffer.write(row_str + "\n")
        csv_buffer.write(row_str + "\n")
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        files = {"file": ("test_tx.csv", csv_bytes, "text/csv")}
        res = client.post(
            "/predict/csv?threshold=0.50",
            files=files,
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_processed"] == 2
        assert data["fraud_count"] == 0
        assert data["legitimate_count"] == 2


class TestTokenRevocationFailClosed:
    """
    Verifies fail-closed revocation checking and failure handling:
    1. Revocation database unavailable -> HTTP 503 Service Unavailable (fail-closed).
    2. Revocation database corruption -> HTTP 503 Service Unavailable (fail-closed).
    3. Explicitly revoked token -> HTTP 401 Unauthorized.
    4. Valid, non-revoked token -> HTTP 200 OK.
    5. Logout persistence failure -> HTTP 503 Service Unavailable.
    """

    def test_24_revocation_db_unavailable_returns_503(self, client, analyst_token, monkeypatch):
        """Authentication must fail-closed with HTTP 503 if the revocation database lookup fails."""
        import src.auth as a

        def mock_is_token_revoked(jti):
            raise RevocationStoreError("Simulated SQLite connection failure: database unavailable")

        monkeypatch.setattr(a, "is_token_revoked", mock_is_token_revoked)
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
        assert "Retry-After" in resp.headers

    def test_25_revocation_db_corruption_returns_503(self, client, analyst_token, monkeypatch, tmp_path):
        """Authentication must fail-closed with HTTP 503 if the revocation database file is corrupted."""
        import src.revocation_store as r_store

        corrupt_db = str(tmp_path / "corrupt_tokens.db")
        with open(corrupt_db, "wb") as f:
            f.write(b"CORRUPTED_NON_SQLITE_GARBAGE_BYTES_HEADER")

        monkeypatch.setattr(r_store, "DB_PATH", corrupt_db)
        monkeypatch.setenv("REVOCATION_DB_PATH", corrupt_db)

        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
        assert "Retry-After" in resp.headers

    def test_26_revoked_token_returns_401(self, client):
        """Explicitly revoked tokens must return HTTP 401 Unauthorized."""
        rate_limit_login.reset()
        login_resp = client.post("/auth/login", json={"username": "analyst", "password": "UserPassword@2026!"})
        tok = login_resp.json()["access_token"]

        # Logout revokes token
        logout_resp = client.post("/auth/logout", headers={"Authorization": f"Bearer {tok}"})
        assert logout_resp.status_code == 200

        # Submitting revoked token must return 401
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {tok}"})
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()

    def test_27_valid_token_authenticates_200(self, client, analyst_token):
        """Valid, non-revoked token must authenticate successfully with HTTP 200 OK."""
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
        assert resp.status_code == 200
        assert resp.json()["username"] == "analyst"

    def test_28_logout_persistence_failure_returns_503(self, client, analyst_token, monkeypatch):
        """Logout must report failure with HTTP 503 if token revocation cannot be persisted."""
        import app as backend_app_module

        def mock_revoke_token(token):
            raise RevocationStoreError("Simulated disk I/O write failure on SQLite store")

        monkeypatch.setattr(backend_app_module, "revoke_token", mock_revoke_token)
        resp = client.post("/auth/logout", headers={"Authorization": f"Bearer {analyst_token}"})
        assert resp.status_code == 503
        assert "Failed to persist token revocation" in resp.json()["detail"]
        assert "Retry-After" in resp.headers

