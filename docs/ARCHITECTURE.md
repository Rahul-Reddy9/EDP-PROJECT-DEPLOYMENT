# FraudGuard AI v2.0 — System Architecture & Component Specification

## 1. High-Level Architecture Overview

FraudGuard AI is an enterprise-grade Credit Card Fraud Detection platform integrating:
1. **High-Throughput Machine Learning Inference Engine:** Pre-calibrated XGBoost model wrapped with isotonic calibration and StandardScaler normalization.
2. **FastAPI Microservice Backend:** Async RESTful API providing JWT authentication, server-side RBAC, Tree SHAP explainability, in-process latency logging, rate limiting, and fail-closed SQLite token revocation.
3. **Streamlit Real-Time Dashboard:** Interactive operational dashboard for fraud risk intelligence, transaction scoring, PCA anomaly visualization, and model governance telemetry.

```
+---------------------------------------------------------------------------------------+
|                                    CLIENT LAYER                                       |
|                                                                                       |
|   +------------------------------------+    +-------------------------------------+   |
|   |    Streamlit Frontend Dashboard    |    |  Automated Payment Gateway / Client |   |
|   | (Real-Time Scoring, Batch Audits)  |    |  (External REST API Consumer)       |   |
|   +-----------------+------------------+    +------------------+------------------+   |
+---------------------|------------------------------------------|----------------------+
                      | HTTPS / Loopback TCP Socket              | HTTPS / Bearer Token
                      v                                          v
+---------------------------------------------------------------------------------------+
|                                    BACKEND MICROSERVICE                               |
|                                                                                       |
|   FastAPI Application Gateway (backend/app.py)                                        |
|   ├── CORS Middleware (Trusted Origins: localhost:8501, *.onrender.com)               |
|   ├── Memory-Efficient In-Memory Sliding-Window Rate Limiter                          |
|   ├── JWT Auth & Role-Based Access Control (RBAC: independent_user, mlops_admin)      |
|   └── Pydantic Schema Validation (Strict 30-feature shape, non-negative Amount)       |
|                                                                                       |
|   Authentication & Session Governance (backend/src/auth.py, revocation_store.py)      |
|   ├── Salted Bcrypt Password Hashing (Cost Factor 12)                                 |
|   ├── Cryptographic HMAC-SHA256 Token Signing & JTI Identification                    |
|   └── Pluggable Token Revocation Store                                                |
|       ├── SQLiteRevocationStore (Default: WAL Mode, Fail-Closed, OS Restart Persistence) |
|       └── RedisRevocationStore (Adapter Template: TTL Native Auto-Pruning)            |
|                                                                                       |
|   Prediction & Inference Engine (backend/src/predict.py)                              |
|   ├── Preprocessing Pipeline (Time & Amount StandardScaler Normalization)             |
|   │   ├── Predict Raw Interface (predict_raw)                                         |
|   │   ├── Predict Preprocessed Interface (predict_preprocessed)                       |
|   │   └── Double-Scaling Prevention & Ambiguity Guardrails                            |
|   ├── Calibrated XGBoost Booster (v2_tuned_xgboost.pkl)                              |
|   │   ├── Posterior Probability Output (Isotonic Calibration)                         |
|   │   └── Dynamic Three-Tier Risk Thresholding (LOW, MEDIUM, HIGH)                    |
|   └── Tree SHAP Feature Attribution (Native Pred-Contrib Matrix)                       |
+---------------------------------------------------------------------------------------+
```

## 2. Component Directory Structure

```
EDP-PROJECT-DEPLOYMENT/
├── backend/
│   ├── app.py                      # FastAPI application gateway & route definitions
│   ├── requirements.txt            # Microservice runtime dependencies
│   ├── models/
│   │   ├── tuned_xgboost.pkl       # Active production v2.0 model
│   │   ├── scaler.pkl              # Active production StandardScaler
│   │   ├── backup_v1_tuned_xgboost.pkl # Preserved v1.0 baseline model
│   │   ├── backup_v1_scaler.pkl    # Preserved v1.0 baseline scaler
│   │   └── registry/               # Candidate model registry and OOF predictions
│   └── src/
│       ├── auth.py                 # JWT cryptography & RBAC dependencies
│       ├── predict.py              # Packaging, double-scaling guard, Tree SHAP
│       ├── revocation_store.py     # SQLite and Redis revocation store adapters
│       ├── rate_limiter.py         # Sliding window rate limiter
│       ├── model_trainer.py        # Model training and artifact serialization
│       └── audit_eval.py           # Evaluation verification script
├── frontend/
│   ├── api_client.py               # Robust HTTP client with retry and error mapping
│   ├── charts.py                   # Plotly risk gauge, PR curves, and anomaly plots
│   ├── components.py               # Decision banners and telemetry cards
│   ├── model_evaluation.py         # Ground-truth test evaluation & bootstrap CI
│   ├── styles.py                   # Custom theme CSS styling
│   └── validators.py               # Client-side input validation
├── data/
│   ├── raw/creditcard.csv          # Protected raw dataset baseline
│   ├── processed/                  # Stratified train and test partitions
│   └── revoked_tokens.db           # Persistent SQLite revocation store
├── tests/                          # 15 automated test suites (146 tests)
├── Dockerfile                      # Production container image definition
├── docker-compose.yml              # Local multi-service orchestration
└── streamlit_app.py                # Streamlit user interface entrypoint
```

## 3. Data Flow Specification

### 3.1 Real-Time Single Transaction Scoring Flow
1. **Payload Submission:** User submits 30 canonical features (`Time`, `V1`–`V28`, `Amount`) via Streamlit UI or external HTTP client.
2. **Client Validation:** `frontend/validators.py` verifies data types, non-null constraints, and bounds.
3. **HTTP Transport:** Payload transmitted via `POST /predict?threshold=0.50&explain=true` with Bearer JWT.
4. **API Gateway Auth:** `backend/src/auth.py` checks token signature, expiration, and consults `revocation_store` to guarantee token has not been revoked.
5. **Schema Validation:** Pydantic `TransactionPayload` validates 30 feature schema.
6. **Inference Pipeline:**
   - `predict_raw()` scales `Time` and `Amount` using training parameters.
   - Boosters compute raw log-odds, calibrated to posterior fraud probability $P(\text{Fraud} \mid X)$.
   - Probability evaluated against active cutoff $\theta$ (default $\theta=0.50$ for auto-decline, $\theta=0.10\text{–}0.15$ for step-up queue).
   - If `explain=True`, native Tree SHAP computes exact Shapley feature contributions.
7. **Response Serialization:** Formatted JSON returned with `prediction`, `fraud_probability`, `risk_level`, `threshold_used`, and `explanation`.
8. **UI Rendering:** Streamlit displays dynamic risk banner (Green = LOW, Amber = MEDIUM, Red = HIGH), calibrated dial gauge, and top anomaly bars.
