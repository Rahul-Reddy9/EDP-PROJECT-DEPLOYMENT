# FraudGuard AI v2.0 — REST API Reference Specification

## 1. Authentication Endpoints

### 1.1 Login & Token Acquisition
- **Endpoint:** `POST /auth/login`
- **Rate Limit:** 5 requests / minute / IP
- **Request Body (`application/json`):**
  ```json
  {
    "username": "analyst",
    "password": "UserPassword@2026!"
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "role": "independent_user",
    "expires_in_minutes": 60
  }
  ```

### 1.2 Session Token Status Validation
- **Endpoint:** `GET /auth/verify`
- **Security:** Bearer JWT required
- **Response (`200 OK`):**
  ```json
  {
    "valid": true,
    "username": "analyst",
    "role": "independent_user",
    "message": "Token is active and unrevoked."
  }
  ```

### 1.3 Logout & Token Revocation
- **Endpoint:** `POST /auth/logout`
- **Security:** Bearer JWT required
- **Response (`200 OK`):**
  ```json
  {
    "status": "success",
    "message": "Session terminated. Token has been permanently revoked."
  }
  ```

---

## 2. Health & System Information Endpoints

### 2.1 Health & Readiness
- **Endpoint:** `GET /health`
- **Security:** Public
- **Response (`200 OK`):**
  ```json
  {
    "status": "healthy",
    "environment": "production",
    "model_loaded": true,
    "scaler_loaded": true,
    "version": "2.0-optimized",
    "revocation_store": "sqlite"
  }
  ```

### 2.2 Sample Preset Generator
- **Endpoint:** `GET /sample?fraud=false`
- **Query Parameters:** `fraud` (boolean, default: `false`)
- **Response (`200 OK`):**
  ```json
  {
    "type": "Legitimate Example",
    "payload": {
      "Time": 31604.21,
      "V1": -0.42015,
      "...": "...",
      "V28": 0.66319,
      "Amount": 149.97
    }
  }
  ```

---

## 3. Real-Time Inference Endpoints

### 3.1 Single Transaction Scoring
- **Endpoint:** `POST /predict`
- **Security:** Bearer JWT required (User or Admin)
- **Rate Limit:** 60 requests / minute / IP
- **Query Parameters:**
  - `threshold` (float, $0.0 \le \theta \le 1.0$, default: `0.50`)
  - `explain` (boolean, default: `false`, enables Tree SHAP)
  - `is_preprocessed` (boolean or null, default: `null`, raw scaling auto-protection)
- **Request Body (`application/json`):**
  ```json
  {
    "Time": 31604.21,
    "V1": -0.42015,
    "V2": -0.43649,
    "...": "...",
    "V28": 0.66319,
    "Amount": 149.97
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "prediction": 0,
    "label": "Legitimate",
    "fraud_probability": 0.000096,
    "legitimate_probability": 0.999904,
    "threshold_used": 0.5,
    "risk_level": "LOW",
    "model_version": "2.0-optimized",
    "explanation": {
      "method": "Tree SHAP (Exact Shapley Additive exPlanations)",
      "bias": -3.8124,
      "top_positive": [{"feature": "V14", "contribution": 0.1245}],
      "top_negative": [{"feature": "V17", "contribution": -0.4512}],
      "summary": "Primary risk-escalating factor is V14; primary mitigating factor is V17."
    }
  }
  ```

### 3.2 Batch Transaction Scoring
- **Endpoint:** `POST /predict/batch`
- **Security:** Bearer JWT required (User or Admin)
- **Limit:** Maximum 400 transactions per batch request
- **Request Body (`application/json`):**
  ```json
  {
    "transactions": [
      { "Time": 100.0, "V1": 0.1, "...": "...", "V28": 0.0, "Amount": 50.0 },
      { "Time": 105.0, "V1": -2.5, "...": "...", "V28": 1.2, "Amount": 820.0 }
    ],
    "threshold": 0.15,
    "explain": false
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "batch_size": 2,
    "fraud_count": 1,
    "legitimate_count": 1,
    "fraud_rate": 0.5,
    "predictions": [ ... ]
  }
  ```

---

## 4. Model Governance & Telemetry Endpoints (Admin RBAC Required)

| Endpoint | Method | Role | Description |
| :--- | :---: | :---: | :--- |
| `/model/info` | `GET` | User/Admin | Returns canonical 30-feature schema, version history, and architecture metadata. |
| `/model/metrics` | `GET` | Admin | Returns confusion matrix, accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC, Brier score, and bootstrap CIs. |
| `/model/threshold` | `GET` | Admin | Returns OOF-derived threshold optimization recommendations and sensitivity sweep data. |
| `/model/compare` | `GET` | Admin | Returns comparative evaluation benchmark across candidate model architectures. |
