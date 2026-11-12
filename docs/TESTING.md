# FraudGuard AI v2.0 — Automated Test Suite & QA Specification

## 1. Test Suite Summary

The project maintains a multi-tiered automated test suite comprising **146 pytest tests** plus dedicated multi-process integration tests.

- **Total Automated Pytest Tests:** **146**
- **Passing:** **146 (100%)**
- **Failing:** **0**
- **Skipped:** **0**
- **Total Execution Time:** ~52.0 seconds

---

## 2. Test Module Breakdown

| Module | Test Count | Scope & Focus Areas |
| :--- | :---: | :--- |
| `tests/test_api_client.py` | 25 | Health check, transient HTTP 502/503/504 retry recovery, checkpoint recovery, URL resolution, Streamlit secrets fallback. |
| `tests/test_api_extended.py` | 6 | `/model/info`, `/model/metrics`, `/model/threshold`, `/model/compare`, Tree SHAP explainability. |
| `tests/test_extended_metrics.py` | 10 | $F_2$-score, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Negative Predictive Value (NPV), Brier score, calibration curves, and stratified bootstrap CI calculation. |
| `tests/test_frontend_ui.py` | 14 | Theme CSS generation, Plotly risk gauge, PR curves, risk tier pills, metric formatting, and Streamlit `AppTest` state flow. |
| `tests/test_full_system_integration.py` | 8 | End-to-end multi-step transaction pipelines, batch recovery, and data partition schema alignment. |
| `tests/test_live_integration.py` | 10 | In-process API client connectivity, authentication guards, and response payload validation. |
| `tests/test_model_comparison.py` | 6 | Candidate model registry verification (Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, XGBoost). |
| `tests/test_model_evaluation.py` | 11 | Confusion matrix computation, threshold sensitivity sweeps, and ground-truth validation. |
| `tests/test_pipeline_optimization.py` | 8 | Pipeline feature sequence adherence, scaler alignment, and v2.0 optimization validation. |
| `tests/test_preprocessing_audit.py` | 13 | Idempotent scaling, double-scaling prevention, negative value detection, ambiguous range handling, `predict_raw` and `predict_preprocessed` interfaces. |
| `tests/test_probability_precision.py` | 3 | Full-precision floating point probability preservation ($P \ge 0.0$ and $P \le 1.0$), ensuring zero rounding degradation. |
| `tests/test_security_rbac.py` | 28 | Salted bcrypt validation, JWT cryptographic expiration, RBAC authorization boundaries, rate limiting enforcement, fail-closed revocation database handling (HTTP 503). |
| `tests/test_threshold_leakage_fix.py` | 4 | Verifies threshold optimization is performed strictly on training OOF predictions ($N=4,032$), and held-out test split remains untouched. |
| `tests/test_validators.py` | 10 | Single and batch transaction schema validation, negative amount rejection, missing column detection, NaN/Infinity rejection. |

---

## 3. Dedicated Multi-Process Integration Test Suites

### 3.1 Real OS Process Restart Persistence
- **Script:** `tests/test_real_restart_persistence.py`
- **Verification:** Spawns Process A on port 8005 -> authenticates -> revokes token via logout -> forcefully kills Process A -> spawns fresh Process B using same SQLite DB -> verifies revoked token returns HTTP 401 across process boundaries.
- **Run Command:**
  ```powershell
  .\.venv\Scripts\python.exe tests/test_real_restart_persistence.py
  ```

### 3.2 Genuine Streamlit-to-FastAPI End-to-End Test
- **Script:** `tests/test_genuine_streamlit_fastapi_e2e.py`
- **Verification:** Spawns live Uvicorn microservice on port 8008 -> connects Streamlit AppTest over loopback network -> performs UI login -> scores Clean Merchant (verifies green LOW ANOMALY RISK banner) -> scores Syndicate Attack (verifies red CRITICAL RISK banner) -> performs live logout.
- **Run Command:**
  ```powershell
  .\.venv\Scripts\python.exe tests/test_genuine_streamlit_fastapi_e2e.py
  ```

---

## 4. Test Execution Instructions

```powershell
# Run full pytest suite with verbose output
.\.venv\Scripts\pytest.exe -v

# Run targeted security test suite
.\.venv\Scripts\pytest.exe tests/test_security_rbac.py -v

# Run targeted preprocessing audit
.\.venv\Scripts\pytest.exe tests/test_preprocessing_audit.py -v

# Run latency benchmark suite
.\.venv\Scripts\python.exe backend/src/benchmark_latency.py
```
