# FraudGuard AI v2.0 — Known Limitations & Production Risk Registry

## 1. Evaluation Sample Size Limitations ($N=8$ Positive Fraud Cases)

### Root Cause
The held-out evaluation dataset (`data/processed/X_test.csv`) contains $N=1,008$ transactions, of which only **8 instances are positive fraudulent transactions** ($0.79\%$).

### Operational Impact
1. **Severe Metric Granularity:**
   Every single false negative or true positive shifts test recall by exactly **$12.5\%$** ($\frac{1}{8}$).
2. **Broad Bootstrap Confidence Intervals:**
   Stratified 95% bootstrap confidence intervals span:
   - Recall: $[12.5\%, 87.5\%]$ (Width: $75.0\%$)
   - F1-Score: $[22.2\%, 93.3\%]$ (Width: $71.1\%$)
   - F2-Score: $[15.2\%, 89.7\%]$ (Width: $74.5\%$)
3. **Statistical Power Limitation:**
   While the model achieves 75.0% recall (6/8 detected) at $\theta=0.10\text{–}0.15$ and 50.0% recall (4/8 detected, 0 FP) at $\theta=0.50$, sample-level performance on 8 cases cannot mathematically establish statistical confidence across millions of live commercial transactions.

### Mandatory Production Remediation:
- The model **must not** be used for automated, irreversible transaction decline decisions on day 1.
- The model **must operate in Shadow Mode / Canary Mode** alongside primary payment rails until at least **100 verified fraud incidents and 50,000 legitimate transactions** are scored and logged.

---

## 2. Token Revocation Storage Boundaries

### 2.1 Ephemeral Container Filesystems (Render / AWS Fargate)
- **Condition:** Default deployment without mounted persistent volumes.
- **Risk:** When a container spins down due to inactivity (e.g. Render free tier after 15 minutes), reboots, or is redeployed, the local filesystem is destroyed. `data/revoked_tokens.db` is recreated empty, effectively un-revoking active JWT tokens that have not yet expired.
- **Remediation:** Mount an SSD persistent disk at `/data` with `REVOCATION_DB_PATH=/data/revoked_tokens.db`.

### 2.2 Horizontal Multi-Replica Scaling
- **Condition:** Scaling backend pods to $> 1$ replica behind a load balancer.
- **Risk:** SQLite is a single-node file database. A token revoked on Replica 1 will not be present in Replica 2's database, allowing a revoked token to succeed if routed to Replica 2.
- **Remediation:** Configure `REDIS_URL` to activate the built-in `RedisRevocationStore` adapter, providing shared distributed revocation across all container replicas.

---

## 3. Frontend Testing Scope (AppTest vs. Full Browser)

- **Tested Scope:** `tests/test_genuine_streamlit_fastapi_e2e.py` exercises full frontend logic via Streamlit `AppTest`, sending real HTTP requests over TCP loopback to a separate Uvicorn subprocess, authenticating, validating UI state transitions, and checking rendered markdown elements.
- **Untested Scope:** `AppTest` executes headlessly within Python and does not use a browser automation driver (Playwright / Selenium). CSS layout rendering, canvas telemetry, and client-side JavaScript execution remain unverified in a live Chromium browser.

---

## 4. Feature Stationarity & Temporal Drift

- **Condition:** The 28 PCA features (`V1`–`V28`) are mathematical projections of historical credit card transactions.
- **Risk:** New fraud syndicates, merchant categories, and changes in consumer spending habits will introduce concept drift.
- **Remediation:** Schedule periodic retraining runs using `backend/src/model_trainer.py` and implement Evidently AI drift monitoring on `Time` and `Amount` distributions.
