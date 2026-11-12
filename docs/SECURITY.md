# FraudGuard AI v2.0 — Security & Token Revocation Architecture

## 1. Authentication & Cryptography Specification

FraudGuard AI employs defense-in-depth security principles across authentication, session management, input sanitization, and access control:

1. **Password Hashing:**
   - Algorithm: Salted `bcrypt` with work factor 12.
   - Built-in validation rejects weak or known insecure passwords (`admin`, `password`, `12345678`, etc.).
2. **Access Token Generation (JWT):**
   - Algorithm: Cryptographic `HMAC-SHA256` (`HS256`).
   - Token Payload: Contains `sub` (username), `role` (RBAC tier), `iat` (issued at), `exp` (expiration), and a cryptographically random `jti` (UUIDv4) identifying the session token.
   - Token Expiration: Default 60 minutes (`ACCESS_TOKEN_EXPIRE_MINUTES`).
3. **Secret Key Management:**
   - Secret key configured via `JWT_SECRET_KEY` environment variable (minimum 32 bytes).
   - In production mode (`ENV=production`), application startup fails immediately if a default or weak secret is detected.

---

## 2. Token Revocation & Fail-Closed Architecture

### 2.1 The Token Revocation Problem
Standard JWT tokens are stateless, meaning a compromised or logged-out token remains valid until its `exp` timestamp elapses. To enforce secure logout and session termination, FraudGuard AI implements an active revocation registry.

```
Incoming Request -> Check JWT Signature & Expiration
                            │
                            ├── If Invalid/Expired -> HTTP 401 Unauthorized
                            │
                            └── If Cryptographically Valid -> Query Revocation Store
                                                                     │
                                  ┌──────────────────────────────────┴──────────────────────────────────┐
                                  │                                                                     │
                         Revocation Record Found                                              Database I/O Error
                                  │                                                                     │
                                  v                                                                     v
                      HTTP 401 Unauthorized                                              HTTP 503 Service Unavailable
                    ("Token has been revoked")                                              ("Revocation check failed")
                                                                                              [Fail-Closed Security]
```

### 2.2 Fail-Closed Principle
If the revocation database is locked, corrupted, or unreachable during token verification, the security dependency raises `RevocationStoreError`, which is mapped to **HTTP 503 Service Unavailable**. The system **never** fails open to allow potentially revoked tokens.

### 2.3 Real Operating System Restart Persistence
Token revocation was proven to survive complete OS process termination via `tests/test_real_restart_persistence.py`:
1. Process A issued a JWT token, accessed a protected endpoint (HTTP 200), and revoked it via `POST /auth/logout` (HTTP 200).
2. Process A was forcefully terminated via OS `kill`.
3. Process B was spawned on the same port pointing to the same SQLite database file.
4. Presenting the revoked token to Process B immediately returned **HTTP 401 Unauthorized** with `{"detail": "Token has been revoked"}`.

---

## 3. Deployment Storage Models: Local vs. Cloud vs. Multi-Replica

| Architecture Tier | Storage Medium | Restart Persistence | Multi-Worker Support | Multi-Replica Support | Production Readiness |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Local Development** | SQLite in `data/revoked_tokens.db` | Yes | Yes (WAL Mode) | No | **Ready** |
| **Container PaaS (Ephemeral)** | Local SQLite in container root | **No (Wiped on reboot)** | Yes | No | **Unsafe without volume** |
| **Single-Instance Staging** | SQLite on Persistent Mounted Disk | Yes | Yes | No | **Ready with volume** |
| **Horizontal Multi-Replica** | Distributed Cache (**Redis**) | Yes | Yes | **Yes (Shared state)** | **Recommended Target** |

### 3.1 SQLite Persistent Volume Configuration (Render / AWS ECS)
When deploying to container environments with SQLite, mount a persistent volume at `/data` and set:
```bash
REVOCATION_DB_PATH=/data/revoked_tokens.db
```

### 3.2 Pluggable Redis Revocation Store Specification
`backend/src/revocation_store.py` defines `BaseRevocationStore` and provides `RedisRevocationStore`. To switch to Redis without changing application logic, configure:
```bash
REDIS_URL=redis://default:password@redis-cluster.internal:6379/0
```
`RedisRevocationStore` stores keys as `revoked_token:{jti}` with native TTL equal to remaining token lifetime (`ex=remaining_seconds`). Expired records are pruned automatically by Redis with zero manual deletion overhead.

---

## 4. Additional Defense-in-Depth Controls

1. **Role-Based Access Control (RBAC):**
   - `independent_user`: Can score single transactions, execute batch CSV inference, and retrieve model info.
   - `mlops_admin`: Granted access to model evaluation metrics (`/model/metrics`), decision threshold optimization (`/model/threshold`), and model comparison registries (`/model/compare`).
2. **Rate Limiting:**
   - In-memory sliding-window limiter on `/auth/login` (5 attempts/min per IP) and `/predict` (60 requests/min per IP). Exceeding limits returns **HTTP 429 Too Many Requests**.
3. **CORS Restrictions:**
   - Allowed origins strictly restricted to trusted local dashboard ports (`localhost:8501`) and production domains (`*.onrender.com`, `*.streamlit.app`).
4. **Input Schema Enforcement:**
   - Pydantic models validate numeric constraints, non-negative amounts ($0.0 \le \text{Amount} \le 1,000,000.0$), non-negative elapsed time ($0.0 \le \text{Time} \le 250,000.0$), and disallow NaN/Infinity.
