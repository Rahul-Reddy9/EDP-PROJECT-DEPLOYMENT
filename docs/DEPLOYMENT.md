# FraudGuard AI v2.0 — Deployment & Operations Guide

## 1. Environment Variables Reference

| Variable | Type | Default Value | Description |
| :--- | :---: | :--- | :--- |
| `ENV` | String | `development` | Environment mode (`development`, `staging`, `production`). In production, weak secrets fail fast. |
| `API_BASE_URL` | String | `https://edp-project-deployment.onrender.com` | Base URL used by the Streamlit frontend to reach the FastAPI service. |
| `JWT_SECRET_KEY` | String | *Auto-generated dynamic dev key* | Cryptographic signing secret (min 32 bytes required in production). |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Integer | `60` | Token lifetime before expiration in minutes. |
| `REVOCATION_DB_PATH` | Path | `data/revoked_tokens.db` | Absolute or relative path to SQLite token revocation database. |
| `REDIS_URL` | String | `None` | Optional Redis connection string for distributed token revocation (e.g. `redis://localhost:6379/0`). |
| `PORT` | Integer | `8000` | Port bound by Uvicorn. |

---

## 2. Local Development Deployment

### Prerequisites:
- Python 3.10, 3.11, or 3.13
- Virtual environment with dependencies installed:
  ```powershell
  pip install -r backend/requirements.txt
  pip install -r requirements.txt
  ```

### Step 1: Start FastAPI Inference Microservice
```powershell
# In terminal 1 (Project root)
.\.venv\Scripts\python.exe -m uvicorn --app-dir backend app:app --host 127.0.0.1 --port 8000 --reload
```
Verify readiness:
```powershell
curl http://127.0.0.1:8000/health
```

### Step 2: Start Streamlit Frontend Dashboard
```powershell
# In terminal 2 (Project root)
$env:API_BASE_URL="http://127.0.0.1:8000"
.\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port=8501
```
Open browser to `http://localhost:8501`.

---

## 3. Docker Container Deployment

### Local Docker Compose
The project includes a multi-service `docker-compose.yml` configuring:
- `backend`: FastAPI microservice on port 8000 with persistent volume for `/app/data`.
- `frontend`: Streamlit dashboard on port 8501 with internal DNS networking to `http://backend:8000`.

```bash
# Build and launch both services
docker compose up -d --build

# View container logs
docker compose logs -f

# Verify backend health
curl http://localhost:8000/health

# Stop containers
docker compose down
```

---

## 4. Cloud Staging Deployment (Render PaaS)

FraudGuard AI is optimized for cloud PaaS deployment on Render:

### 4.1 Backend Web Service Configuration
- **Environment:** Python 3
- **Root Directory:** Empty (Project root)
- **Build Command:** `pip install -r backend/requirements.txt`
- **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 2`
- **Health Check Path:** `/health`
- **Persistent Disk (Critical):**
  Mount an SSD disk at `/data` (1 GB) and set:
  ```bash
  REVOCATION_DB_PATH=/data/revoked_tokens.db
  ```

### 4.2 Frontend Web Service Configuration (Render or Streamlit Community Cloud)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
- **Environment Variables:**
  ```bash
  API_BASE_URL=https://edp-project-deployment.onrender.com
  ```

---

## 5. Enterprise Multi-Replica Production Architecture

For horizontally scaled production clusters (Kubernetes, AWS ECS, Google Cloud Run with > 1 instance):

1. **Distributed Revocation:**
   Deploy a Redis instance (e.g. AWS ElastiCache, Upstash, or Redis Cloud) and configure `REDIS_URL` across all backend worker pods.
2. **Reverse Proxy & TLS Termination:**
   Deploy behind NGINX or AWS ALB with HTTPS certificate termination.
3. **Shadow Mode Ingestion:**
   Configure an asynchronous queue (RabbitMQ / Kafka / Celery) mirroring payment requests to `/predict` in shadow mode, writing evaluation logs to Snowflake or BigQuery for real-world concept drift tracking.
