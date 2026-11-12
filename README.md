# 🛡️ FraudGuard AI – Enterprise Credit Card Fraud Detection System

An end-to-end Machine Learning application for detecting potentially fraudulent credit card transactions through real-time inference, batch processing, risk classification, and model evaluation.

FraudGuard AI integrates a **FastAPI backend**, **Streamlit interactive dashboard**, and a tuned **XGBoost classification pipeline** to provide an integrated fraud detection solution.

The project demonstrates practical implementation of Machine Learning, backend engineering, API development, model management, application security, automated testing, and containerization.

---

## 🚀 Live Application

| Component | URL |
|---|---|
| Streamlit Dashboard | [Open FraudGuard AI](https://edp-fraud-detection-rahul.streamlit.app/) |
| FastAPI Backend | [Backend Service](https://edp-project-deployment.onrender.com) |
| Interactive API Documentation | [Swagger UI](https://edp-project-deployment.onrender.com/docs) |
| Backend Health Check | [Health Endpoint](https://edp-project-deployment.onrender.com/health) |
| GitHub Repository | [EDP-PROJECT-DEPLOYMENT](https://github.com/Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT) |

> **Deployment note:** These are the project's configured deployment URLs. Verify service availability and synchronization with the latest GitHub commit. Render free-tier services may sleep after inactivity, making the first request slower.

---

## ✨ Key Features

### 1. Real-Time Fraud Detection
- Score individual credit card transactions.
- Validate all 30 canonical input features.
- Generate fraud probabilities.
- Classify transactions using a configurable decision threshold.
- Display prediction results and inference latency.

### 2. Batch CSV Processing
- Upload transaction CSV files.
- Validate input schema and numeric feature values.
- Process multiple transactions through the batch prediction API.
- Configure network request chunk sizes.
- Retry supported transient API errors.
- View processed transaction counts.
- Analyze flagged fraud and clean transaction counts.
- Visualize risk distributions and transaction probabilities.
- Generate enriched prediction results.

### 3. Dynamic Decision Threshold
- Adjust the fraud classification threshold.
- Default threshold: `0.50`.
- Supported range: `0.05–0.95`.
- Explore the trade-off between fraud sensitivity and false positives.

Classification logic:

`Fraud if P(fraud) >= threshold`

A lower threshold can increase fraud sensitivity while potentially increasing false positives. A higher threshold can reduce false positives while potentially missing more fraudulent transactions.

### 4. Risk Classification

| Risk Tier | Fraud Probability |
|---|---|
| LOW | Below 40% |
| MEDIUM | 40% to below 75% |
| HIGH | 75% and above |

Risk tiers provide additional interpretation of model predictions and are separate from the binary classification threshold.

### 5. Model Training and Evaluation
- Model training and evaluation workflows.
- Preprocessing and feature transformation.
- Model performance analysis.
- Threshold evaluation.
- Comparison of supported classification models.
- Model artifact management.

### 6. Model Comparison

The project includes a model registry containing multiple model artifacts for comparison.

The registry includes:
- Tuned XGBoost models.
- Balanced Random Forest.
- Balanced Extra Trees.
- Balanced Logistic Regression.
- Histogram-based Gradient Boosting.

The Model Comparison section provides an interface for evaluating supported model alternatives.

### 7. Authentication and Session Management
- JWT-based authentication.
- Role-based access control.
- Session management.
- Token revocation support.
- Password hashing.
- Protected application functionality.

Development credentials should never be reused in production. Configure secure credentials and signing secrets through appropriate environment variables or deployment secrets.

### 8. System Telemetry
- Backend connectivity monitoring.
- Model loading status.
- Scaler availability.
- API response latency.
- Service health information.
- Runtime diagnostics.

### 9. Audit and Verification

The project includes additional audit and verification utilities for:
- Model evaluation.
- Latency benchmarking.
- Genuine end-to-end testing.
- Live integration verification.
- Preprocessing validation.
- Pipeline optimization.
- Restart persistence testing.

### 10. Docker Support

The repository includes:
- `Dockerfile`
- `docker-compose.yml`

These provide containerization configuration for the application. Docker execution should be verified in the target environment before treating a deployment as production-ready.

---

## 🏛️ System Architecture

FraudGuard AI follows a decoupled frontend and backend architecture.

```text
                 USER
                  |
                  v
      +-------------------------+
      |   STREAMLIT DASHBOARD   |
      |                         |
      | Real-Time Scoring       |
      | Batch CSV Processing    |
      | Model Comparison        |
      | System Telemetry        |
      | Authentication          |
      +------------+------------+
                   |
                   v
      +-------------------------+
      |  CLIENT & VALIDATION    |
      |                         |
      | API Client              |
      | Input Validation        |
      | Request Handling        |
      | Batch Chunking          |
      +------------+------------+
                   |
             HTTP / JSON
                   |
                   v
      +-------------------------+
      |       FASTAPI           |
      |                         |
      | REST API                |
      | Authentication          |
      | Authorization           |
      | Prediction Endpoints    |
      | Health Monitoring       |
      +------------+------------+
                   |
                   v
      +-------------------------+
      |    INFERENCE PIPELINE   |
      |                         |
      | Feature Validation      |
      | Preprocessing           |
      | StandardScaler          |
      | XGBoost Classifier      |
      +------------+------------+
                   |
                   v
      +-------------------------+
      |     PREDICTION RESULT   |
      |                         |
      | Fraud Probability       |
      | Classification          |
      | Risk Information        |
      | Inference Latency       |
      +-------------------------+
```

### Architecture Components

| Layer | Technology | Responsibility |
|---|---|---|
| Presentation | Streamlit | Interactive dashboard |
| API Client | Python | Backend communication |
| Validation | Python | Feature schema validation |
| Backend | FastAPI | REST API and request handling |
| Machine Learning | XGBoost | Fraud probability prediction |
| Preprocessing | Scikit-learn | Feature scaling |
| Authentication | JWT | Authentication and session handling |
| Model Management | Pickle / Joblib | Model artifact persistence |
| Containerization | Docker | Application packaging |
| Testing | Pytest | Automated testing |

---

## 📂 Project Structure

```text
EDP-PROJECT-DEPLOYMENT/
│
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── streamlit_app.py
│
├── audit_manifests/
│   └── initial_manifest_20261112.json
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   │
│   ├── models/
│   │   ├── scaler.pkl
│   │   ├── tuned_xgboost.pkl
│   │   ├── backup_v1_scaler.pkl
│   │   ├── backup_v1_tuned_xgboost.pkl
│   │   ├── v2_scaler.pkl
│   │   ├── v2_tuned_xgboost.pkl
│   │   │
│   │   └── registry/
│   │       ├── comparison_manifest.json
│   │       ├── model_balanced_extra_trees.pkl
│   │       ├── model_balanced_logistic_regression.pkl
│   │       ├── model_balanced_random_forest.pkl
│   │       ├── model_histgradientboosting.pkl
│   │       ├── model_tuned_xgboost_baseline_v10.pkl
│   │       ├── model_tuned_xgboost_v20_optimized.pkl
│   │       └── training_oof_predictions.json
│   │
│   └── src/
│       ├── model_trainer.py
│       ├── predict.py
│       ├── revocation_store.py
│       ├── audit_eval.py
│       ├── benchmark_latency.py
│       ├── verify_genuine_e2e.py
│       └── verify_live_e2e.py
│
├── frontend/
│   ├── __init__.py
│   ├── api_client.py
│   └── validators.py
│
├── docs/
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── KNOWN_LIMITATIONS.md
│   ├── MODEL_EVALUATION.md
│   ├── SECURITY.md
│   └── TESTING.md
│
├── tests/
│   ├── test_api_extended.py
│   ├── test_frontend_ui.py
│   ├── test_model_comparison.py
│   ├── test_model_evaluation.py
│   ├── test_threshold_leakage_fix.py
│   ├── test_full_system_integration.py
│   ├── test_genuine_streamlit_fastapi_e2e.py
│   ├── test_pipeline_optimization.py
│   ├── test_preprocessing_audit.py
│   └── test_real_restart_persistence.py
│
└── data/
    └── processed/
        └── Local processed datasets (excluded from Git)
```

*This structure documents the main application components and files included in the project. Additional local files and configuration may exist.*

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Backend Framework | FastAPI |
| Frontend Framework | Streamlit |
| API Communication | REST, HTTP, JSON |
| Model Persistence | Pickle, Joblib |
| Authentication | JWT |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| Version Control | Git, GitHub |
| Backend Deployment | Render |
| Frontend Deployment | Streamlit Community Cloud |

---

## 🧠 Machine Learning Pipeline

### Input Features

The application expects 30 canonical transaction features.

```text
Time
V1, V2, V3, ... V28
Amount
```

| Feature Group | Number of Features |
|---|---:|
| Time | 1 |
| PCA Features (V1–V28) | 28 |
| Amount | 1 |
| **Total** | **30** |

The `Class` label is not part of the prediction input.

### Preprocessing Workflow

1. Receive the transaction input.
2. Validate the required 30-feature schema.
3. Check numeric input values.
4. Apply the saved preprocessing transformation to `Time` and `Amount`.
5. Preserve PCA features `V1` through `V28`.
6. Arrange features in the canonical training order.
7. Pass the processed features to the trained XGBoost classifier.
8. Generate fraud probability.
9. Apply the configured classification threshold.
10. Return the prediction response.

### Model Artifacts

The primary inference artifacts are:

```text
backend/models/
├── scaler.pkl
└── tuned_xgboost.pkl
```

Additional model versions and comparison artifacts are maintained in the model directory and registry.

The backend uses saved model artifacts for inference rather than retraining the model during normal application startup.

---

## 📁 Batch CSV Processing

FraudGuard AI supports batch prediction through CSV ingestion.

### Expected CSV Format

The input file should contain the 30 canonical feature columns:

```csv
Time,V1,V2,V3,...,V28,Amount
```

The input should contain numeric feature values and should not include the actual `Class` target column.

### Batch Processing Workflow

```text
        CSV UPLOAD
             |
             v
      SCHEMA VALIDATION
             |
             v
      NUMERIC VALIDATION
             |
             v
       BATCH REQUESTS
             |
             v
       FASTAPI ENDPOINT
             |
             v
      XGBOOST INFERENCE
             |
             v
      RISK CLASSIFICATION
             |
             v
      RESULTS & ANALYTICS
             |
             v
      ENRICHED CSV EXPORT
```

### Example Functional Test

A five-row batch was successfully processed in the local application.

| Metric | Observed Result |
|---|---:|
| Transactions processed | 5 |
| Flagged as fraud | 2 |
| Classified as clean | 3 |
| Processing time | 0.02 seconds |
| Input schema validation | Passed |

These figures represent one functional batch-processing test, not a general model accuracy measurement.

---

## 🎚️ Decision Threshold

The dashboard provides an adjustable classification threshold.

Default:

```text
Threshold = 0.50
```

Supported range:

```text
0.05 to 0.95
```

Classification logic:

```python
if fraud_probability >= threshold:
    prediction = 1
else:
    prediction = 0
```

### Threshold Interpretation

| Threshold | Expected Behavior |
|---|---|
| Lower | More sensitive fraud detection; potentially more false positives |
| Higher | Fewer transactions flagged; potentially more missed fraud |
| 0.50 | Default classification cutoff |

Threshold selection should be based on validation data and the operational cost of false positives and false negatives.

---

## 🚦 Risk Classification

The application provides three risk tiers:

| Risk Tier | Probability Range |
|---|---|
| LOW | Below 40% |
| MEDIUM | 40% to below 75% |
| HIGH | 75% and above |

The risk tier provides an additional interpretation of the model output.

It does not replace the binary classification threshold.

---

## 🔐 Authentication and Security

The application includes authentication and access-control capabilities.

### Security Components

- JWT-based authentication.
- HS256 token signing.
- Password hashing.
- Role-based access control.
- Session management.
- Token revocation support.
- Input validation.
- API request handling.

### Security Recommendations

For a real deployment:

1. Use strong, unique credentials.
2. Store secrets in environment variables or a secret manager.
3. Never commit `.env` files or passwords.
4. Use HTTPS for external API communication.
5. Restrict access to administrative features.
6. Review model artifacts and evaluation files before publishing.
7. Use production-grade persistence and monitoring.
8. Rotate signing secrets when necessary.

See [docs/SECURITY.md](docs/SECURITY.md) for additional security documentation.

---

## 🩺 System Telemetry

The dashboard includes system diagnostics to monitor the application.

Information includes:

- Backend service connectivity.
- Active model information.
- Model loading status.
- Scaler availability.
- Expected feature count.
- API response time.
- Service health information.

The backend exposes a health endpoint:

```http
GET /health
```

---

## 🔌 API Reference

The FastAPI backend exposes REST endpoints for application communication.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Backend health |
| GET | `/sample` | Sample transaction payloads |
| POST | `/predict` | Single transaction prediction |
| POST | `/predict/batch` | Batch transaction prediction |

### Interactive API Documentation

When running locally:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The Swagger UI allows users to inspect the API schemas and manually test supported endpoints.

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for more information.

---

## ⚙️ Local Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- PowerShell on Windows or Bash on Linux/macOS.
- Required Python dependencies.
- Docker Desktop (optional).

### 1. Clone the Repository

```bash
git clone https://github.com/Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT.git
```

```bash
cd EDP-PROJECT-DEPLOYMENT
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

### 4. Start the FastAPI Backend

From the project root:

```powershell
python -m uvicorn --app-dir backend app:app --host 127.0.0.1 --port 8000
```

The backend should be accessible at:

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

Keep this terminal running.

### 5. Start the Streamlit Dashboard

Open another terminal from the project root.

Configure the local API address:

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
```

Launch Streamlit:

```powershell
streamlit run streamlit_app.py --server.port=8501
```

Open the dashboard:

[http://localhost:8501](http://localhost:8501)

### Local Application Architecture

```text
Browser
   |
   v
Streamlit Dashboard
localhost:8501
   |
   | HTTP / JSON
   v
FastAPI Backend
127.0.0.1:8000
   |
   v
XGBoost + StandardScaler
```

---

## 🐳 Docker Setup

The repository includes Docker configuration files:

```text
Dockerfile
docker-compose.yml
```

Make sure Docker Desktop is running.

From the repository root, use:

```bash
docker compose up --build
```

To stop the containers:

```bash
docker compose down
```

Container startup, networking, environment variables, and health checks should be validated against the supplied Docker configuration.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for deployment details.

---

## 🧪 Testing and Validation

The repository includes tests for multiple application components.

### Test Categories

| Test File | Testing Area |
|---|---|
| `test_api_extended.py` | API behavior and extended backend tests |
| `test_frontend_ui.py` | Streamlit frontend behavior |
| `test_model_comparison.py` | Model comparison functionality |
| `test_model_evaluation.py` | Model evaluation |
| `test_threshold_leakage_fix.py` | Threshold evaluation and leakage checks |
| `test_full_system_integration.py` | Full-system integration |
| `test_genuine_streamlit_fastapi_e2e.py` | Streamlit and FastAPI end-to-end workflow |
| `test_pipeline_optimization.py` | Pipeline optimization |
| `test_preprocessing_audit.py` | Preprocessing validation |
| `test_real_restart_persistence.py` | Persistence across application restarts |

### Run the Test Suite

From the project root:

```bash
python -m pytest tests/ -v
```

Run an individual test module:

```bash
python -m pytest tests/test_api_extended.py -v
```

Run with a summary:

```bash
python -m pytest tests/ -q
```

Test results should be recorded from the actual execution environment. Test availability does not imply that every cloud deployment or production scenario has been validated.

See [docs/TESTING.md](docs/TESTING.md).

---

## 📊 Model Evaluation

Fraud detection involves highly imbalanced classification data.

Model evaluation should consider:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Proportion of predicted fraud cases that are actual fraud |
| Recall | Proportion of actual fraud cases detected |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Ranking performance across classification thresholds |
| PR-AUC | Precision-recall performance |
| Brier Score | Probability calibration assessment |
| Confusion Matrix | Breakdown of TP, FP, TN and FN |

For an imbalanced fraud detection dataset, accuracy alone is insufficient.

The evaluation should also consider false-positive and false-negative costs, threshold selection, class distribution, and validation methodology.

See [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) and [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

---

## ☁️ Cloud Deployment

The project has configured deployment endpoints for the following platforms.

### Backend – Render

| Configuration | Value |
|---|---|
| Platform | Render |
| Service | FastAPI |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health Endpoint | `/health` |

Backend:

https://edp-project-deployment.onrender.com

### Frontend – Streamlit Community Cloud

| Configuration | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| Repository | `Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT` |
| Branch | `main` |
| Main File | `streamlit_app.py` |

Dashboard:

https://edp-fraud-detection-rahul.streamlit.app/

### API Configuration

For the deployed dashboard, configure the backend service address using the deployment's environment variable settings:

```text
API_BASE_URL=https://edp-project-deployment.onrender.com
```

Deployment configuration should be reviewed whenever the application structure or dependency files change.

---

## 📚 Documentation

Additional project documentation is available in the `docs/` directory.

| Document | Description |
|---|---|
| [API_REFERENCE.md](docs/API_REFERENCE.md) | API endpoint reference |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment configuration |
| [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | Known project limitations |
| [MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) | Model evaluation information |
| [SECURITY.md](docs/SECURITY.md) | Security documentation |
| [TESTING.md](docs/TESTING.md) | Testing instructions |

---

## ⚠️ Limitations and Responsible Use

FraudGuard AI is an educational and engineering project demonstrating Machine Learning-based fraud detection.

Important considerations:

- Model predictions are not guaranteed to be correct.
- A predicted fraud probability is not proof of fraudulent activity.
- False positives can affect legitimate transactions.
- False negatives can allow fraudulent transactions to pass undetected.
- Model performance depends on the quality and representativeness of training data.
- Historical transaction patterns may not represent future fraud behavior.
- Deployment availability depends on the hosting platform.
- Production financial decisions require additional validation, monitoring, security, compliance, and human oversight.

This project should not be treated as a production-authorized financial decision system without further independent assessment.

---

## 🔮 Future Improvements

Potential future enhancements include:

- Advanced model monitoring.
- Model drift detection.
- Explainable AI using SHAP.
- Real-time alert notifications.
- Advanced fraud pattern analysis.
- Improved model registry and version management.
- Production-grade database persistence.
- Automated CI/CD pipelines.
- Cloud-native container orchestration.
- More comprehensive security testing.
- Performance evaluation using larger independent datasets.

---

## 🎯 Project Outcomes

FraudGuard AI demonstrates practical experience in:

- End-to-end Machine Learning integration.
- Credit card fraud classification.
- XGBoost model development and inference.
- Data preprocessing and feature validation.
- FastAPI backend development.
- Streamlit application development.
- REST API integration.
- Real-time prediction workflows.
- Batch CSV processing.
- Dynamic classification thresholds.
- Model comparison and registry management.
- JWT authentication and role-based access control.
- Automated testing and verification.
- Docker-based application packaging.
- Git and GitHub version control.
- Cloud application deployment.

---

## 🔗 Important Links

| Resource | Link |
|---|---|
| GitHub Repository | [EDP-PROJECT-DEPLOYMENT](https://github.com/Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT) |
| Live Dashboard | [FraudGuard AI](https://edp-fraud-detection-rahul.streamlit.app/) |
| FastAPI Backend | [Backend Service](https://edp-project-deployment.onrender.com) |
| Swagger Documentation | [API Docs](https://edp-project-deployment.onrender.com/docs) |
| Health Endpoint | [Backend Health](https://edp-project-deployment.onrender.com/health) |

---

## 👨‍💻 Project Information

| Field | Details |
|---|---|
| Project Name | FraudGuard AI |
| Project Title | Credit Card Fraud Detection – Full-Stack AI/ML System |
| Project Type | Machine Learning + Full-Stack Application |
| Developer | **Geggalapally Rahul Reddy** |
| Programming Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Machine Learning Model | Tuned XGBoost |
| Deployment | Render and Streamlit Community Cloud |
| Repository | EDP-PROJECT-DEPLOYMENT |

---

**Built by Geggalapally Rahul Reddy using Python, FastAPI, Streamlit, XGBoost, and practical Machine Learning engineering.**

⭐ If you find this project interesting, consider giving the repository a star!
