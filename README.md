# Credit Card Fraud Detection – Final Internship Project

An end-to-end Machine Learning application for detecting fraudulent credit card transactions in real time. This project integrates a **FastAPI REST API**, a **Streamlit interactive dashboard**, and a tuned **XGBoost classifier** to provide real-time transaction scoring, batch prediction, risk classification, and fraud analysis.

The project demonstrates the complete integration of Machine Learning inference with full-stack application development, API engineering, testing, and cloud deployment.

---

## 🚀 Live Application

| Component | URL |
|---|---|
| **Live Streamlit Dashboard** | https://edp-fraud-detection-rahul.streamlit.app/ |
| **FastAPI Backend** | https://edp-project-deployment.onrender.com |
| **Interactive API Documentation** | https://edp-project-deployment.onrender.com/docs |
| **Backend Health Check** | https://edp-project-deployment.onrender.com/health |
| **GitHub Repository** | https://github.com/Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT |

### Deployment Architecture

- **Frontend:** Streamlit Community Cloud
- **Backend:** Render
- **Machine Learning Model:** Tuned XGBoost Classifier
- **API Communication:** REST API using HTTP and JSON
- **Version Control:** Git and GitHub

> **Note:** The Render free service may enter sleep mode after inactivity. The first request after inactivity may take longer because the service needs to start.

---

## 🏛️ System Architecture

```text
+----------------------------------------------------------+
|                  PRESENTATION TIER                       |
|                                                          |
|                 Streamlit Dashboard                      |
|                 streamlit_app.py                         |
|                                                          |
|  - Real-time transaction scoring                         |
|  - Dynamic classification threshold                      |
|  - Batch CSV upload and processing                       |
|  - Risk tier visualization                               |
|  - Enriched CSV export                                   |
|  - API health monitoring                                 |
+---------------------------+------------------------------+
                            |
                            | HTTP Requests (REST / JSON)
                            v
+----------------------------------------------------------+
|                CLIENT & VALIDATION TIER                  |
|                                                          |
|             frontend/api_client.py                       |
|             frontend/validators.py                       |
|                                                          |
|  - API communication                                     |
|  - Connection retry and timeout handling                 |
|  - 30-feature schema validation                          |
|  - Input sanitization                                    |
|  - Batch request chunking                                |
+---------------------------+------------------------------+
                            |
                            | REST API requests
                            v
+----------------------------------------------------------+
|                    BACKEND REST API                     |
|                                                          |
|                    FastAPI                               |
|                    backend/app.py                        |
|                                                          |
|  GET  /                 API information                  |
|  GET  /health           Service and model health         |
|  GET  /sample           Sample transaction payloads     |
|  POST /predict          Single transaction prediction    |
|  POST /predict/batch    Batch transaction prediction     |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|                    INFERENCE TIER                        |
|                                                          |
|                 backend/src/predict.py                   |
|                                                          |
|  - Feature preprocessing                                 |
|  - Time and Amount scaling                               |
|  - V1–V28 feature preservation                           |
|  - Tuned XGBoost inference                                |
|                                                          |
|                  Model Artifacts                         |
|  - tuned_xgboost.pkl                                      |
|  - scaler.pkl                                             |
+----------------------------------------------------------+
```

---

## 📂 Project Structure

```text
EDP-PROJECT-DEPLOYMENT/
│
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── models/
│   │   ├── tuned_xgboost.pkl
│   │   └── scaler.pkl
│   └── src/
│       ├── __init__.py
│       └── predict.py
│
├── frontend/
│   ├── __init__.py
│   ├── api_client.py
│   └── validators.py
│
├── docs/
│   └── week9_scope.md
│
├── reports/
│   └── week10_report.md
│
└── tests/
    ├── __init__.py
    ├── test_api_client.py
    ├── test_live_integration.py
    └── test_validators.py
```

*The structure above describes the documented project components. Keep the entries aligned with the files actually committed to the GitHub repository.*

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Backend Framework | FastAPI |
| Frontend Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib, Pickle |
| API Communication | REST API, HTTP, JSON |
| Testing | Pytest |
| Backend Deployment | Render |
| Frontend Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## ⚙️ Local Setup and Execution

### Prerequisites

- Python 3.10 or later
- Git
- PowerShell (Windows) or Bash (Linux/macOS)
- Internet connection if using the deployed backend

### 1. Clone the Repository

```powershell
git clone https://github.com/Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT.git

cd EDP-PROJECT-DEPLOYMENT
```

### 2. Install Dependencies

From the repository root:

```powershell
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend

Open the first terminal:

```powershell
cd backend
```

Install backend dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI service:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Verify the backend:

- Health Check: http://127.0.0.1:8000/health
- Interactive API Documentation: http://127.0.0.1:8000/docs

### 4. Launch the Streamlit Dashboard

Open a second terminal from the repository root:

```powershell
streamlit run streamlit_app.py
```

The dashboard will be available at:

http://localhost:8501

**API Configuration:** For local backend testing, configure the dashboard's FastAPI Service URL as:

```text
http://127.0.0.1:8000
```

For the deployed application, use:

```text
https://edp-project-deployment.onrender.com
```

---

## 💡 Application Features

### 🎯 1. Single Transaction Scoring

- Enter transaction amount and elapsed time.
- Provide the 28 anonymized PCA features.
- Load sample legitimate and fraudulent transactions.
- Submit transactions for real-time prediction.
- View fraud probability and classification results.
- Identify the transaction risk category.

### 📁 2. Batch CSV Processing

- Upload CSV files containing the required transaction features.
- Validate the input schema before prediction.
- Process multiple transactions through the batch API.
- View total processed transactions.
- View fraud count and legitimate transaction count.
- Analyze fraud rate and risk distribution.
- Visualize prediction probabilities.
- Download enriched prediction results as a CSV file.

### 🎚️ 3. Dynamic Decision Threshold

The dashboard supports an adjustable fraud classification threshold between **0.05 and 0.95**, with a default threshold of **0.50**.

| Threshold | Expected Effect |
|---|---|
| Lower threshold | Higher fraud sensitivity and potentially more false positives |
| Higher threshold | Fewer transactions flagged and potentially more missed fraud cases |

The final binary classification is determined by comparing the predicted fraud probability with the selected threshold.

### 🚦 4. Risk Classification

The application categorizes transactions into three risk levels:

| Risk Tier | Fraud Probability |
|---|---|
| LOW | Below 40% |
| MEDIUM | 40% to below 75% |
| HIGH | 75% and above |

Risk tiers provide additional interpretation of model predictions and are separate from the binary classification threshold.

### 🩺 5. API Diagnostics and Health Monitoring

The dashboard provides:

- Backend connectivity status.
- API response time.
- Model loading status.
- Scaler loading status.
- Expected feature count.
- Access to interactive API documentation.

---

## 🧠 Machine Learning Inference Pipeline

The application uses a tuned XGBoost classifier with saved preprocessing artifacts.

### Expected Input Features

The model expects 30 transaction features:

```text
Time
V1, V2, V3, ... V28
Amount
```

### Preprocessing

1. Validate the required 30-feature input schema.
2. Scale `Time` and `Amount` using the saved scaler.
3. Preserve PCA features `V1` through `V28` without scaling.
4. Arrange features in the canonical training order.
5. Pass the processed feature vector to the trained XGBoost classifier.
6. Return fraud probability and classification results.

### Model Artifacts

```text
backend/
├── models/
│   ├── tuned_xgboost.pkl
│   └── scaler.pkl
└── src/
    └── predict.py
```

The backend loads the saved model and scaler artifacts for inference. It does not retrain the model during application startup.

---

## 🧪 Testing and Validation

Run the test suite from the repository root:

```powershell
python -m pytest tests/ -v
```

### Test Coverage

| Test Module | Tests | Description |
|---|---:|---|
| `test_api_client.py` | 10 | API communication, error handling, timeout, and batch processing |
| `test_live_integration.py` | 4 | Live backend health, single prediction, and batch prediction |
| `test_validators.py` | 10 | Schema validation, missing values, and invalid inputs |
| **Total** | **24** | **Unit and integration tests** |

### Recorded Test Results

```text
======================= 24 passed in 4.82s =======================
```

The above is the previously recorded test result in the project documentation. Run the test command locally to reproduce it.

---

## ☁️ Cloud Deployment

### Backend Deployment – Render

| Configuration | Value |
|---|---|
| Platform | Render |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health Endpoint | `/health` |

**Live Backend:**

https://edp-project-deployment.onrender.com

### Frontend Deployment – Streamlit Community Cloud

| Configuration | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| GitHub Repository | `Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT` |
| Branch | `main` |
| Main File | `streamlit_app.py` |

**Live Dashboard:**

https://edp-fraud-detection-rahul.streamlit.app/

---

## 🔒 Preservation of the Original Week 8 Project

The original Week 8 project was preserved separately during the final project integration and deployment process.

The deployment repository contains the backend inference files and model artifacts required to run the application.

The original Week 8 project was not intentionally modified or deleted during the deployment workflow.

---

## 🎯 Project Outcomes

This final internship project demonstrates practical implementation of:

- End-to-end Machine Learning application integration.
- REST API development using FastAPI.
- Interactive dashboard development using Streamlit.
- Machine Learning model artifact loading and inference.
- Feature preprocessing and schema validation.
- Real-time and batch prediction workflows.
- Dynamic classification threshold configuration.
- Unit testing and live API integration testing.
- Git-based version control and GitHub collaboration.
- Cloud deployment using Render and Streamlit Community Cloud.
- Decoupled frontend and backend architecture.

---

## 🔗 Important Project Links

- **Live Application:** https://edp-fraud-detection-rahul.streamlit.app/
- **GitHub Repository:** https://github.com/Rahul-Reddy9/EDP-PROJECT-DEPLOYMENT
- **FastAPI Backend:** https://edp-project-deployment.onrender.com
- **API Documentation:** https://edp-project-deployment.onrender.com/docs
- **Backend Health:** https://edp-project-deployment.onrender.com/health

---

## 👨‍💻 Project Information

| Field | Details |
|---|---|
| Project Title | Credit Card Fraud Detection – Full-Stack AI/ML System |
| Project Phase | Final Week |
| Developer | G. Rahul Reddy |
| Repository | EDP-PROJECT-DEPLOYMENT |

**Final Internship Project – Credit Card Fraud Detection using Machine Learning, FastAPI, and Streamlit.**
