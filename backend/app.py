"""
app.py
------
Production FastAPI REST Service for Credit Card Fraud Detection.

Exposes endpoints for health checks, single transaction predictions,
and batch predictions using the packaged tuned XGBoost model and scaler.
Hardened with cryptographic JWT authentication, server-side RBAC,
persistent SQLite token revocation, sliding-window rate limiting,
CORS strict origin control, sanitized health-checks, and DoS limits.
"""

import os
import io
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends, Request, status, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from src.predict import FraudPredictor, FEATURE_ORDER, get_sample_transaction
from src.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    require_admin_role,
    require_user_or_admin,
    revoke_token,
    User,
    LoginPayload,
    TokenResponse,
    UserRole,
    RevocationStoreError,
)
from src.rate_limiter import rate_limit_login, rate_limit_predict

# Configure structured logging
logger = logging.getLogger("fraudguard.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [API] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Environment Mode
IS_PRODUCTION = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower().strip() in ("production", "prod")

# Enforced Security Limits
MAX_REQUEST_SIZE_BYTES = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB limit
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "10000"))  # 10,000 transactions limit

# Initialize FastAPI application
# In production, Swagger UI and OpenAPI documentation endpoints are disabled to prevent API reconnaissance
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Production ML inference service for detecting fraudulent credit card transactions. "
        "Powered by a tuned XGBoost classifier with automated Time and Amount feature scaling, "
        "cryptographic JWT authentication, and server-side RBAC for Independent User and MLOps Administrator."
    ),
    version="1.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# Global Request Size Limit Middleware
@app.middleware("http")
async def enforce_request_size_limit(request: Request, call_next):
    """Enforces global request size limit (10MB) to protect against DoS and memory exhaustion."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
            if length > MAX_REQUEST_SIZE_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Payload Too Large: Request body size ({length} bytes) exceeds "
                            f"the maximum allowed limit of {MAX_REQUEST_SIZE_BYTES} bytes (10MB)."
                        )
                    },
                )
        except ValueError:
            pass
    return await call_next(request)

# Configure CORS: Read allowed origins from environment (disallow wildcards when credentials enabled)
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize predictor singleton
try:
    predictor = FraudPredictor()
except Exception as e:
    predictor = None
    initialization_error = str(e)
    logger.error(f"Predictor initialization failed: {initialization_error}")
else:
    initialization_error = None


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class TransactionPayload(BaseModel):
    """Features for a single credit card transaction."""
    Time: float = Field(..., description="Seconds elapsed since the first transaction", json_schema_extra={"example": 0.0})
    V1: float = Field(..., description="PCA transformed feature V1", json_schema_extra={"example": -1.3598})
    V2: float = Field(..., description="PCA transformed feature V2", json_schema_extra={"example": -0.0728})
    V3: float = Field(..., description="PCA transformed feature V3", json_schema_extra={"example": 2.5363})
    V4: float = Field(..., description="PCA transformed feature V4", json_schema_extra={"example": 1.3782})
    V5: float = Field(..., description="PCA transformed feature V5", json_schema_extra={"example": -0.3383})
    V6: float = Field(..., description="PCA transformed feature V6", json_schema_extra={"example": 0.4624})
    V7: float = Field(..., description="PCA transformed feature V7", json_schema_extra={"example": 0.2396})
    V8: float = Field(..., description="PCA transformed feature V8", json_schema_extra={"example": 0.0987})
    V9: float = Field(..., description="PCA transformed feature V9", json_schema_extra={"example": 0.3638})
    V10: float = Field(..., description="PCA transformed feature V10", json_schema_extra={"example": 0.0908})
    V11: float = Field(..., description="PCA transformed feature V11", json_schema_extra={"example": -0.5516})
    V12: float = Field(..., description="PCA transformed feature V12", json_schema_extra={"example": -0.6178})
    V13: float = Field(..., description="PCA transformed feature V13", json_schema_extra={"example": -0.9914})
    V14: float = Field(..., description="PCA transformed feature V14", json_schema_extra={"example": -0.3112})
    V15: float = Field(..., description="PCA transformed feature V15", json_schema_extra={"example": 1.4682})
    V16: float = Field(..., description="PCA transformed feature V16", json_schema_extra={"example": -0.4704})
    V17: float = Field(..., description="PCA transformed feature V17", json_schema_extra={"example": 0.2080})
    V18: float = Field(..., description="PCA transformed feature V18", json_schema_extra={"example": 0.0258})
    V19: float = Field(..., description="PCA transformed feature V19", json_schema_extra={"example": 0.4040})
    V20: float = Field(..., description="PCA transformed feature V20", json_schema_extra={"example": 0.2514})
    V21: float = Field(..., description="PCA transformed feature V21", json_schema_extra={"example": -0.0183})
    V22: float = Field(..., description="PCA transformed feature V22", json_schema_extra={"example": 0.2778})
    V23: float = Field(..., description="PCA transformed feature V23", json_schema_extra={"example": -0.1105})
    V24: float = Field(..., description="PCA transformed feature V24", json_schema_extra={"example": 0.0669})
    V25: float = Field(..., description="PCA transformed feature V25", json_schema_extra={"example": 0.1285})
    V26: float = Field(..., description="PCA transformed feature V26", json_schema_extra={"example": -0.1891})
    V27: float = Field(..., description="PCA transformed feature V27", json_schema_extra={"example": 0.1336})
    V28: float = Field(..., description="PCA transformed feature V28", json_schema_extra={"example": -0.0211})
    Amount: float = Field(..., description="Transaction amount in currency units", json_schema_extra={"example": 149.62})


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Legitimate, 1 for Fraudulent")
    label: str = Field(..., description="Human-readable classification label")
    fraud_probability: float = Field(..., description="Estimated probability of fraud [0.0 - 1.0]")
    legitimate_probability: float = Field(default=1.0, description="Estimated probability of legitimate [0.0 - 1.0]")
    threshold_used: float = Field(..., description="Decision threshold applied")
    risk_level: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    model_version: str = Field(default="1.0-baseline", description="Active model artifact version")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Tree SHAP feature contribution breakdown")


class BatchTransactionPayload(BaseModel):
    transactions: List[TransactionPayload] = Field(..., description="List of transactions to evaluate")


class BatchPredictionResponse(BaseModel):
    total_processed: int
    fraud_count: int
    legitimate_count: int
    predictions: List[PredictionResponse]


# ---------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"], dependencies=[Depends(rate_limit_login)])
def login(payload: LoginPayload) -> TokenResponse:
    """Authenticates credentials (analyst or admin) and issues a cryptographic JWT token."""
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(user)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        role=user.role,
        username=user.username,
        display_name=user.display_name,
    )


@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"], dependencies=[Depends(rate_limit_login)])
def oauth2_token(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """OAuth2 password form endpoint for client authorization."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(user)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        role=user.role,
        username=user.username,
        display_name=user.display_name,
    )


@app.get("/auth/me", response_model=User, tags=["Authentication"])
def get_current_user_profile(current_user: User = Depends(get_current_user)) -> User:
    """Returns currently authenticated user profile and active RBAC role."""
    return current_user


@app.post("/auth/logout", tags=["Authentication"])
def logout(request: Request, current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    """Terminates session and persistently revokes the active JWT token in SQLite store."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        try:
            persisted = revoke_token(token)
            if not persisted:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or missing token claims for revocation.",
                )
        except RevocationStoreError as exc:
            logger.error(f"Failed to persist token revocation during logout: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service Unavailable: Failed to persist token revocation. Session termination could not be guaranteed.",
                headers={"Retry-After": "30"},
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bearer token required in Authorization header for logout.",
        )
    return {"status": "logged_out", "message": f"Session terminated for user '{current_user.username}'."}


# ---------------------------------------------------------
# Public Metadata & Probe Endpoints
# ---------------------------------------------------------

@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    """Returns general metadata about the inference API."""
    return {
        "service": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "framework": "FastAPI + XGBoost",
        "documentation": "/docs" if not IS_PRODUCTION else "disabled_in_production",
        "health_check": "/health",
        "endpoints": {
            "auth_login": "POST /auth/login",
            "auth_me": "GET /auth/me",
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "predict_csv": "POST /predict/csv",
            "sample_data": "GET /sample",
            "model_info": "GET /model/info",
            "admin_metrics": "GET /model/metrics",
            "admin_threshold": "GET /model/threshold",
            "admin_compare": "GET /model/compare",
        },
    }


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Verifies service readiness and artifact availability without leaking local system paths."""
    if predictor is None or predictor.model is None or predictor.scaler is None:
        logger.error(f"Health check failed due to artifact initialization error: {initialization_error}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: model artifacts could not be loaded. Please check server logs.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
        "model_type": type(predictor.model).__name__,
        "scaler_type": type(predictor.scaler).__name__,
        "expected_features_count": len(FEATURE_ORDER),
    }


@app.get("/sample", tags=["Testing"])
def get_sample(fraud: bool = Query(False, description="Set to true for fraud sample, false for legitimate")) -> Dict[str, Any]:
    """Provides a ready-to-test sample payload."""
    sample = get_sample_transaction(fraud=fraud)
    return {
        "type": "Fraudulent Example" if fraud else "Legitimate Example",
        "payload": sample,
    }


# ---------------------------------------------------------
# Inference Endpoints (Authenticated: User or Admin)
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    dependencies=[Depends(require_user_or_admin), Depends(rate_limit_predict)],
)
def predict_transaction(
    payload: TransactionPayload,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for fraud classification"),
    explain: bool = Query(False, description="Set to true to compute Tree SHAP feature contributions"),
) -> PredictionResponse:
    """Predicts fraud status for a single transaction payload with optional Tree SHAP explainability."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service not initialized.")

    try:
        data_dict = payload.model_dump()
        result = predictor.predict(data_dict, threshold=threshold, explain=explain)[0]
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Single prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail="Inference processing failed. Verify input feature values.")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    dependencies=[Depends(require_user_or_admin), Depends(rate_limit_predict)],
)
def predict_batch(
    payload: BatchTransactionPayload,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for fraud classification"),
) -> BatchPredictionResponse:
    """Predicts fraud status for a batch of transaction payloads with enforced size limit."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service not initialized.")

    if not payload.transactions:
        raise HTTPException(status_code=400, detail="Transactions list cannot be empty.")

    if len(payload.transactions) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size ({len(payload.transactions)}) exceeds maximum allowed limit of {MAX_BATCH_ROWS} transactions.",
        )

    try:
        data_list = [t.model_dump() for t in payload.transactions]
        results = predictor.predict(data_list, threshold=threshold)
        pred_objs = [PredictionResponse(**r) for r in results]

        fraud_count = sum(1 for p in pred_objs if p.prediction == 1)
        legit_count = len(pred_objs) - fraud_count

        return BatchPredictionResponse(
            total_processed=len(pred_objs),
            fraud_count=fraud_count,
            legitimate_count=legit_count,
            predictions=pred_objs,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Batch prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail="Batch inference processing failed. Verify input transaction records.")


@app.post(
    "/predict/csv",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    dependencies=[Depends(require_user_or_admin), Depends(rate_limit_predict)],
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV file containing credit card transactions"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for fraud classification"),
) -> BatchPredictionResponse:
    """
    Accepts an uploaded CSV file with transaction records, enforcing:
    1. CSV file size limit (10MB).
    2. Batch row count limit (10,000 transactions).
    3. Strict feature presence and ordering.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service not initialized.")

    content = await file.read(MAX_REQUEST_SIZE_BYTES + 1)
    if len(content) > MAX_REQUEST_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded CSV exceeds maximum permitted size of {MAX_REQUEST_SIZE_BYTES} bytes (10MB).",
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to parse uploaded CSV file. Ensure valid CSV encoding.")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains 0 data rows.")

    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV row count ({len(df)}) exceeds maximum batch limit of {MAX_BATCH_ROWS} transactions.",
        )

    try:
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        missing = [col for col in FEATURE_ORDER if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV is missing required feature(s): {missing[:5]}")

        data_list = df[FEATURE_ORDER].to_dict(orient="records")
        results = predictor.predict(data_list, threshold=threshold)
        pred_objs = [PredictionResponse(**r) for r in results]

        fraud_count = sum(1 for p in pred_objs if p.prediction == 1)
        legit_count = len(pred_objs) - fraud_count

        return BatchPredictionResponse(
            total_processed=len(pred_objs),
            fraud_count=fraud_count,
            legitimate_count=legit_count,
            predictions=pred_objs,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"CSV batch prediction execution error: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail="CSV inference processing failed. Verify data values and column format.")


# ---------------------------------------------------------
# Model Governance Endpoints
# ---------------------------------------------------------

@app.get("/model/info", tags=["Model Governance"], dependencies=[Depends(require_user_or_admin)])
def get_model_info() -> Dict[str, Any]:
    """
    [PROTECTED: User or Admin]
    Returns technical metadata, feature schema, and configuration of the active model.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    params = {}
    if hasattr(predictor.model, "get_params"):
        raw_params = predictor.model.get_params()
        for k in ["n_estimators", "max_depth", "learning_rate", "scale_pos_weight", "eval_metric", "random_state"]:
            if k in raw_params:
                params[k] = raw_params[k]

    return {
        "status": "active",
        "model_name": "Tuned XGBoost Classifier",
        "model_version": "1.0-baseline",
        "model_type": type(predictor.model).__name__,
        "scaler_type": type(predictor.scaler).__name__,
        "expected_features_count": len(FEATURE_ORDER),
        "feature_order": FEATURE_ORDER,
        "scaled_features": ["Time", "Amount"],
        "hyperparameters": params,
        "default_threshold": 0.50,
        "registry_status": "Production Baseline (Active)",
    }


@app.get("/model/metrics", tags=["Model Governance"], dependencies=[Depends(require_admin_role)])
def get_model_metrics(threshold: float = 0.50) -> Dict[str, Any]:
    """[ADMIN PROTECTED] Returns ground-truth test evaluation metrics, confusion matrix, calibration, and bootstrap 95% CIs."""
    try:
        from frontend.model_evaluation import get_evaluator
        evaluator = get_evaluator()
        th_val = getattr(threshold, "default", threshold)
        th_val = float(th_val)
        metrics = evaluator.compute_metrics(threshold=th_val)
        cis = evaluator.compute_bootstrap_confidence_intervals(threshold=th_val)
        return {
            "status": "success",
            "model_version": "1.0-baseline",
            "threshold": th_val,
            "split": "held_out_test_set",
            "test_records": 1008,
            "test_fraud_cases": 8,
            "metrics": metrics,
            "confusion_matrix": {
                "true_positives": metrics["tp"],
                "false_positives": metrics["fp"],
                "false_negatives": metrics["fn"],
                "true_negatives": metrics["tn"],
            },
            "confidence_intervals_95": cis,
            "calibration": {
                "brier_score": evaluator.brier_score,
                "log_loss": evaluator.log_loss,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to compute evaluation metrics: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error occurred while computing model metrics.")


@app.get("/model/threshold", tags=["Model Governance"], dependencies=[Depends(require_admin_role)])
def get_model_threshold_recommendations() -> Dict[str, Any]:
    """[ADMIN PROTECTED] Returns data-driven decision threshold recommendations and sensitivity analysis table."""
    try:
        from frontend.model_evaluation import get_evaluator
        evaluator = get_evaluator()
        recs = evaluator.get_threshold_recommendations()
        table = evaluator.get_multi_threshold_table([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        return {
            "status": "success",
            "recommendations": recs,
            "sensitivity_table": table.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to generate threshold recommendations: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error occurred while generating threshold recommendations.")


@app.get("/model/compare", tags=["Model Governance"], dependencies=[Depends(require_admin_role)])
def get_model_comparison() -> Dict[str, Any]:
    """[ADMIN PROTECTED] Returns comparative benchmark manifest across candidate models from versioned registry."""
    try:
        from frontend.model_evaluation import get_model_comparison_data
        data = get_model_comparison_data()
        return {
            "status": "success",
            "manifest": data,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to load comparison data: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error occurred while retrieving model comparison manifest.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
