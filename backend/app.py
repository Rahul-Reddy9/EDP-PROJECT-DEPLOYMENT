"""
app.py
------
Week 8: Production FastAPI REST Service for Credit Card Fraud Detection.

Exposes endpoints for health checks, single transaction predictions,
and batch predictions using the packaged tuned XGBoost model and scaler.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import FraudPredictor, FEATURE_ORDER, get_sample_transaction

# Initialize FastAPI application
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Production ML inference service for detecting fraudulent credit card transactions. "
        "Powered by a tuned XGBoost classifier with automated Time and Amount feature scaling."
    ),
    version="1.0.0",
)

# Enable CORS for frontend applications/dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
try:
    predictor = FraudPredictor()
except Exception as e:
    predictor = None
    initialization_error = str(e)
else:
    initialization_error = None


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class TransactionPayload(BaseModel):
    """Features for a single credit card transaction."""
    Time: float = Field(..., description="Seconds elapsed since the first transaction", example=0.0)
    V1: float = Field(..., description="PCA transformed feature V1", example=-1.3598)
    V2: float = Field(..., description="PCA transformed feature V2", example=-0.0728)
    V3: float = Field(..., description="PCA transformed feature V3", example=2.5363)
    V4: float = Field(..., description="PCA transformed feature V4", example=1.3782)
    V5: float = Field(..., description="PCA transformed feature V5", example=-0.3383)
    V6: float = Field(..., description="PCA transformed feature V6", example=0.4624)
    V7: float = Field(..., description="PCA transformed feature V7", example=0.2396)
    V8: float = Field(..., description="PCA transformed feature V8", example=0.0987)
    V9: float = Field(..., description="PCA transformed feature V9", example=0.3638)
    V10: float = Field(..., description="PCA transformed feature V10", example=0.0908)
    V11: float = Field(..., description="PCA transformed feature V11", example=-0.5516)
    V12: float = Field(..., description="PCA transformed feature V12", example=-0.6178)
    V13: float = Field(..., description="PCA transformed feature V13", example=-0.9914)
    V14: float = Field(..., description="PCA transformed feature V14", example=-0.3112)
    V15: float = Field(..., description="PCA transformed feature V15", example=1.4682)
    V16: float = Field(..., description="PCA transformed feature V16", example=-0.4704)
    V17: float = Field(..., description="PCA transformed feature V17", example=0.2080)
    V18: float = Field(..., description="PCA transformed feature V18", example=0.0258)
    V19: float = Field(..., description="PCA transformed feature V19", example=0.4040)
    V20: float = Field(..., description="PCA transformed feature V20", example=0.2514)
    V21: float = Field(..., description="PCA transformed feature V21", example=-0.0183)
    V22: float = Field(..., description="PCA transformed feature V22", example=0.2778)
    V23: float = Field(..., description="PCA transformed feature V23", example=-0.1105)
    V24: float = Field(..., description="PCA transformed feature V24", example=0.0669)
    V25: float = Field(..., description="PCA transformed feature V25", example=0.1285)
    V26: float = Field(..., description="PCA transformed feature V26", example=-0.1891)
    V27: float = Field(..., description="PCA transformed feature V27", example=0.1336)
    V28: float = Field(..., description="PCA transformed feature V28", example=-0.0211)
    Amount: float = Field(..., description="Transaction amount in currency units", example=149.62)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Legitimate, 1 for Fraudulent")
    label: str = Field(..., description="Human-readable classification label")
    fraud_probability: float = Field(..., description="Estimated probability of fraud [0.0 - 1.0]")
    threshold_used: float = Field(..., description="Decision threshold applied")
    risk_level: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")


class BatchTransactionPayload(BaseModel):
    transactions: List[TransactionPayload] = Field(..., description="List of transactions to evaluate")


class BatchPredictionResponse(BaseModel):
    total_processed: int
    fraud_count: int
    legitimate_count: int
    predictions: List[PredictionResponse]


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    """Returns general metadata about the inference API."""
    return {
        "service": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "framework": "FastAPI + XGBoost",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "sample_data": "GET /sample",
        },
    }


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Verifies service readiness and artifact availability."""
    if predictor is None or predictor.model is None or predictor.scaler is None:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: model artifacts could not be loaded ({initialization_error})",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
        "model_type": type(predictor.model).__name__,
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


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(
    payload: TransactionPayload,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for fraud classification"),
) -> PredictionResponse:
    """Predicts fraud status for a single transaction payload."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service not initialized.")

    try:
        data_dict = payload.model_dump()
        result = predictor.predict(data_dict, threshold=threshold)[0]
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(
    payload: BatchTransactionPayload,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for fraud classification"),
) -> BatchPredictionResponse:
    """Predicts fraud status for a batch of transaction payloads."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service not initialized.")

    if not payload.transactions:
        raise HTTPException(status_code=400, detail="Transactions list cannot be empty.")

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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
