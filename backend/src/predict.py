"""
predict.py
----------
Week 8: Core Model Packaging & Inference Pipeline.

Loads the existing tuned XGBoost model (models/tuned_xgboost.pkl) and feature
scaler (models/scaler.pkl). Applies preprocessing (scaling ONLY Time and Amount,
preserving V1-V28 unscaled) and strictly enforces the exact 30-feature sequence:
Time, V1..V28, Amount.
"""

import os
from typing import Dict, List, Union, Any
import numpy as np
import pandas as pd
import joblib

# Canonical feature ordering expected by models/tuned_xgboost.pkl
FEATURE_ORDER = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]

SCALE_COLS = ["Time", "Amount"]

DEFAULT_MODEL_PATH = os.path.join("models", "tuned_xgboost.pkl")
DEFAULT_SCALER_PATH = os.path.join("models", "scaler.pkl")


class FraudPredictor:
    """Production predictor encapsulating feature scaling, column alignment,
    and inference with the tuned XGBoost model.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads the pre-trained model and scaler artifacts."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Ensure Week 5-7 artifacts exist in models/."
            )
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler file not found at {self.scaler_path}. "
                "Ensure models/scaler.pkl exists."
            )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def preprocess(self, data: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame]) -> pd.DataFrame:
        """Preprocesses input transaction data:
        1. Converts input to DataFrame.
        2. Validates presence of all 30 required features.
        3. Scales ONLY 'Time' and 'Amount' using self.scaler.
        4. Leaves 'V1'-'V28' unscaled.
        5. Reorders columns strictly to FEATURE_ORDER.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input data must be a dict, list of dicts, or pandas DataFrame.")

        # If target 'Class' exists in raw data (e.g. from Kaggle), drop it safely
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Check for missing required features
        missing = [f for f in FEATURE_ORDER if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")

        # Re-index to ensure exact column ordering
        df = df[FEATURE_ORDER].copy()

        # Scale only Time and Amount
        df[SCALE_COLS] = self.scaler.transform(df[SCALE_COLS])

        return df

    def predict_proba(self, data: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame]) -> np.ndarray:
        """Returns fraud probability for each transaction."""
        processed_df = self.preprocess(data)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(processed_df)[:, 1]
        elif hasattr(self.model, "predict"):
            return self.model.predict(processed_df).astype(float)
        else:
            raise AttributeError("Loaded model has no predict or predict_proba method.")

    def predict(
        self,
        data: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Makes predictions with risk scoring and structured metadata."""
        probabilities = self.predict_proba(data)
        results = []

        for prob in probabilities:
            prob_float = float(prob)
            is_fraud = int(prob_float >= threshold)

            if prob_float >= 0.75:
                risk_level = "HIGH"
            elif prob_float >= 0.40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            results.append({
                "prediction": is_fraud,
                "label": "Fraudulent" if is_fraud == 1 else "Legitimate",
                "fraud_probability": round(prob_float, 4),
                "threshold_used": threshold,
                "risk_level": risk_level,
            })

        return results


def get_sample_transaction(fraud: bool = False) -> Dict[str, float]:
    """Provides a realistic sample transaction for testing."""
    if fraud:
        # Verified fraudulent transaction from dataset
        return {
            "Time": 99772.46,
            "V1": 1.68568, "V2": -2.32051, "V3": 1.15678, "V4": 1.43683,
            "V5": 2.07263, "V6": 3.43088, "V7": 0.40904, "V8": 0.15500,
            "V9": -0.46354, "V10": 3.18685, "V11": 2.63562, "V12": 2.89936,
            "V13": -0.56984, "V14": 0.13619, "V15": 0.99926, "V16": 2.90565,
            "V17": 1.52876, "V18": 0.32014, "V19": 0.40335, "V20": 0.86711,
            "V21": 1.65071, "V22": 0.11091, "V23": 1.02666, "V24": 1.71387,
            "V25": -0.46354, "V26": -1.10055, "V27": 0.28921, "V28": 0.11728,
            "Amount": 738.54,
        }
    else:
        # Verified legitimate transaction from dataset
        return {
            "Time": 31604.21,
            "V1": -0.42015, "V2": -0.43649, "V3": -1.51860, "V4": 2.98559,
            "V5": -1.63805, "V6": -2.23789, "V7": -2.27306, "V8": -1.00042,
            "V9": -0.78319, "V10": 0.57358, "V11": -0.57348, "V12": 1.55848,
            "V13": 2.59228, "V14": 1.45910, "V15": 0.82946, "V16": -1.08856,
            "V17": 1.44921, "V18": -0.39534, "V19": -1.57161, "V20": 1.14596,
            "V21": 0.23719, "V22": 0.67931, "V23": -0.45521, "V24": 0.28307,
            "V25": 1.18469, "V26": -0.61417, "V27": -1.60533, "V28": 0.66319,
            "Amount": 149.97,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference using Week 8 packaged model.")
    parser.add_argument("--sample", choices=["legit", "fraud"], default="legit", help="Run on a sample transaction.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for fraud class.")
    args = parser.parse_args()

    predictor = FraudPredictor()
    sample = get_sample_transaction(fraud=(args.sample == "fraud"))
    result = predictor.predict(sample, threshold=args.threshold)[0]

    print("\n--- Fraud Prediction Result ---")
    print(f"Sample Type:       {args.sample.upper()}")
    print(f"Prediction:        {result['prediction']} ({result['label']})")
    print(f"Fraud Probability: {result['fraud_probability']:.4%}")
    print(f"Risk Level:        {result['risk_level']}")
    print(f"Threshold:         {result['threshold_used']}")
    print("--------------------------------\n")


if __name__ == "__main__":
    main()
