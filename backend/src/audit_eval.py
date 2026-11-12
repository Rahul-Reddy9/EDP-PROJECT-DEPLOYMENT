"""
audit_eval.py
-------------
Independent evaluation script for FraudGuard AI v1.0 and v2.0.
Computes ground truth metrics, confusion matrices, and calibration scores.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

n_total = len(y_test)
n_fraud = int((y_test == 1).sum())
n_legit = int((y_test == 0).sum())

print("=== PHASE 2: INDEPENDENT MODEL EVALUATION ===")
print(f"Test Records Total: {n_total} (Legitimate: {n_legit}, Fraudulent: {n_fraud})")

v1_model = joblib.load("backend/models/backup_v1_tuned_xgboost.pkl")
v2_model = joblib.load("backend/models/tuned_xgboost.pkl")

v1_probs = v1_model.predict_proba(X_test)[:, 1]
v2_probs = v2_model.predict_proba(X_test)[:, 1]


def evaluate(name, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    return {
        "name": name,
        "threshold": threshold,
        "cm": (tn, fp, fn, tp),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2,
        "specificity": spec,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }


for th in [0.50, 0.30, 0.15, 0.10, 0.05]:
    r1 = evaluate("v1.0 (Baseline)", v1_probs, th)
    r2 = evaluate("v2.0 (Optimized)", v2_probs, th)
    print(f"\n--- THRESHOLD = {th:.2f} ---")
    for r in [r1, r2]:
        tn, fp, fn, tp = r["cm"]
        print(
            f"{r['name']}: TP={tp}, FP={fp}, FN={fn}, TN={tn} | "
            f"Acc={r['accuracy']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}, "
            f"F1={r['f1']:.4f}, F2={r['f2']:.4f}, Spec={r['specificity']:.4f}, "
            f"FPR={r['fpr']:.4f}, ROC={r['roc_auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, "
            f"Brier={r['brier']:.6f}"
        )

# OOF Threshold Analysis
import json
oof_path = "backend/models/registry/training_oof_predictions.json"
y_train_path = "data/processed/y_train.csv"

if os.path.exists(oof_path) and os.path.exists(y_train_path):
    with open(oof_path, "r") as f:
        d = json.load(f)
    oof_probs = np.array(d["oof_probabilities"])
    y_train = pd.read_csv(y_train_path).squeeze().values

    print("\n=======================================================")
    print("PHASE 5: 5-FOLD OUT-OF-FOLD (OOF) THRESHOLD ANALYSIS (N=4,032)")
    print("=======================================================")
    samples_in_window = ((oof_probs >= 0.10) & (oof_probs < 0.15)).sum()
    print(f"OOF Samples in [0.10, 0.15) window: {samples_in_window}")
    print(f"{'Threshold':>10} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>5} | {'Precision':>9} {'Recall':>8} {'F1-Score':>8} {'F2-Score':>8} {'Specificity':>11}")
    print("-" * 80)
    for th in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50]:
        preds = (oof_probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
        p = precision_score(y_train, preds, zero_division=0)
        r = recall_score(y_train, preds, zero_division=0)
        f1 = f1_score(y_train, preds, zero_division=0)
        f2 = fbeta_score(y_train, preds, beta=2, zero_division=0)
        spec = tn / (tn + fp)
        print(f"{th:>10.2f} | {tp:>4} {fp:>4} {fn:>4} {tn:>5} | {p:>9.4f} {r:>8.4f} {f1:>8.4f} {f2:>8.4f} {spec:>11.4f}")

if __name__ == "__main__":
    pass
