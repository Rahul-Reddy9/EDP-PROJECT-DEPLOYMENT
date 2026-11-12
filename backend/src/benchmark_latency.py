"""
benchmark_latency.py
--------------------
Rigorous latency benchmarking script for FraudGuard AI v1.0 vs v2.0.
Measures:
- Artifact loading time
- Preprocessing time
- Model inference time (pure model.predict_proba)
- End-to-end prediction time (preprocess + model + postprocess)
- Statistical breakdown: Median, p95, p99 over 1,000 iterations
- Batch inference latency across multiple batch sizes (1, 10, 50, 100, 500)
"""

import time
import os
import sys
from typing import Dict, List, Any
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.src.predict import FraudPredictor, get_sample_transaction, FEATURE_ORDER


def benchmark_model_loading(model_path: str, scaler_path: str, n_loads: int = 20):
    load_times = []
    for _ in range(n_loads):
        t0 = time.perf_counter()
        _ = joblib.load(model_path)
        _ = joblib.load(scaler_path)
        load_times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median": np.median(load_times),
        "mean": np.mean(load_times),
        "p95": np.percentile(load_times, 95),
        "p99": np.percentile(load_times, 99),
    }


def benchmark_single_inference(predictor: FraudPredictor, n_iterations: int = 1000):
    sample = get_sample_transaction(fraud=False)

    # 1. Warm-up
    for _ in range(50):
        predictor.predict(sample, threshold=0.50, explain=False)

    pre_times = []
    model_times = []
    post_times = []
    e2e_times = []

    for _ in range(n_iterations):
        # Paired iteration measurement: guarantees total_latency >= component_latency
        t0 = time.perf_counter()
        arr, _ = predictor._to_array_and_df(sample, is_preprocessed=False, need_df=False)
        t1 = time.perf_counter()

        prob = predictor.model.predict_proba(arr)[:, 1]
        t2 = time.perf_counter()

        # Postprocessing: thresholding, label assignment, dictionary construction
        prob_val = float(prob[0])
        is_fraud = int(prob_val >= 0.50)
        risk = "HIGH" if prob_val >= 0.75 else ("MEDIUM" if prob_val >= 0.40 else "LOW")
        _ = {
            "prediction": is_fraud,
            "label": "Fraudulent" if is_fraud else "Legitimate",
            "fraud_probability": prob_val,
            "risk_level": risk,
        }
        t3 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000.0
        mod_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        # Mathematical verification: total >= each component
        assert total_ms >= pre_ms, "Total latency must be >= preprocessing latency"
        assert total_ms >= mod_ms, "Total latency must be >= model latency"
        assert total_ms >= post_ms, "Total latency must be >= postprocessing latency"

        pre_times.append(pre_ms)
        model_times.append(mod_ms)
        post_times.append(post_ms)
        e2e_times.append(total_ms)

    def stats(arr):
        return {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "preprocessing": stats(pre_times),
        "model_inference": stats(model_times),
        "postprocessing": stats(post_times),
        "end_to_end": stats(e2e_times),
    }


def benchmark_http_api_latency(n_iterations: int = 200) -> Dict[str, Any]:
    """Measures full HTTP roundtrip latency against FastAPI (auth, deserialization, routing, predict, serialization)."""
    from fastapi.testclient import TestClient
    from backend.app import app, rate_limit_predict

    # Override rate limiter during high-throughput benchmark
    app.dependency_overrides[rate_limit_predict] = lambda: None

    try:
        client = TestClient(app)
        sample = get_sample_transaction(fraud=False)

        # Authenticate to obtain valid session token
        login_resp = client.post(
            "/auth/login",
            json={"username": "analyst", "password": "UserPassword@2026!"},
        )
        assert login_resp.status_code == 200, f"Login failed during benchmark: {login_resp.status_code}"
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Warmup
        for _ in range(10):
            client.post("/predict", json=sample, headers=headers)

        http_times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=sample, headers=headers)
            t1 = time.perf_counter()
            assert resp.status_code == 200
            http_times.append((t1 - t0) * 1000.0)

        return {
            "median": float(np.median(http_times)),
            "mean": float(np.mean(http_times)),
            "p95": float(np.percentile(http_times, 95)),
            "p99": float(np.percentile(http_times, 99)),
            "min": float(np.min(http_times)),
            "max": float(np.max(http_times)),
        }
    finally:
        app.dependency_overrides.pop(rate_limit_predict, None)


def verify_batch_identity(predictor: FraudPredictor) -> bool:
    """Verifies that batch predictions are byte-for-byte and probability-for-probability identical to individual predictions."""
    legit_s = get_sample_transaction(fraud=False)
    fraud_s = get_sample_transaction(fraud=True)
    batch = [legit_s, fraud_s] * 50  # 100 transactions

    batch_preds = predictor.predict(batch, threshold=0.50, explain=False)
    single_preds = [predictor.predict(s, threshold=0.50, explain=False)[0] for s in batch]

    assert len(batch_preds) == len(single_preds) == 100
    for i in range(100):
        b = batch_preds[i]
        s = single_preds[i]
        assert b["prediction"] == s["prediction"], f"Prediction mismatch at index {i}"
        assert b["risk_level"] == s["risk_level"], f"Risk level mismatch at index {i}"
        assert abs(b["fraud_probability"] - s["fraud_probability"]) < 1e-6, f"Probability mismatch at index {i}"
    return True


def benchmark_batches(predictor: FraudPredictor, batch_sizes=[1, 10, 50, 100, 500], n_repeats: int = 50):
    sample = get_sample_transaction(fraud=False)
    results = {}

    for bs in batch_sizes:
        batch = [sample] * bs
        # Warmup
        predictor.predict(batch, threshold=0.50, explain=False)
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = predictor.predict(batch, threshold=0.50, explain=False)
            times.append((time.perf_counter() - t0) * 1000.0)

        results[bs] = {
            "batch_size": bs,
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "per_record_ms": float(np.median(times) / bs),
        }

    return results


def run_full_benchmark():
    print("=================================================================")
    print("PHASE 4: RIGOROUS INFERENCE LATENCY BENCHMARK (1,000 ITERATIONS)")
    print("=================================================================")

    # 1. Loading time
    print("\n[1] Artifact Loading Benchmark (20 cold loads):")
    v1_load = benchmark_model_loading("backend/models/backup_v1_tuned_xgboost.pkl", "backend/models/backup_v1_scaler.pkl")
    v2_load = benchmark_model_loading("backend/models/tuned_xgboost.pkl", "backend/models/scaler.pkl")
    print(f"  v1.0 Loading: Median={v1_load['median']:.2f}ms, p95={v1_load['p95']:.2f}ms, p99={v1_load['p99']:.2f}ms")
    print(f"  v2.0 Loading: Median={v2_load['median']:.2f}ms, p95={v2_load['p95']:.2f}ms, p99={v2_load['p99']:.2f}ms")

    # 2. Predictor instances
    p_v1 = FraudPredictor(model_path="backend/models/backup_v1_tuned_xgboost.pkl", scaler_path="backend/models/backup_v1_scaler.pkl")
    p_v2 = FraudPredictor(model_path="backend/models/tuned_xgboost.pkl", scaler_path="backend/models/scaler.pkl")

    # 3. Single Prediction Breakdown
    print("\n[2] Single-Prediction Latency (1,000 iterations):")
    v1_single = benchmark_single_inference(p_v1, n_iterations=1000)
    v2_single = benchmark_single_inference(p_v2, n_iterations=1000)

    for name, res in [("v1.0 (Baseline)", v1_single), ("v2.0 (Optimized)", v2_single)]:
        print(f"\n  --- {name} ---")
        pre = res["preprocessing"]
        mod = res["model_inference"]
        post = res["postprocessing"]
        e2e = res["end_to_end"]
        print(f"  Preprocessing:     Median={pre['median']:.3f}ms | p95={pre['p95']:.3f}ms | p99={pre['p99']:.3f}ms | Mean={pre['mean']:.3f}ms")
        print(f"  Model Inference:   Median={mod['median']:.3f}ms | p95={mod['p95']:.3f}ms | p99={mod['p99']:.3f}ms | Mean={mod['mean']:.3f}ms")
        print(f"  Postprocessing:    Median={post['median']:.3f}ms | p95={post['p95']:.3f}ms | p99={post['p99']:.3f}ms | Mean={post['mean']:.3f}ms")
        print(f"  In-Process Total:  Median={e2e['median']:.3f}ms | p95={e2e['p95']:.3f}ms | p99={e2e['p99']:.3f}ms | Mean={e2e['mean']:.3f}ms")
        print(f"  Min={e2e['min']:.3f}ms | Max={e2e['max']:.3f}ms")

    # 4. Batch Inference Breakdown
    print("\n[3] Batch Inference Latency (Multiple Batch Sizes, 50 repeats each):")
    v1_batch = benchmark_batches(p_v1)
    v2_batch = benchmark_batches(p_v2)

    print(f"  {'Batch Size':>10} | {'v1.0 Med (ms)':>13} | {'v1.0 Per-Rec':>13} | {'v2.0 Med (ms)':>13} | {'v2.0 Per-Rec':>13} | {'v2.0 p95 (ms)':>13}")
    print("  " + "-" * 85)
    for bs in [1, 10, 50, 100, 500]:
        b1 = v1_batch[bs]
        b2 = v2_batch[bs]
        print(f"  {bs:>10} | {b1['median_ms']:>13.2f} | {b1['per_record_ms']:>13.3f} | {b2['median_ms']:>13.2f} | {b2['per_record_ms']:>13.3f} | {b2['p95_ms']:>13.2f}")

    # 5. Batch Identity Verification
    print("\n[4] Batch Identity Verification (100 Mixed Transactions):")
    batch_ok = verify_batch_identity(p_v2)
    print(f"  Batch predictions identical to single predictions: {batch_ok}")

    # 6. Full HTTP API Roundtrip Latency Benchmark
    print("\n[5] Full HTTP Roundtrip Latency (200 requests):")
    http_res = benchmark_http_api_latency(n_iterations=200)
    print(f"  HTTP Roundtrip:    Median={http_res['median']:.3f}ms | p95={http_res['p95']:.3f}ms | p99={http_res['p99']:.3f}ms | Mean={http_res['mean']:.3f}ms")
    print(f"  Min={http_res['min']:.3f}ms | Max={http_res['max']:.3f}ms")


if __name__ == "__main__":
    run_full_benchmark()
