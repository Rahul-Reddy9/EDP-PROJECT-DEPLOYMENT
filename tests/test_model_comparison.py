"""
test_model_comparison.py
------------------------
Unit and regression tests verifying Phase 3 Multi-Model Development,
the Candidate Model Registry, and artifact integrity.
"""

import os
import json
import pytest
import joblib
import numpy as np
import pandas as pd


BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
REGISTRY_DIR = os.path.join(BACKEND_DIR, "models", "registry")
MANIFEST_PATH = os.path.join(REGISTRY_DIR, "comparison_manifest.json")


class TestCandidateModelRegistry:
    """Verifies that candidate models exist, manifest metadata is valid, and artifacts can infer."""

    def test_01_manifest_exists_and_valid_json(self):
        """Manifest file must exist and contain valid JSON structure."""
        assert os.path.exists(MANIFEST_PATH), f"Manifest file missing at {MANIFEST_PATH}"
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "generated_at" in manifest
        assert "dataset_context" in manifest
        assert "active_production_model" in manifest
        assert "models" in manifest
        assert isinstance(manifest["models"], list)
        assert len(manifest["models"]) >= 5

    def test_02_all_five_candidate_models_registered(self):
        """All 5 candidate algorithms must be registered with CV and test metrics."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        model_names = {m["model_name"] for m in manifest["models"]}
        expected_names = {
            "Tuned XGBoost (Baseline v1.0)",
            "Balanced Logistic Regression",
            "Balanced Random Forest",
            "HistGradientBoosting",
            "Balanced Extra Trees",
        }
        assert expected_names.issubset(model_names)

        for m in manifest["models"]:
            # Check Cross-Validation structure
            assert "cross_validation" in m
            cv = m["cross_validation"]
            assert "f1_mean" in cv
            assert "recall_mean" in cv
            assert "precision_mean" in cv
            assert "roc_auc_mean" in cv
            assert "pr_auc_mean" in cv

            # Check Independent Test structure
            assert "independent_test" in m
            it = m["independent_test"]
            assert "recall" in it
            assert "precision" in it
            assert "f1_score" in it
            assert "roc_auc" in it
            assert "pr_auc" in it
            assert "tp" in it and "fp" in it and "tn" in it and "fn" in it
            # Total test records must sum to 1008
            assert it["tp"] + it["fp"] + it["tn"] + it["fn"] == 1008

    def test_03_candidate_model_files_exist_and_loadable(self):
        """Artifacts referenced in manifest must exist on disk and support predict_proba."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Generate a dummy 30-feature vector
        dummy_input = np.zeros((1, 30))

        for m in manifest["models"]:
            rel_path = m["artifact_file"]
            abs_path = os.path.join(REGISTRY_DIR, rel_path)
            assert os.path.exists(abs_path), f"Candidate model artifact missing: {abs_path}"

            loaded_model = joblib.load(abs_path)
            assert hasattr(loaded_model, "predict_proba"), f"Model {m['model_name']} does not implement predict_proba"
            probas = loaded_model.predict_proba(dummy_input)
            assert probas.shape == (1, 2)
            assert pytest.approx(float(np.sum(probas)), abs=1e-5) == 1.0

    def test_04_baseline_artifacts_and_backups_preserved(self):
        """Tuned XGBoost v1.0 and Scaler in backend/models/ must be intact, along with backups."""
        models_dir = os.path.join(BACKEND_DIR, "models")
        assert os.path.exists(os.path.join(models_dir, "tuned_xgboost.pkl"))
        assert os.path.exists(os.path.join(models_dir, "scaler.pkl"))
        assert os.path.exists(os.path.join(models_dir, "backup_v1_tuned_xgboost.pkl"))
        assert os.path.exists(os.path.join(models_dir, "backup_v1_scaler.pkl"))

    def test_05_sample_size_advisory_contents(self):
        """Manifest advisory must explicitly document the 8 test fraud records limitation."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        ctx = manifest.get("dataset_context", {})
        warning = ctx.get("evaluation_warning", "")
        assert "8" in warning or "eight" in warning.lower()
        assert "12.5%" in warning or "12.5" in warning
