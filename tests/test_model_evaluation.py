"""
test_model_evaluation.py
------------------------
Unit and integration tests for FraudGuard AI's model evaluation engine.
Verifies ground-truth benchmark metrics against the held-out test dataset
(data/processed/X_test.csv, y_test.csv) and tuned XGBoost model artifact.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from frontend.model_evaluation import (
    get_dataset_intelligence,
    get_model_metadata,
    get_evaluator,
    ModelEvaluator,
    DatasetIntelligence,
    ModelMetadata,
)
from frontend.charts import (
    create_evaluation_confusion_matrix_chart,
    create_evaluation_roc_curve_chart,
    create_evaluation_pr_curve_chart,
    create_evaluation_feature_importance_chart,
    create_evaluation_class_distribution_chart,
    create_evaluation_split_comparison_chart,
    create_evaluation_class_probability_distribution_chart,
    create_evaluation_threshold_sweep_chart,
)


class TestDatasetIntelligence:
    """Verifies dataset metadata and ground-truth transaction counts."""

    def test_01_dataset_intelligence_properties(self):
        ds = get_dataset_intelligence()
        assert isinstance(ds, DatasetIntelligence)
        assert ds.raw_filename == "creditcard.csv"
        assert ds.total_records == 5040
        assert ds.raw_legit == 5000
        assert ds.raw_fraud == 40
        assert pytest.approx(ds.raw_fraud_pct, rel=1e-3) == 0.7937
        assert pytest.approx(ds.raw_legit_pct, rel=1e-3) == 99.2063

    def test_02_train_test_split_consistency(self):
        ds = get_dataset_intelligence()
        assert ds.train_records == 4032
        assert ds.train_legit == 4000
        assert ds.train_fraud == 32
        assert ds.test_records == 1008
        assert ds.test_legit == 1000
        assert ds.test_fraud == 8
        assert ds.num_features == 30
        assert "Stratified" in ds.split_strategy
        assert ds.random_state == 42


class TestModelMetadata:
    """Verifies model architecture, tuning parameters, and artifact references."""

    def test_03_model_metadata_properties(self):
        meta = get_model_metadata()
        assert isinstance(meta, ModelMetadata)
        assert "XGBoost" in meta.model_name
        assert meta.num_features == 30
        assert "StandardScaler" in meta.scaler_type
        assert "Time" in meta.scaled_features
        assert "Amount" in meta.scaled_features
        assert len(meta.unscaled_features) == 28

    def test_04_hyperparameter_verification(self):
        meta = get_model_metadata()
        params = meta.hyperparameters
        assert params.get("n_estimators") == 200
        assert params.get("max_depth") == 3
        assert pytest.approx(params.get("learning_rate"), rel=1e-2) == 0.1
        assert pytest.approx(params.get("scale_pos_weight"), rel=1e-2) == 125.0


class TestModelEvaluator:
    """Verifies genuine performance metrics evaluated on the held-out test partition."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        return get_evaluator()

    def test_05_evaluator_initialization(self, evaluator):
        assert isinstance(evaluator, ModelEvaluator)
        assert len(evaluator.X_test) == 1008
        assert len(evaluator.y_test) == 1008
        assert evaluator.y_test.sum() == 8
        assert (evaluator.y_test == 0).sum() == 1000
        assert len(evaluator.y_proba) == 1008
        assert np.all((evaluator.y_proba >= 0.0) & (evaluator.y_proba <= 1.0))

    def test_06_discrimination_curves_auc(self, evaluator):
        # Ground truth ROC-AUC and PR-AUC
        assert pytest.approx(evaluator.roc_auc, rel=1e-3) == 0.9910
        assert pytest.approx(evaluator.pr_auc, rel=1e-3) == 0.7492
        assert len(evaluator.fpr) > 0
        assert len(evaluator.tpr) > 0

    def test_07_default_threshold_metrics(self, evaluator):
        # Canonical threshold θ = 0.50
        m = evaluator.compute_metrics(threshold=0.50)
        assert m["threshold"] == 0.50
        assert m["test_sample_size"] == 1008
        assert m["tp"] == 4
        assert m["fp"] == 1
        assert m["fn"] == 4
        assert m["tn"] == 999

        # Performance scores
        assert pytest.approx(m["accuracy"], rel=1e-3) == 1003 / 1008  # ~99.50%
        assert pytest.approx(m["precision"], rel=1e-3) == 4 / 5        # 80.00%
        assert pytest.approx(m["recall"], rel=1e-3) == 4 / 8           # 50.00%
        assert pytest.approx(m["f1_score"], rel=1e-3) == 0.6154
        assert pytest.approx(m["specificity"], rel=1e-3) == 999 / 1000 # 99.90%
        assert pytest.approx(m["fpr"], rel=1e-3) == 1 / 1000          # 0.10%
        assert pytest.approx(m["fnr"], rel=1e-3) == 4 / 8             # 50.00%

    def test_08_multi_threshold_sensitivity_table(self, evaluator):
        table = evaluator.get_multi_threshold_table([0.30, 0.40, 0.50, 0.60, 0.70])
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 5
        assert list(table.columns) == [
            "Threshold (θ)",
            "TP",
            "FP",
            "FN",
            "TN",
            "Precision",
            "Recall",
            "F1-Score",
            "Specificity",
            "FPR",
            "FNR",
            "Accuracy",
        ]

        # Verify threshold monotonic properties
        tps = list(table["TP"])
        fps = list(table["FP"])
        # As threshold increases, predicted positives should not increase
        assert tps[0] >= tps[-1]
        assert fps[0] >= fps[-1]

    def test_09_feature_importances(self, evaluator):
        fi = evaluator.feature_importances
        assert isinstance(fi, pd.DataFrame)
        assert len(fi) == 30
        assert "Feature" in fi.columns
        assert "Importance" in fi.columns
        assert "Percentage" in fi.columns

        # Amount and V5 are top contributors
        top_feature = fi.iloc[0]["Feature"]
        assert top_feature == "Amount"
        assert pytest.approx(fi.iloc[0]["Percentage"], rel=1e-1) == 24.31
        assert pytest.approx(fi["Percentage"].sum(), rel=1e-3) == 100.0

    def test_10_report_exports(self, evaluator):
        # JSON export test
        json_str = evaluator.export_report_json(active_threshold=0.50)
        parsed = json.loads(json_str)
        assert "dataset_intelligence" in parsed
        assert "model_metadata" in parsed
        assert "ground_truth_metrics" in parsed
        assert parsed["ground_truth_metrics"]["confusion_matrix"]["true_positives"] == 4

        # CSV export test
        csv_str = evaluator.export_report_csv(active_threshold=0.50)
        assert "True Positives (TP),4" in csv_str
        assert "ROC-AUC,0.9910" in csv_str


class TestEvaluationCharts:
    """Verifies that all executive Plotly visualization builders produce valid figures."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        return get_evaluator()

    def test_11_chart_generation(self, evaluator):
        m = evaluator.compute_metrics(threshold=0.50)
        multi_th = evaluator.get_multi_threshold_table()

        fig_cm = create_evaluation_confusion_matrix_chart(m["tp"], m["fp"], m["fn"], m["tn"], 0.50)
        assert fig_cm is not None

        fig_roc = create_evaluation_roc_curve_chart(evaluator.fpr, evaluator.tpr, evaluator.roc_auc)
        assert fig_roc is not None

        fig_pr = create_evaluation_pr_curve_chart(evaluator.pr_precisions, evaluator.pr_recalls, evaluator.pr_auc)
        assert fig_pr is not None

        fig_fi = create_evaluation_feature_importance_chart(evaluator.feature_importances, top_n=15)
        assert fig_fi is not None

        fig_class = create_evaluation_class_distribution_chart(1000, 8)
        assert fig_class is not None

        fig_split = create_evaluation_split_comparison_chart(4000, 32, 1000, 8)
        assert fig_split is not None

        fig_prob = create_evaluation_class_probability_distribution_chart(evaluator.y_test, evaluator.y_proba, 0.50)
        assert fig_prob is not None

        fig_sweep = create_evaluation_threshold_sweep_chart(multi_th)
        assert fig_sweep is not None
