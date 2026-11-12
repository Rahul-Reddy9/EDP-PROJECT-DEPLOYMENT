"""
test_preprocessing_audit.py
---------------------------
Rigorous audit test suite for FraudGuard AI preprocessing and input-scaling resolution:
- Raw transaction inputs (standard range, zero values, extreme values)
- Standardized transaction inputs (negative z-scores, small positive z-scores)
- Ambiguity resolution and explicit is_preprocessed parameter
- Single-record predictions vs batch predictions parity
- Negative feature values in PCA features
- Missing and invalid features validation
- Explicit predict_raw and predict_preprocessed API methods
"""

import os
import sys
import warnings
import pytest
import numpy as np
import pandas as pd

from backend.src.predict import FraudPredictor, FEATURE_ORDER, get_sample_transaction


@pytest.fixture(scope="module")
def predictor():
    return FraudPredictor()


class TestPreprocessingAndDoubleScalingAudit:
    """Verifies that preprocessing is completely idempotent, robust, and never silently distorts inputs."""

    def test_01_raw_transaction_standard_range(self, predictor):
        """Standard raw transaction features must be scaled accurately."""
        sample = get_sample_transaction(fraud=False)
        arr, _ = predictor._to_array_and_df(sample, is_preprocessed=False)
        expected_time_scaled = (sample["Time"] - predictor._time_mean) / predictor._time_scale
        expected_amount_scaled = (sample["Amount"] - predictor._amount_mean) / predictor._amount_scale

        assert pytest.approx(arr[0, 0], abs=1e-4) == expected_time_scaled
        assert pytest.approx(arr[0, 29], abs=1e-4) == expected_amount_scaled

    def test_02_raw_transaction_zero_values(self, predictor):
        """Zero values in Time (midnight) and Amount ($0 auth check) must scale correctly as raw."""
        sample = get_sample_transaction(fraud=False)
        sample["Time"] = 0.0
        sample["Amount"] = 0.0

        arr, _ = predictor._to_array_and_df(sample, is_preprocessed=False)
        expected_time_scaled = (0.0 - predictor._time_mean) / predictor._time_scale
        expected_amount_scaled = (0.0 - predictor._amount_mean) / predictor._amount_scale

        assert pytest.approx(arr[0, 0], abs=1e-4) == expected_time_scaled
        assert pytest.approx(arr[0, 29], abs=1e-4) == expected_amount_scaled
        assert arr[0, 0] < -1.5  # Negative z-score for zero seconds
        assert arr[0, 29] < -1.0  # Negative z-score for zero amount

    def test_03_raw_transaction_extreme_values(self, predictor):
        """Extreme raw values (e.g. 48 hours elapsed, $25,000 transaction) must scale without overflow."""
        sample = get_sample_transaction(fraud=False)
        sample["Time"] = 172800.0  # 48 hours
        sample["Amount"] = 25000.0  # $25,000

        arr, _ = predictor._to_array_and_df(sample, is_preprocessed=False)
        expected_time_scaled = (172800.0 - predictor._time_mean) / predictor._time_scale
        expected_amount_scaled = (25000.0 - predictor._amount_mean) / predictor._amount_scale

        assert pytest.approx(arr[0, 0], abs=1e-3) == expected_time_scaled
        assert pytest.approx(arr[0, 29], abs=1e-3) == expected_amount_scaled
        assert arr[0, 0] > 1.5  # Positive z-score
        assert arr[0, 29] > 100.0  # High positive z-score

    def test_04_preprocessed_negative_z_scores_auto_detection(self, predictor):
        """Preprocessed inputs with negative values must automatically bypass re-scaling."""
        sample = get_sample_transaction(fraud=False)
        sample["Time"] = -0.7682  # Standardized z-score
        sample["Amount"] = -0.1053  # Standardized z-score

        # Auto-detect mode (is_preprocessed=None)
        arr, _ = predictor._to_array_and_df(sample, is_preprocessed=None)
        assert pytest.approx(arr[0, 0], abs=1e-4) == -0.7682
        assert pytest.approx(arr[0, 29], abs=1e-4) == -0.1053

    def test_05_preprocessed_positive_z_scores_explicit_flag(self, predictor):
        """Preprocessed inputs with positive z-scores must be preserved when is_preprocessed=True."""
        sample = get_sample_transaction(fraud=False)
        sample["Time"] = 0.50
        sample["Amount"] = 0.80

        # Explicit preprocessed mode
        arr, _ = predictor._to_array_and_df(sample, is_preprocessed=True)
        assert pytest.approx(arr[0, 0], abs=1e-4) == 0.50
        assert pytest.approx(arr[0, 29], abs=1e-4) == 0.80

    def test_06_preprocessed_batch_auto_detection(self, predictor):
        """A batch of preprocessed records with low mean/variance must bypass re-scaling even if positive."""
        sample = get_sample_transaction(fraud=False)
        # Create a batch of 10 preprocessed records with small positive z-scores
        batch = []
        for i in range(10):
            row = sample.copy()
            row["Time"] = 0.1 * i
            row["Amount"] = 0.05 * i
            batch.append(row)

        arr, _ = predictor._to_array_and_df(batch, is_preprocessed=None)
        # Verify first row
        assert pytest.approx(arr[0, 0], abs=1e-4) == 0.0
        assert pytest.approx(arr[0, 29], abs=1e-4) == 0.0
        # Verify 5th row
        assert pytest.approx(arr[5, 0], abs=1e-4) == 0.5
        assert pytest.approx(arr[5, 29], abs=1e-4) == 0.25

    def test_07_negative_pca_features_preserved(self, predictor):
        """PCA features V1-V28 must never be scaled, shifted, or modified regardless of signs."""
        sample = get_sample_transaction(fraud=True)
        orig_v1 = sample["V1"]
        orig_v2 = sample["V2"]
        orig_v14 = sample["V14"]

        arr, _ = predictor._to_array_and_df(sample, is_preprocessed=False)
        # V1 is index 1, V2 is index 2, V14 is index 14
        assert pytest.approx(arr[0, 1], abs=1e-5) == orig_v1
        assert pytest.approx(arr[0, 2], abs=1e-5) == orig_v2
        assert pytest.approx(arr[0, 14], abs=1e-5) == orig_v14

    def test_08_single_vs_batch_consistency(self, predictor):
        """Single-record prediction and batch prediction must produce identical probabilities."""
        sample_legit = get_sample_transaction(fraud=False)
        sample_fraud = get_sample_transaction(fraud=True)

        prob_legit_single = predictor.predict_proba(sample_legit)[0]
        prob_fraud_single = predictor.predict_proba(sample_fraud)[0]

        batch = [sample_legit, sample_fraud]
        probs_batch = predictor.predict_proba(batch)

        assert pytest.approx(probs_batch[0], abs=1e-7) == prob_legit_single
        assert pytest.approx(probs_batch[1], abs=1e-7) == prob_fraud_single

    def test_09_missing_features_raises_value_error(self, predictor):
        """Missing required features must raise ValueError detailing the missing column names."""
        bad_sample = {"Time": 100.0, "V1": 0.5, "Amount": 50.0}
        with pytest.raises(ValueError) as excinfo:
            predictor.predict(bad_sample)
        assert "Missing required feature(s)" in str(excinfo.value)
        assert "V2" in str(excinfo.value)

    def test_10_explicit_predict_raw_and_preprocessed_methods(self, predictor):
        """predict_raw and predict_preprocessed explicit wrappers operate correctly."""
        raw_sample = get_sample_transaction(fraud=False)
        res_raw = predictor.predict_raw(raw_sample)[0]
        assert "fraud_probability" in res_raw
        assert res_raw["label"] == "Legitimate"

        # Preprocessed sample with negative values
        prep_sample = raw_sample.copy()
        prep_sample["Time"] = -0.5
        prep_sample["Amount"] = 0.2
        res_prep = predictor.predict_preprocessed(prep_sample)[0]
        assert "fraud_probability" in res_prep

    def test_11_ambiguous_input_warns_and_defaults_to_raw(self, predictor):
        """Ambiguous single observation (positive Time and Amount in ambiguous zone)
        with is_preprocessed=None emits a UserWarning and safely scales as raw.
        """
        sample = get_sample_transaction(fraud=False)
        sample["Time"] = 12.0  # In ambiguous zone [0, 35]
        sample["Amount"] = 45.0  # In ambiguous zone [0, 100]

        with pytest.warns(UserWarning, match="Ambiguous transaction scaling detected"):
            arr, _ = predictor._to_array_and_df(sample, is_preprocessed=None)

        expected_time_scaled = (12.0 - predictor._time_mean) / predictor._time_scale
        expected_amount_scaled = (45.0 - predictor._amount_mean) / predictor._amount_scale
        assert pytest.approx(arr[0, 0], abs=1e-4) == expected_time_scaled
        assert pytest.approx(arr[0, 29], abs=1e-4) == expected_amount_scaled

    def test_12_ambiguous_input_explicit_raw_no_warning(self, predictor):
        """Ambiguous inputs with explicit is_preprocessed=False or predict_raw never warn."""
        sample = get_sample_transaction(fraud=False)
        sample["Time"] = 12.0
        sample["Amount"] = 45.0

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise any warning
            arr, _ = predictor._to_array_and_df(sample, is_preprocessed=False)
            res = predictor.predict_raw(sample)

        expected_time_scaled = (12.0 - predictor._time_mean) / predictor._time_scale
        assert pytest.approx(arr[0, 0], abs=1e-4) == expected_time_scaled
        assert len(res) == 1

    def test_13_ambiguous_input_explicit_preprocessed_no_warning(self, predictor):
        """Ambiguous inputs with explicit is_preprocessed=True or predict_preprocessed never warn and stay unscaled."""
        sample = get_sample_transaction(fraud=False)
        sample["Time"] = 0.50
        sample["Amount"] = 0.85

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise any warning
            arr, _ = predictor._to_array_and_df(sample, is_preprocessed=True)
            res = predictor.predict_preprocessed(sample)

        assert pytest.approx(arr[0, 0], abs=1e-4) == 0.50
        assert pytest.approx(arr[0, 29], abs=1e-4) == 0.85
        assert len(res) == 1

