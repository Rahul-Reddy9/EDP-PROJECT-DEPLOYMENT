"""
test_validators.py
------------------
Unit tests for single transaction and batch CSV validation rules.
Ensures consistency with the 30 canonical features and FastAPI constraints.
"""

import unittest
import pandas as pd
import numpy as np

from frontend.validators import (
    FEATURE_ORDER,
    validate_single_transaction,
    validate_batch_dataframe,
    get_sample_preset,
    create_sample_csv_dataframe,
)


class TestValidators(unittest.TestCase):
    def setUp(self):
        self.valid_single = get_sample_preset(fraud=False)

    def test_01_valid_single_transaction(self):
        """A complete, well-formed 30-feature payload must pass validation."""
        is_valid, cleaned, errors = validate_single_transaction(self.valid_single)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(cleaned), 30)
        self.assertEqual(cleaned["Amount"], 149.97)

    def test_02_single_transaction_missing_feature(self):
        """Omitting any canonical feature should trigger a validation error."""
        invalid_data = self.valid_single.copy()
        del invalid_data["V14"]
        del invalid_data["Amount"]

        is_valid, cleaned, errors = validate_single_transaction(invalid_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required features" in err for err in errors))
        self.assertIn("V14", errors[0])

    def test_03_single_transaction_negative_amount(self):
        """Negative amounts violate business logic and must be rejected."""
        invalid_data = self.valid_single.copy()
        invalid_data["Amount"] = -25.50

        is_valid, cleaned, errors = validate_single_transaction(invalid_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("Amount' cannot be negative" in err for err in errors))

    def test_04_single_transaction_nan_infinite(self):
        """NaN or Inf feature values should be rejected."""
        invalid_nan = self.valid_single.copy()
        invalid_nan["V5"] = float("nan")

        is_valid, cleaned, errors = validate_single_transaction(invalid_nan)
        self.assertFalse(is_valid)
        self.assertTrue(any("cannot be NaN or Infinite" in err for err in errors))

    def test_05_valid_batch_dataframe(self):
        """Valid batch dataframe should pass and maintain canonical 30 columns."""
        df = create_sample_csv_dataframe()
        self.assertEqual(len(df), 5)

        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(list(cleaned_df.columns), FEATURE_ORDER)

    def test_06_batch_dataframe_with_class_column(self):
        """Batch dataframe containing 'Class' column retains it when retain_class=True and drops it when retain_class=False."""
        df = create_sample_csv_dataframe()
        df["Class"] = [0, 1, 0, 1, 0]

        # By default retain_class=True preserves Class for post-inference evaluation
        is_valid, cleaned_df_retained, errors = validate_batch_dataframe(df, retain_class=True)
        self.assertTrue(is_valid)
        self.assertIn("Class", cleaned_df_retained.columns)
        self.assertEqual(list(cleaned_df_retained["Class"]), [0, 1, 0, 1, 0])

        # When retain_class=False, Class is dropped
        is_valid_drop, cleaned_df_dropped, errors_drop = validate_batch_dataframe(df, retain_class=False)
        self.assertTrue(is_valid_drop)
        self.assertNotIn("Class", cleaned_df_dropped.columns)
        self.assertEqual(list(cleaned_df_dropped.columns), FEATURE_ORDER)

    def test_07_batch_dataframe_missing_column(self):
        """Batch dataframe missing required columns should fail pre-flight validation."""
        df = create_sample_csv_dataframe()
        df = df.drop(columns=["V12", "Amount"])

        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing 2 required column(s)" in err for err in errors))

    def test_08_batch_dataframe_empty(self):
        """Empty DataFrame should be rejected."""
        empty_df = pd.DataFrame()
        is_valid, cleaned_df, errors = validate_batch_dataframe(empty_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("contains no transaction rows" in err for err in errors))

    def test_09_batch_dataframe_non_numeric(self):
        """Non-numeric values in feature columns should fail validation."""
        df = create_sample_csv_dataframe()
        df["V1"] = df["V1"].astype(object)
        df.loc[0, "V1"] = "corrupted_text"

        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("missing or non-numeric" in err for err in errors))

    def test_10_sample_template_integrity(self):
        """The downloadable batch template should always produce 5 valid rows."""
        sample_df = create_sample_csv_dataframe()
        is_valid, cleaned_df, errors = validate_batch_dataframe(sample_df)
        self.assertTrue(is_valid)
        self.assertEqual(len(cleaned_df), 5)


if __name__ == "__main__":
    unittest.main()
