"""
test_frontend_ui.py
-------------------
Unit tests for the modernized frontend styling, components, and charts.
Validates that all visualization and theming modules instantiate correctly
without runtime exceptions.
"""

import os
import unittest
import pandas as pd
import plotly.graph_objects as go

from frontend.styles import get_fintech_theme_css
from frontend.components import (
    clean_html,
    render_brand_header,
    render_kpi_metrics,
    render_prediction_result_banner,
)
from frontend.validators import (
    FEATURE_ORDER,
    get_sample_preset,
    get_borderline_preset,
    validate_single_transaction,
)
from frontend.charts import (
    create_risk_gauge,
    create_risk_donut_chart,
    create_probability_distribution_chart,
    create_amount_vs_prob_chart,
    create_top_anomalies_chart,
)


class TestFrontendUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_api_url = os.environ.get("API_BASE_URL")
        cls._orig_env = os.environ.get("ENV")
        os.environ["API_BASE_URL"] = "http://127.0.0.1:8000"
        os.environ["ENV"] = "test"

    @classmethod
    def tearDownClass(cls):
        if cls._orig_api_url is not None:
            os.environ["API_BASE_URL"] = cls._orig_api_url
        else:
            os.environ.pop("API_BASE_URL", None)

        if cls._orig_env is not None:
            os.environ["ENV"] = cls._orig_env
        else:
            os.environ.pop("ENV", None)

    def test_01_theme_css_generation(self):
        """Custom fintech CSS should contain core dark navy and glassmorphic selectors."""
        css = get_fintech_theme_css()
        self.assertIsInstance(css, str)
        self.assertIn(".brand-container", css)
        self.assertIn(".risk-banner-high", css)
        self.assertIn(".kpi-card", css)
        self.assertIn("070B14", css)

    def test_02_borderline_preset_validity(self):
        """Borderline preset must contain all 30 features and pass validation."""
        preset = get_borderline_preset()
        self.assertEqual(len(preset), 30)
        self.assertEqual(sorted(list(preset.keys())), sorted(FEATURE_ORDER))
        is_valid, cleaned, errors = validate_single_transaction(preset)
        self.assertTrue(is_valid, f"Validation errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_03_create_risk_gauge(self):
        """Risk gauge should generate a valid Plotly Figure with threshold indicator."""
        fig = create_risk_gauge(probability=0.88, threshold=0.5)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "indicator")
        self.assertEqual(fig.data[0].value, 88.0)

    def test_04_create_risk_donut_chart(self):
        """Risk donut chart should generate a valid Plotly Figure."""
        counts = {"LOW": 100, "MEDIUM": 15, "HIGH": 5}
        fig = create_risk_donut_chart(counts)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.data[0].type, "pie")
        self.assertEqual(fig.data[0].hole, 0.68)

    def test_05_create_probability_distribution_chart(self):
        """Probability distribution chart should handle list of floats cleanly."""
        probs = [0.01, 0.05, 0.45, 0.78, 0.99]
        fig = create_probability_distribution_chart(probs, threshold=0.5)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.data[0].type, "histogram")

    def test_06_create_amount_vs_prob_chart(self):
        """Amount vs Probability scatter plot should map dataframe columns."""
        df = pd.DataFrame({
            "Amount": [10.0, 50.0, 200.0],
            "Fraud_Probability": [0.01, 0.45, 0.95],
            "Risk_Level": ["LOW", "MEDIUM", "HIGH"],
        })
        fig = create_amount_vs_prob_chart(df)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 1)

    def test_07_create_top_anomalies_chart(self):
        """Top anomalies chart should parse V1-V28 features and return horizontal bar chart."""
        sample = get_sample_preset(fraud=True)
        fig = create_top_anomalies_chart(sample)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.data[0].type, "bar")

    def test_08_clean_html_strips_indentation_and_blank_lines(self):
        """clean_html should remove leading indentation and empty lines to prevent markdown code blocks."""
        raw_html = """
            <div class="kpi-card">
                <span class="pulse-dot online"></span>
            </div>
        """
        cleaned = clean_html(raw_html)
        lines = cleaned.splitlines()
        for line in lines:
            self.assertEqual(line, line.strip())
            self.assertFalse(line.startswith("    "))

    def test_09_render_brand_header_no_pre_code(self):
        """Brand header HTML should render as clean markup without pre or code wrapping."""
        import markdown_it
        md = markdown_it.MarkdownIt()
        rendered = []

        class MockSt:
            def markdown(self, text, unsafe_allow_html=False):
                rendered.append((text, unsafe_allow_html))

        import frontend.components as fc
        orig_st = fc.st
        fc.st = MockSt()
        try:
            render_brand_header(is_healthy=True, latency_ms=18.0, base_url="http://localhost:8000")
            render_brand_header(is_healthy=False, latency_ms=0.0, base_url="http://localhost:8000")
        finally:
            fc.st = orig_st

        self.assertEqual(len(rendered), 2)
        for html_str, unsafe in rendered:
            self.assertTrue(unsafe)
            self.assertIn("brand-container", html_str)
            parsed = md.render(html_str)
            self.assertNotIn("<pre>", parsed)
            self.assertNotIn("<code>", parsed)

    def test_10_render_kpi_metrics_no_pre_code(self):
        """KPI executive cards should render without triggering markdown code blocks."""
        import markdown_it
        md = markdown_it.MarkdownIt()
        rendered = []

        class MockSt:
            def markdown(self, text, unsafe_allow_html=False):
                rendered.append((text, unsafe_allow_html))

        import frontend.components as fc
        orig_st = fc.st
        fc.st = MockSt()
        try:
            render_kpi_metrics(total_tx=1250, fraud_tx=15, legit_tx=1235, total_volume=45000.0, latency_ms=25.0)
        finally:
            fc.st = orig_st

        self.assertEqual(len(rendered), 1)
        html_str, unsafe = rendered[0]
        self.assertTrue(unsafe)
        self.assertIn("kpi-container", html_str)
        self.assertIn("kpi-card", html_str)
        parsed = md.render(html_str)
        self.assertNotIn("<pre>", parsed)
        self.assertNotIn("<code>", parsed)

    def test_11_render_prediction_result_banner_no_pre_code(self):
        """Prediction result banners across all risk tiers should render clean HTML without pre/code."""
        import markdown_it
        md = markdown_it.MarkdownIt()

        import frontend.components as fc
        orig_st = fc.st

        for tier, pred in [("LOW", 0), ("MEDIUM", 0), ("HIGH", 1)]:
            rendered = []
            class MockSt:
                def markdown(self, text, unsafe_allow_html=False):
                    rendered.append((text, unsafe_allow_html))

            fc.st = MockSt()
            try:
                render_prediction_result_banner(
                    result={"prediction": pred, "label": "Fraudulent" if pred == 1 else "Legitimate", "fraud_probability": 0.85, "risk_level": tier},
                    amount=150.0,
                    threshold=0.50,
                )
            finally:
                fc.st = orig_st

            self.assertEqual(len(rendered), 1)
            html_str, unsafe = rendered[0]
            self.assertTrue(unsafe)
            self.assertIn("risk-banner", html_str)
            parsed = md.render(html_str)
            self.assertNotIn("<pre>", parsed)
            self.assertNotIn("<code>", parsed)

    def test_12_theme_css_starts_with_style_no_pre_code(self):
        """Fintech theme CSS must render directly as a <style> block and never as pre/code."""
        import markdown_it
        md = markdown_it.MarkdownIt()
        css = get_fintech_theme_css()
        parsed = md.render(clean_html(css))
        self.assertNotIn("<pre>", parsed)
        self.assertNotIn("<code>", parsed)
        self.assertTrue(parsed.strip().startswith("<style>"))

    def test_13_apptest_model_evaluation_navigation(self):
        """Streamlit app should navigate to Model Training & Evaluation view without exceptions."""
        from streamlit.testing.v1 import AppTest
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Initial run exceptions: {at.exception}")

        # Switch to Model Training & Evaluation tab
        for r in at.radio:
            if "📈 Model Training & Evaluation" in r.options:
                r.set_value("📈 Model Training & Evaluation")
        at.run()
        self.assertEqual(len(at.exception), 0, f"Evaluation tab exceptions: {at.exception}")

    def test_14_apptest_batch_processing_navigation(self):
        """Streamlit app should navigate to Batch CSV Processing view without exceptions."""
        from streamlit.testing.v1 import AppTest
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Initial run exceptions: {at.exception}")

        # Switch to Batch CSV Processing tab
        for r in at.radio:
            if "📁 Batch CSV Processing" in r.options:
                r.set_value("📁 Batch CSV Processing")
        at.run()
        self.assertEqual(len(at.exception), 0, f"Batch tab exceptions: {at.exception}")


if __name__ == "__main__":
    unittest.main()
