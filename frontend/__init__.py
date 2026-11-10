"""
frontend package
----------------
Streamlit client, styling, components, charts, and validation utilities
for the Credit Card Fraud Detection enterprise deployment.
"""

from frontend.api_client import (
    APIClient,
    resolve_default_api_url,
    DEFAULT_RENDER_BACKEND_URL,
    DEFAULT_LOCAL_BACKEND_URL,
)
from frontend.components import clean_html
from frontend.model_evaluation import (
    get_evaluator,
    get_dataset_intelligence,
    get_model_metadata,
    ModelEvaluator,
    DatasetIntelligence,
    ModelMetadata,
)
from frontend.validators import (
    FEATURE_ORDER,
    validate_single_transaction,
    validate_batch_dataframe,
    get_sample_preset,
    get_borderline_preset,
    create_sample_csv_dataframe,
)
from frontend.charts import (
    create_risk_gauge,
    create_risk_donut_chart,
    create_probability_distribution_chart,
    create_amount_vs_prob_chart,
    create_top_anomalies_chart,
    create_evaluation_confusion_matrix_chart,
    create_evaluation_roc_curve_chart,
    create_evaluation_pr_curve_chart,
    create_evaluation_feature_importance_chart,
    create_evaluation_class_distribution_chart,
    create_evaluation_split_comparison_chart,
    create_evaluation_class_probability_distribution_chart,
    create_evaluation_threshold_sweep_chart,
)

__all__ = [
    "APIClient",
    "resolve_default_api_url",
    "DEFAULT_RENDER_BACKEND_URL",
    "DEFAULT_LOCAL_BACKEND_URL",
    "clean_html",
    "get_evaluator",
    "get_dataset_intelligence",
    "get_model_metadata",
    "ModelEvaluator",
    "DatasetIntelligence",
    "ModelMetadata",
    "FEATURE_ORDER",
    "validate_single_transaction",
    "validate_batch_dataframe",
    "get_sample_preset",
    "get_borderline_preset",
    "create_sample_csv_dataframe",
    "create_risk_gauge",
    "create_risk_donut_chart",
    "create_probability_distribution_chart",
    "create_amount_vs_prob_chart",
    "create_top_anomalies_chart",
    "create_evaluation_confusion_matrix_chart",
    "create_evaluation_roc_curve_chart",
    "create_evaluation_pr_curve_chart",
    "create_evaluation_feature_importance_chart",
    "create_evaluation_class_distribution_chart",
    "create_evaluation_split_comparison_chart",
    "create_evaluation_class_probability_distribution_chart",
    "create_evaluation_threshold_sweep_chart",
]
