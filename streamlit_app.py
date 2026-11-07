"""
streamlit_app.py
----------------
Week 10: Production Streamlit Dashboard for Credit Card Fraud Detection.
Integrates directly with the Week 8 FastAPI inference service.

Features:
- Live API connectivity monitoring & status indicators.
- Single transaction prediction with sample presets & dynamic threshold tuning.
- Batch CSV file ingestion, validation, batch inference, visual analytics & export.
- Full input validation & defensive error handling.
"""

import io
import time
import pandas as pd
import streamlit as st

from frontend.api_client import APIClient
from frontend.validators import (
    FEATURE_ORDER,
    validate_single_transaction,
    validate_batch_dataframe,
    get_sample_preset,
    create_sample_csv_dataframe,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI | Fraud Detection System",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge-online {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-offline {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .card-risk-high {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 18px;
        margin-top: 10px;
    }
    .card-risk-medium {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 18px;
        margin-top: 10px;
    }
    .card-risk-low {
        background-color: #F0FDF4;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 18px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "api_url" not in st.session_state:
    st.session_state.api_url = "https://edp-project-deployment.onrender.com"

if "single_inputs" not in st.session_state:
    # Initialize with default legitimate preset
    st.session_state.single_inputs = get_sample_preset(fraud=False)

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# -----------------------------------------------------------------------------
# Sidebar: API Connection & Global Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### âš™ï¸ System Configuration")

    api_url = st.text_input(
        "FastAPI Service URL",
        value=st.session_state.api_url,
        help="URL of the backend inference server (FastAPI)",
    )
    st.session_state.api_url = api_url

    client = APIClient(base_url=api_url, timeout=10.0)

    # Health check
    is_healthy, health_data, health_error = client.check_health()

    if is_healthy:
        latency = health_data.get("latency_ms", 0)
        model_type = health_data.get("model_type", "Unknown")
        st.markdown(
            f'<div class="status-badge-online">ðŸŸ¢ Online ({latency} ms)</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"**Model:** {model_type} | **Features:** {health_data.get('expected_features_count', 30)}")
    else:
        st.markdown(
            '<div class="status-badge-offline">ðŸ”´ Backend Offline</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Reason: {health_error[:90] if health_error else 'Connection refused'}...")

    st.divider()

    st.markdown("### ðŸŽšï¸ Decision Threshold")
    threshold = st.slider(
        "Fraud Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help=(
            "Probability cutoff for declaring a transaction fraudulent. "
            "Lower values increase sensitivity (higher recall); "
            "higher values decrease false alarms (higher precision)."
        ),
    )

    st.caption(
        f"**Active Threshold:** `{threshold:.2f}`\n\n"
        f"â€¢ Prob â‰¥ `{threshold:.2f}`: **Fraudulent**\n\n"
        f"â€¢ Prob < `{threshold:.2f}`: **Legitimate**"
    )

    st.divider()
    st.markdown("### ðŸ“– System Info")
    st.markdown(
        """
        - **Pipeline:** Week 8 Model Packaging
        - **Model:** Tuned XGBoost
        - **Scaler:** Robust Time & Amount Scaler
        - **Architecture:** Decoupled FastAPI + Streamlit
        """
    )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">ðŸ›¡ï¸ FraudGuard AI Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-grade credit card fraud detection interface '
    'powered by tuned XGBoost machine learning inference.</div>',
    unsafe_allow_html=True,
)

if not is_healthy:
    st.warning(
        f"âš ï¸ **FastAPI Service Unavailable at `{api_url}`**\n\n"
        "To start the FastAPI backend, open a separate terminal and run:\n\n"
        "```powershell\n"
        "cd C:\\IntershipProject\\EDP-AI-ML-Week9-10\\EDP-Internship-Week-8\n"
        "uvicorn app:app --host 127.0.0.1 --port 8000\n"
        "```\n\n"
        "Once running, refresh this page or check the sidebar indicator."
    )


# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_single, tab_batch, tab_health, tab_docs = st.tabs([
    "ðŸŽ¯ Single Transaction Scoring",
    "ðŸ“ Batch CSV Processing",
    "ðŸ©º API Diagnostics & Health",
    "ðŸ“š Documentation & Specifications",
])


# =============================================================================
# TAB 1: Single Transaction Prediction
# =============================================================================
with tab_single:
    st.subheader("Real-Time Transaction Evaluation")
    st.write(
        "Enter transaction features or load verified historical samples to assess fraud probability."
    )

    # Preset Controls
    col_pre1, col_pre2, col_pre3 = st.columns([1, 1, 2])
    with col_pre1:
        if st.button("ðŸŸ¢ Load Legitimate Sample", use_container_width=True):
            if is_healthy:
                sample_data, _ = client.get_sample(fraud=False)
                if sample_data:
                    st.session_state.single_inputs = sample_data
                else:
                    st.session_state.single_inputs = get_sample_preset(fraud=False)
            else:
                st.session_state.single_inputs = get_sample_preset(fraud=False)
            st.rerun()

    with col_pre2:
        if st.button("ðŸ”´ Load Fraud Sample", use_container_width=True):
            if is_healthy:
                sample_data, _ = client.get_sample(fraud=True)
                if sample_data:
                    st.session_state.single_inputs = sample_data
                else:
                    st.session_state.single_inputs = get_sample_preset(fraud=True)
            else:
                st.session_state.single_inputs = get_sample_preset(fraud=True)
            st.rerun()

    with col_pre3:
        if st.button("ðŸ”„ Reset Inputs to Zero", use_container_width=True):
            st.session_state.single_inputs = {col: 0.0 for col in FEATURE_ORDER}
            st.rerun()

    st.markdown("---")

    # Transaction Input Form
    with st.form("single_transaction_form"):
        st.markdown("#### Primary Features")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            amount_val = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=1000000.0,
                value=float(st.session_state.single_inputs.get("Amount", 149.97)),
                step=1.0,
                format="%.2f",
                help="Actual transaction amount in dollars. Will be automatically scaled by the backend scaler.",
            )
        with pcol2:
            time_val = st.number_input(
                "Elapsed Time (Seconds)",
                min_value=0.0,
                max_value=200000.0,
                value=float(st.session_state.single_inputs.get("Time", 31604.21)),
                step=1.0,
                format="%.2f",
                help="Elapsed seconds since dataset start time benchmark.",
            )

        st.markdown("#### PCA Anonymized Features (V1 â€“ V28)")
        with st.expander("Show / Edit Principal Components (V1 to V28)", expanded=True):
            st.caption(
                "Features V1 through V28 represent confidential bank account PCA transforms. "
                "These are preserved unscaled by the pipeline."
            )
            v_inputs = {}
            # Render 4 columns of 7 features each
            for row in range(7):
                cols = st.columns(4)
                for col_idx in range(4):
                    feature_num = row * 4 + col_idx + 1
                    feature_name = f"V{feature_num}"
                    default_v = float(st.session_state.single_inputs.get(feature_name, 0.0))
                    with cols[col_idx]:
                        v_inputs[feature_name] = st.number_input(
                            feature_name,
                            value=default_v,
                            step=0.1,
                            format="%.5f",
                            key=f"input_{feature_name}",
                        )

        predict_button = st.form_submit_button(
            "âš¡ Score Transaction",
            use_container_width=True,
            type="primary",
        )

    # Submission & Prediction
    if predict_button:
        # Build payload dictionary
        payload = {"Time": time_val, "Amount": amount_val}
        payload.update(v_inputs)

        # Validate inputs
        is_valid, cleaned_payload, validation_errors = validate_single_transaction(payload)

        if not is_valid:
            st.error("Validation Failed:")
            for err in validation_errors:
                st.write(f"- {err}")
        else:
            if not is_healthy:
                st.error(
                    "âŒ Cannot score transaction: The backend FastAPI service is currently offline. "
                    "Please launch `uvicorn app:app --port 8000` in `EDP-Internship-Week-8`."
                )
            else:
                with st.spinner("Executing model inference..."):
                    result, err = client.predict_single(cleaned_payload, threshold=threshold)

                if err:
                    st.error(f"Prediction Error: {err}")
                elif result:
                    prediction = result.get("prediction", 0)
                    label = result.get("label", "Unknown")
                    prob = result.get("fraud_probability", 0.0)
                    risk_level = result.get("risk_level", "LOW")
                    thresh_used = result.get("threshold_used", threshold)

                    # Display Result Card
                    if prediction == 1:
                        card_class = "card-risk-high"
                        badge_color = "#DC2626"
                        icon = "ðŸš¨"
                    elif risk_level == "MEDIUM":
                        card_class = "card-risk-medium"
                        badge_color = "#D97706"
                        icon = "âš ï¸"
                    else:
                        card_class = "card-risk-low"
                        badge_color = "#16A34A"
                        icon = "âœ…"

                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0; color: {badge_color};">{icon} Classification: {label.upper()}</h3>
                                <span style="background-color: {badge_color}; color: white; padding: 6px 14px; border-radius: 9999px; font-weight: bold;">
                                    RISK: {risk_level}
                                </span>
                            </div>
                            <hr style="margin: 12px 0; border: none; border-top: 1px solid rgba(0,0,0,0.1);">
                            <div style="display: flex; gap: 40px; margin-top: 10px;">
                                <div>
                                    <div style="font-size: 0.85rem; color: #64748B;">Fraud Probability</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: {badge_color};">{prob:.2%}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.85rem; color: #64748B;">Decision Threshold</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: #334155;">{thresh_used:.2f}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.85rem; color: #64748B;">Transaction Value</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: #334155;">${amount_val:,.2f}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.progress(min(max(prob, 0.0), 1.0))

                    with st.expander("Inspect Raw API JSON Response"):
                        st.json(result)


# =============================================================================
# TAB 2: Batch CSV Prediction
# =============================================================================
with tab_batch:
    st.subheader("Bulk Batch CSV Transaction Scoring")
    st.write(
        "Upload a batch CSV file with transaction records to score multiple transactions concurrently."
    )

    # Sample template download
    sample_df = create_sample_csv_dataframe()
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    bcol1, bcol2 = st.columns([2, 1])
    with bcol1:
        uploaded_file = st.file_uploader(
            "Upload Transactions CSV File",
            type=["csv"],
            help="CSV must contain columns: Time, V1 to V28, and Amount.",
        )
    with bcol2:
        st.markdown("**Need test data?**")
        st.download_button(
            label="ðŸ“¥ Download Sample Batch CSV (5 Records)",
            data=csv_bytes,
            file_name="sample_transactions_batch.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Contains verified legitimate and fraudulent records for instant testing.")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded File:** `{uploaded_file.name}` ({len(input_df)} rows, {len(input_df.columns)} columns)")

            # Preview
            with st.expander("Preview Uploaded Data (First 5 Rows)", expanded=False):
                st.dataframe(input_df.head(), use_container_width=True)

            # Pre-flight Validation
            is_valid, cleaned_batch_df, validation_errors = validate_batch_dataframe(input_df)

            if not is_valid:
                st.error("âŒ Batch Schema Validation Failed:")
                for err in validation_errors:
                    st.write(f"- {err}")
            else:
                st.success("âœ… Schema check passed: All 30 canonical features present and numeric.")

                start_batch = st.button(
                    f"ðŸš€ Evaluate {len(cleaned_batch_df)} Transactions (Threshold = {threshold:.2f})",
                    type="primary",
                    use_container_width=True,
                )

                if start_batch:
                    if not is_healthy:
                        st.error(
                            "Cannot execute batch: FastAPI backend service is offline. "
                            "Start the backend service to continue."
                        )
                    else:
                        records = cleaned_batch_df.to_dict(orient="records")

                        progress_bar = st.progress(0, text="Sending batch to FastAPI service...")
                        start_time = time.time()

                        batch_result, err = client.predict_batch(
                            records,
                            threshold=threshold,
                            chunk_size=500,
                        )

                        elapsed = round(time.time() - start_time, 2)
                        progress_bar.progress(100, text=f"Completed batch in {elapsed}s")

                        if err:
                            st.error(f"Batch Scoring Error: {err}")
                        elif batch_result:
                            st.session_state.batch_results = {
                                "response": batch_result,
                                "original_df": cleaned_batch_df,
                            }

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    # Display Batch Results
    if st.session_state.batch_results is not None:
        batch_response = st.session_state.batch_results["response"]
        original_df = st.session_state.batch_results["original_df"]

        total = batch_response.get("total_processed", 0)
        frauds = batch_response.get("fraud_count", 0)
        legits = batch_response.get("legitimate_count", 0)
        fraud_rate = batch_response.get("fraud_rate", 0.0)
        preds = batch_response.get("predictions", [])

        st.markdown("---")
        st.markdown("### ðŸ“Š Batch Evaluation Summary")

        # Metric Tiles
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Total Transactions", f"{total:,}")
        mcol2.metric("Flagged Fraudulent", f"{frauds:,}", delta=f"{fraud_rate:.2%}", delta_color="inverse")
        mcol3.metric("Approved Legitimate", f"{legits:,}")
        mcol4.metric("Fraud Rate", f"{fraud_rate:.2%}")

        # Construct Enriched Results DataFrame
        results_df = original_df.copy()
        pred_labels = [p.get("label") for p in preds]
        pred_probs = [p.get("fraud_probability") for p in preds]
        pred_risks = [p.get("risk_level") for p in preds]
        pred_vals = [p.get("prediction") for p in preds]

        results_df["Prediction"] = pred_vals
        results_df["Status"] = pred_labels
        results_df["Fraud_Probability"] = pred_probs
        results_df["Risk_Level"] = pred_risks
        results_df["Threshold_Used"] = threshold

        # Analytics Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### Risk Level Breakdown")
            risk_counts = results_df["Risk_Level"].value_counts()
            st.bar_chart(risk_counts, color="#3B82F6")

        with chart_col2:
            st.markdown("##### Fraud Probability Distribution")
            # Group into probability bins
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
            prob_binned = pd.cut(results_df["Fraud_Probability"], bins=bins, labels=labels, include_lowest=True)
            bin_counts = prob_binned.value_counts().sort_index()
            st.bar_chart(bin_counts, color="#EF4444")

        # Table Filter
        st.markdown("##### Detailed Transaction Results")
        filter_risk = st.multiselect(
            "Filter by Risk Level",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
        )

        filtered_table = results_df[results_df["Risk_Level"].isin(filter_risk)]
        st.dataframe(
            filtered_table[
                ["Status", "Risk_Level", "Fraud_Probability", "Amount", "Time"]
                + [c for c in filtered_table.columns if c.startswith("V")]
            ],
            use_container_width=True,
            height=320,
        )

        # Enriched CSV Download
        out_buf = io.StringIO()
        results_df.to_csv(out_buf, index=False)
        st.download_button(
            label="ðŸ’¾ Download Enriched Predictions CSV",
            data=out_buf.getvalue().encode("utf-8"),
            file_name=f"fraud_predictions_results_thresh_{threshold:.2f}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )


# =============================================================================
# TAB 3: API Diagnostics & Health
# =============================================================================
with tab_health:
    st.subheader("FastAPI Service Diagnostics & Health Check")
    st.write("Inspect backend microservice readiness, model artifact status, and response latency.")

    hcol1, hcol2 = st.columns([1, 2])
    with hcol1:
        if st.button("ðŸ”„ Re-test Backend Connection", use_container_width=True):
            st.rerun()

    with hcol2:
        st.write(f"Testing URL: `{api_url}/health`")

    is_healthy, health_info, health_err = client.check_health()

    if is_healthy:
        st.success("âœ… **FastAPI Inference Microservice is Healthy and Responding.**")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Latency", f"{health_info.get('latency_ms', 0)} ms")
        col_m2.metric("Model Architecture", health_info.get("model_type", "Unknown"))
        col_m3.metric("Expected Feature Count", health_info.get("expected_features_count", 30))

        st.markdown("#### Detailed Metadata Response")
        st.json(health_info)
    else:
        st.error(f"âŒ **Connection Failed:** {health_err}")
        st.markdown(
            """
            #### Troubleshooting Checklist:
            1. Ensure the FastAPI application is running:
               ```powershell
               cd C:\\IntershipProject\\EDP-AI-ML-Week9-10\\EDP-Internship-Week-8
               uvicorn app:app --host 127.0.0.1 --port 8000
               ```
            2. Verify the host and port in the sidebar match `http://127.0.0.1:8000`.
            3. Check that model artifacts `models/tuned_xgboost.pkl` and `models/scaler.pkl` exist in `EDP-Internship-Week-8`.
            """
        )


# =============================================================================
# TAB 4: Documentation & Architecture
# =============================================================================
with tab_docs:
    st.subheader("System Architecture & Pipeline Reference")

    st.markdown(
        """
        ### ðŸ—ï¸ End-to-End Pipeline Architecture
        The Credit Card Fraud Detection platform follows a decoupled, three-tier architecture:

        1. **Presentation Layer (Streamlit)**:
           - Provides intuitive risk analytics for compliance and fraud investigators.
           - Supports single transaction assessment and batch file processing.
           - Implements input sanitization and defensive network error recovery.
        2. **Service Layer (FastAPI)**:
           - Exposes REST endpoints (`/predict`, `/predict/batch`, `/sample`, `/health`).
           - Validates incoming JSON payloads against strict Pydantic schemas.
        3. **Inference Layer (XGBoost + Preprocessing Pipeline)**:
           - Selectively scales `Time` and `Amount` using the pre-fitted scaler.
           - Keeps `V1` through `V28` unscaled to preserve mathematical PCA coordinates.
           - Executes fast binary classification with calibrated probability scores.

        ---

        ### ðŸ“ Schema & Canonical Feature Ordering
        The inference pipeline strictly enforces the canonical 30-feature ordering:
        ```text
        1. Time
        2-29. V1 to V28 (PCA Principal Components)
        30. Amount
        ```

        ---

        ### ðŸŽ¯ Risk Tier Stratification
        - **LOW RISK (< 40% Probability)**: Transaction appears standard with no significant anomalies.
        - **MEDIUM RISK (40% - 75% Probability)**: Borderline transaction, recommended for secondary two-factor authentication or manual review.
        - **HIGH RISK (â‰¥ 75% Probability)**: High probability of fraudulent behavior; immediate block recommended.
        """
    )

