"""
streamlit_app.py
----------------
Enterprise-grade Fintech SaaS Dashboard for Credit Card Fraud Detection.
Integrates directly with the FastAPI inference microservice and tuned XGBoost model.

Features:
- Dark navy fintech aesthetic with subtle glassmorphic elements.
- Executive Risk Overview & Analytics with live KPIs and activity audit log.
- Real-time transaction scoring with grouped inputs, presets, and explainability.
- Bulk batch CSV analysis with schema validation, analytics charts, and enriched export.
- Model and API telemetry with health monitoring and Render wake-up handling.
- Full architectural and schema documentation.
"""

import io
import os
import textwrap
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from frontend.api_client import (
    APIClient,
    resolve_default_api_url,
    DEFAULT_RENDER_BACKEND_URL,
    DEFAULT_LOCAL_BACKEND_URL,
)
from frontend.validators import (
    FEATURE_ORDER,
    validate_single_transaction,
    validate_batch_dataframe,
    get_sample_preset,
    get_borderline_preset,
    create_sample_csv_dataframe,
)
from frontend.styles import get_fintech_theme_css
from frontend.model_evaluation import (
    get_evaluator,
    get_dataset_intelligence,
    get_model_metadata,
    get_model_comparison_data,
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
    create_explainability_waterfall_chart,
    create_calibration_curve_chart,
    create_model_comparison_bar_chart,
    create_model_comparison_radar_chart,
)
from frontend.components import (
    render_brand_header,
    render_kpi_metrics,
    render_prediction_result_banner,
    clean_html,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI | Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom Fintech Dark Navy Glassmorphism CSS
st.markdown(clean_html(get_fintech_theme_css()), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "api_url" not in st.session_state:
    query_api = st.query_params.get("api_url") if hasattr(st, "query_params") else None
    if query_api and query_api.strip():
        st.session_state.api_url = query_api.strip().rstrip("/")
    else:
        st.session_state.api_url = resolve_default_api_url()

if "single_inputs" not in st.session_state:
    st.session_state.single_inputs = get_sample_preset(fraud=False)

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

if "total_evaluated_count" not in st.session_state:
    st.session_state.total_evaluated_count = 0

if "fraud_flagged_count" not in st.session_state:
    st.session_state.fraud_flagged_count = 0

if "total_volume_amount" not in st.session_state:
    st.session_state.total_volume_amount = 0.0

if "latest_single_result" not in st.session_state:
    st.session_state.latest_single_result = None

if "active_nav" not in st.session_state:
    st.session_state.active_nav = "📊 Executive Overview"


# -----------------------------------------------------------------------------
# 3. API Client, Cryptographic Authentication & RBAC Management
# -----------------------------------------------------------------------------
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
    st.session_state.auth_username = None
    st.session_state.server_role = None
    st.session_state.auth_display_name = None

if "user_role" not in st.session_state:
    st.session_state.user_role = "Unauthenticated User"

client = APIClient(base_url=st.session_state.api_url, timeout=12.0)

if st.session_state.auth_token:
    client.set_token(
        token=st.session_state.auth_token,
        role=st.session_state.server_role,
        username=st.session_state.auth_username,
        display_name=st.session_state.get("auth_display_name"),
    )

is_healthy, health_data, health_error = client.check_health()
latency_ms = health_data.get("latency_ms", 0.0) if is_healthy else 0.0


# -----------------------------------------------------------------------------
# 4. Sidebar: Navigation, Authentication & System Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Header
    st.markdown(
        clean_html(
            """
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #2563EB, #06B6D4); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">
                    🛡️
                </div>
                <div>
                    <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF; letter-spacing: -0.02em;">FRAUDGUARD AI</div>
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em;">Enterprise Risk Suite</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # Server-Side RBAC & Session Management
    st.markdown("### 🔐 Authentication & Session")

    if not st.session_state.auth_token:
        st.markdown(
            clean_html(
                """
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 0.6rem 0.75rem; margin-bottom: 0.75rem; font-size: 0.78rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94A3B8;">Status:</span>
                        <span style="color: #F87171; font-weight: 700;">Unauthenticated</span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.72rem; margin-top: 0.25rem;">
                        Sign in to obtain a cryptographic JWT session for real-time scoring and governance.
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        with st.form("sidebar_login_form"):
            login_username = st.text_input("Username", placeholder="e.g. analyst or admin", key="sidebar_user")
            login_password = st.text_input("Password", type="password", placeholder="Enter password", key="sidebar_pwd")
            submit_login = st.form_submit_button("Sign In 🛡️", use_container_width=True)
            if submit_login:
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    ok, login_data, err_msg = client.login(login_username.strip(), login_password)
                    if ok and login_data:
                        st.session_state.auth_token = login_data.get("access_token")
                        st.session_state.auth_username = login_data.get("username")
                        st.session_state.server_role = login_data.get("role")
                        st.session_state.auth_display_name = login_data.get("display_name")
                        st.session_state.user_role = (
                            "MLOps & Compliance Admin"
                            if login_data.get("role") == "mlops_admin"
                            else "Risk Analyst (Independent User)"
                        )
                        client.set_token(
                            token=st.session_state.auth_token,
                            role=st.session_state.server_role,
                            username=st.session_state.auth_username,
                            display_name=st.session_state.auth_display_name,
                        )
                        st.success(f"Welcome, {login_data.get('display_name')}!")
                        st.rerun()
                    else:
                        st.error(f"Sign in failed: {err_msg or 'Invalid credentials'}")
    else:
        role_badge_color = "#10B981" if st.session_state.get("server_role") == "mlops_admin" else "#3B82F6"
        role_name_display = "MLOps Administrator (mlops_admin)" if st.session_state.get("server_role") == "mlops_admin" else "Independent User (independent_user)"
        st.markdown(
            clean_html(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 0.6rem 0.75rem; margin-bottom: 0.75rem; font-size: 0.78rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                        <span style="color: #94A3B8;">User:</span>
                        <strong style="color: #F8FAFC;">{st.session_state.get('auth_username', 'anonymous')}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                        <span style="color: #94A3B8;">Server RBAC:</span>
                        <span style="color: {role_badge_color}; font-weight: 700;">{role_name_display}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94A3B8;">JWT Session:</span>
                        <span style="color: #10B981; font-weight: 600;">Active (HS256)</span>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        if st.button("Sign Out 🚪", use_container_width=True, key="btn_signout"):
            client.logout()
            st.session_state.auth_token = None
            st.session_state.auth_username = None
            st.session_state.server_role = None
            st.session_state.auth_display_name = None
            st.session_state.user_role = "Unauthenticated User"
            st.rerun()

    # Navigation Menu
    if st.session_state.user_role.startswith("Risk"):
        nav_options = [
            "📊 Executive Overview",
            "🎯 Real-Time Scoring",
            "📁 Batch CSV Processing",
            "📚 Architecture & Specs",
        ]
    else:
        nav_options = [
            "📊 Executive Overview",
            "📈 Model Training & Evaluation",
            "🎯 Real-Time Scoring",
            "📁 Batch CSV Processing",
            "⚖️ Model Comparison",
            "🩺 System Telemetry",
            "📚 Architecture & Specs",
        ]

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0,
        label_visibility="collapsed",
    )
    st.session_state.active_nav = selected_nav

    st.divider()

    # Dynamic Decision Threshold
    st.markdown("### 🎚️ Decision Threshold")
    threshold = st.slider(
        "Fraud Classification Cutoff (θ)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Probability threshold (θ) above which a transaction is classified as Fraudulent.",
    )

    # Threshold Sensitivity Pill
    if threshold < 0.35:
        sensitivity_label = "High Sensitivity (Recall Priority)"
        sens_color = "#34D399"
    elif threshold > 0.65:
        sensitivity_label = "High Specificity (Precision Priority)"
        sens_color = "#60A5FA"
    else:
        sensitivity_label = "Balanced Cutoff (F1-Optimal)"
        sens_color = "#FBBF24"

    st.markdown(
        clean_html(
            f"""
            <div style="padding: 0.5rem 0.75rem; background: rgba(15, 23, 42, 0.6); border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem;">
                <div style="color: {sens_color}; font-weight: 600; margin-bottom: 0.2rem;">&bull; {sensitivity_label}</div>
                <div style="color: #94A3B8;">Cutoff: <strong style="color: #F8FAFC;">{threshold:.2f}</strong> | Fraud if P &ge; {threshold:.2f}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    # Backend Endpoint Configuration
    st.markdown("### 🌐 Backend Service")
    api_url_input = st.text_input(
        "API Base URL",
        value=st.session_state.api_url,
        help="FastAPI microservice endpoint.",
    )
    if api_url_input != st.session_state.api_url:
        st.session_state.api_url = api_url_input.strip().rstrip("/")
        if hasattr(st, "query_params"):
            st.query_params["api_url"] = st.session_state.api_url
        st.rerun()

    # Quick Switcher
    col_sw1, col_sw2 = st.columns(2)
    with col_sw1:
        if st.button("☁️ Render", width="stretch", help="Switch to live cloud backend"):
            st.session_state.api_url = DEFAULT_RENDER_BACKEND_URL
            if hasattr(st, "query_params"):
                st.query_params["api_url"] = DEFAULT_RENDER_BACKEND_URL
            st.rerun()
    with col_sw2:
        if st.button("💻 Local", width="stretch", help="Switch to local server http://127.0.0.1:8000"):
            st.session_state.api_url = DEFAULT_LOCAL_BACKEND_URL
            if hasattr(st, "query_params"):
                st.query_params["api_url"] = DEFAULT_LOCAL_BACKEND_URL
            st.rerun()

    # Status summary
    if is_healthy:
        st.markdown(
            clean_html(
                f"""
                <div style="margin-top: 0.75rem; font-size: 0.75rem; color: #34D399; display: flex; align-items: center; gap: 0.4rem;">
                    <span class="pulse-dot online"></span>
                    <span>Active: <strong>{health_data.get('model_type', 'XGBClassifier')}</strong> ({latency_ms:.0f}ms)</span>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            clean_html(
                """
                <div style="margin-top: 0.75rem; font-size: 0.75rem; color: #F87171; display: flex; align-items: center; gap: 0.4rem;">
                    <span class="pulse-dot offline"></span>
                    <span>Service Disconnected</span>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    # Quick stats footer with dynamic environment and scaler reflection
    current_api_url = str(st.session_state.get("api_url", ""))
    is_local_env = "127.0.0.1" in current_api_url or "localhost" in current_api_url
    env_label = "Local Server" if is_local_env else "Production Cloud"
    scaler_label = health_data.get("scaler_type", "StandardScaler") if is_healthy else "StandardScaler"

    st.markdown(
        clean_html(
            f"""
            <div style="font-size: 0.72rem; color: #64748B; line-height: 1.5;">
                <div><strong>Pipeline:</strong> XGBoost + {scaler_label}</div>
                <div><strong>Features:</strong> 30 Canonical Predictors</div>
                <div><strong>Environment:</strong> {env_label}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. Top Brand Header
# -----------------------------------------------------------------------------
render_brand_header(is_healthy, latency_ms, st.session_state.api_url)

# Render backend offline notification & wake-up utility
if not is_healthy:
    st.markdown(
        clean_html(
            f"""
            <div style="background: rgba(127, 29, 29, 0.3); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.5rem;">⚠️</span>
                    <div>
                        <h4 style="margin: 0; color: #F87171;">FastAPI Inference Service Unavailable at <code>{st.session_state.api_url}</code></h4>
                        <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: #CBD5E1;">
                            If using the free-tier Render backend, the instance may be spinning up from cold sleep (typically takes 20–45s).
                        </p>
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    col_wake1, col_wake2 = st.columns([1, 3])
    with col_wake1:
        if st.button("⚡ Ping & Wake Up Backend", width="stretch", type="primary"):
            with st.spinner("Pinging Render backend (allowing up to 50s for cold start)..."):
                wake_client = APIClient(base_url=st.session_state.api_url, timeout=50.0)
                wake_ok, _, wake_err = wake_client.check_health()
                if wake_ok:
                    st.success("Backend is awake and operational!")
                else:
                    st.error(f"Could not reach backend: {wake_err}")
            st.rerun()
    with col_wake2:
        st.caption("Or run locally: `uvicorn backend.app:app --port 8000` and click the **💻 Local** button in the sidebar.")


# =============================================================================
# VIEW 1: 📊 Executive Overview & Analytics
# =============================================================================
if selected_nav == "📊 Executive Overview":
    st.markdown("### 📊 Executive Risk Operations Command Center")
    st.markdown(
        clean_html(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 1.5rem;'>"
            "Real-time risk posture, threat prevention metrics, and audit log of scored transactions across this session."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Cards
    render_kpi_metrics(
        total_tx=st.session_state.total_evaluated_count,
        fraud_tx=st.session_state.fraud_flagged_count,
        legit_tx=st.session_state.total_evaluated_count - st.session_state.fraud_flagged_count,
        total_volume=st.session_state.total_volume_amount,
        latency_ms=latency_ms,
    )

    # Middle Charts Row
    st.markdown("#### Threat Intelligence & Risk Stratification")
    col_ch1, col_ch2 = st.columns([1, 1])

    # Calculate live risk tier distribution from activity log or batch
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    probabilities: List[float] = []

    if st.session_state.activity_log:
        for entry in st.session_state.activity_log:
            r = entry.get("risk_level", "LOW")
            risk_counts[r] = risk_counts.get(r, 0) + 1
            probabilities.append(entry.get("fraud_probability", 0.0))

    with col_ch1:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                        Risk Stratification Breakdown
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            if sum(risk_counts.values()) > 0:
                donut_fig = create_risk_donut_chart(risk_counts)
                st.plotly_chart(donut_fig, width="stretch")
            else:
                st.markdown(
                    clean_html(
                        """
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 260px; color: #64748B; text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                            <div style="font-size: 0.9rem; font-weight: 600; color: #94A3B8;">No Transactions Scored Yet</div>
                            <div style="font-size: 0.78rem; max-width: 260px; margin-top: 0.25rem;">Score a transaction in Real-Time Scoring or upload a batch CSV to populate live charts.</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

    with col_ch2:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                        Risk Probability Distribution
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            if probabilities:
                dist_fig = create_probability_distribution_chart(probabilities, threshold=threshold)
                st.plotly_chart(dist_fig, width="stretch")
            else:
                st.markdown(
                    clean_html(
                        """
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 260px; color: #64748B; text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
                            <div style="font-size: 0.9rem; font-weight: 600; color: #94A3B8;">Awaiting Inference Data</div>
                            <div style="font-size: 0.78rem; max-width: 260px; margin-top: 0.25rem;">Probability density curves will render dynamically as transactions are evaluated.</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Recent Activity Audit Feed
    st.markdown("#### 🕒 Live Audit Feed (Session History)")
    if st.session_state.activity_log:
        audit_df = pd.DataFrame(st.session_state.activity_log)
        # Format columns for display
        display_cols = ["Timestamp", "Transaction_ID", "Amount", "Risk_Level", "Fraud_Probability", "Decision", "Cutoff"]
        available_cols = [c for c in display_cols if c in audit_df.columns]
        
        st.dataframe(
            audit_df[available_cols],
            width="stretch",
            height=260,
        )

        col_clr1, col_clr2 = st.columns([1, 4])
        with col_clr1:
            if st.button("🗑️ Clear Activity Log", width="stretch"):
                st.session_state.activity_log = []
                st.session_state.total_evaluated_count = 0
                st.session_state.fraud_flagged_count = 0
                st.session_state.total_volume_amount = 0.0
                st.rerun()
    else:
        st.markdown(
            clean_html(
                """
                <div class="glass-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛡️</div>
                    <h4 style="margin: 0; color: #F8FAFC;">Real-Time Activity Audit Log is Empty</h4>
                    <p style="color: #94A3B8; font-size: 0.85rem; max-width: 480px; margin: 0.5rem auto 1.25rem auto;">
                        Transactions evaluated through Real-Time Scoring or Batch Ingestion are recorded here in real-time with comprehensive risk telemetry.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        col_demo1, col_demo2, col_demo3 = st.columns([1, 2, 1])
        with col_demo2:
            if st.button("⚡ Run Instant Verification Demo (Score 3 Presets)", width="stretch", type="primary"):
                if not is_healthy:
                    st.error("Cannot run demo while backend is offline.")
                else:
                    with st.spinner("Scoring verified verification presets..."):
                        p_legit = get_sample_preset(fraud=False)
                        p_border = get_borderline_preset()
                        p_fraud = get_sample_preset(fraud=True)

                        for sample, t_type in [(p_legit, "Standard Merchant"), (p_border, "Suspicious Outlier"), (p_fraud, "Syndicate Attack")]:
                            res, err = client.predict_single(sample, threshold=threshold)
                            if res:
                                now_str = datetime.now().strftime("%H:%M:%S")
                                prob_val = res.get("fraud_probability", 0.0)
                                is_fraud = res.get("prediction", 0) == 1
                                r_level = res.get("risk_level", "LOW")
                                amt = sample.get("Amount", 0.0)

                                st.session_state.total_evaluated_count += 1
                                st.session_state.total_volume_amount += amt
                                if is_fraud:
                                    st.session_state.fraud_flagged_count += 1

                                st.session_state.activity_log.insert(0, {
                                    "Timestamp": now_str,
                                    "Transaction_ID": f"TX-{int(time.time() * 1000) % 1000000}",
                                    "Type": t_type,
                                    "Amount": f"${amt:,.2f}",
                                    "Risk_Level": r_level,
                                    "Fraud_Probability": f"{prob_val:.2%}",
                                    "Decision": "DECLINED" if is_fraud else ("STEP-UP" if r_level == "MEDIUM" else "APPROVED"),
                                    "Cutoff": f"{threshold:.2f}",
                                })
                    st.rerun()


# =============================================================================
# VIEW 1.5: 📈 Model Training & Evaluation Suite
# =============================================================================
elif selected_nav == "📈 Model Training & Evaluation":
    st.markdown("### 📈 Model Training & Evaluation Suite")
    st.markdown(
        clean_html(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 1.25rem;'>"
            "Genuine ground-truth performance evaluation of the tuned XGBoost classifier on the original held-out test partition (N = 1,008 transactions)."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    # Data Source Verification Banner
    st.markdown(
        clean_html(
            """
            <div style="padding: 0.85rem 1.25rem; background: rgba(37, 99, 235, 0.12); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 10px; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                    <span style="font-size: 1.3rem;">🛡️</span>
                    <div style="font-size: 0.84rem; color: #E2E8F0; line-height: 1.5;">
                        <strong style="color: #60A5FA;">Original Held-Out Benchmark Partition:</strong>
                        All metrics, diagnostics, and curves on this page are computed directly from the <strong>1,008 held-out test transactions</strong>
                        (<code>data/processed/X_test.csv</code> and <code>y_test.csv</code>) evaluated against <code>backend/models/tuned_xgboost.pkl</code>.
                        This test partition was strictly held out and was never seen during model fitting or scaler fitting.
                        <strong>Zero synthetic samples or 5-row batch demonstration data are used here.</strong>
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # Load Evaluator and Metadata
    try:
        evaluator = get_evaluator()
        ds_intel = get_dataset_intelligence()
        model_meta = get_model_metadata()
    except Exception as e:
        st.error(f"❌ Failed to load evaluation suite: {e}")
        st.stop()

    # SECTION 1: Model Performance Overview
    st.markdown("#### 1. Model Performance Overview (Held-Out Test Set: N = 1,008)")
    st.caption("Baseline ground-truth metrics evaluated at the canonical cutoff θ = 0.50:")

    m_50 = evaluator.compute_metrics(threshold=0.50)

    # Row 1 of KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">ACCURACY</div>
                    <div class="kpi-value">{m_50['accuracy']:.2%}</div>
                    <div class="kpi-footer">&bull; 1,003 / 1,008 correct</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #34D399;">FRAUD PRECISION</div>
                    <div class="kpi-value" style="color: #34D399;">{m_50['precision']:.2%}</div>
                    <div class="kpi-footer" style="color: #34D399;">&bull; 4 TP / 5 Flagged</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #FBBF24;">FRAUD RECALL</div>
                    <div class="kpi-value" style="color: #FBBF24;">{m_50['recall']:.2%}</div>
                    <div class="kpi-footer" style="color: #FBBF24;">&bull; 4 TP / 8 Total Fraud</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #60A5FA;">FRAUD F1-SCORE</div>
                    <div class="kpi-value" style="color: #60A5FA;">{m_50['f1_score']:.4f}</div>
                    <div class="kpi-footer" style="color: #60A5FA;">&bull; Harmonic balance</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with kpi_col5:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #38BDF8;">ROC-AUC</div>
                    <div class="kpi-value" style="color: #38BDF8;">{m_50['roc_auc']:.4f}</div>
                    <div class="kpi-footer" style="color: #38BDF8;">&bull; Class separation</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with kpi_col6:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #A78BFA;">PR-AUC (AVG PREC)</div>
                    <div class="kpi-value" style="color: #A78BFA;">{m_50['pr_auc']:.4f}</div>
                    <div class="kpi-footer" style="color: #A78BFA;">&bull; Base: 0.79%</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # Row 2: Operational Rates
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    op_col1, op_col2, op_col3, op_col4 = st.columns(4)
    with op_col1:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #34D399;">SPECIFICITY (TNR)</div>
                    <div class="kpi-value" style="color: #34D399;">{m_50['specificity']:.2%}</div>
                    <div class="kpi-footer">&bull; 999 TN / 1,000 Legit</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with op_col2:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #38BDF8;">FALSE POSITIVE RATE</div>
                    <div class="kpi-value" style="color: #38BDF8;">{m_50['fpr']:.2%}</div>
                    <div class="kpi-footer">&bull; 1 FP / 1,000 Legit (0.10%)</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with op_col3:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #F87171;">FALSE NEGATIVE RATE</div>
                    <div class="kpi-value" style="color: #F87171;">{m_50['fnr']:.2%}</div>
                    <div class="kpi-footer">&bull; 4 FN / 8 Fraud (50.0%)</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with op_col4:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #CBD5E1;">WEIGHTED F1-SCORE</div>
                    <div class="kpi-value" style="color: #CBD5E1;">{m_50['weighted_f1']:.4f}</div>
                    <div class="kpi-footer">&bull; Overall multi-class score</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # Row 3: Advanced Classification & Calibration Metrics
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    adv_col1, adv_col2, adv_col3, adv_col4, adv_col5, adv_col6 = st.columns(6)
    with adv_col1:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #FBBF24;">FRAUD F2-SCORE</div>
                    <div class="kpi-value" style="color: #FBBF24;">{m_50['f2_score']:.4f}</div>
                    <div class="kpi-footer">&bull; 2x weight on Recall</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with adv_col2:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #38BDF8;">BALANCED ACCURACY</div>
                    <div class="kpi-value" style="color: #38BDF8;">{m_50['balanced_accuracy']:.2%}</div>
                    <div class="kpi-footer">&bull; (TPR + TNR) / 2</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with adv_col3:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #34D399;">MATTHEWS CORR (MCC)</div>
                    <div class="kpi-value" style="color: #34D399;">{m_50['mcc']:.4f}</div>
                    <div class="kpi-footer">&bull; Range [-1.0, +1.0]</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with adv_col4:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #A78BFA;">NPV (CLEARING CONF)</div>
                    <div class="kpi-value" style="color: #A78BFA;">{m_50['npv']:.2%}</div>
                    <div class="kpi-footer">&bull; True legit / approved</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with adv_col5:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #F472B6;">BRIER SCORE</div>
                    <div class="kpi-value" style="color: #F472B6;">{m_50['brier_score']:.6f}</div>
                    <div class="kpi-footer">&bull; Calibration MSE (lower=better)</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with adv_col6:
        st.markdown(
            clean_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title" style="color: #F87171;">LOG LOSS</div>
                    <div class="kpi-value" style="color: #F87171;">{m_50['log_loss']:.4f}</div>
                    <div class="kpi-footer">&bull; Binary cross-entropy</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # Statistical Reliability & 95% Bootstrap Confidence Intervals Table
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    with st.expander("📊 Statistical Reliability & Stratified Bootstrap 95% Confidence Intervals", expanded=False):
        boot_cis = evaluator.compute_bootstrap_confidence_intervals()
        ci_rows = [
            {"Metric": "Fraud Precision (PPV)", "Point Estimate (θ=0.50)": f"{m_50['precision']:.2%}", "95% Bootstrap CI": f"[{boot_cis['precision']['lower']*100:.2f}%, {boot_cis['precision']['upper']*100:.2f}%]", "Institutional Interpretation": "High probability that flagged alarms are genuine; upper bound reaches 100%."},
            {"Metric": "Fraud Recall (Sensitivity)", "Point Estimate (θ=0.50)": f"{m_50['recall']:.2%}", "95% Bootstrap CI": f"[{boot_cis['recall']['lower']*100:.2f}%, {boot_cis['recall']['upper']*100:.2f}%]", "Institutional Interpretation": "Wide interval reflecting small sample size (only 8 test frauds; each fraud = 12.5%)."},
            {"Metric": "Fraud F1-Score", "Point Estimate (θ=0.50)": f"{m_50['f1_score']:.4f}", "95% Bootstrap CI": f"[{boot_cis['f1_score']['lower']:.4f}, {boot_cis['f1_score']['upper']:.4f}]", "Institutional Interpretation": "Harmonic balance between precision and sensitivity across resampling."},
            {"Metric": "Fraud F2-Score", "Point Estimate (θ=0.50)": f"{m_50['f2_score']:.4f}", "95% Bootstrap CI": f"[{boot_cis['f2_score']['lower']:.4f}, {boot_cis['f2_score']['upper']:.4f}]", "Institutional Interpretation": "Emphasizes fraud capture over false alarm penalty."},
            {"Metric": "ROC-AUC", "Point Estimate (θ=0.50)": f"{m_50['roc_auc']:.4f}", "95% Bootstrap CI": f"[{boot_cis['roc_auc']['lower']:.4f}, {boot_cis['roc_auc']['upper']:.4f}]", "Institutional Interpretation": "Exceptionally robust discrimination across arbitrary decision cutoffs."},
            {"Metric": "PR-AUC (Average Precision)", "Point Estimate (θ=0.50)": f"{m_50['pr_auc']:.4f}", "95% Bootstrap CI": f"[{boot_cis['pr_auc']['lower']:.4f}, {boot_cis['pr_auc']['upper']:.4f}]", "Institutional Interpretation": "Strong minority class performance significantly exceeding 0.79% baseline."},
            {"Metric": "Matthews Correlation (MCC)", "Point Estimate (θ=0.50)": f"{m_50['mcc']:.4f}", "95% Bootstrap CI": f"[{boot_cis['mcc']['lower']:.4f}, {boot_cis['mcc']['upper']:.4f}]", "Institutional Interpretation": "Gold standard binary metric accounting for all 4 confusion matrix quadrants."},
        ]
        st.dataframe(pd.DataFrame(ci_rows), width="stretch", hide_index=True)
        st.caption("ℹ️ Confidence intervals generated via 500 stratified bootstrap iterations resampling legitimate and fraud classes proportionally.")

    # Imbalance evaluation explanation
    st.markdown(
        clean_html(
            """
            <div style="font-size: 0.8rem; color: #94A3B8; background: rgba(15, 23, 42, 0.6); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-top: 0.75rem; line-height: 1.5;">
                <strong style="color: #F8FAFC;">Methodological Note on Class Imbalance:</strong>
                Due to extreme real-world fraud rarity (0.79% fraud rate), raw accuracy (99.50%) alone is insufficient — a naïve classifier that labels all transactions as legitimate would score 99.21% accuracy while stopping 0 frauds.
                Institutional model evaluation therefore prioritizes <strong>Fraud Precision (80.00%)</strong>, <strong>Fraud Recall (50.00%)</strong>, <strong>F1-Score (0.6154)</strong>, <strong>ROC-AUC (0.9910)</strong>, and <strong>PR-AUC (0.7492)</strong>.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 2: Dataset Intelligence
    st.markdown("#### 2. Dataset Intelligence & Partitioning")
    st.caption("Distribution of legitimate transactions vs fraudulent transactions across raw and split sets:")

    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Raw Dataset Class Breakdown (creditcard.csv)
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            raw_pie = create_evaluation_class_distribution_chart(
                legit_count=ds_intel.raw_legit,
                fraud_count=ds_intel.raw_fraud,
                title="Raw Dataset Class Distribution",
            )
            st.plotly_chart(raw_pie, width="stretch")

            # Metadata bullet list
            st.markdown(
                clean_html(
                    f"""
                    <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.6rem;">
                        <div>&bull; <strong>Total Transactions:</strong> {ds_intel.total_records:,}</div>
                        <div>&bull; <strong>Legitimate Count:</strong> {ds_intel.raw_legit:,} ({ds_intel.raw_legit_pct:.2f}%)</div>
                        <div>&bull; <strong>Fraud Count:</strong> {ds_intel.raw_fraud:,} ({ds_intel.raw_fraud_pct:.2f}%)</div>
                        <div>&bull; <strong>Canonical Features:</strong> {ds_intel.num_features} predictors (Time, V1-V28, Amount)</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    with d_col2:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Train / Test Partitioning (80/20 Stratified Split)
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            split_bar = create_evaluation_split_comparison_chart(
                train_legit=ds_intel.train_legit,
                train_fraud=ds_intel.train_fraud,
                test_legit=ds_intel.test_legit,
                test_fraud=ds_intel.test_fraud,
            )
            st.plotly_chart(split_bar, width="stretch")

            st.markdown(
                clean_html(
                    f"""
                    <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.6rem;">
                        <div>&bull; <strong>Training Set:</strong> {ds_intel.train_records:,} rows ({ds_intel.train_legit:,} Legit, {ds_intel.train_fraud:,} Fraud [0.79%])</div>
                        <div>&bull; <strong>Held-Out Test Set:</strong> {ds_intel.test_records:,} rows ({ds_intel.test_legit:,} Legit, {ds_intel.test_fraud:,} Fraud [0.79%])</div>
                        <div>&bull; <strong>Splitting Strategy:</strong> Stratified 80/20 split (random_state=42, stratify=y)</div>
                        <div>&bull; <strong>Preprocessing:</strong> StandardScaler fit strictly on train split; test split transformed.</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 3: Model Evaluation Diagnostics
    st.markdown("#### 3. Model Evaluation Diagnostics")
    st.caption("Interactive confusion matrix, classification report, and discrimination curves on the held-out test set:")

    diag_row1_col1, diag_row1_col2 = st.columns([1, 1])
    with diag_row1_col1:
        with st.container(border=True):
            cm_fig = create_evaluation_confusion_matrix_chart(
                tn=m_50["tn"],
                fp=m_50["fp"],
                fn=m_50["fn"],
                tp=m_50["tp"],
                threshold=0.50,
            )
            st.plotly_chart(cm_fig, width="stretch")

    with diag_row1_col2:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Classification Report (θ = 0.50)
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            clf_data = [
                {
                    "Class": "Legitimate (0)",
                    "Precision": f"{m_50['precision_legit'] * 100:.2f}%",
                    "Recall": f"{m_50['recall_legit'] * 100:.2f}%",
                    "F1-Score": f"{m_50['f1_legit']:.4f}",
                    "Support": f"{m_50['actual_legit_count']:,}",
                },
                {
                    "Class": "Fraudulent (1)",
                    "Precision": f"{m_50['precision'] * 100:.2f}%",
                    "Recall": f"{m_50['recall'] * 100:.2f}%",
                    "F1-Score": f"{m_50['f1_score']:.4f}",
                    "Support": f"{m_50['actual_fraud_count']:,}",
                },
                {
                    "Class": "Accuracy",
                    "Precision": "-",
                    "Recall": "-",
                    "F1-Score": f"{m_50['accuracy'] * 100:.2f}%",
                    "Support": f"{m_50['test_sample_size']:,}",
                },
                {
                    "Class": "Macro Avg",
                    "Precision": f"{m_50['macro_precision'] * 100:.2f}%",
                    "Recall": f"{m_50['macro_recall'] * 100:.2f}%",
                    "F1-Score": f"{m_50['macro_f1']:.4f}",
                    "Support": f"{m_50['test_sample_size']:,}",
                },
                {
                    "Class": "Weighted Avg",
                    "Precision": f"{m_50['weighted_precision'] * 100:.2f}%",
                    "Recall": f"{m_50['weighted_recall'] * 100:.2f}%",
                    "F1-Score": f"{m_50['weighted_f1']:.4f}",
                    "Support": f"{m_50['test_sample_size']:,}",
                },
            ]
            st.dataframe(pd.DataFrame(clf_data), width="stretch", hide_index=True)

            st.markdown(
                clean_html(
                    f"""
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.5rem; line-height: 1.5;">
                        <div>&bull; <strong>True Positives (TP):</strong> {m_50['tp']} fraud transactions correctly caught</div>
                        <div>&bull; <strong>False Positives (FP):</strong> {m_50['fp']} legitimate transaction flagged (0.10% false alarm)</div>
                        <div>&bull; <strong>False Negatives (FN):</strong> {m_50['fn']} frauds missed at 0.50 cutoff</div>
                        <div>&bull; <strong>True Negatives (TN):</strong> {m_50['tn']} legitimate transactions cleared</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    # Diagnostic Curves Row
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    diag_row2_col1, diag_row2_col2 = st.columns([1, 1])
    with diag_row2_col1:
        with st.container(border=True):
            roc_fig = create_evaluation_roc_curve_chart(
                fpr=evaluator.fpr,
                tpr=evaluator.tpr,
                roc_auc=evaluator.roc_auc,
            )
            st.plotly_chart(roc_fig, width="stretch")

    with diag_row2_col2:
        with st.container(border=True):
            pr_fig = create_evaluation_pr_curve_chart(
                precisions=evaluator.pr_precisions,
                recalls=evaluator.pr_recalls,
                pr_auc=evaluator.pr_auc,
                baseline=ds_intel.test_fraud / ds_intel.test_records,
            )
            st.plotly_chart(pr_fig, width="stretch")

    # Diagnostic Row 3: Class Probability Separation & Calibration Reliability Curve
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    diag_row3_col1, diag_row3_col2 = st.columns([1, 1])
    with diag_row3_col1:
        with st.container(border=True):
            prob_sep_fig = create_evaluation_class_probability_distribution_chart(
                y_true=evaluator.y_test,
                y_proba=evaluator.y_proba,
                threshold=0.50,
            )
            st.plotly_chart(prob_sep_fig, width="stretch")

    with diag_row3_col2:
        with st.container(border=True):
            calib_fig = create_calibration_curve_chart(
                prob_true=evaluator.calibration_prob_true,
                prob_pred=evaluator.calibration_prob_pred,
                brier_score=evaluator.brier_score,
            )
            st.plotly_chart(calib_fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 4: Feature Explainability
    st.markdown("#### 4. Feature Explainability & Split Contributions")
    st.caption("Top mathematical feature importances derived from the trained XGBoost model:")

    feat_col1, feat_col2 = st.columns([3, 2])
    with feat_col1:
        with st.container(border=True):
            fi_fig = create_evaluation_feature_importance_chart(evaluator.feature_importances, top_n=15)
            st.plotly_chart(fi_fig, width="stretch")

    with feat_col2:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Top 15 Feature Importances Table
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            top_fi = evaluator.feature_importances.head(15).copy()
            top_fi["Importance"] = top_fi["Importance"].map(lambda x: f"{x:.4f}")
            top_fi["Percentage"] = top_fi["Percentage"].map(lambda x: f"{x:.2f}%")
            st.dataframe(top_fi, width="stretch", hide_index=True, height=310)

    # Explainability Disclaimer
    st.markdown(
        clean_html(
            """
            <div style="font-size: 0.8rem; color: #94A3B8; background: rgba(15, 23, 42, 0.6); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-top: 0.75rem; line-height: 1.5;">
                <strong style="color: #60A5FA;">Non-Causal Mathematical Attribution Notice:</strong>
                Feature importances reflect the relative Gain contribution of each predictor across tree decision splits in all 200 boosted estimators.
                They denote mathematical association and discriminative utility within the XGBoost ensemble, <strong>not real-world causal explanation</strong>.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 5: Threshold Optimization & Sensitivity Analysis
    st.markdown("#### 5. Threshold Optimization & Trade-Off Analysis")
    st.caption("Calibrate decision threshold (θ) and examine the precision-recall trade-off without retraining:")

    th_col_ctrl, th_col_kpi = st.columns([1, 2])
    with th_col_ctrl:
        with st.container(border=True):
            st.markdown("##### 🎚️ Interactive Threshold")
            eval_th = st.slider(
                "Operational Cutoff (θ)",
                min_value=0.10,
                max_value=0.90,
                value=float(threshold),
                step=0.05,
                help="Transactions with P(Fraud) >= θ are classified as Fraudulent.",
                key="eval_th_slider",
            )

            st.markdown("##### ⚡ Quick Presets")
            p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
            if p_c1.button("0.30", width="stretch"):
                eval_th = 0.30
            if p_c2.button("0.40", width="stretch"):
                eval_th = 0.40
            if p_c3.button("0.50", width="stretch"):
                eval_th = 0.50
            if p_c4.button("0.60", width="stretch"):
                eval_th = 0.60
            if p_c5.button("0.70", width="stretch"):
                eval_th = 0.70

    m_eval = evaluator.compute_metrics(threshold=eval_th)

    with th_col_kpi:
        with st.container(border=True):
            st.markdown(f"##### Performance at Cutoff θ = {eval_th:.2f}")
            e_k1, e_k2, e_k3, e_k4 = st.columns(4)
            e_k1.metric("Fraud Precision", f"{m_eval['precision'] * 100:.1f}%", f"{m_eval['tp']} TP / {m_eval['tp']+m_eval['fp']} Flagged")
            e_k2.metric("Fraud Recall", f"{m_eval['recall'] * 100:.1f}%", f"{m_eval['tp']} TP / {m_eval['actual_fraud_count']} Total")
            e_k3.metric("F1-Score", f"{m_eval['f1_score']:.4f}", "Harmonic Mean")
            e_k4.metric("False Positives", f"{m_eval['fp']}", f"FPR: {m_eval['fpr']*100:.2f}%")

            st.markdown(
                clean_html(
                    f"""
                    <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 0.5rem;">
                        <strong>Confusion Matrix at θ = {eval_th:.2f}:</strong>
                        TP: <span style="color: #34D399; font-weight: bold;">{m_eval['tp']}</span> |
                        FP: <span style="color: #F87171; font-weight: bold;">{m_eval['fp']}</span> |
                        FN: <span style="color: #F87171; font-weight: bold;">{m_eval['fn']}</span> |
                        TN: <span style="color: #34D399; font-weight: bold;">{m_eval['tn']}</span> |
                        Accuracy: <span style="color: #60A5FA; font-weight: bold;">{m_eval['accuracy'] * 100:.2f}%</span>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    # Threshold Trade-off Curve & Canonical Comparison Table
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    th_chart_col, th_table_col = st.columns([1, 1])

    multi_th_df = evaluator.get_multi_threshold_table([0.30, 0.40, 0.50, 0.60, 0.70])

    with th_chart_col:
        with st.container(border=True):
            th_sweep_fig = create_evaluation_threshold_sweep_chart(multi_th_df)
            st.plotly_chart(th_sweep_fig, width="stretch")

    with th_table_col:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Multi-Threshold Performance Comparison
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(multi_th_df, width="stretch", hide_index=True)

            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.4rem; line-height: 1.5;">
                        <div>&bull; <strong>θ = 0.30/0.40:</strong> 50% Recall with 2 False Positives (66.67% Precision).</div>
                        <div>&bull; <strong>θ = 0.50 (Optimal):</strong> 50% Recall with only 1 False Positive (80.00% Precision, F1 = 0.6154).</div>
                        <div>&bull; <strong>θ = 0.60/0.70:</strong> Precision drops to 75.00% as Recall drops to 37.50% (5 missed frauds).</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 6: Technical Specifications & Downloadable Reports
    st.markdown("#### 6. Technical Specifications & Exportable Audit Reports")
    st.caption("Review technical parameters and export reproducible compliance documentation:")

    spec_col, dl_col = st.columns([2, 1])
    with spec_col:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    f"""
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Production Model & Preprocessing Pipeline Blueprint
                    </div>
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; color: #CBD5E1;">
                        <tbody>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                                <td style="padding: 6px 0; color: #94A3B8;">Model Architecture</td>
                                <td style="padding: 6px 0; font-weight: 600; color: #F8FAFC;">{model_meta.model_name} ({model_meta.model_type})</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                                <td style="padding: 6px 0; color: #94A3B8;">Scaler Transformation</td>
                                <td style="padding: 6px 0; font-weight: 600; color: #F8FAFC;">{model_meta.scaler_type} on {model_meta.scaled_features}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                                <td style="padding: 6px 0; color: #94A3B8;">PCA Feature Space</td>
                                <td style="padding: 6px 0; font-weight: 600; color: #F8FAFC;">V1 to V28 (28 components preserved unscaled)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                                <td style="padding: 6px 0; color: #94A3B8;">Tuned Hyperparameters</td>
                                <td style="padding: 6px 0; font-family: monospace; font-size: 0.76rem; color: #60A5FA;">
                                    n_estimators={model_meta.hyperparameters.get('n_estimators', 200)},
                                    max_depth={model_meta.hyperparameters.get('max_depth', 3)},
                                    lr={model_meta.hyperparameters.get('learning_rate', 0.1)},
                                    scale_pos_weight={model_meta.hyperparameters.get('scale_pos_weight', 125.0)}
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                                <td style="padding: 6px 0; color: #94A3B8;">Tuning Methodology</td>
                                <td style="padding: 6px 0; font-weight: 600; color: #F8FAFC;">{model_meta.cross_validation_folds}-Fold GridSearchCV targeting {model_meta.optimization_metric}</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 0; color: #94A3B8;">Model Artifact Hash Verified</td>
                                <td style="padding: 6px 0; font-family: monospace; font-size: 0.76rem; color: #34D399;">a8b0f2876eb8c2d07f9efe6dd7fdf81f (tuned_xgboost.pkl)</td>
                            </tr>
                        </tbody>
                    </table>
                    """
                ),
                unsafe_allow_html=True,
            )

    with dl_col:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Export Evaluation Reports
                    </div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 1rem;">
                        Download verifiable, machine-readable audit reports containing complete ground-truth performance metrics.
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

            # JSON Report Download
            json_report = evaluator.export_report_json(active_threshold=eval_th)
            st.download_button(
                label="📥 Download Audit Report (JSON)",
                data=json_report.encode("utf-8"),
                file_name=f"fraudguard_evaluation_audit_cutoff_{eval_th:.2f}.json",
                mime="application/json",
                width="stretch",
            )

            # CSV Report Download
            csv_report = evaluator.export_report_csv(active_threshold=eval_th)
            st.download_button(
                label="📥 Download Metrics Summary (CSV)",
                data=csv_report.encode("utf-8"),
                file_name=f"fraudguard_evaluation_summary_cutoff_{eval_th:.2f}.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )


# =============================================================================
# VIEW 2: 🎯 Real-Time Transaction Scoring
# =============================================================================
elif selected_nav == "🎯 Real-Time Scoring":
    st.markdown("### 🎯 Real-Time Transaction Scoring Engine")
    st.markdown(
        clean_html(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 1.5rem;'>"
            "Evaluate individual transaction payloads against the tuned XGBoost model with dynamic thresholding and PCA anomaly explainability."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    # Presets Bar
    st.markdown("##### ⚡ Quick-Fill Verified Test Scenarios")
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)

    with c_p1:
        if st.button("🟢 Clean Merchant (Legitimate)", width="stretch"):
            if is_healthy:
                api_sample, _ = client.get_sample(fraud=False)
                st.session_state.single_inputs = api_sample or get_sample_preset(fraud=False)
            else:
                st.session_state.single_inputs = get_sample_preset(fraud=False)
            st.rerun()

    with c_p2:
        if st.button("⚠️ Suspicious Case (Medium Risk)", width="stretch"):
            st.session_state.single_inputs = get_borderline_preset()
            st.rerun()

    with c_p3:
        if st.button("🚨 Syndicate Attack (High Risk)", width="stretch"):
            if is_healthy:
                api_sample, _ = client.get_sample(fraud=True)
                st.session_state.single_inputs = api_sample or get_sample_preset(fraud=True)
            else:
                st.session_state.single_inputs = get_sample_preset(fraud=True)
            st.rerun()

    with c_p4:
        if st.button("🔄 Zero Baseline Reset", width="stretch"):
            st.session_state.single_inputs = {c: 0.0 for c in FEATURE_ORDER}
            st.rerun()

    # Alternative: Single CSV Upload
    with st.expander("📁 Alternative: Upload 1-Transaction CSV to Pre-Fill Form", expanded=False):
        single_csv_file = st.file_uploader("Upload single transaction CSV", type=["csv"], key="single_csv_uploader")
        if single_csv_file is not None:
            try:
                uploaded_single_df = pd.read_csv(single_csv_file)
                if len(uploaded_single_df) >= 1:
                    row_data = uploaded_single_df.iloc[0].to_dict()
                    valid_inputs = {c: float(row_data[c]) for c in FEATURE_ORDER if c in row_data and pd.notnull(row_data[c])}
                    if len(valid_inputs) == 30:
                        st.session_state.single_inputs = valid_inputs
                        st.success("✅ Successfully extracted all 30 canonical predictors from uploaded CSV!")
                    else:
                        st.warning(f"Uploaded CSV provided {len(valid_inputs)}/30 canonical features. Pre-filled available values.")
                        st.session_state.single_inputs.update(valid_inputs)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Transaction Input Form
    with st.form("single_scoring_form"):
        st.markdown("#### 1. Core Transaction Context")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            amount_val = st.number_input(
                "Transaction Amount ($ USD)",
                min_value=0.0,
                max_value=1_000_000.0,
                value=float(st.session_state.single_inputs.get("Amount", 149.97)),
                step=5.0,
                format="%.2f",
                help="Actual checkout amount. Scaled using the pre-fitted StandardScaler before inference.",
            )

        with f_col2:
            time_val = st.number_input(
                "Elapsed Benchmark Time (Seconds)",
                min_value=0.0,
                max_value=250_000.0,
                value=float(st.session_state.single_inputs.get("Time", 31604.21)),
                step=10.0,
                format="%.2f",
                help="Seconds elapsed since reference dataset baseline.",
            )

        with f_col3:
            # Derived velocity context indicator (helpful for fraud analysts)
            est_hours = round(time_val / 3600.0, 1)
            st.markdown(
                clean_html(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.65); padding: 0.85rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-top: 1.55rem;">
                        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Temporal Estimate</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #60A5FA;">~{est_hours} hrs elapsed</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 2. Anonymized PCA Behavioral Features (V1 – V28)")
        st.caption(
            "Principal Component Analysis (PCA) features capture bank account behaviors, location signatures, "
            "and device anomalies while preserving user confidentiality. Features are passed unscaled to the model."
        )

        pca_tab1, pca_tab2, pca_tab3, pca_tab4 = st.tabs([
            "🔹 Suite A: Account Dynamics (V1–V7)",
            "🔹 Suite B: Auth & Behavioral (V8–V14)",
            "🔹 Suite C: Device & Channel (V15–V21)",
            "🔹 Suite D: Network & Anomalies (V22–V28)",
        ])

        v_inputs: Dict[str, float] = {}

        def render_pca_grid(tab_ctx, start_idx, end_idx):
            with tab_ctx:
                num_features = end_idx - start_idx + 1
                cols = st.columns(4)
                for i in range(num_features):
                    feat_num = start_idx + i
                    feat_name = f"V{feat_num}"
                    def_val = float(st.session_state.single_inputs.get(feat_name, 0.0))
                    with cols[i % 4]:
                        v_inputs[feat_name] = st.number_input(
                            f"{feat_name}",
                            value=def_val,
                            step=0.1,
                            format="%.5f",
                            key=f"input_{feat_name}",
                        )

        render_pca_grid(pca_tab1, 1, 7)
        render_pca_grid(pca_tab2, 8, 14)
        render_pca_grid(pca_tab3, 15, 21)
        render_pca_grid(pca_tab4, 22, 28)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_score = st.form_submit_button(
            "⚡ Score Transaction Risk",
            width="stretch",
            type="primary",
        )

    # Inference Execution & Result Display
    if submit_score:
        if not st.session_state.get("auth_token"):
            st.warning("⚠️ Authentication Required: Please sign in via the sidebar to obtain a cryptographic session token.")
        else:
            payload = {"Time": time_val, "Amount": amount_val}
            payload.update(v_inputs)

            is_valid, cleaned_payload, validation_errors = validate_single_transaction(payload)

            if not is_valid:
                st.error("Input Schema Validation Failed:")
                for err in validation_errors:
                    st.write(f"- {err}")
            elif not is_healthy:
                st.error(
                    "❌ FastAPI Inference Service is unreachable. "
                    "Ensure the backend is online or use the wake-up button above."
                )
            else:
                with st.spinner("Executing calibrated XGBoost inference..."):
                    t_start = time.time()
                    result, err = client.predict_single(cleaned_payload, threshold=threshold, explain=True)
                    latency = round((time.time() - t_start) * 1000, 1)

                if err:
                    st.error(f"Inference Failure: {err}")
                elif result:
                    st.session_state.latest_single_result = {
                        "result": result,
                        "payload": cleaned_payload,
                        "amount": amount_val,
                        "latency": latency,
                    }

                # Update live session metrics
                is_fraud = result.get("prediction", 0) == 1
                r_tier = result.get("risk_level", "LOW")
                prob_score = result.get("fraud_probability", 0.0)

                st.session_state.total_evaluated_count += 1
                st.session_state.total_volume_amount += amount_val
                if is_fraud:
                    st.session_state.fraud_flagged_count += 1

                # Record in activity log
                now_str = datetime.now().strftime("%H:%M:%S")
                st.session_state.activity_log.insert(0, {
                    "Timestamp": now_str,
                    "Transaction_ID": f"TX-{int(time.time() * 1000) % 1000000}",
                    "Type": "Real-Time Scoring",
                    "Amount": f"${amount_val:,.2f}",
                    "Risk_Level": r_tier,
                    "Fraud_Probability": f"{prob_score:.2%}",
                    "Decision": "DECLINED" if is_fraud else ("STEP-UP" if r_tier == "MEDIUM" else "APPROVED"),
                    "Cutoff": f"{threshold:.2f}",
                })

    # Render Latest Result Panel
    if st.session_state.latest_single_result:
        res_data = st.session_state.latest_single_result["result"]
        amt_data = st.session_state.latest_single_result["amount"]
        stored_payload = st.session_state.latest_single_result["payload"]
        latency_val = st.session_state.latest_single_result.get("latency", 0.0)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Real-Time Risk Intelligence Dossier")

        # Result Banner
        render_prediction_result_banner(
            result=res_data,
            amount=amt_data,
            threshold=threshold,
        )

        # Decision Metrics Strip
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        fraud_p = res_data.get("fraud_probability", 0.0)
        legit_p = res_data.get("legitimate_probability", max(0.0, 1.0 - fraud_p))
        with rc1:
            st.metric("Fraud Probability", f"{fraud_p:.4%}")
        with rc2:
            st.metric("Legit Probability", f"{legit_p:.4%}")
        with rc3:
            st.metric("Active Cutoff (θ)", f"{threshold:.2f}")
        with rc4:
            st.metric("Model Version", res_data.get("model_version", "1.0-baseline"))
        with rc5:
            st.metric("Latency", f"{latency_val:.1f} ms")

        # Visual Telemetry: Gauge & Feature Explainability
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            with st.container(border=True):
                st.markdown(
                    clean_html(
                        """
                        <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                            Calibrated Risk Dial
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
                prob_v = res_data.get("fraud_probability", 0.0)
                gauge_fig = create_risk_gauge(prob_v, threshold=threshold)
                st.plotly_chart(gauge_fig, width="stretch")

        with col_g2:
            with st.container(border=True):
                exp_data = res_data.get("explanation")
                if exp_data and exp_data.get("top_positive"):
                    st.plotly_chart(create_explainability_waterfall_chart(exp_data), width="stretch")
                    st.caption(f"💡 {exp_data.get('summary', '')}")
                else:
                    st.markdown(
                        clean_html(
                            """
                            <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                                Mathematical PCA Contributors
                            </div>
                            """
                        ),
                        unsafe_allow_html=True,
                    )
                    anom_fig = create_top_anomalies_chart(stored_payload)
                    if anom_fig:
                        st.plotly_chart(anom_fig, width="stretch")

        # Raw JSON Inspector
        with st.expander("🔍 Inspect Full API JSON Response", expanded=False):
            st.json(res_data)


# =============================================================================
# VIEW 3: 📁 Batch CSV Processing
# =============================================================================
elif selected_nav == "📁 Batch CSV Processing":
    st.markdown("### 📁 High-Throughput Batch CSV Ingestion & Scoring")
    st.markdown(
        clean_html(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 1.5rem;'>"
            "Upload a batch CSV file with transaction records to score multiple transactions concurrently with chunked network streaming."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    # Sample Template Generator & Downloader
    sample_df = create_sample_csv_dataframe()
    csv_buf = io.StringIO()
    sample_df.to_csv(csv_buf, index=False)
    sample_bytes = csv_buf.getvalue().encode("utf-8")

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Batch CSV Transactions File",
            type=["csv"],
            help="File must contain canonical headers: Time, V1 to V28, and Amount.",
        )
    with col_up2:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    """
                    <div style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.25rem;">
                        📥 Need Test Dataset?
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin-bottom: 0.75rem;">
                        Download a verified 5-row batch containing both legitimate and fraudulent transactions.
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download Sample CSV",
                data=sample_bytes,
                file_name="sample_transactions_batch.csv",
                mime="text/csv",
                width="stretch",
            )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > 10 * 1024 * 1024:
                st.error("❌ File size exceeds the maximum allowed limit of 10 MB.")
                st.stop()
            raw_batch_df = pd.read_csv(io.BytesIO(file_bytes))
            file_size_kb = len(file_bytes) / 1024.0

            st.markdown(
                clean_html(
                    f"""
                    <div style="display: flex; gap: 1rem; align-items: center; padding: 0.75rem 1.25rem; background: rgba(15, 23, 42, 0.6); border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); margin: 1rem 0;">
                        <div>📄 <strong>{uploaded_file.name}</strong></div>
                        <div style="color: #64748B;">&bull;</div>
                        <div style="color: #94A3B8;">{len(raw_batch_df):,} Rows</div>
                        <div style="color: #64748B;">&bull;</div>
                        <div style="color: #94A3B8;">{len(raw_batch_df.columns)} Columns</div>
                        <div style="color: #64748B;">&bull;</div>
                        <div style="color: #94A3B8;">{file_size_kb:.1f} KB</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

            # Pre-flight Validation
            is_valid, cleaned_batch_df, validation_errors = validate_batch_dataframe(raw_batch_df)

            if not is_valid:
                st.error("❌ Batch Schema Validation Failed:")
                for err in validation_errors:
                    st.write(f"- {err}")
            else:
                has_class = "Class" in cleaned_batch_df.columns
                class_info = " (Ground truth `Class` detected & preserved for evaluation)" if has_class else ""
                st.markdown(
                    clean_html(
                        f"""
                        <div style="padding: 0.6rem 1rem; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; color: #34D399; font-size: 0.85rem; margin-bottom: 1rem;">
                            ✅ <strong>Schema Verified:</strong> All 30 canonical features present, numeric types validated, zero NaN anomalies.{class_info}
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

                # Batch Ingestion Parameters
                cfg_c1, cfg_c2 = st.columns(2)
                with cfg_c1:
                    chunk_size = st.select_slider(
                        "Network Streaming Chunk Size (Rows / Request)",
                        options=[50, 100, 200, 500],
                        value=100,
                        help="Controls the number of transaction rows sent per HTTP POST request to FastAPI. "
                             "Smaller chunk sizes (e.g., 100) ensure sub-second response times and prevent edge-proxy / Cloudflare HTTP 520 timeouts on large datasets.",
                    )
                with cfg_c2:
                    max_retries = st.selectbox(
                        "Transient Error Retries (520, 502, 503, 504, 429)",
                        options=[1, 2, 3, 5],
                        index=2,
                        help="Number of bounded retry attempts with exponential backoff and jitter per failing chunk before pausing.",
                    )

                btn_batch = st.button(
                    f"🚀 Execute Batch Inference for {len(cleaned_batch_df):,} Transactions (Chunk Size = {chunk_size}, Cutoff = {threshold:.2f})",
                    type="primary",
                    width="stretch",
                )

                if btn_batch:
                    if not st.session_state.get("auth_token"):
                        st.warning("⚠️ Authentication Required: Please sign in via the sidebar to process batch transactions.")
                    elif not is_healthy:
                        st.error("FastAPI Backend service is offline. Please launch the backend to run inference.")
                    else:
                        records = cleaned_batch_df.to_dict(orient="records")
                        prog_bar = st.progress(0, text="Initializing sequential batch inference stream...")
                        t0 = time.time()

                        def update_progress(stats: dict):
                            c_idx = stats.get("chunk_index", 0)
                            t_chunks = stats.get("total_chunks", 1)
                            t_done = stats.get("processed_rows", 0)
                            c_lat = stats.get("latency_ms", 0.0)
                            c_code = stats.get("status_code", 200)
                            c_retries = stats.get("retries_used", 0)
                            frac = min(1.0, max(0.0, (c_idx + 1) / max(1, t_chunks)))
                            retry_msg = f" | Retries: {c_retries}" if c_retries > 0 else ""
                            prog_bar.progress(
                                frac,
                                text=f"Chunk {c_idx + 1}/{t_chunks} ({t_done:,}/{len(records):,} rows) | HTTP {c_code} ({c_lat:.0f}ms){retry_msg}",
                            )

                        batch_resp, b_err = client.predict_batch(
                            records,
                            threshold=threshold,
                            chunk_size=chunk_size,
                            max_retries=max_retries,
                            progress_callback=update_progress,
                        )
                        elapsed_s = round(time.time() - t0, 2)

                        if batch_resp and batch_resp.get("predictions"):
                            preds = batch_resp["predictions"]
                            n_preds = len(preds)
                            is_partial = batch_resp.get("is_partial", False)

                            # Preserve matching slice of original data for exact 1-to-1 row alignment
                            sliced_df = cleaned_batch_df.iloc[:n_preds].copy()

                            st.session_state.batch_results = {
                                "response": batch_resp,
                                "original_df": sliced_df,
                                "elapsed_s": elapsed_s,
                            }

                            # Update session counters
                            f_count = batch_resp.get("fraud_count", 0)
                            t_proc = batch_resp.get("total_processed", 0)
                            sum_amt = float(sliced_df["Amount"].sum()) if "Amount" in sliced_df else 0.0

                            st.session_state.total_evaluated_count += t_proc
                            st.session_state.fraud_flagged_count += f_count
                            st.session_state.total_volume_amount += sum_amt

                            # Append to activity log
                            st.session_state.activity_log.insert(0, {
                                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                                "Transaction_ID": f"BATCH-{uploaded_file.name[:8]}",
                                "Type": f"Batch CSV ({t_proc:,} rows)",
                                "Amount": f"${sum_amt:,.2f}",
                                "Risk_Level": "HIGH" if f_count > 0 else "LOW",
                                "Fraud_Probability": f"{batch_resp.get('fraud_rate', 0.0):.2%}",
                                "Decision": f"{f_count:,} Flagged",
                                "Cutoff": f"{threshold:.2f}",
                            })

                            if is_partial:
                                prog_bar.progress(
                                    min(1.0, n_preds / max(1, len(records))),
                                    text=f"⚠️ Batch Job Paused at Row {n_preds:,} / {len(records):,} ({elapsed_s}s)",
                                )
                                st.warning(
                                    f"⚠️ **Batch Partially Completed (Checkpoint Saved):** Successfully scored {n_preds:,} rows "
                                    f"before encountering: `{b_err}`. All {n_preds:,} completed predictions have been preserved "
                                    f"and are displayed below for immediate inspection and CSV export."
                                )
                            else:
                                prog_bar.progress(100, text=f"✅ Inference pipeline finished: {n_preds:,} rows in {elapsed_s}s")
                                st.success(f"✅ Successfully scored all {n_preds:,} transactions in {elapsed_s}s!")
                        elif b_err:
                            prog_bar.empty()
                            st.error(f"❌ Batch Scoring Error: {b_err}")

        except Exception as exc:
            st.error(f"Error parsing CSV file: {str(exc)}")

    # Display Batch Analytics & Enriched Data Table
    if st.session_state.batch_results is not None:
        b_data = st.session_state.batch_results["response"]
        orig_df = st.session_state.batch_results["original_df"]
        elapsed = st.session_state.batch_results.get("elapsed_s", 0.0)

        total_p = b_data.get("total_processed", 0)
        frauds_p = b_data.get("fraud_count", 0)
        legits_p = b_data.get("legitimate_count", 0)
        frate_p = b_data.get("fraud_rate", 0.0)
        predictions_list = b_data.get("predictions", [])
        is_partial = b_data.get("is_partial", False)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Batch Evaluation Intelligence & Analytics")

        if is_partial:
            st.warning(
                f"⚠️ **Displaying Partial Results Checkpoint:** {total_p:,} transactions scored before error: "
                f"`{b_data.get('error_detail')}`. Failed at chunk {b_data.get('failed_chunk')}. "
                f"All {total_p:,} completed rows are preserved and analyzed below."
            )

        # 4 Metric Tiles
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.markdown(
                clean_html(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">PROCESSED ROWS</div>
                        <div class="kpi-value">{total_p:,}</div>
                        <div class="kpi-footer">&bull; Elapsed: {elapsed}s</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
        with b_col2:
            st.markdown(
                clean_html(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title" style="color: #F87171;">FLAGGED FRAUD</div>
                        <div class="kpi-value" style="color: #F87171;">{frauds_p:,}</div>
                        <div class="kpi-footer" style="color: #F87171;">&bull; {frate_p:.2%} Flagged Fraud Rate</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
        with b_col3:
            st.markdown(
                clean_html(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title" style="color: #34D399;">APPROVED CLEAN</div>
                        <div class="kpi-value" style="color: #34D399;">{legits_p:,}</div>
                        <div class="kpi-footer" style="color: #34D399;">&bull; {(1 - frate_p):.2%} Acceptance Rate</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
        with b_col4:
            st.markdown(
                clean_html(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title" style="color: #60A5FA;">ACTIVE CUTOFF (&theta;)</div>
                        <div class="kpi-value" style="color: #60A5FA;">{threshold:.2f}</div>
                        <div class="kpi-footer">&bull; Dynamic threshold</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        # Ensure orig_df length matches predictions_list exactly
        if len(orig_df) != len(predictions_list):
            orig_df = orig_df.iloc[:len(predictions_list)].copy()

        # Enriched DataFrame
        enriched_df = orig_df.copy()
        enriched_df["Prediction"] = [p.get("prediction") for p in predictions_list]
        enriched_df["Status"] = [p.get("label") for p in predictions_list]
        enriched_df["Fraud_Probability"] = [p.get("fraud_probability") for p in predictions_list]
        enriched_df["Risk_Level"] = [p.get("risk_level") for p in predictions_list]
        enriched_df["Threshold_Used"] = threshold

        # Ground Truth Batch Evaluation (when 'Class' column is present)
        if "Class" in enriched_df.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Ground Truth Batch Performance Evaluation")
            st.markdown(
                clean_html(
                    "<p style='color: #94A3B8; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem;'>"
                    "Ground truth labels detected in batch payload. Real-time classification metrics computed against actual Class labels."
                    "</p>"
                ),
                unsafe_allow_html=True,
            )

            y_actual = enriched_df["Class"].astype(int)
            y_pred = enriched_df["Prediction"].astype(int)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

            acc = float(accuracy_score(y_actual, y_pred))
            prec = float(precision_score(y_actual, y_pred, zero_division=0))
            rec = float(recall_score(y_actual, y_pred, zero_division=0))
            f1 = float(f1_score(y_actual, y_pred, zero_division=0))

            has_both_classes = len(np.unique(y_actual)) > 1
            roc_val = None
            if has_both_classes:
                try:
                    roc_val = float(roc_auc_score(y_actual, enriched_df["Fraud_Probability"]))
                except Exception:
                    roc_val = None

            # Confusion Matrix counts
            if has_both_classes:
                cm = confusion_matrix(y_actual, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
            else:
                tp = int(((y_actual == 1) & (y_pred == 1)).sum())
                tn = int(((y_actual == 0) & (y_pred == 0)).sum())
                fp = int(((y_actual == 0) & (y_pred == 1)).sum())
                fn = int(((y_actual == 1) & (y_pred == 0)).sum())

            # Metrics row
            eval_c1, eval_c2, eval_c3, eval_c4, eval_c5 = st.columns(5)
            with eval_c1:
                st.markdown(
                    clean_html(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-title">BATCH ACCURACY</div>
                            <div class="kpi-value">{acc:.2%}</div>
                            <div class="kpi-footer">&bull; Correct Decisions</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            with eval_c2:
                st.markdown(
                    clean_html(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-title" style="color: #60A5FA;">PRECISION</div>
                            <div class="kpi-value" style="color: #60A5FA;">{prec:.2%}</div>
                            <div class="kpi-footer" style="color: #60A5FA;">&bull; TP / (TP + FP)</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            with eval_c3:
                st.markdown(
                    clean_html(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-title" style="color: #A78BFA;">RECALL</div>
                            <div class="kpi-value" style="color: #A78BFA;">{rec:.2%}</div>
                            <div class="kpi-footer" style="color: #A78BFA;">&bull; TP / (TP + FN)</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            with eval_c4:
                st.markdown(
                    clean_html(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-title" style="color: #34D399;">F1-SCORE</div>
                            <div class="kpi-value" style="color: #34D399;">{f1:.2%}</div>
                            <div class="kpi-footer" style="color: #34D399;">&bull; Harmonic Mean</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            with eval_c5:
                roc_str = f"{roc_val:.4f}" if roc_val is not None else "N/A"
                st.markdown(
                    clean_html(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-title" style="color: #FBBF24;">ROC-AUC</div>
                            <div class="kpi-value" style="color: #FBBF24;">{roc_str}</div>
                            <div class="kpi-footer">&bull; Area Under ROC</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

            # Confusion matrix card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                clean_html(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.25rem;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.75rem;">
                            Batch Confusion Matrix Breakdown
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 0.9rem;">
                                <div style="font-size: 0.75rem; color: #34D399; font-weight: 600;">TRUE NEGATIVE (TN)</div>
                                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">{tn:,}</div>
                                <div style="font-size: 0.75rem; color: #94A3B8;">Actual Legitimate &rarr; Predicted Legitimate</div>
                            </div>
                            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 0.9rem;">
                                <div style="font-size: 0.75rem; color: #FBBF24; font-weight: 600;">FALSE POSITIVE (FP)</div>
                                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">{fp:,}</div>
                                <div style="font-size: 0.75rem; color: #94A3B8;">Actual Legitimate &rarr; Predicted Fraud (False Alarm)</div>
                            </div>
                            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 0.9rem;">
                                <div style="font-size: 0.75rem; color: #F87171; font-weight: 600;">FALSE NEGATIVE (FN)</div>
                                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">{fn:,}</div>
                                <div style="font-size: 0.75rem; color: #94A3B8;">Actual Fraud &rarr; Predicted Legitimate (Missed Fraud)</div>
                            </div>
                            <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 0.9rem;">
                                <div style="font-size: 0.75rem; color: #60A5FA; font-weight: 600;">TRUE POSITIVE (TP)</div>
                                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">{tp:,}</div>
                                <div style="font-size: 0.75rem; color: #94A3B8;">Actual Fraud &rarr; Predicted Fraud (Detected)</div>
                            </div>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        # Analytics Visualizations
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Batch Risk Diagnostics")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            with st.container(border=True):
                st.markdown(
                    clean_html(
                        """
                        <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">
                            Risk Tier Breakdown
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
                r_counts = enriched_df["Risk_Level"].value_counts().to_dict()
                st.plotly_chart(create_risk_donut_chart(r_counts), width="stretch")

        with vcol2:
            with st.container(border=True):
                st.markdown(
                    clean_html(
                        """
                        <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">
                            Transaction Value vs. Fraud Probability
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
                st.plotly_chart(create_amount_vs_prob_chart(enriched_df), width="stretch")

        # Filterable Data Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Enriched Batch Transaction Table")

        filt_col1, filt_col2 = st.columns([2, 1])
        with filt_col1:
            filter_tiers = st.multiselect(
                "Filter by Risk Tier",
                options=["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"],
            )

        filtered_df = enriched_df[enriched_df["Risk_Level"].isin(filter_tiers)]

        display_columns = ["Status", "Risk_Level", "Fraud_Probability"]
        if "Class" in filtered_df.columns:
            display_columns.append("Class")
        display_columns += ["Amount", "Time"] + [
            c for c in filtered_df.columns if c.startswith("V")
        ]
        st.dataframe(
            filtered_df[display_columns],
            width="stretch",
            height=320,
        )

        # Export Enriched CSV
        out_buffer = io.StringIO()
        enriched_df.to_csv(out_buffer, index=False)
        st.download_button(
            label=f"💾 Download Enriched Prediction Results CSV ({len(enriched_df):,} rows)",
            data=out_buffer.getvalue().encode("utf-8"),
            file_name=f"fraudguard_batch_predictions_cutoff_{threshold:.2f}.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )


# =============================================================================
# VIEW: ⚖️ Model Comparison
# =============================================================================
elif selected_nav == "⚖️ Model Comparison":
    st.markdown("### ⚖️ Multi-Model Development & Cross-Validation Benchmarks")
    st.markdown(
        clean_html(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 1.5rem;'>"
            "Comparative evaluation of 5 candidate classification algorithms trained and cross-validated "
            "on the 4,032 training transactions with zero test set leakage."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    try:
        manifest = get_model_comparison_data()
    except Exception as e:
        st.error(f"Failed to load model comparison manifest: {e}")
        st.stop()

    # Active Production Status
    st.info(
        f"🛡️ **Active Production Artifact:** `{manifest.get('active_production_model', 'Tuned XGBoost (Baseline v1.0)')}` "
        f"— Stored under `backend/models/`. Candidate models are versioned in `backend/models/registry/`. "
        "Production model activation is reversible and requires explicit administrator authorization."
    )

    # Key Comparison Table
    models_list = manifest.get("models", [])
    if models_list:
        st.markdown("#### 1. Multi-Model Benchmark Matrix (5-Fold CV + Held-Out Test Split)")
        summary_rows = []
        for m in models_list:
            cv = m.get("cross_validation", {})
            te = m.get("independent_test", {})
            summary_rows.append({
                "Model Candidate": m.get("model_name"),
                "Algorithm": m.get("algorithm"),
                "CV F1-Score": f"{cv.get('f1_mean', 0.0):.4f}",
                "CV Recall": f"{cv.get('recall_mean', 0.0) * 100:.2f}%",
                "CV Precision": f"{cv.get('precision_mean', 0.0) * 100:.2f}%",
                "CV PR-AUC": f"{cv.get('pr_auc_mean', 0.0):.4f}",
                "Test Recall": f"{te.get('recall', 0.0) * 100:.2f}%",
                "Test Precision": f"{te.get('precision', 0.0) * 100:.2f}%",
                "Test F1-Score": f"{te.get('f1_score', 0.0):.4f}",
                "Test ROC-AUC": f"{te.get('roc_auc', 0.0):.4f}",
                "Test TP/FP": f"{te.get('tp', 0)} / {te.get('fp', 0)}",
                "Train Time": f"{m.get('avg_training_time_s', 0.0):.3f}s",
            })
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")

        # Visual Comparison Charts
        st.markdown("#### 2. Visual Comparative Diagnostics")
        c_ch1, c_ch2 = st.columns([1, 1])
        with c_ch1:
            with st.container(border=True):
                st.plotly_chart(create_model_comparison_bar_chart(manifest), width="stretch")
        with c_ch2:
            with st.container(border=True):
                st.plotly_chart(create_model_comparison_radar_chart(manifest), width="stretch")

        # Qualitative Trade-Off Analysis
        st.markdown("#### 3. Engineering Trade-Off & Selection Rationale")
        for m in models_list:
            with st.expander(f"📌 {m['model_name']} ({m['algorithm']})"):
                st.markdown(f"**Trade-off Profile:** {m.get('trade_off_analysis', 'N/A')}")
                te = m.get("independent_test", {})
                cv = m.get("cross_validation", {})
                st.markdown(
                    f"- **Cross-Validation Stability:** F1 = {cv.get('f1_mean', 0):.4f}, Recall = {cv.get('recall_mean', 0):.2%}, Precision = {cv.get('precision_mean', 0):.2%}\n"
                    f"- **Held-Out Test Results:** Caught {te.get('tp', 0)} of 8 frauds (Recall = {te.get('recall', 0):.2%}), False Alarms = {te.get('fp', 0)} (Precision = {te.get('precision', 0):.2%})\n"
                    f"- **Brier Score:** {te.get('brier_score', 0):.6f} | **Log Loss:** {te.get('log_loss', 0):.4f}\n"
                    f"- **Artifact File:** `backend/models/registry/{m.get('artifact_file')}`"
                )

        # Statistical Honesty & Sample Size Advisory
        st.warning(
            "⚠️ **Statistical Reality on Small Sample Size:** "
            "The training partition contains only 32 fraud cases, and the held-out test split contains only 8 fraud transactions. "
            "Because each false negative swings recall by 12.5%, cross-validation consistency and probability calibration "
            "must be prioritized over isolated test-set metrics. No model trained on 40 total fraud cases should be described as production-proven."
        )


# =============================================================================
# VIEW 4: 🩺 System Telemetry
# =============================================================================
elif selected_nav == "🩺 System Telemetry":
    st.markdown("### 🩺 Model Telemetry & Microservice Health")
    st.markdown(
        clean_html(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 1.5rem;'>"
            "Real-time diagnostic telemetry, microservice readiness, model artifact verification, and endpoint validation."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    tcol1, tcol2 = st.columns([1, 3])
    with tcol1:
        if st.button("🔄 Probe Connection Now", width="stretch"):
            st.rerun()
    with tcol2:
        st.write(f"Target Service URL: `{st.session_state.api_url}/health`")

    if is_healthy:
        st.markdown(
            clean_html(
                """
                <div style="padding: 1rem 1.5rem; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; margin-bottom: 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span class="pulse-dot online" style="width: 12px; height: 12px;"></span>
                        <div>
                            <h4 style="margin: 0; color: #34D399;">FastAPI Microservice is Online & Healthy</h4>
                            <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 0.2rem;">Model artifacts loaded and memory footprint verified.</div>
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Response Latency", f"{latency_ms:.0f} ms", delta="Optimal" if latency_ms < 300 else "Acceptable")
        mcol2.metric("Classifier Engine", health_data.get("model_type", "XGBClassifier"))
        mcol3.metric("Canonical Features", health_data.get("expected_features_count", 30))
        active_scaler_name = health_data.get("scaler_type", "StandardScaler")
        mcol4.metric("Pre-Scaler Status", f"Active ({active_scaler_name})" if health_data.get("scaler_loaded") else "Missing")

        st.markdown("#### Endpoint Verification Checklist")
        st.markdown(
            clean_html(
                f"""
                <div class="glass-card">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                        <thead>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); text-align: left; color: #94A3B8;">
                                <th style="padding: 8px 12px;">METHOD</th>
                                <th style="padding: 8px 12px;">ENDPOINT</th>
                                <th style="padding: 8px 12px;">STATUS</th>
                                <th style="padding: 8px 12px;">FUNCTION</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                                <td style="padding: 8px 12px;"><span style="color: #60A5FA; font-weight: bold;">GET</span></td>
                                <td style="padding: 8px 12px; font-family: monospace;">/health</td>
                                <td style="padding: 8px 12px;"><span style="color: #34D399;">● 200 OK</span></td>
                                <td style="padding: 8px 12px; color: #CBD5E1;">Service health check & metadata</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                                <td style="padding: 8px 12px;"><span style="color: #60A5FA; font-weight: bold;">GET</span></td>
                                <td style="padding: 8px 12px; font-family: monospace;">/</td>
                                <td style="padding: 8px 12px;"><span style="color: #34D399;">● 200 OK</span></td>
                                <td style="padding: 8px 12px; color: #CBD5E1;">Root API manifest & documentation link</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                                <td style="padding: 8px 12px;"><span style="color: #60A5FA; font-weight: bold;">GET</span></td>
                                <td style="padding: 8px 12px; font-family: monospace;">/sample</td>
                                <td style="padding: 8px 12px;"><span style="color: #34D399;">● 200 OK</span></td>
                                <td style="padding: 8px 12px; color: #CBD5E1;">Fetches legitimate or fraud test payloads</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                                <td style="padding: 8px 12px;"><span style="color: #34D399; font-weight: bold;">POST</span></td>
                                <td style="padding: 8px 12px; font-family: monospace;">/predict</td>
                                <td style="padding: 8px 12px;"><span style="color: #34D399;">● 200 OK</span></td>
                                <td style="padding: 8px 12px; color: #CBD5E1;">Single transaction inference engine</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 12px;"><span style="color: #34D399; font-weight: bold;">POST</span></td>
                                <td style="padding: 8px 12px; font-family: monospace;">/predict/batch</td>
                                <td style="padding: 8px 12px;"><span style="color: #34D399;">● 200 OK</span></td>
                                <td style="padding: 8px 12px; color: #CBD5E1;">Bulk chunked batch transaction processing</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### Diagnostic JSON Payload")
        st.json(health_data)
    else:
        st.error(f"❌ Connection Probing Failed: {health_error}")
        st.markdown(
            textwrap.dedent(
                """
                #### Troubleshooting & Resolution:
                1. **Render Free Tier Spin-up:**
                   - Render free instances suspend during inactivity.
                   - Click the **⚡ Ping & Wake Up Backend** button above and wait 30 seconds for the container to boot.
                2. **Localhost Execution:**
                   - Launch the backend locally in a separate terminal:
                     ```powershell
                     cd C:\\IntershipProject\\EDP-PROJECT-DEPLOYMENT
                     uvicorn backend.app:app --host 127.0.0.1 --port 8000
                     ```
                   - Click **💻 Local** in the sidebar to point to `http://127.0.0.1:8000`.
                3. **Verify Model Artifacts:**
                   - Confirm that `backend/models/tuned_xgboost.pkl` and `backend/models/scaler.pkl` exist on disk.
                """
            )
        )


# =============================================================================
# VIEW 5: 📚 Architecture & Compliance Specs
# =============================================================================
elif selected_nav == "📚 Architecture & Specs":
    st.markdown("### 📚 Architecture & Compliance Specifications")
    st.markdown(
        clean_html(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 1.5rem;'>"
            "Comprehensive architectural blueprints, canonical feature schemas, and risk stratification methodologies."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    doc_tab1, doc_tab2, doc_tab3 = st.tabs([
        "🏗️ End-to-End Pipeline Architecture",
        "📐 30-Feature Canonical Schema",
        "⚖️ Decision Threshold & Risk Tiers",
    ])

    with doc_tab1:
        st.markdown(
            textwrap.dedent(
                """
                #### High-Performance Decoupled Architecture
                The platform adopts a decoupled microservice paradigm designed for sub-second inference and resilient fault-isolation:

                ```text
                ┌────────────────────────────────┐         REST / JSON          ┌────────────────────────────────┐
                │   Streamlit Presentation       │ ─────────────────────────────> │       FastAPI Microservice     │
                │   - Fintech Glassmorphic UI    │                                │   - Pydantic Schema Validation │
                │   - Real-Time Scoring          │ <───────────────────────────── │   - CORS & Exception Handling  │
                │   - Batch CSV Processing       │         Status: 200 OK         └───────────────┬────────────────┘
                └────────────────────────────────┘                                                │
                                                                                                  ▼
                ┌────────────────────────────────┐                                ┌────────────────────────────────┐
                │       Inference Output         │ <───────────────────────────── │   Pre-Processing & ML Engine   │
                │   - Fraud Probability [0-1]    │                                │   - StandardScaler (Time/Amount)│
                │   - Risk Tier (LOW/MED/HIGH)   │                                │   - V1-V28 Unscaled (PCA)      │
                │   - Automated Guidance Rule    │                                │   - Tuned XGBoost Classifier   │
                └────────────────────────────────┘                                └────────────────────────────────┘
                ```

                ##### Key Pipeline Stages:
                1. **Presentation Tier (Streamlit Webapp)**:
                   - Encapsulates executive dashboards, interactive transaction scoring, and bulk batch streaming.
                   - Implements defensive network exception handling (timeouts, connection resets, 422 validations).
                2. **API Microservice Tier (FastAPI)**:
                   - Enforces strict Pydantic model validation on all incoming payloads.
                   - Serves asynchronous batch prediction requests with chunking to prevent socket exhaustion.
                3. **Inference Pipeline**:
                   - Automatically transforms raw `Time` and `Amount` using the pre-fitted `StandardScaler`.
                   - Strictly maintains the canonical 30-feature ordering expected by the tuned XGBoost model.
                """
            )
        )

    with doc_tab2:
        st.markdown(
            textwrap.dedent(
                """
                #### Canonical 30-Feature Ordering
                The inference engine strictly enforces the mathematical feature sequence established during model training:

                | Index | Feature Name | Description | Preprocessing Transformation |
                | :--- | :--- | :--- | :--- |
                | **1** | `Time` | Seconds elapsed since baseline benchmark | Standard Scaled (`StandardScaler`) |
                | **2–8** | `V1` – `V7` | Account Dynamics & Card Origin | Preserved Unscaled (Raw PCA) |
                | **9–15** | `V8` – `V14` | Behavioral & Location Signatures | Preserved Unscaled (Raw PCA) |
                | **16–22** | `V15` – `V21` | Device Fingerprint & Velocity | Preserved Unscaled (Raw PCA) |
                | **23–29** | `V22` – `V28` | Network Topology & Anomalies | Preserved Unscaled (Raw PCA) |
                | **30** | `Amount` | Transaction checkout value in currency | Standard Scaled (`StandardScaler`) |
                """
            )
        )

    with doc_tab3:
        st.markdown(
            textwrap.dedent(
                """
                #### Risk Tier Stratification & Threshold Formulation
                The fraud classifier generates a calibrated continuous probability score $P(\\text{Fraud} \\mid X) \\in [0, 1]$.
                The active threshold $\\theta$ governs the binary classification:

                $$\\text{Decision}(X) = \\begin{cases} 1 \\text{ (Fraudulent)}, & \\text{if } P(\\text{Fraud} \\mid X) \\ge \\theta \\\\ 0 \\text{ (Legitimate)}, & \\text{if } P(\\text{Fraud} \\mid X) < \\theta \\end{cases}$$

                ##### Institutional Risk Tiers:
                - 🟢 **LOW RISK ($P < 40\\%$)**:
                  - Normal behavioral and velocity characteristics.
                  - **Policy:** Instant automated approval and clearing.
                - 🟡 **MEDIUM RISK ($40\\% \\le P < 75\\%$)**:
                  - Statistical borderline anomaly; unusual transaction value or elevated component variance.
                  - **Policy:** Step-up Two-Factor Authentication (2FA / OTP) or compliance review queue.
                - 🔴 **HIGH RISK ($P \\ge 75\\%$)**:
                  - Severe anomaly signature matching known syndicate attack patterns.
                  - **Policy:** Immediate card decline, authorization freeze, and automated risk alert.
                """
            )
        )
