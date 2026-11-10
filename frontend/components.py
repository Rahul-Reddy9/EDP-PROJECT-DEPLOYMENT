"""
components.py
-------------
Reusable, beautifully styled HTML/CSS component templates for the
FraudGuard AI fintech dashboard.
"""

from typing import Dict, List, Any, Optional
import streamlit as st


def clean_html(html: str) -> str:
    """
    Cleans an HTML string for safe rendering via st.markdown(..., unsafe_allow_html=True).
    Strips leading and trailing whitespace from every line and removes empty lines so that
    Markdown parsers (e.g. CommonMark / markdown-it) never treat indented HTML as code blocks.
    """
    if not html:
        return ""
    return "\n".join(line.strip() for line in html.strip().splitlines() if line.strip())


def render_brand_header(is_healthy: bool, latency_ms: float, base_url: str) -> None:
    """Renders the top branding header with live status pill."""
    if is_healthy:
        status_html = f"""
        <div class="status-pill status-pill-online">
            <span class="pulse-dot online"></span>
            <span>SYSTEM OPERATIONAL &bull; {latency_ms:.0f} ms</span>
        </div>
        """
    else:
        status_html = """
        <div class="status-pill status-pill-offline">
            <span class="pulse-dot offline"></span>
            <span>BACKEND DISCONNECTED</span>
        </div>
        """

    html = f"""
    <div class="brand-container">
        <div class="brand-left">
            <div class="brand-logo-glow">🛡️</div>
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <h1 class="brand-title">FRAUDGUARD AI</h1>
                    <span class="brand-tag">FINTECH SAAS</span>
                    <span class="brand-tag" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border-color: rgba(16, 185, 129, 0.3);">ENTERPRISE RISK</span>
                </div>
                <div class="brand-subtitle">
                    <span>Autonomous Credit Card Fraud Detection & Real-Time Risk Telemetry</span>
                    <span>&bull;</span>
                    <span class="mono-font" style="font-size: 0.75rem; color: #64748B;">FastAPI + Tuned XGBoost</span>
                </div>
            </div>
        </div>
        <div>
            {clean_html(status_html)}
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_kpi_metrics(
    total_tx: int,
    fraud_tx: int,
    legit_tx: int,
    total_volume: float = 0.0,
    latency_ms: float = 0.0,
) -> None:
    """Renders the 4 top-line KPI executive cards."""
    fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0.0

    html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">TOTAL EVALUATED</span>
                <div class="kpi-icon-wrap">💳</div>
            </div>
            <div class="kpi-value">{total_tx:,}</div>
            <div class="kpi-footer">
                <span class="kpi-badge-positive">&bull; Active Session</span>
                <span>All scored transactions</span>
            </div>
        </div>

        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">THREATS INTERCEPTED</span>
                <div class="kpi-icon-wrap" style="background: rgba(239, 68, 68, 0.15); color: #F87171; border-color: rgba(239, 68, 68, 0.3);">🚨</div>
            </div>
            <div class="kpi-value" style="color: #F87171;">{fraud_tx:,}</div>
            <div class="kpi-footer">
                <span class="kpi-badge-negative">{fraud_rate:.2f}% Flagged Rate</span>
                <span>Above decision cutoff</span>
            </div>
        </div>

        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">PROTECTED VOLUME</span>
                <div class="kpi-icon-wrap" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border-color: rgba(16, 185, 129, 0.3);">🛡️</div>
            </div>
            <div class="kpi-value" style="color: #34D399;">${total_volume:,.2f}</div>
            <div class="kpi-footer">
                <span class="kpi-badge-positive">&bull; Capital Secured</span>
                <span>Cumulative transaction sum</span>
            </div>
        </div>

        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">INFERENCE ENGINE</span>
                <div class="kpi-icon-wrap" style="background: rgba(99, 102, 241, 0.15); color: #818CF8; border-color: rgba(99, 102, 241, 0.3);">⚡</div>
            </div>
            <div class="kpi-value">{latency_ms:.0f}<span style="font-size: 1rem; color: #94A3B8; font-weight: 500;"> ms</span></div>
            <div class="kpi-footer">
                <span class="kpi-badge-positive">&bull; Tuned XGBoost</span>
                <span>Optimized latency</span>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_prediction_result_banner(
    result: Dict[str, Any],
    amount: float,
    threshold: float,
) -> None:
    """Renders the high-impact prediction result banner based on risk tier."""
    prediction = result.get("prediction", 0)
    label = result.get("label", "Legitimate")
    prob = result.get("fraud_probability", 0.0)
    risk_level = result.get("risk_level", "LOW")
    thresh_used = result.get("threshold_used", threshold)

    if risk_level == "HIGH" or prediction == 1:
        banner_class = "risk-banner-high"
        pill_class = "risk-tier-pill-high"
        icon = "🚨"
        action_title = "CRITICAL RISK: TRANSACTION FREEZE RECOMMENDED"
        action_desc = "Probability exceeds high-risk threshold. Automated defense protocols recommend immediately declining authorization and flagging associated account."
        color_val = "#F87171"
    elif risk_level == "MEDIUM":
        banner_class = "risk-banner-medium"
        pill_class = "risk-tier-pill-medium"
        icon = "⚠️"
        action_title = "SUSPICIOUS ACTIVITY: STEP-UP 2FA / OTP REQUIRED"
        action_desc = "Transaction exhibits borderline statistical anomalies. Recommended action: Prompt cardholder for secondary authentication or queue for analyst audit."
        color_val = "#FBBF24"
    else:
        banner_class = "risk-banner-low"
        pill_class = "risk-tier-pill-low"
        icon = "✅"
        action_title = "TRANSACTION VERIFIED: LOW ANOMALY RISK"
        action_desc = "Normal behavioral signature consistent with legitimate cardholder patterns. Safe for instant clearing and settlement."
        color_val = "#34D399"

    html = f"""
    <div class="risk-banner {banner_class}">
        <div class="risk-header-row">
            <div>
                <h2 class="risk-title-badge" style="color: {color_val};">
                    <span>{icon}</span>
                    <span>DECISION: {label.upper()}</span>
                </h2>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.25rem;">
                    Inference complete &bull; Multi-factor XGBoost evaluation
                </div>
            </div>
            <div>
                <span class="risk-tier-pill {pill_class}">
                    RISK TIER: {risk_level}
                </span>
            </div>
        </div>

        <div class="risk-metrics-grid">
            <div>
                <div class="risk-submetric-label">Fraud Probability</div>
                <div class="risk-submetric-val" style="color: {color_val};">{prob:.2%}</div>
            </div>
            <div>
                <div class="risk-submetric-label">Decision Cutoff (&theta;)</div>
                <div class="risk-submetric-val" style="color: #60A5FA;">{thresh_used:.2f}</div>
            </div>
            <div>
                <div class="risk-submetric-label">Transaction Value</div>
                <div class="risk-submetric-val" style="color: #F8FAFC;">${amount:,.2f}</div>
            </div>
            <div>
                <div class="risk-submetric-label">Confidence Delta</div>
                <div class="risk-submetric-val" style="color: #CBD5E1;">{abs(prob - thresh_used):.2f}</div>
            </div>
        </div>

        <div class="action-guidance-box">
            <span style="font-size: 1.25rem;">📌</span>
            <div>
                <strong style="color: #F8FAFC;">{action_title}</strong>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.15rem;">
                    {action_desc}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)
