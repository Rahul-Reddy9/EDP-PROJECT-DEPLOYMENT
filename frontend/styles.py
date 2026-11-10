"""
styles.py
---------
Custom CSS theme definitions for the FraudGuard AI Fintech SaaS Dashboard.
Provides a modern dark navy glassmorphic aesthetic inspired by institutional
fraud monitoring and risk operations platforms.
"""

def get_fintech_theme_css() -> str:
    """Returns custom CSS for dark navy fintech UI."""
    return """<style>
    /* -------------------------------------------------------------
       1. Global Resets & Typography
       ------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #F1F5F9;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background overrides */
    .stApp {
        background: radial-gradient(circle at 15% 10%, rgba(30, 58, 138, 0.22) 0%, transparent 45%),
                    radial-gradient(circle at 85% 85%, rgba(13, 148, 136, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
                    #070B14 !important;
        background-attachment: fixed !important;
    }

    /* Smooth transitions */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    /* -------------------------------------------------------------
       2. Header & Branding Banner
       ------------------------------------------------------------- */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .brand-logo-glow {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 0 20px rgba(37, 99, 235, 0.5);
    }

    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(180deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94A3B8;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .brand-tag {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* -------------------------------------------------------------
       3. Status Badges & Pulse Indicators
       ------------------------------------------------------------- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .status-pill-online {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
    }

    .status-pill-offline {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.15);
    }

    .status-pill-warning {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        position: relative;
    }

    .pulse-dot.online {
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }

    .pulse-dot.online::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        border-radius: 50%;
        background-color: #10B981;
        animation: pulse-ring 1.8s cubic-bezier(0.24, 0, 0.38, 1) infinite;
    }

    .pulse-dot.offline {
        background-color: #EF4444;
        box-shadow: 0 0 8px #EF4444;
    }

    @keyframes pulse-ring {
        0% { transform: scale(1); opacity: 0.8; }
        100% { transform: scale(2.8); opacity: 0; }
    }

    /* -------------------------------------------------------------
       4. Fintech Glassmorphism Cards & KPIs
       ------------------------------------------------------------- */
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.55), 0 0 15px rgba(59, 130, 246, 0.08);
    }

    /* Glassmorphic border container styling for native Streamlit containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4) !important;
        padding: 1rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(59, 130, 246, 0.3) !important;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.55), 0 0 15px rgba(59, 130, 246, 0.08) !important;
    }

    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.45) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3B82F6, transparent);
        opacity: 0.7;
    }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }

    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin: 0;
    }

    .kpi-icon-wrap {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }

    .kpi-footer {
        font-size: 0.75rem;
        color: #64748B;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .kpi-badge-positive {
        color: #34D399;
        font-weight: 600;
    }

    .kpi-badge-negative {
        color: #F87171;
        font-weight: 600;
    }

    /* -------------------------------------------------------------
       5. Risk Assessment Results Cards
       ------------------------------------------------------------- */
    .risk-banner {
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        position: relative;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.6);
        overflow: hidden;
    }

    .risk-banner-low {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 10px 30px -5px rgba(16, 185, 129, 0.18);
    }

    .risk-banner-low::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #10B981, #34D399, #10B981);
    }

    .risk-banner-medium {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 10px 30px -5px rgba(245, 158, 11, 0.18);
    }

    .risk-banner-medium::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #F59E0B, #FBBF24, #F59E0B);
    }

    .risk-banner-high {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 10px 30px -5px rgba(239, 68, 68, 0.25);
    }

    .risk-banner-high::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #EF4444, #F87171, #EF4444);
    }

    .risk-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }

    .risk-title-badge {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .risk-tier-pill {
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 6px 16px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .risk-tier-pill-low {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    .risk-tier-pill-medium {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
    }

    .risk-tier-pill-high {
        background: rgba(239, 68, 68, 0.25);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.6);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.35);
        animation: pulse-high 2s infinite ease-in-out;
    }

    @keyframes pulse-high {
        0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.6); }
    }

    .risk-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1.25rem;
        padding: 1.25rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .risk-submetric-label {
        font-size: 0.76rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    .risk-submetric-val {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .action-guidance-box {
        margin-top: 1.25rem;
        padding: 0.9rem 1.25rem;
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.06);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.88rem;
    }

    /* -------------------------------------------------------------
       6. Form Styling & Input Groups
       ------------------------------------------------------------- */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* Primary Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
        transform: translateY(-1px);
    }

    /* Secondary Buttons */
    button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.65) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }

    button[kind="secondary"]:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
        color: #FFFFFF !important;
    }

    /* -------------------------------------------------------------
       7. Tabs Navigation
       ------------------------------------------------------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 0.4rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.15) !important;
        color: #60A5FA !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }

    /* -------------------------------------------------------------
       8. File Uploader Styling
       ------------------------------------------------------------- */
    [data-testid="stFileUploader"] section {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(59, 130, 246, 0.35) !important;
        border-radius: 14px !important;
        padding: 2rem !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #3B82F6 !important;
        background-color: rgba(30, 58, 138, 0.15) !important;
    }

    /* -------------------------------------------------------------
       9. Sidebar & Controls
       ------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background: #0B1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* -------------------------------------------------------------
       10. Activity Table & DataFrame
       ------------------------------------------------------------- */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #070B14;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Clean progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 50%, #EF4444 100%) !important;
        border-radius: 9999px !important;
    }

    /* Expander card */
    .streamlit-expanderHeader {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #E2E8F0 !important;
    }
    </style>
    """
