"""
charts.py
---------
Interactive, dark-navy fintech visualizations using Plotly.
Designed to blend seamlessly with the glassmorphic SaaS theme.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Theme constants
BG_COLOR = "rgba(0, 0, 0, 0)"
CARD_BG = "rgba(15, 23, 42, 0.75)"
TEXT_COLOR = "#F1F5F9"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
COLOR_LOW = "#10B981"
COLOR_MED = "#F59E0B"
COLOR_HIGH = "#EF4444"
COLOR_BLUE = "#3B82F6"


def create_risk_gauge(probability: float, threshold: float = 0.5) -> go.Figure:
    """Generates an executive gauge dial for real-time transaction scoring."""
    prob_pct = round(probability * 100, 2)
    thresh_pct = round(threshold * 100, 1)

    # Determine needle / bar color
    if probability >= 0.75:
        bar_color = COLOR_HIGH
    elif probability >= 0.40:
        bar_color = COLOR_MED
    else:
        bar_color = COLOR_LOW

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 38, "color": bar_color, "family": "Inter"}},
            title={
                "text": "FRAUD RISK PROBABILITY<br><span style='font-size:0.75em;color:#94A3B8'>Calibrated ML Decision Metric</span>",
                "font": {"size": 13, "color": "#94A3B8", "family": "Inter"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#64748B",
                    "tickfont": {"size": 10, "color": "#94A3B8"},
                    "ticksuffix": "%",
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.04)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [40, 75], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#60A5FA", "width": 3},
                    "thickness": 0.85,
                    "value": thresh_pct,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin={"l": 25, "r": 25, "t": 50, "b": 20},
        height=220,
    )
    return fig


def create_risk_donut_chart(risk_counts: Dict[str, int]) -> go.Figure:
    """Generates a modern donut breakdown chart of risk tiers."""
    labels = ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
    keys = ["LOW", "MEDIUM", "HIGH"]
    values = [risk_counts.get(k, 0) for k in keys]
    colors = [COLOR_LOW, COLOR_MED, COLOR_HIGH]

    total = sum(values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker={"colors": colors, "line": {"color": "#0F172A", "width": 2}},
                textinfo="percent+label",
                textposition="inside",
                insidetextfont={"color": "#FFFFFF", "size": 11, "family": "Inter"},
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5, "font": {"color": "#94A3B8", "size": 11}},
        margin={"l": 15, "r": 15, "t": 20, "b": 40},
        height=260,
        annotations=[
            {
                "text": f"<b>{total}</b><br><span style='font-size:10px;color:#94A3B8'>Transactions</span>",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 18, "color": "#F8FAFC", "family": "Inter"},
                "showarrow": False,
            }
        ],
    )
    return fig


def create_probability_distribution_chart(probabilities: List[float], threshold: float = 0.5) -> go.Figure:
    """Generates a probability distribution histogram with threshold marker."""
    df = pd.DataFrame({"Probability": probabilities})

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["Probability"],
            nbinsx=20,
            marker={
                "color": "#3B82F6",
                "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1},
            },
            opacity=0.85,
            name="Transactions",
        )
    )

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
        annotation_font_color="#F87171",
        annotation_font_size=11,
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin={"l": 40, "r": 20, "t": 30, "b": 35},
        height=260,
        xaxis={
            "title": "Fraud Probability Score",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [0, 1],
        },
        yaxis={
            "title": "Count",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
        },
    )
    return fig


def create_amount_vs_prob_chart(df: pd.DataFrame) -> go.Figure:
    """Generates a scatter plot of Transaction Amount vs Fraud Probability."""
    color_map = {"LOW": COLOR_LOW, "MEDIUM": COLOR_MED, "HIGH": COLOR_HIGH}

    fig = go.Figure()
    for risk in ["LOW", "MEDIUM", "HIGH"]:
        subset = df[df["Risk_Level"] == risk]
        if not subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=subset["Amount"],
                    y=subset["Fraud_Probability"],
                    mode="markers",
                    name=f"{risk} Risk",
                    marker={
                        "color": color_map.get(risk, COLOR_BLUE),
                        "size": 8,
                        "opacity": 0.8,
                        "line": {"color": "rgba(255,255,255,0.2)", "width": 1},
                    },
                    hovertext=[
                        f"Amount: ${row['Amount']:,.2f}<br>Prob: {row['Fraud_Probability']:.2%}<br>Risk: {row['Risk_Level']}"
                        for _, row in subset.iterrows()
                    ],
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin={"l": 40, "r": 20, "t": 30, "b": 35},
        height=260,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.28, "xanchor": "center", "x": 0.5, "font": {"color": "#94A3B8", "size": 10}},
        xaxis={
            "title": "Transaction Amount ($)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
        },
        yaxis={
            "title": "Fraud Probability",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [0, 1.05],
        },
    )
    return fig


def create_top_anomalies_chart(transaction_dict: Dict[str, float]) -> Optional[go.Figure]:
    """Identifies the top PCA features deviating from normal distribution (explainability)."""
    v_features = {k: v for k, v in transaction_dict.items() if k.startswith("V")}
    if not v_features:
        return None

    # Sort by absolute magnitude
    sorted_v = sorted(v_features.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    names = [x[0] for x in reversed(sorted_v)]
    values = [x[1] for x in reversed(sorted_v)]
    colors = [COLOR_HIGH if abs(v) > 3.0 else (COLOR_MED if abs(v) > 1.5 else COLOR_BLUE) for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
            text=[f"{v:+.2f}" for v in values],
            textposition="auto",
            textfont={"size": 10, "color": "#FFFFFF"},
        )
    )

    fig.update_layout(
        title={
            "text": "TOP ANOMALOUS PCA COMPONENTS<br><span style='font-size:0.75em;color:#94A3B8'>Primary mathematical drivers of current inference score</span>",
            "font": {"size": 12, "color": "#94A3B8", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin={"l": 45, "r": 20, "t": 45, "b": 25},
        height=240,
        xaxis={
            "title": "Normalized Component Value",
            "title_font": {"size": 10, "color": "#94A3B8"},
            "tickfont": {"size": 9, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
        },
        yaxis={"tickfont": {"size": 10, "color": "#CBD5E1"}},
    )
    return fig


def create_evaluation_confusion_matrix_chart(
    tn: int, fp: int, fn: int, tp: int, threshold: float = 0.50
) -> go.Figure:
    """Generates an executive interactive confusion matrix with raw counts and normalized rates."""
    total_neg = tn + fp
    total_pos = fn + tp

    tn_pct = (tn / total_neg * 100.0) if total_neg > 0 else 0.0
    fp_pct = (fp / total_neg * 100.0) if total_neg > 0 else 0.0
    fn_pct = (fn / total_pos * 100.0) if total_pos > 0 else 0.0
    tp_pct = (tp / total_pos * 100.0) if total_pos > 0 else 0.0

    z_values = [[tn, fp], [fn, tp]]
    text_annotations = [
        [
            f"<b>{tn:,}</b><br><span style='font-size:11px;color:#94A3B8'>TN ({tn_pct:.1f}%)</span>",
            f"<b>{fp:,}</b><br><span style='font-size:11px;color:#FCA5A5'>FP ({fp_pct:.1f}%)</span>",
        ],
        [
            f"<b>{fn:,}</b><br><span style='font-size:11px;color:#FCA5A5'>FN ({fn_pct:.1f}%)</span>",
            f"<b>{tp:,}</b><br><span style='font-size:11px;color:#86EFAC'>TP ({tp_pct:.1f}%)</span>",
        ],
    ]

    x_labels = ["Predicted Legitimate", "Predicted Fraudulent"]
    y_labels = ["Actual Legitimate", "Actual Fraudulent"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 15, "family": "Inter", "color": "#FFFFFF"},
            colorscale=[
                [0.0, "rgba(15, 23, 42, 0.95)"],
                [0.05, "rgba(30, 41, 59, 0.95)"],
                [0.5, "rgba(30, 58, 138, 0.8)"],
                [1.0, "rgba(37, 99, 235, 0.9)"],
            ],
            showscale=False,
            hoverinfo="none",
        )
    )

    fig.update_layout(
        title={
            "text": f"CONFUSION MATRIX (θ = {threshold:.2f})<br><span style='font-size:0.75em;color:#94A3B8'>Held-Out Test Set (N = {tn+fp+fn+tp:,} transactions)</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=280,
        margin={"l": 30, "r": 20, "t": 55, "b": 30},
        xaxis={
            "tickfont": {"size": 11, "color": "#E2E8F0"},
            "side": "bottom",
            "gridcolor": GRID_COLOR,
        },
        yaxis={
            "tickfont": {"size": 11, "color": "#E2E8F0"},
            "autorange": "reversed",
            "gridcolor": GRID_COLOR,
        },
    )
    return fig


def create_evaluation_roc_curve_chart(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float
) -> go.Figure:
    """Generates an executive ROC Curve with shaded area under curve."""
    fig = go.Figure()

    # Diagonal random guess baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "rgba(148, 163, 184, 0.5)", "width": 1.5, "dash": "dash"},
            name="Random Baseline (AUC = 0.5000)",
            hoverinfo="skip",
        )
    )

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line={"color": "#38BDF8", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            name=f"Tuned XGBoost (AUC = {roc_auc:.4f})",
            hovertemplate="FPR: %{x:.4f}<br>TPR (Recall): %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"RECEIVER OPERATING CHARACTERISTIC (ROC)<br><span style='font-size:0.75em;color:#38BDF8'><b>ROC-AUC: {roc_auc:.4f}</b> (Outstanding Discrimination)</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=280,
        margin={"l": 40, "r": 20, "t": 55, "b": 35},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center", "font": {"size": 10, "color": "#94A3B8"}},
        xaxis={
            "title": "False Positive Rate (1 - Specificity)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [-0.02, 1.02],
        },
        yaxis={
            "title": "True Positive Rate (Recall)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [-0.02, 1.05],
        },
    )
    return fig


def create_evaluation_pr_curve_chart(
    precisions: np.ndarray, recalls: np.ndarray, pr_auc: float, baseline: float = 8 / 1008
) -> go.Figure:
    """Generates an executive Precision-Recall Curve with baseline prevalence."""
    fig = go.Figure()

    # Baseline prevalence line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            line={"color": "rgba(239, 68, 68, 0.5)", "width": 1.5, "dash": "dash"},
            name=f"Baseline Prevalence ({baseline * 100:.2f}%)",
            hoverinfo="skip",
        )
    )

    # PR curve
    fig.add_trace(
        go.Scatter(
            x=recalls,
            y=precisions,
            mode="lines",
            line={"color": "#10B981", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)",
            name=f"Tuned XGBoost (PR-AUC = {pr_auc:.4f})",
            hovertemplate="Recall: %{x:.4f}<br>Precision: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"PRECISION-RECALL (PR) CURVE<br><span style='font-size:0.75em;color:#10B981'><b>PR-AUC / Avg Precision: {pr_auc:.4f}</b> (Prevalence: {baseline * 100:.2f}%)</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=280,
        margin={"l": 40, "r": 20, "t": 55, "b": 35},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center", "font": {"size": 10, "color": "#94A3B8"}},
        xaxis={
            "title": "Recall (True Positive Rate)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [-0.02, 1.02],
        },
        yaxis={
            "title": "Precision (Positive Predictive Value)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [-0.02, 1.05],
        },
    )
    return fig


def create_evaluation_feature_importance_chart(
    fi_df: pd.DataFrame, top_n: int = 15
) -> go.Figure:
    """Generates an executive horizontal bar chart of top XGBoost feature importances."""
    subset = fi_df.head(top_n).iloc[::-1]  # Reverse for ascending order in horizontal bar

    fig = go.Figure(
        go.Bar(
            x=subset["Percentage"],
            y=subset["Feature"],
            orientation="h",
            marker={
                "color": subset["Percentage"],
                "colorscale": [
                    [0.0, "#1E3A8A"],
                    [0.3, "#2563EB"],
                    [0.7, "#06B6D4"],
                    [1.0, "#10B981"],
                ],
                "line": {"color": "rgba(255, 255, 255, 0.15)", "width": 1},
            },
            text=[f"{p:.2f}%" for p in subset["Percentage"]],
            textposition="inside",
            insidetextfont={"size": 10, "color": "#FFFFFF", "family": "Inter"},
            hovertemplate="<b>%{y}</b><br>Gain Importance: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"TOP {top_n} XGBOOST FEATURE IMPORTANCES<br><span style='font-size:0.75em;color:#94A3B8'>Relative tree split contribution across all 200 boosted estimators</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=380,
        margin={"l": 60, "r": 20, "t": 55, "b": 35},
        xaxis={
            "title": "Relative Importance (%)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
        },
        yaxis={"tickfont": {"size": 10, "color": "#E2E8F0"}},
    )
    return fig


def create_evaluation_class_distribution_chart(
    legit_count: int, fraud_count: int, title: str = "Class Imbalance Breakdown"
) -> go.Figure:
    """Generates an executive donut chart showing class distribution and extreme imbalance."""
    total = legit_count + fraud_count
    legit_pct = legit_count / total * 100.0 if total > 0 else 0.0
    fraud_pct = fraud_count / total * 100.0 if total > 0 else 0.0

    fig = go.Figure(
        go.Pie(
            labels=["Legitimate (0)", "Fraudulent (1)"],
            values=[legit_count, fraud_count],
            hole=0.68,
            marker={"colors": [COLOR_BLUE, COLOR_HIGH], "line": {"color": "#0F172A", "width": 2}},
            textinfo="label+percent",
            textposition="inside",
            insidetextfont={"color": "#FFFFFF", "size": 10, "family": "Inter"},
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"{title}<br><span style='font-size:0.75em;color:#94A3B8'>N = {total:,} | Fraud Prevalence: {fraud_pct:.2f}%</span>",
            "font": {"size": 12, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=260,
        margin={"l": 15, "r": 15, "t": 50, "b": 20},
        legend={"orientation": "h", "y": -0.15, "x": 0.5, "xanchor": "center", "font": {"size": 10, "color": "#94A3B8"}},
        annotations=[
            {
                "text": f"<b>{total:,}</b><br><span style='font-size:10px;color:#94A3B8'>Total Records</span>",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 15, "color": "#F8FAFC", "family": "Inter"},
                "showarrow": False,
            }
        ],
    )
    return fig


def create_evaluation_split_comparison_chart(
    train_legit: int, train_fraud: int, test_legit: int, test_fraud: int
) -> go.Figure:
    """Compares the distribution of classes across the 80/20 train/test split."""
    splits = ["Training Split (80%)", "Testing Split (20%)"]
    legit_counts = [train_legit, test_legit]
    fraud_counts = [train_fraud, test_fraud]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Legitimate (Class 0)",
            x=splits,
            y=legit_counts,
            marker={"color": "#3B82F6", "line": {"color": "rgba(255,255,255,0.1)", "width": 1}},
            text=[f"{c:,}" for c in legit_counts],
            textposition="auto",
            textfont={"size": 11, "color": "#FFFFFF"},
        )
    )
    fig.add_trace(
        go.Bar(
            name="Fraudulent (Class 1)",
            x=splits,
            y=fraud_counts,
            marker={"color": "#EF4444", "line": {"color": "rgba(255,255,255,0.1)", "width": 1}},
            text=[f"{c:,}" for c in fraud_counts],
            textposition="auto",
            textfont={"size": 11, "color": "#FFFFFF"},
            yaxis="y2",
        )
    )

    fig.update_layout(
        title={
            "text": "STRATIFIED TRAIN / TEST SPLIT COMPARISON<br><span style='font-size:0.75em;color:#94A3B8'>Preserves exact 0.79% fraud ratio across both partitions (random_state=42)</span>",
            "font": {"size": 12, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        barmode="group",
        height=260,
        margin={"l": 40, "r": 40, "t": 55, "b": 30},
        legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center", "font": {"size": 10, "color": "#94A3B8"}},
        xaxis={"tickfont": {"size": 11, "color": "#E2E8F0"}},
        yaxis={
            "title": "Legitimate Count",
            "title_font": {"size": 10, "color": "#60A5FA"},
            "tickfont": {"size": 10, "color": "#60A5FA"},
            "gridcolor": GRID_COLOR,
        },
        yaxis2={
            "title": "Fraud Count",
            "title_font": {"size": 10, "color": "#F87171"},
            "tickfont": {"size": 10, "color": "#F87171"},
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(239, 68, 68, 0.05)",
        },
    )
    return fig


def create_evaluation_class_probability_distribution_chart(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.50
) -> go.Figure:
    """Generates an executive probability distribution chart separated by true class."""
    legit_probas = y_proba[y_true == 0]
    fraud_probas = y_proba[y_true == 1]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=legit_probas,
            nbinsx=40,
            marker={"color": "#3B82F6", "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
            opacity=0.7,
            name=f"Actual Legitimate (N = {len(legit_probas):,})",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=fraud_probas,
            nbinsx=20,
            marker={"color": "#EF4444", "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
            opacity=0.85,
            name=f"Actual Fraudulent (N = {len(fraud_probas):,})",
        )
    )

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text=f"θ = {threshold:.2f}",
        annotation_position="top right",
        annotation_font_color="#FBBF24",
        annotation_font_size=11,
    )

    fig.update_layout(
        title={
            "text": f"PREDICTED PROBABILITY DISTRIBUTION BY GROUND TRUTH<br><span style='font-size:0.75em;color:#94A3B8'>Separation of legitimate transactions vs fraud signals (θ = {threshold:.2f})</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        barmode="overlay",
        height=280,
        margin={"l": 40, "r": 20, "t": 55, "b": 35},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center", "font": {"size": 10, "color": "#94A3B8"}},
        xaxis={
            "title": "Model Fraud Probability Score P(y=1)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [-0.02, 1.02],
        },
        yaxis={
            "title": "Transaction Count",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
        },
    )
    return fig


def create_evaluation_threshold_sweep_chart(multi_th_df: pd.DataFrame) -> go.Figure:
    """Plots Precision, Recall, and F1-Score trade-off curve across operational thresholds."""
    # Convert string percentages back to floats for clean plotting
    th_floats = [float(x) for x in multi_th_df["Threshold (θ)"]]
    precisions = [float(x.replace("%", "")) for x in multi_th_df["Precision"]]
    recalls = [float(x.replace("%", "")) for x in multi_th_df["Recall"]]
    f1s = [float(x) * 100.0 for x in multi_th_df["F1-Score"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=th_floats,
            y=precisions,
            mode="lines+markers",
            name="Fraud Precision",
            line={"color": "#10B981", "width": 2.5},
            marker={"size": 7},
            hovertemplate="θ: %{x:.2f}<br>Precision: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=th_floats,
            y=recalls,
            mode="lines+markers",
            name="Fraud Recall",
            line={"color": "#F59E0B", "width": 2.5},
            marker={"size": 7},
            hovertemplate="θ: %{x:.2f}<br>Recall: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=th_floats,
            y=f1s,
            mode="lines+markers",
            name="Fraud F1-Score",
            line={"color": "#38BDF8", "width": 2.5},
            marker={"size": 7},
            hovertemplate="θ: %{x:.2f}<br>F1-Score: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "THRESHOLD SENSITIVITY TRADE-OFF ANALYSIS<br><span style='font-size:0.75em;color:#94A3B8'>Precision vs Recall balance across operational cutoffs θ ∈ [0.30, 0.70]</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=280,
        margin={"l": 40, "r": 20, "t": 55, "b": 35},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center", "font": {"size": 10, "color": "#94A3B8"}},
        xaxis={
            "title": "Decision Threshold (θ)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
        },
        yaxis={
            "title": "Metric (%)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [0, 105],
        },
    )
    return fig


def create_explainability_waterfall_chart(explanation: Dict[str, Any]) -> go.Figure:
    """Generates an intuitive horizontal bar chart of Tree SHAP feature contributions."""
    if not explanation or "top_positive" not in explanation:
        fig = go.Figure()
        fig.update_layout(
            title={"text": "EXPLAINABILITY UNAVAILABLE", "font": {"size": 12, "color": "#94A3B8"}},
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR, height=220,
        )
        return fig

    pos_items = explanation.get("top_positive", [])
    neg_items = explanation.get("top_negative", [])

    features = []
    values = []
    colors = []

    for item in reversed(neg_items):
        features.append(item["feature"])
        values.append(item["contribution"])
        colors.append(COLOR_LOW)

    for item in pos_items:
        features.append(item["feature"])
        values.append(item["contribution"])
        colors.append(COLOR_HIGH)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.2)", "width": 1}},
            hovertemplate="Feature: %{y}<br>SHAP Contribution: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "DECISION EXPLAINABILITY (TREE SHAP CONTRIBUTIONS)<br><span style='font-size:0.75em;color:#94A3B8'>Red: Increases Fraud Risk | Green: Reduces Fraud Risk</span>",
            "font": {"size": 12, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=280,
        margin={"l": 60, "r": 20, "t": 50, "b": 35},
        xaxis={
            "title": "Log-Odds Margin Contribution",
            "title_font": {"size": 10, "color": "#94A3B8"},
            "tickfont": {"size": 9, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "zeroline": True,
            "zerolinecolor": "rgba(255,255,255,0.3)",
        },
        yaxis={"tickfont": {"size": 10, "color": "#F1F5F9"}},
    )
    return fig


def create_calibration_curve_chart(
    prob_true: np.ndarray, prob_pred: np.ndarray, brier_score: float
) -> go.Figure:
    """Generates a probability calibration reliability curve against the perfectly calibrated diagonal."""
    fig = go.Figure()

    # Ideal calibration line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line={"color": "rgba(255, 255, 255, 0.4)", "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )

    # Empirical calibration curve
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name=f"Tuned XGBoost (Brier={brier_score:.4f})",
            line={"color": "#06B6D4", "width": 2.5},
            marker={"size": 8, "color": "#06B6D4", "line": {"color": "#FFFFFF", "width": 1.5}},
            hovertemplate="Mean Pred Prob: %{x:.3f}<br>Observed True Fraction: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"PROBABILITY CALIBRATION DIAGRAM<br><span style='font-size:0.75em;color:#94A3B8'>Reliability curve vs Perfect Calibration | Brier Score: {brier_score:.6f}</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=320,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center", "font": {"size": 10, "color": "#94A3B8"}},
        xaxis={
            "title": "Mean Predicted Fraud Probability",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [-0.02, 1.02],
        },
        yaxis={
            "title": "Fraction of Actual Frauds",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [-0.02, 1.02],
        },
    )
    return fig


def create_model_comparison_bar_chart(manifest: Dict[str, Any]) -> go.Figure:
    """Generates comparative grouped bar chart across candidate models."""
    models = manifest.get("models", [])
    if not models:
        fig = go.Figure()
        fig.update_layout(title="No model comparison records found", paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR)
        return fig

    names = [m["model_name"] for m in models]
    test_recalls = [m["independent_test"]["recall"] * 100.0 for m in models]
    test_precisions = [m["independent_test"]["precision"] * 100.0 for m in models]
    test_f1s = [m["independent_test"]["f1_score"] * 100.0 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=test_recalls, name="Test Recall (%)", marker_color="#F59E0B"))
    fig.add_trace(go.Bar(x=names, y=test_precisions, name="Test Precision (%)", marker_color="#10B981"))
    fig.add_trace(go.Bar(x=names, y=test_f1s, name="Test F1-Score (%)", marker_color="#38BDF8"))

    fig.update_layout(
        title={
            "text": "MULTI-MODEL PERFORMANCE BENCHMARK COMPARISON<br><span style='font-size:0.75em;color:#94A3B8'>Held-Out Test Set Performance (Threshold θ = 0.50)</span>",
            "font": {"size": 13, "color": "#F1F5F9", "family": "Inter"},
        },
        barmode="group",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=340,
        margin={"l": 40, "r": 20, "t": 60, "b": 60},
        legend={"orientation": "h", "y": -0.3, "x": 0.5, "xanchor": "center", "font": {"size": 11, "color": "#94A3B8"}},
        xaxis={"tickfont": {"size": 10, "color": "#F1F5F9"}},
        yaxis={
            "title": "Score (%)",
            "title_font": {"size": 11, "color": "#94A3B8"},
            "tickfont": {"size": 10, "color": "#94A3B8"},
            "gridcolor": GRID_COLOR,
            "range": [0, 105],
        },
    )
    return fig


def create_model_comparison_radar_chart(manifest: Dict[str, Any]) -> go.Figure:
    """Generates multi-metric radar chart for candidate models."""
    models = manifest.get("models", [])
    categories = ["Recall", "Precision", "F1-Score", "ROC-AUC", "PR-AUC"]
    palette = ["#38BDF8", "#F59E0B", "#10B981", "#EC4899", "#8B5CF6"]

    fig = go.Figure()
    for idx, m in enumerate(models):
        t = m["independent_test"]
        values = [t["recall"], t["precision"], t["f1_score"], t["roc_auc"], t["pr_auc"]]
        # Close radar loop
        values.append(values[0])
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=f"rgba({int(palette[idx%len(palette)][1:3],16)}, {int(palette[idx%len(palette)][3:5],16)}, {int(palette[idx%len(palette)][5:7],16)}, 0.12)",
                line={"color": palette[idx % len(palette)], "width": 2},
                name=m["model_name"],
            )
        )

    fig.update_layout(
        polar={
            "bgcolor": "rgba(15, 23, 42, 0.5)",
            "radialaxis": {"visible": True, "range": [0, 1.0], "tickfont": {"size": 8, "color": "#94A3B8"}},
            "angularaxis": {"tickfont": {"size": 10, "color": "#F1F5F9"}},
        },
        paper_bgcolor=BG_COLOR,
        height=340,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center", "font": {"size": 9, "color": "#94A3B8"}},
    )
    return fig


