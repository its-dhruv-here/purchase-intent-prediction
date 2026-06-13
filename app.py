"""
E-Commerce Conversion Intelligence Platform — Streamlit Application.

Enterprise-grade session analytics and conversion prediction engine backed by a
serialized scikit-learn pipeline.

Usage:
    streamlit run app.py
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Conversion Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG         = "#0E1117"
SURFACE    = "#161B22"
CARD       = "#1C2333"
BORDER     = "#30363D"
ACCENT     = "#58A6FF"
ACCENT2    = "#3FB950"
WARN       = "#D29922"
DANGER     = "#F85149"
TEXT       = "#E6EDF3"
TEXT_DIM   = "#8B949E"
TEXT_FAINT = "#484F58"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

/* ---- main area ---- */
.stApp {{
    background: {BG};
}}
.main .block-container {{
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}}

/* ---- sidebar compact ---- */
section[data-testid="stSidebar"] {{
    background: {SURFACE};
    width: 220px !important;
    min-width: 220px !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    padding: 1rem 0.75rem;
}}

/* ---- tabs ---- */
button[data-baseweb="tab"] {{
    color: {TEXT_DIM} !important;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    padding: 8px 16px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom-color: {ACCENT} !important;
}}

/* ---- KPI card ---- */
.kpi {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 14px 12px 12px;
    text-align: center;
}}
.kpi-val {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1.15;
}}
.kpi-lbl {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}}

/* ---- result card ---- */
.res-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 6px;
    padding: 18px 16px;
}}

/* ---- misc ---- */
.section-hdr {{
    font-size: 0.75rem;
    font-weight: 700;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}}
hr {{
    border-color: {BORDER} !important;
    margin: 12px 0 !important;
}}

/* hide default header/footer */
header[data-testid="stHeader"] {{
    background: {BG};
}}

/* button overrides */
.stButton > button {{
    background: {ACCENT};
    color: #0D1117;
    font-weight: 600;
    border: none;
    border-radius: 4px;
}}
.stButton > button:hover {{
    background: #79C0FF;
}}

/* dataframe */
.stDataFrame {{
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load(os.path.join("models", "pipeline.pkl"))
    except Exception as e:
        logger.error("Pipeline load failed: %s", e)
        return None

@st.cache_data
def load_metrics() -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join("reports", "metrics.json")) as f:
            return json.load(f)
    except Exception:
        return None

pipeline = load_pipeline()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Compact sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<p style='font-size:0.9rem;font-weight:700;color:{TEXT};margin:0;'>🛒 CIP</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_DIM};margin-top:2px;'>Conversion Intelligence Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    if metrics:
        bm = metrics.get("best_model", {})
        ds = metrics.get("dataset", {})
        st.markdown(f"<p class='section-hdr'>Production Model</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.78rem;color:{TEXT};margin:0;'>{bm.get('Model','—')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.72rem;color:{TEXT_DIM};margin:2px 0 12px;'>{bm.get('Accuracy',0)*100:.1f}% acc · {bm.get('Weighted F1',0)*100:.1f}% F1</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='section-hdr'>Dataset</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.72rem;color:{TEXT_DIM};margin:0;'>{ds.get('Total Rows',0):,} sessions</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<p style='font-size:0.62rem;color:{TEXT_FAINT};'>v1.0 · scikit-learn · Streamlit</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex; align-items:baseline; gap:10px; margin-bottom:4px;">
    <span style="font-size:1.25rem; font-weight:800; color:{TEXT};">E-Commerce Conversion Intelligence Platform</span>
    <span style="font-size:0.72rem; color:{TEXT_DIM}; font-weight:500;">Session Purchase Intent Engine</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Executive KPI row
# ---------------------------------------------------------------------------
if metrics:
    bm = metrics["best_model"]
    ds = metrics["dataset"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, f"{bm.get('Accuracy',0)*100:.1f}%", "Accuracy"),
        (k2, f"{bm.get('Weighted F1',0)*100:.1f}%", "Weighted F1"),
        (k3, f"{bm.get('ROC AUC',0):.3f}", "ROC AUC"),
        (k4, f"{bm.get('Precision',0)*100:.1f}%", "Precision"),
        (k5, f"{ds.get('Total Rows',0):,}", "Training Sessions"),
        (k6, f"18", "Features Extracted"),
    ]
    for col, val, lbl in kpis:
        col.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_infer, tab_analytics, tab_arch = st.tabs(["INFERENCE ENGINE", "ANALYTICS DASHBOARD", "ARCHITECTURE & DATA"])

# =====================================================================
# TAB 1 — INFERENCE ENGINE
# =====================================================================
with tab_infer:
    # ---- Input row ----
    st.markdown(f'<div class="section-hdr">Session Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        admin = st.number_input("Administrative Pages", min_value=0, value=2, step=1)
        admin_dur = st.number_input("Admin Duration (s)", min_value=0.0, value=45.0, step=10.0)
        info = st.number_input("Informational Pages", min_value=0, value=0, step=1)
        info_dur = st.number_input("Info Duration (s)", min_value=0.0, value=0.0, step=10.0)
    with col2:
        prod = st.number_input("Product Related Pages", min_value=0, value=15, step=1)
        prod_dur = st.number_input("Product Duration (s)", min_value=0.0, value=650.0, step=50.0)
        bounce = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        exit_r = st.number_input("Exit Rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    with col3:
        page_val = st.number_input("Page Values", min_value=0.0, value=12.5, step=1.0)
        special = st.number_input("Special Day Proximity", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        month = st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=8)
        visitor = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'], index=0)
    with col4:
        os_val = st.number_input("Operating System", min_value=1, value=2, step=1)
        browser = st.number_input("Browser", min_value=1, value=2, step=1)
        region = st.number_input("Region", min_value=1, value=1, step=1)
        traffic = st.number_input("Traffic Type", min_value=1, value=2, step=1)
        weekend = st.checkbox("Weekend Session", value=False)

    predict_btn = st.button("▶ Run Session Inference", use_container_width=True)

    if predict_btn:
        if pipeline is None:
            st.error("Model not loaded.")
        else:
            t0 = time.time()
            input_dict = {
                'Administrative': [admin], 'Administrative_Duration': [admin_dur],
                'Informational': [info], 'Informational_Duration': [info_dur],
                'ProductRelated': [prod], 'ProductRelated_Duration': [prod_dur],
                'BounceRates': [bounce], 'ExitRates': [exit_r], 'PageValues': [page_val],
                'SpecialDay': [special], 'Month': [month], 'OperatingSystems': [os_val],
                'Browser': [browser], 'Region': [region], 'TrafficType': [traffic],
                'VisitorType': [visitor], 'Weekend': [weekend]
            }
            df_in = pd.DataFrame(input_dict)
            
            probas = pipeline.predict_proba(df_in)[0]
            conv_prob = probas[1] * 100
            pred_class = pipeline.predict(df_in)[0]
            latency_ms = (time.time() - t0) * 1000

            if conv_prob >= 75:
                tier_label = "HIGH INTENT"
                tier_color = ACCENT2
                action = "No intervention needed."
            elif conv_prob >= 35:
                tier_label = "MEDIUM INTENT"
                tier_color = WARN
                action = "Deploy targeted discount offer."
            else:
                tier_label = "LOW INTENT"
                tier_color = DANGER
                action = "Trigger re-engagement drip campaign."

            st.markdown("---")
            r1, r2 = st.columns([2, 3])

            with r1:
                st.markdown(f"""
                <div class="res-card">
                    <div class="section-hdr">Inference Result</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{TEXT}; margin:6px 0 4px;">{"Purchase Projected" if pred_class else "Abandonment Projected"}</div>
                    <div style="display:flex; gap:16px; margin-top:10px;">
                        <div>
                            <div class="kpi-lbl">Conversion Probability</div>
                            <div style="font-size:1.1rem; font-weight:700; color:{ACCENT};">{conv_prob:.1f}%</div>
                        </div>
                        <div>
                            <div class="kpi-lbl">Intent Tier</div>
                            <div style="font-size:0.85rem; font-weight:700; color:{tier_color};">{tier_label}</div>
                        </div>
                        <div>
                            <div class="kpi-lbl">Latency</div>
                            <div style="font-size:0.85rem; font-weight:600; color:{TEXT_DIM};">{latency_ms:.0f}ms</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with r2:
                 st.markdown(f"""
                <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:4px; padding:10px 14px; height: 100%;">
                    <strong style="color:{TEXT_DIM}; font-size: 0.75rem; text-transform: uppercase;">Automated System Action</strong>
                    <p style="margin-top: 8px; font-weight: 500; font-size: 0.9rem;">{action}</p>
                </div>
                """, unsafe_allow_html=True)


# =====================================================================
# TAB 2 — ANALYTICS DASHBOARD
# =====================================================================
with tab_analytics:
    if not metrics:
        st.info("Run `python src/train.py` to generate analytics data.")
    else:
        bm = metrics["best_model"]
        ds = metrics["dataset"]

        # ---- Metrics cards ----
        st.markdown(f'<div class="section-hdr">Production Model Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        metric_items = [
            (m1, "Accuracy",    bm.get("Accuracy", 0)),
            (m2, "Precision",   bm.get("Precision", 0)),
            (m3, "Recall",      bm.get("Recall", 0)),
            (m4, "Weighted F1", bm.get("Weighted F1", 0)),
            (m5, "ROC AUC",     bm.get("ROC AUC", 0)),
            (m6, "Inference Time (ms)", bm.get("Inference Time (s)", 0) * 1000),
        ]
        for col, label, val in metric_items:
            display = f"{val*100:.1f}%" if "Time" not in label and "ROC" not in label else (f"{val:.3f}" if "ROC" in label else f"{val:.1f}")
            col.markdown(f'<div class="kpi"><div class="kpi-val">{display}</div><div class="kpi-lbl">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ---- Benchmark table ----
        st.markdown(f'<div class="section-hdr">Model Benchmark Comparison</div>', unsafe_allow_html=True)
        if "benchmarks" in metrics:
            df_b = pd.DataFrame(metrics["benchmarks"])
            st.dataframe(
                df_b.style
                    .highlight_max(subset=["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1", "ROC AUC"], color="#1a3a2a")
                    .format({
                        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                        "Macro F1": "{:.4f}", "Weighted F1": "{:.4f}", "ROC AUC": "{:.4f}",
                        "Train Time (s)": "{:.2f}", "Inference Time (s)": "{:.4f}",
                    }),
                use_container_width=True, hide_index=True, height=150,
            )

        st.markdown("---")

        # ---- Charts ----
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown(f'<div class="section-hdr">Benchmark Comparison</div>', unsafe_allow_html=True)
            p = os.path.join("reports", "figures", "benchmark_comparison.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)
        with ch2:
            st.markdown(f'<div class="section-hdr">Confusion Matrix (Production Model)</div>', unsafe_allow_html=True)
            # Find the best model's confusion matrix
            best_model_file = f"cm_{bm.get('Model').replace(' ', '_').lower()}.png"
            p = os.path.join("reports", "figures", best_model_file)
            if os.path.exists(p):
                st.image(p, use_container_width=True)

# =====================================================================
# TAB 3 — ARCHITECTURE & DATA
# =====================================================================
with tab_arch:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f'<div class="section-hdr">Machine Learning Pipeline</div>', unsafe_allow_html=True)
        pipe_img = os.path.join("reports", "pipeline_diagram.png")
        if os.path.exists(pipe_img):
            st.image(pipe_img, use_container_width=True)
        else:
            st.info("Pipeline diagram not generated.")
            
    with a2:
        st.markdown(f'<div class="section-hdr">System Architecture</div>', unsafe_allow_html=True)
        arch_img = os.path.join("reports", "architecture_diagram.png")
        if os.path.exists(arch_img):
            st.image(arch_img, use_container_width=True)
        else:
            st.info("Architecture diagram not generated.")
    
    st.markdown("---")
    st.markdown(f'<div class="section-hdr">Explainability (SHAP Analysis)</div>', unsafe_allow_html=True)
    shap_img = os.path.join("reports", "figures", "shap_summary.png")
    if os.path.exists(shap_img):
        st.image(shap_img, use_container_width=False, width=800)
    else:
        st.info("SHAP diagram not generated.")
