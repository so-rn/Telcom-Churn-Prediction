"""
Telcom Churn - premium Streamlit app (layer 2 on the trained model).

Run from the project root:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import streamlit as st

import core
from theme import inject_css, register_plotly_template
from views import dashboard, insights, scorer, intelligence, business

st.set_page_config(
    page_title="Telcom Churn",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
register_plotly_template()

# Pre-warm the heavy caches once per session behind a branded spinner, so the
# first page the presenter opens is instant instead of looking frozen.
if "warmed" not in st.session_state:
    with st.spinner("⚡ Booting the churn engine — loading model and scoring the customer base…"):
        core.load_models()
        core.load_json_artifacts()
        core.load_scored_population()
    st.session_state["warmed"] = True

# --- Pages (default nav hidden; we render a custom sidebar below) -----------
pages = [
    st.Page(dashboard.render,    title="Executive Dashboard", icon=":material/dashboard:",
            url_path="dashboard", default=True),
    st.Page(insights.render,     title="Churn Insights", icon=":material/query_stats:",
            url_path="insights"),
    st.Page(scorer.render,       title="Customer Risk Scorer", icon=":material/person_search:",
            url_path="scorer"),
    st.Page(intelligence.render, title="Model Intelligence",   icon=":material/insights:",
            url_path="intelligence"),
    st.Page(business.render,     title="Business Impact & ROI", icon=":material/payments:",
            url_path="business"),
]
nav = st.navigation(pages, position="hidden")

# --- Custom sidebar ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-logo"><span class="brand-mono">TC</span></div>
          <div>
            <div class="brand-title">Telcom Churn</div>
            <div class="brand-sub">Retention Intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider" style="margin:.6rem 0 1rem">', unsafe_allow_html=True)

    st.markdown('<div class="side-label">Navigate</div>', unsafe_allow_html=True)
    for p in pages:
        st.page_link(p)

    st.markdown('<hr class="divider" style="margin:1rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Engine</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="badges">
          <span class="chip">LightGBM</span>
          <span class="chip">SMOTE</span>
          <span class="chip v">Calibrated</span>
          <span class="chip v">SHAP-explained</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider" style="margin:1rem 0 .8rem">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="credit-pin">
          <div class="credit-card">
            <div class="avatar">K</div>
            <div>
              <div class="credit-name">Built by Karzan</div>
              <div class="credit-role">Telcom Churn Prediction &middot; 2026</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

nav.run()
