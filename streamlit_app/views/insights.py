"""Churn Insights - the core EDA: why customers actually leave."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core
from theme import COLORS, hero, kpi_card, style_fig


def _rate_bar(df, col: str, title: str) -> go.Figure:
    """Horizontal churn-rate bar for one categorical driver."""
    g = core.churn_rate_by(df, col)
    fig = go.Figure(go.Bar(
        x=g["rate"], y=g[col].astype(str), orientation="h",
        marker=dict(color=g["rate"],
                    colorscale=[[0, COLORS["green"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
                    line=dict(width=0)),
        text=[f"{v:.0f}%" for v in g["rate"]], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% churn<extra></extra>"))
    fig.update_layout(xaxis_title="Churn rate (%)", yaxis_title="",
                      xaxis=dict(range=[0, max(g["rate"]) * 1.18]))
    return style_fig(fig, 250)


def render() -> None:
    df = core.load_raw_with_churn()
    base = df["churn"].mean() * 100

    st.markdown(hero(
        "The core analysis",
        'Why customers <span class="grad-text">leave</span>',
        "Churn rate broken down by the factors that matter — the story behind the model.",
    ), unsafe_allow_html=True)

    # --- Headline drivers as KPIs -------------------------------------------
    m2m = df[df["Contract"] == "Month-to-month"]["churn"].mean() * 100
    fiber = df[df["InternetService"] == "Fiber optic"]["churn"].mean() * 100
    echeck = df[df["PaymentMethod"] == "Electronic check"]["churn"].mean() * 100
    new6 = df[df["tenure"] <= 6]["churn"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Overall churn", f"{base:.1f}%", color="cyan",
                         delta="baseline", icon="groups"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Month-to-month", f"{m2m:.0f}%", color="red",
                         delta=f"{m2m/base:.1f}× baseline", icon="event_repeat"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("New (≤6 mo)", f"{new6:.0f}%", color="amber",
                         delta=f"{new6/base:.1f}× baseline", icon="hourglass_top"),
                unsafe_allow_html=True)
    c4.markdown(kpi_card("Electronic check", f"{echeck:.0f}%", color="violet",
                         delta=f"{echeck/base:.1f}× baseline", icon="receipt_long"),
                unsafe_allow_html=True)
    from theme import animate_counters
    animate_counters()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- Driver bars (the main analysis) ------------------------------------
    st.markdown('<div class="section">Churn rate by key driver</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.caption("Contract type")
        st.plotly_chart(_rate_bar(df, "Contract", "Contract"), width="stretch")
        st.caption("Internet service")
        st.plotly_chart(_rate_bar(df, "InternetService", "Internet"), width="stretch")
    with r1c2:
        st.caption("Tenure group")
        st.plotly_chart(_rate_bar(df, "tenure_group", "Tenure"), width="stretch")
        st.caption("Payment method")
        st.plotly_chart(_rate_bar(df, "PaymentMethod", "Payment"), width="stretch")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- Distributions: churned vs retained ---------------------------------
    st.markdown('<div class="section">How leavers differ — distributions</div>',
                unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    def _overlap_hist(field: str, title: str, xlab: str) -> go.Figure:
        stay = df[df["churn"] == 0][field].dropna()
        leave = df[df["churn"] == 1][field].dropna()
        fig = go.Figure()
        fig.add_histogram(x=stay, name="Stayed", opacity=0.75, nbinsx=30,
                          marker=dict(color=COLORS["green"]))
        fig.add_histogram(x=leave, name="Churned", opacity=0.7, nbinsx=30,
                          marker=dict(color=COLORS["red"]))
        fig.update_layout(barmode="overlay", xaxis_title=xlab, yaxis_title="Customers",
                          legend=dict(orientation="h", y=1.12))
        return style_fig(fig, 300)

    with d1:
        st.caption("Tenure (months)")
        st.plotly_chart(_overlap_hist("tenure", "Tenure", "Tenure (months)"),
                        width="stretch")
    with d2:
        st.caption("Monthly charges ($)")
        st.plotly_chart(_overlap_hist("MonthlyCharges", "Monthly charges", "Monthly charges ($)"),
                        width="stretch")

    # --- Key takeaways -------------------------------------------------------
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass" style="border-left:3px solid {COLORS['cyan']}">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
               font-weight:600;margin-bottom:.6rem">&#128161; Key takeaways</div>
          <ul style="margin:0;padding-left:1.1rem;line-height:1.9;color:{COLORS['text']}">
            <li><b style="color:{COLORS['red']}">Month-to-month contracts</b> churn ~{m2m/base:.1f}× the
                average — the single strongest signal. Long contracts barely churn.</li>
            <li><b style="color:{COLORS['amber']}">New customers (≤6 months)</b> leave far more; churn
                drops sharply with tenure — onboarding is critical.</li>
            <li><b style="color:{COLORS['violet']}">Fiber-optic + electronic-check</b> customers are
                high-risk — price sensitivity and friction-heavy billing.</li>
            <li>Leavers cluster at <b>high monthly charges & low tenure</b> — the "expensive newcomer" profile.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True)
    st.caption("Exploratory analysis on the full dataset — these patterns are what the model learns to detect.")
