"""Customer Risk Scorer - the interactive heart of the app."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core
from theme import COLORS, TIER_COLORS, hero, tier_badge, style_fig


def _seed_state() -> None:
    """Initialise the form fields once from a balanced default preset."""
    if "f_tenure" in st.session_state:
        return
    for k, v in core.PRESETS["Borderline mid-tenure"].items():
        st.session_state[f"f_{k}"] = v


def _apply_preset(name: str) -> None:
    for k, v in core.PRESETS[name].items():
        st.session_state[f"f_{k}"] = v


def _collect_inputs() -> dict:
    return {
        "gender": st.session_state["f_gender"],
        "SeniorCitizen": st.session_state["f_SeniorCitizen"],
        "Partner": st.session_state["f_Partner"],
        "Dependents": st.session_state["f_Dependents"],
        "tenure": st.session_state["f_tenure"],
        "PhoneService": st.session_state["f_PhoneService"],
        "MultipleLines": st.session_state["f_MultipleLines"],
        "InternetService": st.session_state["f_InternetService"],
        "OnlineSecurity": st.session_state["f_OnlineSecurity"],
        "OnlineBackup": st.session_state["f_OnlineBackup"],
        "DeviceProtection": st.session_state["f_DeviceProtection"],
        "TechSupport": st.session_state["f_TechSupport"],
        "StreamingTV": st.session_state["f_StreamingTV"],
        "StreamingMovies": st.session_state["f_StreamingMovies"],
        "Contract": st.session_state["f_Contract"],
        "PaperlessBilling": st.session_state["f_PaperlessBilling"],
        "PaymentMethod": st.session_state["f_PaymentMethod"],
        "MonthlyCharges": st.session_state["f_MonthlyCharges"],
        "TotalCharges": st.session_state["f_TotalCharges"],
    }


def _gauge(prob: float, tier: str) -> go.Figure:
    color = TIER_COLORS[tier]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(size=46, family="Space Grotesk", color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=COLORS["muted"],
                      tickfont=dict(color=COLORS["muted"], size=11)),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(255,255,255,0.03)", borderwidth=0,
            steps=[
                dict(range=[0, 30], color="rgba(46,212,122,0.18)"),
                dict(range=[30, 55], color="rgba(255,181,71,0.18)"),
                dict(range=[55, 75], color="rgba(255,138,76,0.20)"),
                dict(range=[75, 100], color="rgba(255,92,124,0.22)"),
            ],
            threshold=dict(line=dict(color=COLORS["text"], width=3), thickness=0.78,
                           value=prob * 100),
        ),
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=0))
    return fig


def _drivers_chart(drivers) -> go.Figure:
    d = drivers.iloc[::-1]  # smallest at bottom for horizontal bar
    colors = [COLORS["red"] if s > 0 else COLORS["green"] for s in d["shap"]]
    labels = [core.humanize_feature(f) for f in d["feature"]]
    fig = go.Figure(go.Bar(
        x=d["shap"], y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}<br>impact: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
    fig.update_layout(xaxis_title="← lowers risk      raises risk →",
                      yaxis_title="", height=320)
    return fig


def render() -> None:
    _seed_state()

    st.markdown(hero(
        "Live risk scoring",
        'Score any customer in <span class="grad-text">real time</span>',
        "Adjust the profile on the left; the model re-scores instantly and explains why.",
    ), unsafe_allow_html=True)

    form_col, result_col = st.columns([1, 1.05], gap="large")

    # ----------------------------------------------------------------- inputs
    with form_col:
        st.markdown('<div class="section">Customer profile</div>', unsafe_allow_html=True)

        pc1, pc2 = st.columns([2, 1])
        preset = pc1.selectbox("Quick preset", list(core.PRESETS.keys()),
                               label_visibility="collapsed")
        if pc2.button("Apply preset", width="stretch"):
            _apply_preset(preset)
            st.rerun()

        with st.container():
            a, b = st.columns(2)
            a.selectbox("Gender", core.CATEGORICAL_FIELDS["gender"], key="f_gender")
            b.selectbox("Senior citizen", [0, 1], key="f_SeniorCitizen",
                        format_func=lambda x: "Yes" if x else "No")
            a.selectbox("Partner", core.CATEGORICAL_FIELDS["Partner"], key="f_Partner")
            b.selectbox("Dependents", core.CATEGORICAL_FIELDS["Dependents"], key="f_Dependents")

            a.selectbox("Contract", core.CATEGORICAL_FIELDS["Contract"], key="f_Contract")
            b.selectbox("Internet service", core.CATEGORICAL_FIELDS["InternetService"],
                        key="f_InternetService")
            a.selectbox("Payment method", core.CATEGORICAL_FIELDS["PaymentMethod"],
                        key="f_PaymentMethod")
            b.selectbox("Paperless billing", core.CATEGORICAL_FIELDS["PaperlessBilling"],
                        key="f_PaperlessBilling")

            st.slider("Tenure (months)", 0, 72, key="f_tenure")
            a, b = st.columns(2)
            a.number_input("Monthly charges ($)", 0.0, 200.0, step=1.0, key="f_MonthlyCharges")
            b.number_input("Total charges ($)", 0.0, 9000.0, step=10.0, key="f_TotalCharges")

            with st.expander("Add-on services"):
                s1, s2 = st.columns(2)
                s1.selectbox("Phone", core.CATEGORICAL_FIELDS["PhoneService"], key="f_PhoneService")
                s2.selectbox("Multiple lines", core.CATEGORICAL_FIELDS["MultipleLines"],
                             key="f_MultipleLines")
                s1.selectbox("Online security", core.CATEGORICAL_FIELDS["OnlineSecurity"],
                             key="f_OnlineSecurity")
                s2.selectbox("Online backup", core.CATEGORICAL_FIELDS["OnlineBackup"],
                             key="f_OnlineBackup")
                s1.selectbox("Device protection", core.CATEGORICAL_FIELDS["DeviceProtection"],
                             key="f_DeviceProtection")
                s2.selectbox("Tech support", core.CATEGORICAL_FIELDS["TechSupport"],
                             key="f_TechSupport")
                s1.selectbox("Streaming TV", core.CATEGORICAL_FIELDS["StreamingTV"],
                             key="f_StreamingTV")
                s2.selectbox("Streaming movies", core.CATEGORICAL_FIELDS["StreamingMovies"],
                             key="f_StreamingMovies")

    # ---------------------------------------------------------------- results
    with result_col:
        res = core.score_customer(_collect_inputs())
        prob, tier = res["probability"], res["tier"]

        verdict = "WILL LIKELY CHURN" if res["prediction"] else "LIKELY TO STAY"
        vcolor = COLORS["red"] if res["prediction"] else COLORS["green"]
        st.markdown(
            f"""<div class="glass" style="text-align:center;padding:1.1rem 1.2rem">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    {tier_badge(tier)}
                    <span style="color:{vcolor};font-weight:700;font-family:'Space Grotesk'">
                      {verdict}</span>
                  </div>
                </div>""",
            unsafe_allow_html=True)

        st.plotly_chart(style_fig(_gauge(prob, tier), 300), width="stretch")

        st.markdown('<div class="section">Why &mdash; top drivers (SHAP)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(style_fig(_drivers_chart(res["drivers"]), 320),
                        width="stretch")

        # Recommendation
        recs = {
            "Critical": "Immediate outreach. Offer a contract upgrade + loyalty credit within 48h.",
            "High": "Proactive call this week. Bundle tech support / security to raise stickiness.",
            "Medium": "Monitor. Trigger a satisfaction check-in and targeted offer next cycle.",
            "Low": "No action needed. Healthy, low-risk customer.",
        }
        st.markdown(
            f"""<div class="glass" style="border-left:3px solid {TIER_COLORS[tier]}">
                  <div style="color:{COLORS['muted']};font-size:.78rem;text-transform:uppercase;
                       letter-spacing:1px">Recommended action</div>
                  <div style="margin-top:.35rem;font-size:.98rem">{recs[tier]}</div>
                </div>""",
            unsafe_allow_html=True)
