"""Business Impact & ROI - turn model scores into money, interactively."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core
from theme import COLORS, hero, kpi_card, style_fig


def _economics(prob, actual, t, cost_fn, cost_fp, success):
    pred = prob >= t
    tp = int(np.sum(pred & (actual == 1)))
    fp = int(np.sum(pred & (actual == 0)))
    fn = int(np.sum(~pred & (actual == 1)))
    cost = fn * cost_fn + fp * cost_fp
    revenue_saved = tp * cost_fn * success
    retention_spend = (tp + fp) * cost_fp
    net = revenue_saved - retention_spend
    roi = (net / retention_spend * 100) if retention_spend else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, cost=cost, revenue_saved=revenue_saved,
                retention_spend=retention_spend, net=net, roi=roi, recall=recall)


def render() -> None:
    df = core.load_scored_population()
    arts = core.load_json_artifacts()
    cfg = arts["config"]
    opt_thr = arts["threshold"].get("threshold", 0.5)

    prob = df["churn_probability"].values
    actual = df["actual_churn"].values

    st.markdown(hero(
        "From probabilities to profit",
        'The <span class="grad-text">business case</span>, in your hands',
        "Move the levers; the retention economics recompute live across the customer base.",
    ), unsafe_allow_html=True)

    # --- Controls ------------------------------------------------------------
    with st.container():
        st.markdown('<div class="section">Assumptions</div>', unsafe_allow_html=True)
        a, b, c, d = st.columns(4)
        t = a.slider("Decision threshold", 0.05, 0.95, float(opt_thr), 0.01)
        cost_fn = b.slider("Cost of losing a customer ($)", 100, 3000,
                           int(cfg.get("cost_fn", 500)), 50)
        cost_fp = c.slider("Cost of a retention contact ($)", 10, 500,
                           int(cfg.get("cost_fp", 100)), 10)
        success = d.slider("Retention success rate", 0.1, 1.0,
                           float(cfg.get("retention_success_rate", 0.7)), 0.05)

    e = _economics(prob, actual, t, cost_fn, cost_fp, success)
    eo = _economics(prob, actual, opt_thr, cost_fn, cost_fp, success)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- KPI row -------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_card("Churners caught", f"{e['tp']:,}",
                         color="cyan", delta=f"recall {e['recall']:.0%}"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Revenue saved", f"${e['revenue_saved']/1e6:.2f}M",
                         color="green"), unsafe_allow_html=True)
    k3.markdown(kpi_card("Retention spend", f"${e['retention_spend']/1e3:.0f}K",
                         color="amber"), unsafe_allow_html=True)
    k4.markdown(kpi_card("Net savings", f"${e['net']/1e6:.2f}M",
                         color="violet"), unsafe_allow_html=True)
    k5.markdown(kpi_card("ROI", f"{e['roi']:.0f}%",
                         color="green" if e["roi"] >= 0 else "red"), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.1])

    # --- Waterfall -----------------------------------------------------------
    with left:
        st.markdown('<div class="section">Where the money goes</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Revenue saved", "Retention spend", "Net savings"],
            y=[e["revenue_saved"], -e["retention_spend"], 0],
            connector=dict(line=dict(color="rgba(255,255,255,0.2)")),
            increasing=dict(marker=dict(color=COLORS["green"])),
            decreasing=dict(marker=dict(color=COLORS["red"])),
            totals=dict(marker=dict(color=COLORS["cyan"])),
            text=[f"${e['revenue_saved']/1e6:.2f}M", f"-${e['retention_spend']/1e3:.0f}K",
                  f"${e['net']/1e6:.2f}M"], textposition="outside"))
        fig.update_layout(yaxis_title="USD", showlegend=False)
        st.plotly_chart(style_fig(fig, 360), width="stretch")

    # --- Cost vs threshold curve --------------------------------------------
    with right:
        st.markdown('<div class="section">Net savings vs decision threshold</div>',
                    unsafe_allow_html=True)
        grid = np.arange(0.05, 0.96, 0.02)
        nets = [_economics(prob, actual, g, cost_fn, cost_fp, success)["net"] for g in grid]
        fig = go.Figure()
        fig.add_scatter(x=grid, y=np.array(nets) / 1e6, mode="lines",
                        line=dict(color=COLORS["cyan"], width=3),
                        fill="tozeroy", fillcolor="rgba(0,229,255,0.08)",
                        hovertemplate="thr %{x:.2f}: $%{y:.2f}M<extra></extra>")
        # Annotations placed top vs bottom so they never collide, even when the
        # two thresholds coincide.
        fig.add_vline(x=t, line=dict(color=COLORS["violet"], width=2, dash="dash"),
                      annotation_text=f"your {t:.2f}", annotation_position="top left",
                      annotation_font_color=COLORS["violet"])
        fig.add_vline(x=opt_thr, line=dict(color=COLORS["green"], width=1.5, dash="dot"),
                      annotation_text=f"optimal {opt_thr:.2f}", annotation_position="bottom right",
                      annotation_font_color=COLORS["green"])
        fig.update_layout(xaxis_title="Decision threshold", yaxis_title="Net savings ($M)")
        st.plotly_chart(style_fig(fig, 360), width="stretch")

    # --- Comparison note -----------------------------------------------------
    delta = e["net"] - eo["net"]
    msg = ("matches" if abs(delta) < 1 else
           (f"beats by ${delta/1e3:,.0f}K" if delta > 0 else
            f"trails by ${-delta/1e3:,.0f}K"))
    color = COLORS["green"] if delta >= 0 else COLORS["amber"]
    st.markdown(
        f"""<div class="glass" style="border-left:3px solid {color}">
              Your threshold of <b>{t:.2f}</b> yields <b>${e['net']/1e6:.2f}M</b> net savings &mdash;
              this <b style="color:{color}">{msg}</b> the model-optimal threshold of
              <b>{opt_thr:.2f}</b> (${eo['net']/1e6:.2f}M).
              The optimal point is chosen to minimise business cost while meeting the recall target.
            </div>""",
        unsafe_allow_html=True)
    st.caption("Economics computed across the full active base for interactivity. "
               "Headline out-of-sample numbers live on Model Intelligence.")
