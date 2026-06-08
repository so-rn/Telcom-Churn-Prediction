"""Executive Dashboard - the at-a-glance overview of churn risk."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core
from theme import COLORS, TIER_COLORS, animate_counters, hero, kpi_card, style_fig


def render() -> None:
    inject = st.markdown
    df = core.load_scored_population()
    arts = core.load_json_artifacts()
    biz = arts["business"]

    inject(hero(
        "Executive overview",
        'Who is about to <span class="grad-text">leave</span> &mdash; and what it costs',
        "Every active customer scored by the calibrated model, ranked by churn risk.",
    ), unsafe_allow_html=True)

    # --- KPI row -------------------------------------------------------------
    total = len(df)
    churn_rate = df["actual_churn"].mean() * 100
    flagged = int(df["predicted_churn"].sum())
    critical = int((df["risk_tier"] == "Critical").sum())
    annual_at_risk = df.loc[df["predicted_churn"] == 1, "MonthlyCharges"].sum() * 12
    roi = biz.get("roi_percent", 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("Active customers", f"{total:,}", color="cyan", icon="groups"),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("Observed churn rate", f"{churn_rate:.1f}%", color="violet", icon="trending_down"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("Flagged at risk", f"{flagged:,}",
                         color="amber", delta=f"{critical:,} critical", icon="warning"),
                unsafe_allow_html=True)
    c4.markdown(kpi_card("Annual revenue at risk", f"${annual_at_risk/1e6:.2f}M",
                         color="red", icon="payments"),
                unsafe_allow_html=True)
    c5.markdown(kpi_card("Retention ROI", f"{roi:.0f}%",
                         color="green", delta="at optimal threshold", icon="trending_up"),
                unsafe_allow_html=True)
    animate_counters()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- Charts row 1 --------------------------------------------------------
    left, right = st.columns([1, 1.25])

    with left:
        st.markdown('<div class="section">Risk-tier distribution</div>', unsafe_allow_html=True)
        order = ["Low", "Medium", "High", "Critical"]
        counts = df["risk_tier"].value_counts().reindex(order).fillna(0)
        fig = go.Figure(go.Pie(
            labels=order, values=counts.values, hole=0.62,
            marker=dict(colors=[TIER_COLORS[t] for t in order],
                        line=dict(color="#0B0E14", width=2)),
            textinfo="percent", textfont=dict(size=13),
            hovertemplate="%{label}: %{value:,} customers<extra></extra>",
        ))
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", y=-0.08),
                          annotations=[dict(text=f"{total:,}<br><span style='font-size:12px;color:{COLORS['muted']}'>customers</span>",
                                            x=0.5, y=0.5, font=dict(size=22, family="Space Grotesk"),
                                            showarrow=False)])
        st.plotly_chart(style_fig(fig, 340), width="stretch")

    with right:
        st.markdown('<div class="section">Churn-probability landscape</div>', unsafe_allow_html=True)
        prob = df["churn_probability"].values
        counts, edges = np.histogram(prob, bins=40, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure()
        fig.add_bar(
            x=centers, y=counts, width=(edges[1] - edges[0]) * 0.92,
            marker=dict(
                color=centers,
                colorscale=[[0, COLORS["green"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
                line=dict(width=0)),
            hovertemplate="churn P %{x:.2f}: %{y} customers<extra></extra>")
        thr = arts["threshold"].get("threshold", 0.5)
        fig.add_vline(x=thr, line=dict(color=COLORS["cyan"], width=2, dash="dash"),
                      annotation_text=f"  decision threshold {thr:.2f}",
                      annotation_font_color=COLORS["cyan"])
        fig.update_layout(bargap=0.04, xaxis_title="Calibrated churn probability",
                          yaxis_title="Customers")
        st.plotly_chart(style_fig(fig, 340), width="stretch")

    # --- Charts row 2 --------------------------------------------------------
    left2, right2 = st.columns([1.25, 1])

    with left2:
        st.markdown('<div class="section">Churn rate by contract type</div>', unsafe_allow_html=True)
        g = (df.groupby("Contract")["actual_churn"].mean() * 100).sort_values()
        fig = go.Figure(go.Bar(
            x=g.values, y=g.index, orientation="h",
            marker=dict(color=g.values, colorscale=[[0, COLORS["green"]], [1, COLORS["red"]]],
                        line=dict(width=0)),
            text=[f"{v:.0f}%" for v in g.values], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}% churn<extra></extra>"))
        fig.update_layout(xaxis_title="Churn rate (%)", yaxis_title="")
        st.plotly_chart(style_fig(fig, 300), width="stretch")

    with right2:
        st.markdown('<div class="section">Revenue exposure by tier</div>', unsafe_allow_html=True)
        rev = (df.assign(annual=df["MonthlyCharges"] * 12)
                 .groupby("risk_tier")["annual"].sum()
                 .reindex(["Low", "Medium", "High", "Critical"]).fillna(0) / 1e6)
        fig = go.Figure(go.Bar(
            x=rev.index, y=rev.values,
            marker=dict(color=[TIER_COLORS[t] for t in rev.index], line=dict(width=0)),
            text=[f"${v:.1f}M" for v in rev.values], textposition="outside",
            hovertemplate="%{x}: $%{y:.2f}M / yr<extra></extra>"))
        fig.update_layout(yaxis_title="Annual revenue ($M)", xaxis_title="")
        st.plotly_chart(style_fig(fig, 300), width="stretch")

    # --- Watchlist -----------------------------------------------------------
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    base = (df[df["risk_tier"].isin(["Critical", "High"])]
            .sort_values("churn_probability", ascending=False)
            .assign(annual_value=lambda d: (d["MonthlyCharges"] * 12).round(0)))
    st.markdown('<div class="section">&#128293; Priority watchlist</div>', unsafe_allow_html=True)

    # --- Filters -------------------------------------------------------------
    f1, f2, f3 = st.columns([1, 1.2, 1.5])
    tier_sel = f1.selectbox("Risk tier", ["All tiers", "Critical", "High"])
    contract_opts = ["All contracts"] + sorted(base["Contract"].unique().tolist())
    contract_sel = f2.selectbox("Contract", contract_opts)
    min_risk = f3.slider("Minimum churn risk", 0, 100, 0, 5, format="%d%%")

    view = base
    if tier_sel != "All tiers":
        view = view[view["risk_tier"] == tier_sel]
    if contract_sel != "All contracts":
        view = view[view["Contract"] == contract_sel]
    if min_risk > 0:
        view = view[view["churn_probability"] >= min_risk / 100]

    st.markdown(
        f'<div style="color:{COLORS["muted"]};font-size:.85rem;margin:.1rem 0 .5rem">'
        f'Showing <b style="color:{COLORS["text"]}">{len(view):,}</b> of {len(base):,} '
        f'flagged customers</div>', unsafe_allow_html=True)

    # --- Scrollable, theme-matched table ------------------------------------
    rows = ""
    for _, r in view.iterrows():
        p = float(r["churn_probability"])
        tier = r["risk_tier"]
        c = TIER_COLORS[tier]
        annual = r["MonthlyCharges"] * 12
        rows += f"""<tr>
          <td><div class="risk-wrap">
            <div class="risk-bar"><div class="risk-fill" style="width:{p*100:.0f}%;background:{c}"></div></div>
            <span class="risk-val" style="color:{c}">{p:.0%}</span></div></td>
          <td><span class="wl-badge" style="background:{c}1F;color:{c};border:1px solid {c}55">{tier}</span></td>
          <td>{int(r['tenure'])} mo</td>
          <td>{r['Contract']}</td>
          <td class="wl-num">${annual:,.0f}/yr</td>
        </tr>"""
    if not rows:
        rows = ('<tr><td colspan="5" style="text-align:center;'
                f'color:{COLORS["muted"]};padding:1.6rem">No customers match these filters.</td></tr>')
    st.markdown(
        f"""<div class="glass" style="padding:.3rem 1rem">
          <div class="wl-scroll">
            <table class="wl">
              <thead><tr>
                <th>Churn risk</th><th>Tier</th><th>Tenure</th><th>Contract</th>
                <th class="wl-num">Value at risk</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </div>""",
        unsafe_allow_html=True)

    # --- Export (matches the active filters) --------------------------------
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    export_cols = ([col for col in ["customerID"] if col in view.columns] +
                   ["churn_probability", "risk_tier", "tenure", "Contract",
                    "InternetService", "MonthlyCharges", "annual_value"])
    csv = view[export_cols].to_csv(index=False).encode("utf-8")
    dl, cap = st.columns([1, 2.6])
    dl.download_button(
        f"⬇  Export these {len(view):,} (CSV, with IDs)",
        csv, file_name="churn_watchlist.csv", mime="text/csv",
        width="stretch",
    )
    cap.caption("Filter and scroll to browse any segment in-app; the export matches your filters. "
                "Out-of-sample metrics on Model Intelligence.")
