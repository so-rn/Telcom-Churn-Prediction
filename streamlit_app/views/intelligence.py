"""Model Intelligence - how the model performs and what it learned."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core
from theme import COLORS, animate_counters, hero, kpi_card, style_fig


def render() -> None:
    models = core.load_models()
    arts = core.load_json_artifacts()
    metrics = arts["metrics"]
    cv = arts["cv"].get("summary", {})
    stab = arts["stability"].get("summary", {})

    st.markdown(hero(
        "Under the hood",
        'Model <span class="grad-text">intelligence</span> &amp; performance',
        "Out-of-sample metrics, what drives predictions, and proof the model is stable.",
    ), unsafe_allow_html=True)

    # --- Headline test metrics ----------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Recall (test)", f"{metrics.get('recall', 0):.3f}",
                         color="cyan", delta="churners caught"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Precision (test)", f"{metrics.get('precision', 0):.3f}",
                         color="violet", delta="flag accuracy"), unsafe_allow_html=True)
    c3.markdown(kpi_card("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}",
                         color="green"), unsafe_allow_html=True)
    c4.markdown(kpi_card("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}",
                         color="amber"), unsafe_allow_html=True)
    animate_counters()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Feature drivers", "Performance", "Stability"])

    # ----------------------------------------------------- feature importance
    with tab1:
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section">Top features by model gain</div>',
                        unsafe_allow_html=True)
            imp = models["lgbm"].feature_importances_
            names = models["feat_names"]
            idx = np.argsort(imp)[::-1][:15][::-1]
            fig = go.Figure(go.Bar(
                x=imp[idx], y=[core.humanize_feature(names[i]) for i in idx],
                orientation="h",
                marker=dict(color=imp[idx], colorscale=[[0, COLORS["violet"]], [1, COLORS["cyan"]]],
                            line=dict(width=0)),
                hovertemplate="%{y}: gain %{x:.0f}<extra></extra>"))
            fig.update_layout(xaxis_title="LightGBM gain", yaxis_title="")
            st.plotly_chart(style_fig(fig, 430), width="stretch")

        with right:
            st.markdown('<div class="section">Mean impact on output (SHAP)</div>',
                        unsafe_allow_html=True)
            bg = core.joblib.load(core.ART / "shap_background.joblib")
            Xb = bg["X_background"]
            sv = models["explainer"].shap_values(Xb)
            if isinstance(sv, list):
                sv = sv[1]
            mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
            names = bg["feature_names"]
            idx = np.argsort(mean_abs)[::-1][:15][::-1]
            fig = go.Figure(go.Bar(
                x=mean_abs[idx],
                y=[core.humanize_feature(names[i].split("__")[-1]) for i in idx],
                orientation="h",
                marker=dict(color=mean_abs[idx],
                            colorscale=[[0, COLORS["green"]], [1, COLORS["cyan"]]],
                            line=dict(width=0)),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(xaxis_title="mean |SHAP value|", yaxis_title="")
            st.plotly_chart(style_fig(fig, 430), width="stretch")

    # ---------------------------------------------------------- performance
    with tab2:
        left, right = st.columns([1, 1])
        with left:
            st.markdown('<div class="section">Confusion matrix (held-out test)</div>',
                        unsafe_allow_html=True)
            cm = metrics.get("confusion_matrix", {})
            z = [[cm.get("tn", 0), cm.get("fp", 0)],
                 [cm.get("fn", 0), cm.get("tp", 0)]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["Pred: Stay", "Pred: Churn"], y=["Actual: Stay", "Actual: Churn"],
                colorscale=[[0, COLORS["bg_soft"]], [1, COLORS["cyan"]]],
                showscale=False,
                text=[[f"{v:,}" for v in row] for row in z],
                texttemplate="%{text}", textfont=dict(size=22, family="Space Grotesk")))
            st.plotly_chart(style_fig(fig, 360), width="stretch")

        with right:
            st.markdown('<div class="section">Cross-validation (5-fold)</div>',
                        unsafe_allow_html=True)
            keys = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
            means = [cv.get(k, {}).get("mean", 0) for k in keys]
            stds = [cv.get(k, {}).get("std", 0) for k in keys]
            fig = go.Figure(go.Bar(
                x=[k.replace("_", "-").upper() for k in keys], y=means,
                error_y=dict(type="data", array=stds, color=COLORS["muted"]),
                marker=dict(color=means, colorscale=[[0, COLORS["violet"]], [1, COLORS["cyan"]]],
                            line=dict(width=0)),
                text=[f"{m:.2f}" for m in means], textposition="outside",
                hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
            fig.update_layout(yaxis=dict(range=[0, 1]), yaxis_title="Score", xaxis_title="")
            st.plotly_chart(style_fig(fig, 360), width="stretch")
            st.caption("Mean ± std across folds. Tight bars = consistent learning.")

    # ------------------------------------------------------------- stability
    with tab3:
        st.markdown('<div class="section">Robustness across random seeds</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{COLORS['muted']};margin-bottom:1rem'>"
            "Hyperparameters held fixed; only the random seed (split, SMOTE, bagging) "
            "varies. Low spread proves the model is robust, not lucky on one split.</div>",
            unsafe_allow_html=True)

        per = arts["stability"].get("per_seed", [])
        if per:
            seeds = [r["seed"] for r in per]
            fig = go.Figure()
            for key, col in [("cv_pr_auc", COLORS["violet"]),
                             ("test_roc_auc", COLORS["cyan"]),
                             ("test_pr_auc", COLORS["green"])]:
                fig.add_bar(name=key.replace("_", " ").upper(),
                            x=[str(s) for s in seeds], y=[r[key] for r in per],
                            marker=dict(color=col, line=dict(width=0)))
            fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]),
                              xaxis_title="Random seed", yaxis_title="Score",
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(style_fig(fig, 340), width="stretch")

            cols = st.columns(3)
            labels = {"cv_pr_auc": "CV PR-AUC", "test_roc_auc": "Test ROC-AUC",
                      "test_pr_auc": "Test PR-AUC"}
            for (k, lab), c in zip(labels.items(), cols):
                m = stab.get(k, {})
                c.markdown(kpi_card(lab, f"{m.get('mean', 0):.3f}",
                                    color="cyan", delta=f"± {m.get('std', 0):.3f} across seeds"),
                           unsafe_allow_html=True)
        else:
            st.info("Run `python stability_check.py` to populate stability results.")
