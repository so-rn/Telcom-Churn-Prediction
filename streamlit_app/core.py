"""
Data & model access layer for the Streamlit app.

Loads the artifacts produced by train.py, scores customers through the
calibrated pipeline, and exposes per-customer SHAP explanations. All heavy
objects are cached so the UI stays snappy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Make the project root importable so we reuse train.py's exact feature logic.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib          # noqa: E402
import shap            # noqa: E402
import train           # noqa: E402  (reuse add_features / assign_risk_tiers / CONFIG)

ART = PROJECT_ROOT / "artifacts"
DATA = PROJECT_ROOT / "data" / "telco_churn.csv"


# ---------------------------------------------------------------------------
# Raw input schema for the Risk Scorer form
# ---------------------------------------------------------------------------
YES_NO = ["Yes", "No"]
INTERNET_DEP = ["Yes", "No", "No internet service"]

CATEGORICAL_FIELDS: dict[str, list[str]] = {
    "gender":           ["Male", "Female"],
    "Partner":          YES_NO,
    "Dependents":       YES_NO,
    "PhoneService":     YES_NO,
    "MultipleLines":    ["Yes", "No", "No phone service"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   INTERNET_DEP,
    "OnlineBackup":     INTERNET_DEP,
    "DeviceProtection": INTERNET_DEP,
    "TechSupport":      INTERNET_DEP,
    "StreamingTV":      INTERNET_DEP,
    "StreamingMovies":  INTERNET_DEP,
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod":    ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Demo presets - one click to show a striking contrast
PRESETS: dict[str, dict[str, Any]] = {
    "High-risk newcomer": {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 94.5, "TotalCharges": 189.0,
    },
    "Loyal long-term": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 68, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 4420.0,
    },
    "Borderline mid-tenure": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 24, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 84.0, "TotalCharges": 2010.0,
    },
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, Any]:
    """Load fitted model, calibrator and a SHAP explainer (cached)."""
    model = joblib.load(ART / "model.joblib")
    calibrator = joblib.load(ART / "calibrator.joblib")
    pre = model.named_steps["preprocessor"]
    lgbm = model.named_steps["model"]
    explainer = shap.TreeExplainer(lgbm)
    feat_names = [n.split("__")[-1] for n in pre.get_feature_names_out()]
    return {"model": model, "calibrator": calibrator, "pre": pre,
            "lgbm": lgbm, "explainer": explainer, "feat_names": feat_names}


@st.cache_data(show_spinner=False)
def load_json_artifacts() -> dict[str, Any]:
    """Load all JSON artifacts (metrics, business, cv, threshold, ...)."""
    def _read(name: str) -> Any:
        p = ART / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {
        "threshold":  _read("optimal_threshold.json"),
        "metrics":    _read("final_metrics.json"),
        "business":   _read("business_impact.json"),
        "cv":         _read("cv_results.json"),
        "best_params": _read("best_params.json"),
        "stability":  _read("stability.json"),
        "config":     _read("config.json"),
    }


@st.cache_data(show_spinner=False)
def load_scored_population() -> pd.DataFrame:
    """Load the full customer base, score it, and attach probability + tier."""
    models = load_models()
    arts = load_json_artifacts()
    thr = arts["threshold"].get("threshold", 0.5)

    raw = pd.read_csv(DATA)
    df = train.clean_data(train.load_and_validate(str(DATA)))
    y = df[train.CONFIG["target_column"]]
    X = train.add_features(df.drop(columns=[train.CONFIG["target_column"]]))

    proba = models["calibrator"].predict_proba(X)[:, 1]
    out = df.copy()
    # clean_data drops customerID for modelling; re-attach it (same row order)
    # so the watchlist export carries a real identifier.
    if "customerID" in raw.columns:
        out.insert(0, "customerID", raw["customerID"].values)
    out["churn_probability"] = proba
    out["risk_tier"] = train.assign_risk_tiers(proba)
    out["predicted_churn"] = (proba >= thr).astype(int)
    out["actual_churn"] = y.values
    return out


@st.cache_data(show_spinner=False)
def load_raw_with_churn() -> pd.DataFrame:
    """Cleaned dataframe with a readable 'Churn' label, for EDA charts."""
    df = train.clean_data(train.load_and_validate(str(DATA)))
    df = df.rename(columns={train.CONFIG["target_column"]: "churn"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 1000],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)
    return df


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate (%) and lift vs base, per category of `col`, sorted."""
    base = df["churn"].mean()
    g = df.groupby(col)["churn"].agg(["mean", "count"]).reset_index()
    g["rate"] = g["mean"] * 100
    g["lift"] = g["mean"] / base
    return g.sort_values("rate")


# ---------------------------------------------------------------------------
# Per-customer scoring + explanation
# ---------------------------------------------------------------------------
def score_customer(raw: dict[str, Any]) -> dict[str, Any]:
    """Score one raw customer dict; return prob, tier, prediction, SHAP drivers."""
    models = load_models()
    arts = load_json_artifacts()
    thr = arts["threshold"].get("threshold", 0.5)

    row = pd.DataFrame([raw])
    row = train.add_features(row)

    proba = float(models["calibrator"].predict_proba(row)[:, 1][0])
    tier = str(train.assign_risk_tiers(np.array([proba]))[0])

    # SHAP from the uncalibrated tree model (calibration is monotonic, so the
    # ranking of drivers is preserved). Transform through the same preprocessor.
    x_trans = models["pre"].transform(row)
    sv = models["explainer"].shap_values(x_trans)
    if isinstance(sv, list):          # [class0, class1] in some shap versions
        sv = sv[1]
    sv = np.asarray(sv).reshape(-1)

    drivers = pd.DataFrame({"feature": models["feat_names"], "shap": sv})
    drivers["abs"] = drivers["shap"].abs()
    drivers = drivers.sort_values("abs", ascending=False).head(8)

    return {
        "probability": proba,
        "tier": tier,
        "prediction": int(proba >= thr),
        "threshold": thr,
        "drivers": drivers,
    }


def humanize_feature(name: str) -> str:
    """Turn an encoded feature name into a readable label."""
    name = name.replace("_", " ")
    return name[0].upper() + name[1:] if name else name
