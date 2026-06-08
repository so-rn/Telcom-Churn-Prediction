"""
Telcom Customer Churn Prediction - Training Pipeline
Version: 3.6 (Student + Portfolio Ready)

Implements 14 phases:
  1.  Data Loading & Validation
  2.  Data Cleaning
  3.  Train/Test Split
  4.  Feature Engineering (row-based only)
  5.  Preprocessing Pipeline
  6.  Class Imbalance with SMOTE
  7.  Hyperparameter Optimization (Optuna)
  8.  Cross-Validation Evaluation
  9.  Probability Calibration
  10. Threshold Optimization
  11. SHAP Explainability
  12. Risk Segmentation
  13. Final Hold-Out Evaluation
  14. Artifact Export
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "data_path":               "data/telco_churn.csv",
    "target_column":           "Churn",
    "random_state":            42,
    "test_size":               0.20,
    "calibration_size":        0.15,
    "cv_folds":                5,
    "smote_ratio":             0.5,
    "optuna_trials":           50,
    "optuna_timeout":          600,
    "threshold_min":           0.05,
    "threshold_max":           0.95,
    "threshold_step":          0.01,
    "recall_target":           0.75,
    "cost_fn":                 500,
    "cost_fp":                 100,
    "retention_success_rate":  0.70,
    "artifacts_dir":           "artifacts/",
    "plots_dir":               "plots/",
}

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

SERVICE_COLUMN_NAMES = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("churn")


# ---------------------------------------------------------------------------
# Phase 1 - Data loading & validation
# ---------------------------------------------------------------------------
def load_and_validate(path: str) -> pd.DataFrame:
    """Load dataset from path, fail fast if missing or malformed."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p.resolve()}")

    df = pd.read_csv(p)
    log.info("Loaded dataset: shape=%s", df.shape)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    log.info("Missing values per column:\n%s",
             df.isna().sum()[df.isna().sum() > 0].to_string() or "  (none)")
    log.info("Target class distribution: %s",
             df[CONFIG["target_column"]].value_counts().to_dict())
    churn_rate = (df[CONFIG["target_column"]] == "Yes").mean() * 100
    log.info("Churn rate: %.2f%%", churn_rate)
    return df


# ---------------------------------------------------------------------------
# Phase 2 - Data cleaning (deterministic only)
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only deterministic cleaning - no statistics learned here."""
    df = df.copy()

    # Drop identifier-like columns
    drop_cols = [c for c in df.columns if c.lower() in {"customerid", ""}]
    drop_cols += [c for c in df.columns if c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.info("Dropped identifier/index columns: %s", drop_cols)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    target = CONFIG["target_column"]
    df[target] = df[target].map({"Yes": 1, "No": 0}).astype(int)

    log.info("Post-clean shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Phase 4 - Row-based feature engineering
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, row-based business features. Safe after split.

    All features are pure functions of a single row's values - no statistics
    are learned across rows, so applying this to train and test independently
    introduces zero leakage.
    """
    df = df.copy()

    # Count of "active" services (anything that isn't No / No internet / No phone)
    service_cols_present = [c for c in SERVICE_COLUMN_NAMES if c in df.columns]
    has_service = df[service_cols_present].apply(
        lambda col: (~col.isin(["No", "No internet service",
                                "No phone service"])).astype(int)
    )
    df["services_count"] = has_service.sum(axis=1)

    # Tenure-based: new customers churn more, long-tenured rarely churn.
    # NOTE: tenure_bucket was dropped - it had ~0 gain (raw tenure already
    # carries the signal; bucketing only discarded information).
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["is_long_term"]    = (df["tenure"] >= 48).astype(int)

    # Charge ratios - price sensitivity proxies
    df["monthly_to_total_ratio"] = (
        df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)
    )
    df["avg_monthly_charge"] = (
        df["TotalCharges"] / df["tenure"].replace(0, 1)
    )
    df["charge_per_service"] = (
        df["MonthlyCharges"] / df["services_count"].replace(0, 1)
    )
    # Monthly spend relative to how long they've stayed. High monthly charge
    # on a short tenure is the classic "expensive newcomer" churn profile.
    # Univariate AUC ~0.80 on train - one of the strongest single signals.
    df["monthly_per_tenure"] = (
        df["MonthlyCharges"] / (df["tenure"] + 1)
    )

    # Contract risk - month-to-month customers churn dramatically more
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_two_year"]       = (df["Contract"] == "Two year").astype(int)

    # Fiber-optic customers churn more in this dataset (known pattern).
    # has_no_internet was dropped - it had 0 gain (fully redundant with the
    # many "No internet service" one-hot columns).
    df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    # Electronic check payment is associated with higher churn
    df["pays_electronic_check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)
    df["pays_automatic"] = df["PaymentMethod"].str.contains(
        "automatic", case=False, na=False
    ).astype(int)

    # No security/support combo - "naked" customers leave more
    df["no_security_no_support"] = (
        (df["OnlineSecurity"] == "No") & (df["TechSupport"] == "No")
    ).astype(int)

    # Interaction: month-to-month + fiber is the highest-risk combo
    df["m2m_and_fiber"] = (
        df["is_month_to_month"] * df["has_fiber"]
    ).astype(int)

    # Household features
    df["has_partner_or_deps"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    # Composite risk score: count of known churn risk factors. Each component
    # is an established churn driver in this dataset; summing them gives the
    # tree a clean monotonic feature. Univariate AUC ~0.80 on train - the
    # single strongest signal, beating every raw column.
    df["risk_factor_count"] = (
        (df["Contract"] == "Month-to-month").astype(int)
        + (df["InternetService"] == "Fiber optic").astype(int)
        + (df["PaymentMethod"] == "Electronic check").astype(int)
        + (df["OnlineSecurity"] == "No").astype(int)
        + (df["TechSupport"] == "No").astype(int)
        + (df["Dependents"] == "No").astype(int)
        + (df["SeniorCitizen"] == 1).astype(int)
        + (df["PaperlessBilling"] == "Yes").astype(int)
    )

    return df


# ---------------------------------------------------------------------------
# Phase 5 - Preprocessor builder
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer with imputers + scaler + one-hot encoder."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    log.info("Numeric features:    %d", len(numeric_cols))
    log.info("Categorical features: %d", len(categorical_cols))

    pre = ColumnTransformer([
        ("num", SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
        ]), numeric_cols),
        ("cat", SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False)),
        ]), categorical_cols),
    ])
    return pre


def build_pipeline(preprocessor: ColumnTransformer,
                   model_params: dict[str, Any]) -> ImbPipeline:
    """Build full pipeline: preprocessor -> SMOTE -> LightGBM."""
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(
            sampling_strategy=CONFIG["smote_ratio"],
            random_state=CONFIG["random_state"],
        )),
        ("model", lgb.LGBMClassifier(**model_params)),
    ])


def fit_pipeline_early_stopping(X: pd.DataFrame, y: pd.Series,
                                model_params: dict[str, Any]
                                ) -> tuple[ImbPipeline, int]:
    """Fit a pipeline using LightGBM early stopping, with zero leakage.

    Early stopping needs a validation set whose features go through the SAME
    preprocessor as training, but which is NEVER touched by SMOTE. A plain
    pipeline.fit(..., model__eval_set=...) would pass the eval set to LightGBM
    *raw* (un-preprocessed) - a silent bug. So we wire the steps by hand:

        1. carve a stratified early-stopping slice from X (10%)
        2. fit the preprocessor on the fit slice only
        3. SMOTE the transformed fit slice only
        4. fit LightGBM with eval_set = transformed (un-SMOTEd) val slice

    The fitted preprocessor + model are reassembled into a normal pipeline so
    downstream code (calibration, predict, joblib) treats it like any other.
    Returns the pipeline and the early-stopped iteration count.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y,
        test_size=0.10,
        stratify=y,
        random_state=CONFIG["random_state"],
    )

    pre = build_preprocessor(X_fit)
    X_fit_t = pre.fit_transform(X_fit)
    X_val_t = pre.transform(X_val)

    smote = SMOTE(
        sampling_strategy=CONFIG["smote_ratio"],
        random_state=CONFIG["random_state"],
    )
    X_res, y_res = smote.fit_resample(X_fit_t, y_fit)

    # Give early stopping plenty of headroom; it picks the real count.
    es_params = dict(model_params)
    es_params["n_estimators"] = max(model_params.get("n_estimators", 400), 1000)
    model = lgb.LGBMClassifier(**es_params)
    model.fit(
        X_res, y_res,
        eval_set=[(X_val_t, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    best_iter = int(model.best_iteration_ or es_params["n_estimators"])

    # Reassemble a standard pipeline from the already-fitted components. The
    # SMOTE step is inert at predict time (imblearn skips samplers), so the
    # pipeline behaves exactly like a fitted build_pipeline() result.
    pipe = ImbPipeline([
        ("preprocessor", pre),
        ("smote", smote),
        ("model", model),
    ])
    return pipe, best_iter


# ---------------------------------------------------------------------------
# Phase 7 - Optuna optimization
# ---------------------------------------------------------------------------
def run_optuna(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Run Optuna search, maximizing mean PR-AUC across StratifiedKFold."""
    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True,
        random_state=CONFIG["random_state"],
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Capacity (tightened vs the original 20-100 / 100-500 ranges to
            # reduce overfitting; the dataset is small and near its ceiling).
            "num_leaves":        trial.suggest_int("num_leaves", 15, 50),
            "learning_rate":     trial.suggest_float("learning_rate",
                                                    0.01, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            # Regularization knobs - the levers that actually close the gap.
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "min_split_gain":    trial.suggest_float("min_split_gain",
                                                    0.0, 0.2),
            "reg_alpha":         trial.suggest_float("reg_alpha",
                                                    1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",
                                                    1e-3, 10.0, log=True),
            # Stochastic bagging. NOTE: in LightGBM `subsample` is ignored
            # unless `subsample_freq >= 1` - set it so bagging actually runs.
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree",
                                                    0.6, 1.0),
            "random_state":      CONFIG["random_state"],
            "verbosity":         -1,
        }
        pre = build_preprocessor(X_train)
        pipe = build_pipeline(pre, params)
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=skf, scoring="average_precision", n_jobs=1,
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=CONFIG["random_state"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=CONFIG["optuna_trials"],
        timeout=CONFIG["optuna_timeout"],
        show_progress_bar=False,
    )
    log.info("Best PR-AUC: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    # Optuna history plot
    fig, ax = plt.subplots(figsize=(8, 4))
    values = [t.value for t in study.trials if t.value is not None]
    ax.plot(range(1, len(values) + 1), values, marker="o")
    ax.set_xlabel("Trial")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Optuna optimization history")
    fig.tight_layout()
    fig.savefig(Path(CONFIG["plots_dir"]) / "optuna_history.png", dpi=120)
    plt.close(fig)

    best = dict(study.best_params)
    best.update({
        "subsample_freq": 1,   # keep bagging active (see objective note)
        "random_state":   CONFIG["random_state"],
        "verbosity":      -1,
    })
    return best


# ---------------------------------------------------------------------------
# Phase 8 - CV evaluation
# ---------------------------------------------------------------------------
def evaluate_cv(X_train: pd.DataFrame, y_train: pd.Series,
                best_params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the tuned pipeline with stratified 5-fold CV."""
    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True,
        random_state=CONFIG["random_state"],
    )

    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx]
        y_va = y_train.iloc[va_idx]

        pipe = build_pipeline(build_preprocessor(X_tr), best_params)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        fold_metrics.append({
            "fold":      fold_idx,
            "recall":    recall_score(y_va, pred),
            "precision": precision_score(y_va, pred, zero_division=0),
            "f1":        f1_score(y_va, pred),
            "roc_auc":   roc_auc_score(y_va, proba),
            "pr_auc":    average_precision_score(y_va, proba),
        })
        log.info("Fold %d: recall=%.3f  prec=%.3f  f1=%.3f  roc=%.3f  pr=%.3f",
                 fold_idx,
                 fold_metrics[-1]["recall"],
                 fold_metrics[-1]["precision"],
                 fold_metrics[-1]["f1"],
                 fold_metrics[-1]["roc_auc"],
                 fold_metrics[-1]["pr_auc"])

    keys = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
    summary = {
        k: {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std":  float(np.std([m[k] for m in fold_metrics])),
        } for k in keys
    }
    log.info("CV summary: %s",
             {k: f"{v['mean']:.3f}±{v['std']:.3f}" for k, v in summary.items()})

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    means = [summary[k]["mean"] for k in keys]
    stds  = [summary[k]["std"]  for k in keys]
    ax.bar(keys, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("5-fold CV results (mean ± std)")
    fig.tight_layout()
    fig.savefig(Path(CONFIG["plots_dir"]) / "cv_results.png", dpi=120)
    plt.close(fig)

    return {"per_fold": fold_metrics, "summary": summary}


# ---------------------------------------------------------------------------
# Phase 9 - Calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Standard ECE with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1
                                  else y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc  = y_true[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(conf - acc)
    return float(ece)


def calibrate_model(X_train: pd.DataFrame, y_train: pd.Series,
                    best_params: dict[str, Any]
                    ) -> tuple[CalibratedClassifierCV, ImbPipeline]:
    """Split a calibration set, fit the pipeline, then calibrate (isotonic)."""
    X_tr, X_calib, y_tr, y_calib = train_test_split(
        X_train, y_train,
        test_size=CONFIG["calibration_size"],
        stratify=y_train,
        random_state=CONFIG["random_state"],
    )

    base_pipe, best_iter = fit_pipeline_early_stopping(X_tr, y_tr, best_params)
    log.info("Early stopping selected %d trees (ceiling was %d)",
             best_iter, max(best_params.get("n_estimators", 400), 1000))

    raw_prob = base_pipe.predict_proba(X_calib)[:, 1]
    ece_raw = expected_calibration_error(y_calib.to_numpy(), raw_prob)

    iso = CalibratedClassifierCV(base_pipe, method="isotonic", cv="prefit")
    iso.fit(X_calib, y_calib)
    iso_prob = iso.predict_proba(X_calib)[:, 1]
    ece_iso = expected_calibration_error(y_calib.to_numpy(), iso_prob)

    if ece_iso > ece_raw:
        log.warning("Isotonic worsened ECE (%.4f -> %.4f). Falling back "
                    "to sigmoid.", ece_raw, ece_iso)
        cal = CalibratedClassifierCV(base_pipe, method="sigmoid", cv="prefit")
        cal.fit(X_calib, y_calib)
        cal_prob = cal.predict_proba(X_calib)[:, 1]
    else:
        cal = iso
        cal_prob = iso_prob

    ece_final = expected_calibration_error(y_calib.to_numpy(), cal_prob)
    log.info("ECE: raw=%.4f  calibrated=%.4f", ece_raw, ece_final)

    # Reliability diagram
    frac_pos, mean_pred = calibration_curve(y_calib, cal_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    ax.plot(mean_pred, frac_pos, marker="o", label="Calibrated model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(CONFIG["plots_dir"]) / "calibration_curve.png", dpi=120)
    plt.close(fig)

    joblib.dump(cal, Path(CONFIG["artifacts_dir"]) / "calibrator.joblib")
    return cal, base_pipe


# ---------------------------------------------------------------------------
# Phase 10 - Threshold optimization
# ---------------------------------------------------------------------------
def business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Business cost = FN * cost_fn + FP * cost_fp."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * CONFIG["cost_fn"] + fp * CONFIG["cost_fp"])


def optimize_threshold(X_train: pd.DataFrame, y_train: pd.Series,
                       best_params: dict[str, Any]) -> dict[str, Any]:
    """Pick a threshold from CALIBRATED OOF probabilities.

    Wraps the pipeline in CalibratedClassifierCV (isotonic, inner cv=3) so the
    probabilities are on the same scale as the final calibrator used at test
    time. This makes the chosen threshold directly applicable in Phase 13.
    """
    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True,
        random_state=CONFIG["random_state"],
    )
    pipe = build_pipeline(build_preprocessor(X_train), best_params)
    cal_pipe = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    oof = cross_val_predict(
        cal_pipe, X_train, y_train, cv=skf,
        method="predict_proba", n_jobs=1,
    )[:, 1]

    y = y_train.to_numpy()
    thresholds = np.arange(
        CONFIG["threshold_min"],
        CONFIG["threshold_max"] + 1e-9,
        CONFIG["threshold_step"],
    )

    rows = []
    for t in thresholds:
        pred = (oof >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "recall":    recall_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "f1":        f1_score(y, pred, zero_division=0),
            "cost":      business_cost(y, pred),
        })
    grid = pd.DataFrame(rows)

    feasible = grid[grid["recall"] >= CONFIG["recall_target"]]
    if feasible.empty:
        log.warning("No threshold met recall>=%.2f. "
                    "Falling back to max-recall threshold.",
                    CONFIG["recall_target"])
        chosen = grid.sort_values(["recall", "f1"],
                                  ascending=[False, False]).iloc[0]
    else:
        chosen = feasible.sort_values(
            ["cost", "f1"], ascending=[True, False]
        ).iloc[0]

    log.info("Selected threshold: %.2f | recall=%.3f  cost=%.0f  f1=%.3f",
             chosen["threshold"], chosen["recall"],
             chosen["cost"], chosen["f1"])

    # Plot threshold vs cost
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(grid["threshold"], grid["cost"], color="crimson",
             label="Business cost")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Business cost", color="crimson")
    ax1.axvline(chosen["threshold"], color="black", linestyle="--",
                label=f"Chosen={chosen['threshold']:.2f}")
    ax2 = ax1.twinx()
    ax2.plot(grid["threshold"], grid["recall"], color="steelblue",
             label="Recall")
    ax2.axhline(CONFIG["recall_target"], color="steelblue", linestyle=":")
    ax2.set_ylabel("Recall", color="steelblue")
    ax1.set_title("Threshold sweep (OOF)")
    fig.tight_layout()
    fig.savefig(Path(CONFIG["plots_dir"]) / "threshold_vs_cost.png", dpi=120)
    plt.close(fig)

    return {
        "threshold": float(chosen["threshold"]),
        "recall":    float(chosen["recall"]),
        "precision": float(chosen["precision"]),
        "f1":        float(chosen["f1"]),
        "cost":      float(chosen["cost"]),
    }


# ---------------------------------------------------------------------------
# Phase 11 - SHAP explainability
# ---------------------------------------------------------------------------
def shap_explain(base_pipe: ImbPipeline, X_train: pd.DataFrame,
                 y_train: pd.Series) -> None:
    """Generate SHAP values on a 200-sample stratified subset."""
    n = min(200, len(X_train))
    sub_idx, _ = train_test_split(
        np.arange(len(X_train)),
        train_size=n,
        stratify=y_train,
        random_state=CONFIG["random_state"],
    )
    X_sub = X_train.iloc[sub_idx]

    pre   = base_pipe.named_steps["preprocessor"]
    model = base_pipe.named_steps["model"]
    X_trans = pre.transform(X_sub)
    feature_names = pre.get_feature_names_out().tolist()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(Path(CONFIG["plots_dir"]) / "shap_summary.png", dpi=120)
    plt.close(fig)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in order][::-1], mean_abs[order][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(Path(CONFIG["plots_dir"]) / "feature_importance.png", dpi=120)
    plt.close(fig)

    joblib.dump({"X_background": X_trans,
                 "feature_names": feature_names},
                Path(CONFIG["artifacts_dir"]) / "shap_background.joblib")
    joblib.dump(feature_names,
                Path(CONFIG["artifacts_dir"]) / "feature_names.joblib")


# ---------------------------------------------------------------------------
# Phase 12 - Risk segmentation
# ---------------------------------------------------------------------------
def assign_risk_tiers(prob: np.ndarray) -> np.ndarray:
    """Map calibrated probabilities to four risk tiers."""
    tiers = np.empty(len(prob), dtype=object)
    tiers[prob < 0.30] = "Low"
    tiers[(prob >= 0.30) & (prob < 0.55)] = "Medium"
    tiers[(prob >= 0.55) & (prob < 0.75)] = "High"
    tiers[prob >= 0.75] = "Critical"
    return tiers


# ---------------------------------------------------------------------------
# Phase 13 - Final hold-out evaluation
# ---------------------------------------------------------------------------
def evaluate_holdout(calibrator: CalibratedClassifierCV,
                     X_test: pd.DataFrame, y_test: pd.Series,
                     threshold: float,
                     cv_mean_recall: float) -> dict[str, Any]:
    """Single-shot evaluation on the held-out test set."""
    proba = calibrator.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = recall_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    f1   = f1_score(y_test, pred)
    roc  = roc_auc_score(y_test, proba)
    pr   = average_precision_score(y_test, proba)
    cost = business_cost(y_test.to_numpy(), pred)

    revenue_saved   = tp * CONFIG["cost_fn"] * CONFIG["retention_success_rate"]
    retention_spend = (tp + fp) * CONFIG["cost_fp"]
    net_savings     = revenue_saved - retention_spend
    roi = (net_savings / retention_spend * 100.0) if retention_spend else 0.0

    log.info("Test recall=%.3f  prec=%.3f  f1=%.3f  roc=%.3f  pr=%.3f",
             rec, prec, f1, roc, pr)
    log.info("Business cost=%.0f  net savings=%.0f  ROI=%.1f%%",
             cost, net_savings, roi)
    log.info("\n%s", classification_report(y_test, pred, digits=3))

    if abs(cv_mean_recall - rec) > 0.05:
        log.warning("CV vs Test recall gap > 5%% "
                    "(CV=%.3f, Test=%.3f) - possible overfit",
                    cv_mean_recall, rec)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No churn", "Churn"],
                yticklabels=["No churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(Path(CONFIG["plots_dir"]) / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    final_metrics = {
        "threshold": float(threshold),
        "recall":    float(rec),
        "precision": float(prec),
        "f1":        float(f1),
        "roc_auc":   float(roc),
        "pr_auc":    float(pr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_test, pred, digits=3, output_dict=True
        ),
    }
    business_impact = {
        "business_cost":   float(cost),
        "revenue_saved":   float(revenue_saved),
        "retention_spend": float(retention_spend),
        "net_savings":     float(net_savings),
        "roi_percent":     float(roi),
        "assumptions": {
            "cost_fn": CONFIG["cost_fn"],
            "cost_fp": CONFIG["cost_fp"],
            "retention_success_rate": CONFIG["retention_success_rate"],
        },
    }
    return {"metrics": final_metrics, "business": business_impact,
            "test_proba": proba}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def manifest(artifacts_dir: Path) -> None:
    """Print final manifest of saved artifacts."""
    log.info("--- Artifact manifest ---")
    for p in sorted(artifacts_dir.iterdir()):
        if p.is_file():
            kb = p.stat().st_size / 1024
            log.info("  %-32s %8.1f KB", p.name, kb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    art_dir = Path(CONFIG["artifacts_dir"])
    plt_dir = Path(CONFIG["plots_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    plt_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    log.info("=== Phase 1: Data Loading & Validation ===")
    df = load_and_validate(CONFIG["data_path"])

    # Phase 2
    log.info("=== Phase 2: Data Cleaning ===")
    df = clean_data(df)

    # Phase 3
    log.info("=== Phase 3: Train/Test Split ===")
    y = df[CONFIG["target_column"]]
    X = df.drop(columns=[CONFIG["target_column"]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_state"],
    )
    log.info("Train: %s  churn=%.2f%%", X_train.shape, y_train.mean() * 100)
    log.info("Test:  %s  churn=%.2f%%", X_test.shape,  y_test.mean()  * 100)

    # Phase 4 - applied to train and test independently
    log.info("=== Phase 4: Feature Engineering ===")
    X_train = add_features(X_train)
    X_test  = add_features(X_test)

    # Phase 7
    log.info("=== Phase 7: Optuna Hyperparameter Search ===")
    best_params = run_optuna(X_train, y_train)
    save_json(best_params, art_dir / "best_params.json")

    # Phase 8
    log.info("=== Phase 8: Cross-Validation Evaluation ===")
    cv_results = evaluate_cv(X_train, y_train, best_params)
    save_json(cv_results, art_dir / "cv_results.json")

    # Phase 9
    log.info("=== Phase 9: Probability Calibration ===")
    calibrator, base_pipe = calibrate_model(X_train, y_train, best_params)

    # Phase 10
    log.info("=== Phase 10: Threshold Optimization ===")
    threshold_info = optimize_threshold(X_train, y_train, best_params)
    save_json(threshold_info, art_dir / "optimal_threshold.json")

    # Phase 11
    log.info("=== Phase 11: SHAP Explainability ===")
    shap_explain(base_pipe, X_train, y_train)

    # Final pipeline trained on full training set, for prediction at scale.
    log.info("Fitting final pipeline (early stopping) on training set...")
    final_pipe, final_iter = fit_pipeline_early_stopping(
        X_train, y_train, best_params
    )
    log.info("Final model: %d trees after early stopping", final_iter)
    joblib.dump(final_pipe, art_dir / "model.joblib")

    # Phase 12
    log.info("=== Phase 12: Risk Segmentation (train) ===")
    train_proba = calibrator.predict_proba(X_train)[:, 1]
    tiers = assign_risk_tiers(train_proba)
    tier_counts = pd.Series(tiers).value_counts().to_dict()
    log.info("Train tier distribution: %s", tier_counts)

    # Phase 13
    # Use OOF recall at the CHOSEN threshold as the CV baseline. That makes
    # the CV-vs-test gap check apples-to-apples (same threshold, same
    # calibrated probability scale).
    log.info("=== Phase 13: Final Hold-Out Evaluation ===")
    holdout = evaluate_holdout(
        calibrator, X_test, y_test,
        threshold_info["threshold"], threshold_info["recall"],
    )
    save_json(holdout["metrics"], art_dir / "final_metrics.json")
    save_json(holdout["business"], art_dir / "business_impact.json")

    # Risk tier distribution on test
    test_tiers = assign_risk_tiers(holdout["test_proba"])
    log.info("Test tier distribution: %s",
             pd.Series(test_tiers).value_counts().to_dict())

    # Phase 14
    log.info("=== Phase 14: Artifact Export ===")
    save_json(CONFIG, art_dir / "config.json")
    manifest(art_dir)
    log.info("Done.")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
