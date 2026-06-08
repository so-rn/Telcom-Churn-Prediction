"""
Stability check: hold the tuned hyperparameters fixed and vary only the random
seed (data split + bagging + SMOTE). Low variance across seeds is evidence the
model is robust rather than lucky on one particular split.

AUC metrics are reported because they are independent of threshold and
calibration - the cleanest measure of discriminative stability.

Run:  python stability_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import train  # reuse the exact same functions as the training pipeline

SEEDS = [42, 7, 2024]


def run_one_seed(seed: int, best_params: dict) -> dict[str, float]:
    """Re-split, re-fit (early stopping) and evaluate for a single seed."""
    train.CONFIG["random_state"] = seed
    params = dict(best_params)
    params["random_state"] = seed

    df = train.clean_data(train.load_and_validate(train.CONFIG["data_path"]))
    y = df[train.CONFIG["target_column"]]
    X = df.drop(columns=[train.CONFIG["target_column"]])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=train.CONFIG["test_size"],
        stratify=y, random_state=seed,
    )
    X_tr = train.add_features(X_tr)
    X_te = train.add_features(X_te)

    # CV PR-AUC with fixed params (no re-tuning)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipe = train.build_pipeline(train.build_preprocessor(X_tr), params)
    cv_pr = cross_val_score(
        pipe, X_tr, y_tr, cv=skf, scoring="average_precision", n_jobs=1
    ).mean()

    # Test AUCs from the early-stopped model (calibration-free ranking metrics)
    fitted, n_trees = train.fit_pipeline_early_stopping(X_tr, y_tr, params)
    proba = fitted.predict_proba(X_te)[:, 1]

    return {
        "seed":        seed,
        "cv_pr_auc":   float(cv_pr),
        "test_roc_auc": float(roc_auc_score(y_te, proba)),
        "test_pr_auc":  float(average_precision_score(y_te, proba)),
        "n_trees":     int(n_trees),
    }


def main() -> None:
    best_params = json.loads(
        Path("artifacts/best_params.json").read_text(encoding="utf-8")
    )

    rows = [run_one_seed(s, best_params) for s in SEEDS]
    res = pd.DataFrame(rows)

    summary = {
        "cv_pr_auc":    (res["cv_pr_auc"].mean(),   res["cv_pr_auc"].std()),
        "test_roc_auc": (res["test_roc_auc"].mean(), res["test_roc_auc"].std()),
        "test_pr_auc":  (res["test_pr_auc"].mean(),  res["test_pr_auc"].std()),
    }

    print("\n=== Per-seed results ===")
    print(res.to_string(index=False))
    print("\n=== Stability summary (mean +/- std across seeds) ===")
    for k, (m, s) in summary.items():
        print(f"  {k:14s} {m:.4f} +/- {s:.4f}")

    out = {
        "seeds": SEEDS,
        "per_seed": rows,
        "summary": {k: {"mean": float(m), "std": float(s)}
                    for k, (m, s) in summary.items()},
    }
    Path("artifacts/stability.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\nSaved -> artifacts/stability.json")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
