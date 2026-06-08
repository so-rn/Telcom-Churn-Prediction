# Telcom Customer Churn Prediction

A clean, leakage-free machine-learning pipeline that predicts which telcom
customers are about to churn, optimised for a real business objective (catch
churners cheaply) rather than raw accuracy.

Built with LightGBM + SMOTE + isotonic calibration, tuned with Optuna, and
hardened against overfitting with early stopping and L1/L2 regularization.

---

## Results (held-out test set, 1,198 customers)

The test set is touched **exactly once**, at the very end. Operating threshold
is chosen on calibrated out-of-fold probabilities, never on the test set.

| Metric | Value |
|---|---|
| Recall (churners caught) | **0.871** |
| Precision | 0.438 |
| F1 | 0.583 |
| ROC-AUC | 0.833 |
| PR-AUC | 0.627 |
| Operating threshold | 0.19 |

**Business impact** (assumptions in `CONFIG`: `cost_fn=$500`, `cost_fp=$100`,
`retention_success_rate=0.70`):

| | |
|---|---|
| Revenue saved | $96,950 |
| Retention spend | $63,300 |
| **Net savings** | **$33,650** |
| **ROI** | **53.2%** |

Confusion matrix: TP=277, FN=41, FP=356, TN=524 — we miss only 41 of 318
churners while keeping false alarms at a level the business case can absorb.

---

## Stability across random seeds

To show the model is robust rather than lucky on one split, the **tuned
hyperparameters are held fixed** and only the random seed (data split, SMOTE,
bagging) is varied. Low spread = trustworthy model. Run with
`python stability_check.py`.

| Seed | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Trees (early stop) |
|---|---|---|---|---|
| 42   | 0.672 | 0.830 | 0.653 | 24 |
| 7    | 0.655 | 0.847 | 0.683 | 158 |
| 2024 | 0.653 | 0.859 | 0.686 | 48 |
| **Mean ± std** | **0.660 ± 0.011** | **0.845 ± 0.014** | **0.674 ± 0.018** | — |

The ~1–2% standard deviation confirms the pipeline generalises consistently.
Early stopping adapts the tree count to each split (24–158) instead of using a
fixed, overfit-prone count.

---

## Why these numbers (and not "99% accuracy")

This dataset (≈5,990 rows, 20 columns) has a well-known performance ceiling of
roughly **ROC-AUC ~0.84–0.85** and **PR-AUC ~0.65–0.68**. We sit right at it.
Any claim far above that range on this dataset almost always means data leakage.
The goal here was a model that is **honest, calibrated, and defensible**, not an
inflated score.

A low operating threshold (0.19) is intentional: with `cost_fn` (losing a
customer) higher than `cost_fp` (an unnecessary retention contact), the optimal
business decision is to flag aggressively. The threshold is derived from the
cost formula in `CONFIG`, not hand-picked.

---

## Engineering decisions worth highlighting

- **Split before anything is learned.** Train/test split happens in Phase 3,
  before feature engineering and before any imputer/scaler/encoder is fitted.
  Every learned transform is fitted inside CV folds only — zero leakage.
- **Row-based features only.** Engineered features are pure functions of a
  single row (ratios, flags, a risk-factor count), so applying them to train
  and test independently cannot leak.
- **The strongest features were engineered, not raw.** `monthly_per_tenure`,
  the charge ratios, and a composite `risk_factor_count` rank as the top
  features by LightGBM gain — they beat every raw column.
- **An honest lesson on feature value.** A composite feature with high
  *univariate* AUC (~0.80) did not raise model AUC, because a gradient-boosted
  tree could already reconstruct it from the raw components. High univariate
  signal ≠ guaranteed model lift.
- **Calibration before threshold.** Probabilities are isotonic-calibrated
  (sigmoid fallback if ECE worsens) so the threshold search and the risk tiers
  operate on trustworthy probabilities.
- **Early stopping + regularization for stability.** L1/L2 (`reg_alpha`,
  `reg_lambda`), `min_split_gain`, tighter capacity bounds, and early stopping
  cut the final model from hundreds of trees to ~24, closing the CV/test gap to
  ~0.6 points and reducing fold-to-fold variance.
- **Bug fixed along the way.** LightGBM's `subsample` is silently ignored
  unless `subsample_freq >= 1`. Enabling bagging properly improved test PR-AUC
  (0.606 → 0.627) — a one-line fix that mattered more than more Optuna trials.

---

## Pipeline phases

| Phase | What it does |
|---|---|
| 1 | Data loading & schema validation |
| 2 | Deterministic cleaning (drop IDs, `TotalCharges`→numeric, target→0/1) |
| 3 | Stratified train/test split (test locked until the end) |
| 4 | Row-based feature engineering |
| 5 | Preprocessing (`ColumnTransformer`: impute + scale + one-hot) |
| 6 | SMOTE on training folds only |
| 7 | Optuna hyperparameter search (maximise CV PR-AUC) |
| 8 | 5-fold stratified CV evaluation |
| 9 | Isotonic probability calibration (sigmoid fallback) |
| 10 | Threshold optimisation on calibrated OOF probabilities |
| 11 | SHAP explainability |
| 12 | Risk segmentation (Low / Medium / High / Critical) |
| 13 | Single-shot hold-out evaluation + business impact |
| 14 | Artifact export |

---

## How to run

```bash
# 1. Create and activate a Python 3.12 virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train (data must be at data/telco_churn.csv)
python train.py

# 4. (Optional) Stability check across seeds
python stability_check.py

# 5. Launch the interactive app
streamlit run streamlit_app/app.py
```

Training runs end-to-end in a few minutes on a laptop CPU.

---

## Interactive app (layer 2)

A premium **Streamlit** dashboard sits on top of the trained artifacts -
dark-glassmorphism design, custom Plotly theme, four pages:

| Page | What it shows |
|---|---|
| **Executive Dashboard** | Portfolio KPIs, risk-tier mix, probability landscape, revenue-at-risk, priority watchlist |
| **Customer Risk Scorer** | Live single-customer scoring: animated risk gauge + per-prediction SHAP drivers + recommended action |
| **Model Intelligence** | Test metrics, feature importance, global SHAP, confusion matrix, CV results, multi-seed stability |
| **Business Impact & ROI** | Interactive what-if: move the threshold and cost assumptions, watch net savings / ROI recompute live |

```
streamlit_app/
├── app.py            # entry + navigation + global theme
├── theme.py          # design tokens, CSS, Plotly template
├── core.py           # cached model/data loaders, scoring, SHAP
└── views/            # one module per page
```

The app reads only the saved artifacts, so retrain anytime and the app reflects
the new model on next launch.

---

## Project structure

```
Telecom-Churn-Prediction/
├── train.py              # Full 14-phase training pipeline
├── stability_check.py    # Multi-seed robustness check
├── requirements.txt
├── data/
│   └── telco_churn.csv
├── artifacts/            # Models, metrics, params (JSON + joblib)
└── plots/                # All generated figures
```

### Artifacts produced

| File | Contents |
|---|---|
| `model.joblib` | Fitted pipeline (preprocessor + SMOTE + LightGBM) |
| `calibrator.joblib` | Fitted `CalibratedClassifierCV` |
| `optimal_threshold.json` | Chosen threshold + its recall/precision/cost |
| `best_params.json` | Best Optuna hyperparameters |
| `cv_results.json` | Per-fold + summary CV metrics |
| `final_metrics.json` | Hold-out test metrics |
| `business_impact.json` | Cost, revenue saved, net savings, ROI |
| `stability.json` | Multi-seed stability summary |
| `shap_background.joblib` | 200-sample background for SHAP |
| `feature_names.joblib` | Feature names after preprocessing |
| `config.json` | Full CONFIG snapshot |

### Plots

`optuna_history.png` · `cv_results.png` · `calibration_curve.png` ·
`threshold_vs_cost.png` · `shap_summary.png` · `feature_importance.png` ·
`confusion_matrix.png`

---

## Notes

- All randomness is seeded (`random_state=42`) for reproducibility.
- Business assumptions live in `CONFIG` — change `cost_fn` / `cost_fp` /
  `recall_target` to re-optimise for a different cost structure.
- A harmless `_count_physical_cores` traceback may print on Windows (joblib
  trying to count CPU cores); it does not affect results.
