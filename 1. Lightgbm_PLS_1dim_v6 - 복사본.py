"""
MFC LightGBM Regression with Optuna Hyperparameter Optimization
----------------------------------------------------------------
Predicts max_power_density_w_m2 from PLS 1-dimensional embeddings
using LightGBM with log-transformed target and Bayesian HP tuning.

Pipeline:
  1. Load dataset & log1p-transform the target
  2. Hold-out split: 85% dev / 15% test (seed=97)
  3. Dev split: 65% train_sub / 35% val_sub for Optuna
  4. Optuna TPE optimization (score = 0.65*val_R² + 0.35*train_R²)
  5. Evaluate on hold-out test set with bootstrap confidence intervals
  6. Visualize: Predicted vs Actual plot + SHAP summary plot
"""

import os
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. Configuration
# ==========================================

DATA_PATH   = r"C:\Users\shimk\Desktop\LLM-BES 코드\머신러닝\PLS\1 dim\MFC_dataset_embedded_1dim.csv"
TARGET_COL  = "max_power_density_w_m2"
TEST_SIZE   = 0.15   # hold-out test fraction (from full dataset)
VAL_SIZE    = 0.35   # validation fraction (from the 85% dev set)
SPLIT_SEED  = 97

N_TRIALS_PER_RESTART = 500
N_RESTARTS           = 2

OUTPUT_CSV  = "LGBM_PLS_1dim_results.csv"
FIG_PRED    = "LGBM_PLS_1dim_pred_vs_actual.png"
FIG_SHAP    = "LGBM_PLS_1dim_shap_summary.png"


# ==========================================
# 2. Bootstrap Evaluation
# ==========================================

def bootstrap_metrics(y_true, y_pred, metric_func, n_iterations=1000, random_state=42):
    """
    Stratified Prediction Bootstrapping for mean ± std of a metric.
    Stratifies on outlier threshold (y > 2.0) to preserve distribution.
    """
    np.random.seed(random_state)
    scores = []

    outlier_idx = np.where(y_true > 2.0)[0]
    normal_idx  = np.where(y_true <= 2.0)[0]

    if len(outlier_idx) == 0:
        outlier_idx = np.arange(len(y_true))
        normal_idx  = []

    for _ in range(n_iterations):
        if len(normal_idx) > 0:
            samp_outlier = resample(outlier_idx, replace=True)
            samp_normal  = resample(normal_idx, replace=True)
            indices = np.concatenate([samp_outlier, samp_normal])
        else:
            indices = resample(outlier_idx, replace=True)

        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)

    return np.mean(scores), np.std(scores)


# ==========================================
# 3. Optuna Optimization
# ==========================================

def optimize_hyperparameters(X_dev, y_dev):
    """
    Bayesian hyperparameter optimization using Optuna TPE sampler.
    Splits the dev set (85% of total) into train_sub (65%) / val_sub (35%)
    to avoid leaking the hold-out test set into HP search.
    Objective: weighted composite of val R² (65%) and train R² (35%).
    Multiple restarts with different sampler seeds for search diversity.
    """

    # ── Dev → train_sub / val_sub (65:35, fixed seed) ──
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_dev, y_dev, test_size=VAL_SIZE, random_state=SPLIT_SEED, shuffle=True
    )
    print(f"  Optuna dev split : {X_train_sub.shape[0]} train_sub / "
          f"{X_val.shape[0]} val_sub  (65:35 of dev set)")

    def objective(trial):
        obj_type = trial.suggest_categorical("obj_type", ["regression", "huber", "fair"])

        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 2000),
            "max_depth":         trial.suggest_int("max_depth", 2, 12),
            "num_leaves":        trial.suggest_int("num_leaves", 4, 127),
            "learning_rate":     trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":    trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.3),
            "path_smooth":       trial.suggest_float("path_smooth", 0.0, 1.0),
            "max_bin":           trial.suggest_int("max_bin", 31, 512),
            "objective":    obj_type,
            "random_state": 42,
            "n_jobs":       -1,
            "verbosity":    -1,
        }

        if params["num_leaves"] > 2 ** params["max_depth"]:
            raise optuna.TrialPruned()

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_sub, y_train_sub)

        train_r2 = r2_score(y_train_sub, model.predict(X_train_sub))
        val_r2   = r2_score(y_val,       model.predict(X_val))

        score = 0.65 * val_r2 + 0.35 * train_r2
        if train_r2 < 0.3:
            score -= 0.5 * (0.3 - train_r2)

        return score

    best_study = None
    best_value = -np.inf

    for restart_idx in range(N_RESTARTS):
        sampler_seed = 42 + restart_idx * 77
        print(f"  Restart {restart_idx + 1}/{N_RESTARTS} "
              f"(sampler_seed={sampler_seed}, trials={N_TRIALS_PER_RESTART})")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=sampler_seed,
                multivariate=True,
                group=True,
                n_startup_trials=50,
            )
        )
        study.optimize(objective, n_trials=N_TRIALS_PER_RESTART, show_progress_bar=False)

        if study.best_value > best_value:
            best_value = study.best_value
            best_study = study

    best_params = best_study.best_params
    best_params["objective"] = best_params.pop("obj_type")
    best_params["random_state"] = 42
    best_params["n_jobs"]       = -1
    best_params["verbosity"]    = -1

    return best_params, best_value


# ==========================================
# 4. Visualization
# ==========================================

def plot_pred_vs_actual(y_train_orig, pred_train, y_test_orig, pred_test,
                        train_r2, test_r2):
    """
    Scatter plot of Predicted vs Actual values for train and test sets.
    Includes 45° perfect-prediction line and R² annotations.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_train_orig, pred_train, c="#4C72B0", alpha=0.7,
              edgecolors="white", linewidths=0.5, s=80, label=f"Train (R²={train_r2:.4f})", zorder=3)
    ax.scatter(y_test_orig, pred_test, c="#DD5143", alpha=0.85,
              edgecolors="white", linewidths=0.5, s=100, marker="D", label=f"Test (R²={test_r2:.4f})", zorder=4)

    all_vals = np.concatenate([y_train_orig, pred_train, y_test_orig, pred_test])
    lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
    ax.plot([lo, hi], [lo, hi], "--", color="#333333", linewidth=1.5, alpha=0.6, label="Ideal (y = x)", zorder=2)

    ax.set_xlabel("Actual Max Power Density (W/m²)", fontsize=13)
    ax.set_ylabel("Predicted Max Power Density (W/m²)", fontsize=13)
    ax.set_title("LightGBM — Predicted vs Actual", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PRED, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_PRED}")


def plot_shap_summary(model, X_train, feature_names):
    """
    SHAP beeswarm summary plot showing feature importance and direction.
    Uses TreeExplainer for LightGBM (fast, exact).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("LightGBM — SHAP Summary (log-scale target)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_SHAP, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {FIG_SHAP}")


# ==========================================
# 5. Main Pipeline
# ==========================================

def main():
    print("=" * 60)
    print("  MFC LightGBM — PLS 1-dim Embedding")
    print("=" * 60)

    # ── Load Data ──
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    feature_names = df.drop(columns=[TARGET_COL]).columns.tolist()
    X = df[feature_names].values
    y = df[TARGET_COL].values

    print(f"\n  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Split   : {1 - TEST_SIZE:.0%} dev / {TEST_SIZE:.0%} hold-out test (seed={SPLIT_SEED})")

    # ── Log1p Transform ──
    y_log = np.log1p(y)
    print(f"  Target  : log1p transform (skew {pd.Series(y).skew():.1f} → {pd.Series(y_log).skew():.1f})")

    # ── Dev / Hold-out Test Split (85:15) ──
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=SPLIT_SEED, shuffle=True
    )
    print(f"  Dev     : {X_dev.shape[0]} samples  (used for Optuna + final training)")
    print(f"  Test    : {X_test.shape[0]} samples  (hold-out, untouched during HP search)\n")

    # ── Hyperparameter Optimization (uses only dev set) ──
    print("─" * 60)
    print("  Optuna Hyperparameter Search")
    print("─" * 60)

    best_params, composite_score = optimize_hyperparameters(X_dev, y_dev)

    # ── Train Final Model on full dev set ──
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_dev, y_dev)

    pred_dev_log  = model.predict(X_dev)
    pred_test_log = model.predict(X_test)

    # ── Inverse Transform ──
    pred_dev   = np.clip(np.expm1(pred_dev_log),  0, None)
    pred_test  = np.clip(np.expm1(pred_test_log), 0, None)
    y_dev_orig  = np.expm1(y_dev)
    y_test_orig = np.expm1(y_test)

    # ── Metrics (Original Scale) ──
    dev_r2    = r2_score(y_dev_orig,  pred_dev)
    test_r2   = r2_score(y_test_orig, pred_test)
    dev_rmse  = np.sqrt(mean_squared_error(y_dev_orig, pred_dev))
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, pred_test))
    dev_mae   = mean_absolute_error(y_dev_orig, pred_dev)
    test_mae  = mean_absolute_error(y_test_orig, pred_test)

    # ── Metrics (Log Scale) ──
    dev_r2_log  = r2_score(y_dev, pred_dev_log)
    test_r2_log = r2_score(y_test, pred_test_log)

    # ── Bootstrap Confidence Intervals ──
    _, dev_r2_std    = bootstrap_metrics(y_dev_orig,  pred_dev,  r2_score)
    _, test_r2_std   = bootstrap_metrics(y_test_orig, pred_test, r2_score)
    _, dev_rmse_std  = bootstrap_metrics(y_dev_orig,  pred_dev,
                                         lambda y, p: np.sqrt(mean_squared_error(y, p)))
    _, test_rmse_std = bootstrap_metrics(y_test_orig, pred_test,
                                         lambda y, p: np.sqrt(mean_squared_error(y, p)))
    _, dev_mae_std   = bootstrap_metrics(y_dev_orig,  pred_dev,  mean_absolute_error)
    _, test_mae_std  = bootstrap_metrics(y_test_orig, pred_test, mean_absolute_error)

    # ── Save Results ──
    results = {
        "split_seed": SPLIT_SEED,
        "composite_score": composite_score,
        "dev_r2":  dev_r2,  "dev_r2_std":  dev_r2_std,
        "test_r2": test_r2, "test_r2_std": test_r2_std,
        "dev_r2_log":  dev_r2_log,
        "test_r2_log": test_r2_log,
        "dev_rmse":  dev_rmse,  "dev_rmse_std":  dev_rmse_std,
        "test_rmse": test_rmse, "test_rmse_std": test_rmse_std,
        "dev_mae":  dev_mae,  "dev_mae_std":  dev_mae_std,
        "test_mae": test_mae, "test_mae_std": test_mae_std,
        "best_params": str(best_params),
    }
    pd.DataFrame([results]).to_csv(OUTPUT_CSV, index=False)

    # ── Print Report ──
    print(f"\n{'═' * 60}")
    print(f"  RESULTS")
    print(f"{'═' * 60}")
    print(f"  Test  R² (orig)    : {test_r2:.4f} ± {test_r2_std:.4f}")
    print(f"  Test  R² (log)     : {test_r2_log:.4f}")
    print(f"{'─' * 60}")
    print(f"  Test  RMSE         : {test_rmse:.4f} ± {test_rmse_std:.4f}")
    print(f"  Test  MAE          : {test_mae:.4f} ± {test_mae_std:.4f}")
    print(f"{'─' * 60}")
    print(f"  Best Params        : {best_params}")
    print(f"{'═' * 60}")
    print(f"\n  Results saved to: {OUTPUT_CSV}")

    # ── Visualizations ──
    print(f"\n{'─' * 60}")
    print(f"  Generating Visualizations")
    print(f"{'─' * 60}")
    plot_pred_vs_actual(y_dev_orig, pred_dev, y_test_orig, pred_test,
                        dev_r2, test_r2)
    plot_shap_summary(model, X_dev, feature_names)
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
