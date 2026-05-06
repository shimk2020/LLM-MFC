# -*- coding: utf-8 -*-
"""
Model Interpretability Analysis
==============================================
For LightGBM with LLM Embedding (PLS 1-dim) + Numerical Features.

Standard SHAP beeswarm was not meaningful for embedding features
because "high/low embedding value" has no interpretable direction.
-->Methods INDEPENDENT of embedding value direction.

Methods:
  1. Permutation Feature Importance (group & individual)
  2. Controlled Category Impact (Partial Dependence for categories)
  3. Pairwise Counterfactual Analysis (test set)
  4. Group SHAP Contribution (text vs numerical)
  5. Comprehensive Dashboard Visualization
  
================================================================================
I will fix the data leakage from data imputation (can't impute the target features),
and will apply this interpretability analysis to the new model (if this method is okay...)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration (from lightGBM.ipynb)
# ==========================================

EMBEDDED_PATH = r"C:\Users\shimk\Desktop\LLM-BES 코드\머신러닝\PLS\1 dim\MFC_dataset_embedded_1dim.csv"
ORIGINAL_PATH = r"C:\Users\shimk\Desktop\LLM-BES 코드\머신러닝\MFC_dataset_imputed.csv"
TARGET_COL    = "max_power_density_w_m2"
TEST_SIZE     = 0.15
SPLIT_SEED    = 97

BEST_PARAMS = {
    'n_estimators': 1815, 'max_depth': 10, 'num_leaves': 105,
    'learning_rate': 0.16273608907354092,
    'subsample': 0.954958048911006, 'subsample_freq': 1,
    'colsample_bytree': 0.8580907820278014, 'min_child_samples': 6,
    'reg_alpha': 5.337179673315628e-06, 'reg_lambda': 0.13127897500060157,
    'min_split_gain': 0.10638481829595507, 'path_smooth': 0.0688167720487593,
    'max_bin': 50, 'objective': 'huber',
    'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
}

TEXT_COLS = {
    "substrate_type":   "substrate_type_1",
    "anode_material":   "anode_material_1",
    "cathode_material": "cathode_material_1",
}

NUMERICAL_COLS = [
    "initial_conc_g_cod_l", "anode_surface_area_m2",
    "anolyte_volume_ml", "cathode_surface_area_m2", "catholyte_volume_ml",
]

OUTPUT_DIR = "interpretability_results"


# ==========================================
# 2. Data Loading & Model Training
# ==========================================

def load_data():
    df_embedded = pd.read_csv(EMBEDDED_PATH)
    try:
        df_original = pd.read_csv(ORIGINAL_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df_original = pd.read_csv(ORIGINAL_PATH, encoding="cp949")
    assert len(df_embedded) == len(df_original)
    return df_embedded, df_original


def train_model(df_embedded):
    feature_names = df_embedded.drop(columns=[TARGET_COL]).columns.tolist()
    X = df_embedded[feature_names].values
    y_log = np.log1p(df_embedded[TARGET_COL].values)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=SPLIT_SEED, shuffle=True
    )
    idx_all = np.arange(len(X))
    idx_dev, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=SPLIT_SEED, shuffle=True
    )

    model = lgb.LGBMRegressor(**BEST_PARAMS)
    model.fit(X_dev, y_dev)

    print(f"  Train R² (log): {r2_score(y_dev, model.predict(X_dev)):.4f}")
    print(f"  Test  R² (log): {r2_score(y_test, model.predict(X_test)):.4f}")

    return model, feature_names, X, y_log, X_dev, X_test, y_dev, y_test, idx_dev, idx_test


def get_category_embedding_map(df_original, df_embedded, text_col, embed_col):
    """Build mapping: text_label -> embedding_value."""
    cat_map = {}
    for i in range(len(df_original)):
        cat_map[df_original[text_col].iloc[i]] = df_embedded[embed_col].iloc[i]
    return cat_map


# ==========================================
# Analysis 1: Permutation Feature Importance
# ==========================================

def analysis_permutation_importance(model, X_dev, y_dev, X_test, y_test, feature_names):
    """Group-level and individual permutation importance."""
    print("\n" + "=" * 70)
    print("  Analysis 1: Permutation Feature Importance")
    print("=" * 70)

    # Individual
    perm = permutation_importance(model, X_test, y_test,
                                  n_repeats=50, random_state=42, scoring='r2')
    sorted_idx = np.argsort(perm.importances_mean)[::-1]

    print("\n  Individual (ΔR² when shuffled):")
    for rank, idx in enumerate(sorted_idx, 1):
        tag = "[TEXT]" if feature_names[idx] in TEXT_COLS.values() else "[NUM] "
        print(f"    {rank}. {tag} {feature_names[idx]:<28} "
              f"ΔR²={perm.importances_mean[idx]:+.4f} ± {perm.importances_std[idx]:.4f}")

    # Group-level
    text_idx = [feature_names.index(v) for v in TEXT_COLS.values()]
    num_idx = [feature_names.index(c) for c in NUMERICAL_COLS if c in feature_names]
    baseline_r2 = r2_score(y_test, model.predict(X_test))
    rng = np.random.RandomState(42)

    text_drops, num_drops = [], []
    for _ in range(50):
        Xp = X_test.copy()
        for i in text_idx:
            Xp[:, i] = rng.permutation(Xp[:, i])
        text_drops.append(baseline_r2 - r2_score(y_test, model.predict(Xp)))

        Xp = X_test.copy()
        for i in num_idx:
            Xp[:, i] = rng.permutation(Xp[:, i])
        num_drops.append(baseline_r2 - r2_score(y_test, model.predict(Xp)))

    td, nd = np.mean(text_drops), np.mean(num_drops)
    ts, ns = np.std(text_drops), np.std(num_drops)
    print(f"\n  Group-level:")
    print(f"    Text (3 features):      ΔR² = {td:+.4f} ± {ts:.4f}")
    print(f"    Numerical (5 features): ΔR² = {nd:+.4f} ± {ns:.4f}")
    if td + nd > 0:
        print(f"    → Text contributes {td/(td+nd)*100:.1f}%, Numerical {nd/(td+nd)*100:.1f}%")

    return perm, td, nd, ts, ns


# ==========================================
# Analysis 2: Controlled Category Impact
# ==========================================

def analysis_category_impact(model, X, feature_names, df_embedded, df_original):
    """
    Partial Dependence for categories: replace ALL samples' text embedding
    with each category's embedding, average predictions.
    Independent of embedding direction — purely outcome-based.
    """
    print("\n" + "=" * 70)
    print("  Analysis 2: Controlled Category Impact (PD-based)")
    print("=" * 70)

    results = {}
    baseline_mean = np.mean(model.predict(X))

    for text_col, embed_col in TEXT_COLS.items():
        feat_idx = feature_names.index(embed_col)
        cat_map = get_category_embedding_map(df_original, df_embedded, text_col, embed_col)

        effects = []
        for cat, emb_val in cat_map.items():
            X_mod = X.copy()
            X_mod[:, feat_idx] = emb_val
            preds = model.predict(X_mod)
            count = (df_original[text_col] == cat).sum()
            effects.append({
                "category": cat,
                "mean_pred_log": np.mean(preds),
                "mean_pred_orig": np.mean(np.clip(np.expm1(preds), 0, None)),
                "delta": np.mean(preds) - baseline_mean,
                "count": count,
            })

        df_eff = pd.DataFrame(effects).sort_values("delta", ascending=False)
        results[text_col] = df_eff

        print(f"\n  {text_col.upper().replace('_', ' ')} (baseline={baseline_mean:.4f}):")
        print(f"  Top-5 POSITIVE:")
        for i, (_, r) in enumerate(df_eff.head(5).iterrows(), 1):
            print(f"    {i}. {r['category'][:48]:<50} Δ={r['delta']:>+.4f} (n={int(r['count'])})")
        print(f"  Top-5 NEGATIVE:")
        for i, (_, r) in enumerate(df_eff.sort_values("delta").head(5).iterrows(), 1):
            print(f"    {i}. {r['category'][:48]:<50} Δ={r['delta']:>+.4f} (n={int(r['count'])})")

    return results


# ==========================================
# Analysis 3: Counterfactual (Test Set)
# ==========================================

def analysis_counterfactual(model, X, feature_names, df_embedded, df_original, idx_test):
    """
    For each test sample: try every category, measure prediction change.
    Answers: "If we used material X instead, what happens?"
    """
    print("\n" + "=" * 70)
    print("  Analysis 3: Counterfactual Analysis (Test Set)")
    print("=" * 70)

    X_test = X[idx_test]
    orig_preds = model.predict(X_test)
    results = {}

    for text_col, embed_col in TEXT_COLS.items():
        feat_idx = feature_names.index(embed_col)
        cat_map = get_category_embedding_map(df_original, df_embedded, text_col, embed_col)

        cf = []
        for cat, emb_val in cat_map.items():
            X_cf = X_test.copy()
            X_cf[:, feat_idx] = emb_val
            deltas = model.predict(X_cf) - orig_preds
            count = (df_original[text_col] == cat).sum()
            cf.append({
                "category": cat, "mean_delta": np.mean(deltas),
                "median_delta": np.median(deltas), "std_delta": np.std(deltas),
                "count": count,
            })

        df_cf = pd.DataFrame(cf).sort_values("mean_delta", ascending=False)
        results[text_col] = df_cf

        print(f"\n  {text_col.upper().replace('_', ' ')}:")
        print(f"  'Switching to X changes prediction by Δ (avg over test set)'")
        print(f"  Top-5 BEST:")
        for i, (_, r) in enumerate(df_cf.head(5).iterrows(), 1):
            print(f"    {i}. {r['category'][:48]:<50} Δ={r['mean_delta']:>+.4f}±{r['std_delta']:.4f} (n={int(r['count'])})")
        print(f"  Top-5 WORST:")
        for i, (_, r) in enumerate(df_cf.tail(5).iterrows(), 1):
            print(f"    {i}. {r['category'][:48]:<50} Δ={r['mean_delta']:>+.4f}±{r['std_delta']:.4f} (n={int(r['count'])})")

    return results


# ==========================================
# Analysis 4: Group SHAP
# ==========================================

def analysis_group_shap(model, X_dev, feature_names):
    """SHAP aggregated by feature group (text vs numerical)."""
    print("\n" + "=" * 70)
    print("  Analysis 4: Group SHAP Contribution")
    print("=" * 70)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_dev)

    text_idx = [feature_names.index(v) for v in TEXT_COLS.values()]
    num_idx = [feature_names.index(c) for c in NUMERICAL_COLS if c in feature_names]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    text_shap = sum(mean_abs[i] for i in text_idx)
    num_shap = sum(mean_abs[i] for i in num_idx)
    total = text_shap + num_shap

    print(f"\n  Mean |SHAP| contribution:")
    print(f"    Text:      {text_shap:.4f} ({text_shap/total*100:.1f}%)")
    print(f"    Numerical: {num_shap:.4f} ({num_shap/total*100:.1f}%)")

    print(f"\n  Per-feature:")
    for i, fn in enumerate(feature_names):
        tag = "[TEXT]" if fn in TEXT_COLS.values() else "[NUM] "
        print(f"    {tag} {fn:<28} {mean_abs[i]:.4f}")

    return shap_values, mean_abs


# ==========================================
# Visualization: Comprehensive Dashboard
# ==========================================

def plot_dashboard(perm_result, text_drop, num_drop, text_std, num_std,
                   category_results, shap_values, mean_abs_shap,
                   feature_names, X_dev, df_original):
    """Generate multi-panel dashboard figure."""

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("Comprehensive Model Interpretability — LightGBM (PLS 1-dim)",
                 fontsize=18, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                           top=0.94, bottom=0.05, left=0.06, right=0.97)

    # ── Panel 1: Group Importance (Permutation) ──
    ax1 = fig.add_subplot(gs[0, 0])
    groups = ["Text\n(3 features)", "Numerical\n(5 features)"]
    vals = [text_drop, num_drop]
    stds = [text_std, num_std]
    colors = ["#E8636E", "#5B9BD5"]
    bars = ax1.bar(groups, vals, yerr=stds, color=colors, edgecolor="white",
                   linewidth=1.5, capsize=8, width=0.5)
    ax1.set_ylabel("ΔR² (Permutation Importance)", fontsize=11)
    ax1.set_title("(a) Feature Group Importance", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Panel 2: Individual Permutation Importance ──
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_idx = np.argsort(perm_result.importances_mean)
    fnames = [feature_names[i] for i in sorted_idx]
    fvals = [perm_result.importances_mean[i] for i in sorted_idx]
    fstds = [perm_result.importances_std[i] for i in sorted_idx]
    fcolors = ["#E8636E" if feature_names[i] in TEXT_COLS.values() else "#5B9BD5"
               for i in sorted_idx]
    ax2.barh(range(len(fnames)), fvals, xerr=fstds, color=fcolors,
             edgecolor="white", linewidth=0.8, capsize=3)
    ax2.set_yticks(range(len(fnames)))
    ax2.set_yticklabels(fnames, fontsize=9)
    ax2.set_xlabel("ΔR²", fontsize=11)
    ax2.set_title("(b) Individual Feature Importance", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    ax2.legend([Patch(color="#E8636E"), Patch(color="#5B9BD5")],
               ["Text Embedding", "Numerical"], fontsize=9, loc="lower right")

    # ── Panel 3: Group SHAP ──
    ax3 = fig.add_subplot(gs[0, 2])
    text_idx = [feature_names.index(v) for v in TEXT_COLS.values()]
    num_idx = [feature_names.index(c) for c in NUMERICAL_COLS if c in feature_names]
    text_shap_total = sum(mean_abs_shap[i] for i in text_idx)
    num_shap_total = sum(mean_abs_shap[i] for i in num_idx)
    sizes = [text_shap_total, num_shap_total]
    labels = [f"Text\n{text_shap_total:.3f}\n({text_shap_total/(text_shap_total+num_shap_total)*100:.1f}%)",
              f"Numerical\n{num_shap_total:.3f}\n({num_shap_total/(text_shap_total+num_shap_total)*100:.1f}%)"]
    ax3.pie(sizes, labels=labels, colors=["#E8636E", "#5B9BD5"],
            autopct="", startangle=90, textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax3.set_title("(c) Group SHAP Contribution\n(Mean |SHAP|)", fontsize=13, fontweight="bold")

    # ── Panels 4-6: Category Impact for each text feature ──
    text_col_list = list(TEXT_COLS.keys())
    panel_labels = ["(d)", "(e)", "(f)"]

    for panel_idx, text_col in enumerate(text_col_list):
        ax = fig.add_subplot(gs[1, panel_idx])
        df_eff = category_results[text_col].copy()

        # Show top-7 and bottom-7
        n_show = min(7, len(df_eff))
        top = df_eff.head(n_show)
        bot = df_eff.sort_values("delta").head(n_show)
        combined = pd.concat([top, bot]).drop_duplicates(subset="category")
        combined = combined.sort_values("delta")

        colors_bar = ["#E8636E" if v < 0 else "#5B9BD5" for v in combined["delta"]]
        ax.barh(range(len(combined)), combined["delta"], color=colors_bar,
                edgecolor="white", linewidth=0.5)
        labels_y = [f"{c[:35]}.. (n={int(n)})" if len(c) > 35 else f"{c} (n={int(n)})"
                    for c, n in zip(combined["category"], combined["count"])]
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(labels_y, fontsize=7)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Δ Prediction (log scale)", fontsize=10)
        title = text_col.replace("_", " ").title()
        ax.set_title(f"{panel_labels[panel_idx]} Category Impact: {title}",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    # ── Panels 7-9: Counterfactual heatmaps ──
    # Use the third row for a summary table or additional analysis
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis("off")

    # Create summary table
    summary_data = []
    for text_col in text_col_list:
        df_eff = category_results[text_col]
        best = df_eff.iloc[0]
        worst = df_eff.iloc[-1]
        spread = best["delta"] - worst["delta"]
        summary_data.append([
            text_col.replace("_", " ").title(),
            f"{best['category'][:40]}",
            f"{best['delta']:+.4f}",
            f"{worst['category'][:40]}",
            f"{worst['delta']:+.4f}",
            f"{spread:.4f}",
        ])

    col_labels = ["Feature", "Best Category", "Best Δ",
                  "Worst Category", "Worst Δ", "Spread"]
    table = ax_summary.table(cellText=summary_data, colLabels=col_labels,
                              loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax_summary.set_title("(g) Summary: Best vs Worst Category per Text Feature",
                         fontsize=13, fontweight="bold", pad=20)

    fname = os.path.join(OUTPUT_DIR, "comprehensive_dashboard.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_category_detail(category_results, counterfactual_results):
    """Individual detailed bar plots for each text feature."""
    for text_col in TEXT_COLS.keys():
        df_pd = category_results[text_col].sort_values("delta")
        df_cf = counterfactual_results[text_col].sort_values("mean_delta")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, len(df_pd)*0.35)))
        fig.suptitle(f"Category Analysis: {text_col.replace('_',' ').title()}",
                     fontsize=15, fontweight="bold")

        # Left: PD-based
        colors1 = ["#E8636E" if v < 0 else "#5B9BD5" for v in df_pd["delta"]]
        ax1.barh(range(len(df_pd)), df_pd["delta"], color=colors1,
                 edgecolor="white", linewidth=0.5)
        lbls1 = [f"{c[:45]} (n={int(n)})" for c, n in zip(df_pd["category"], df_pd["count"])]
        ax1.set_yticks(range(len(df_pd)))
        ax1.set_yticklabels(lbls1, fontsize=7)
        ax1.axvline(0, color="black", linewidth=0.8)
        ax1.set_xlabel("Δ from baseline (log scale)", fontsize=10)
        ax1.set_title("Controlled Category Impact\n(Partial Dependence)", fontsize=12)
        ax1.grid(axis="x", alpha=0.3)

        # Right: Counterfactual
        colors2 = ["#E8636E" if v < 0 else "#5B9BD5" for v in df_cf["mean_delta"]]
        ax2.barh(range(len(df_cf)), df_cf["mean_delta"], color=colors2,
                 edgecolor="white", linewidth=0.5,
                 xerr=df_cf["std_delta"], capsize=2)
        lbls2 = [f"{c[:45]} (n={int(n)})" for c, n in zip(df_cf["category"], df_cf["count"])]
        ax2.set_yticks(range(len(df_cf)))
        ax2.set_yticklabels(lbls2, fontsize=7)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Δ from original (log scale)", fontsize=10)
        ax2.set_title("Counterfactual Analysis\n(Test Set Average)", fontsize=12)
        ax2.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"category_detail_{text_col}.png")
        fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ==========================================
# Save Results to CSV
# ==========================================

def save_results(category_results, counterfactual_results, perm_result, feature_names):
    """Save all analysis results to CSV."""
    # Category impact
    for text_col, df_eff in category_results.items():
        fname = os.path.join(OUTPUT_DIR, f"category_impact_{text_col}.csv")
        df_eff.to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"  Saved: {fname}")

    # Counterfactual
    for text_col, df_cf in counterfactual_results.items():
        fname = os.path.join(OUTPUT_DIR, f"counterfactual_{text_col}.csv")
        df_cf.to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"  Saved: {fname}")

    # Permutation importance
    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
        "is_text": [fn in TEXT_COLS.values() for fn in feature_names],
    }).sort_values("importance_mean", ascending=False)
    fname = os.path.join(OUTPUT_DIR, "permutation_importance.csv")
    perm_df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"  Saved: {fname}")


# ==========================================
# Main
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Comprehensive Model Interpretability Analysis")
    print("=" * 70)

    # Load
    print("\n  Loading datasets...")
    df_embedded, df_original = load_data()
    print(f"  Embedded: {df_embedded.shape}, Original: {df_original.shape}")

    # Train
    print(f"\n{'-' * 70}")
    print("  Training LightGBM (latest params from notebook)...")
    print(f"{'-' * 70}")
    (model, feature_names, X, y_log,
     X_dev, X_test, y_dev, y_test, idx_dev, idx_test) = train_model(df_embedded)

    # Analysis 1
    perm, td, nd, ts, ns = analysis_permutation_importance(
        model, X_dev, y_dev, X_test, y_test, feature_names)

    # Analysis 2
    cat_results = analysis_category_impact(
        model, X, feature_names, df_embedded, df_original)

    # Analysis 3
    cf_results = analysis_counterfactual(
        model, X, feature_names, df_embedded, df_original, idx_test)

    # Analysis 4
    shap_values, mean_abs = analysis_group_shap(model, X_dev, feature_names)

    # Visualizations
    print(f"\n{'─' * 70}")
    print("  Generating Visualizations...")
    print(f"{'─' * 70}")
    plot_dashboard(perm, td, nd, ts, ns, cat_results,
                   shap_values, mean_abs, feature_names, X_dev, df_original)
    plot_category_detail(cat_results, cf_results)

    # Save CSVs
    print(f"\n{'─' * 70}")
    print("  Saving CSV results...")
    print(f"{'─' * 70}")
    save_results(cat_results, cf_results, perm, feature_names)

    print(f"\n{'=' * 70}")
    print("  Analysis complete! Results in: " + OUTPUT_DIR)
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
