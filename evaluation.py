"""
evaluation.py
-------------
Comprehensive evaluation suite for the ad bidding system.
Produces:
  - Per-model metrics table
  - Per-user-profile CTR analysis
  - Per-bidder auction stats
  - Matplotlib report figures saved to outputs/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve


# ── visual style ──────────────────────────────────────────────────────────────
PALETTE = ["#00C9FF", "#FF6B6B", "#FFD93D", "#6BCB77", "#845EC2",
           "#F9A825", "#26C6DA", "#EF5350"]
plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#C9D1D9",
    "grid.color":       "#21262D",
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
})


def _save(fig, name: str, out_dir: str = "outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fp = f"{out_dir}/{name}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Eval] Saved → {fp}")
    return fp


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Model comparison dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results: dict,
                          out_dir: str = "outputs") -> str:
    """
    Bar charts comparing all trained models on key metrics.
    results = {"model_name": {"val": {...}, "test": {...}}, ...}
    """
    metrics = ["auc_roc", "auc_pr", "log_loss", "lift_top10pct"]
    labels  = ["AUC-ROC", "AUC-PR", "Log Loss ↓", "Lift@10%"]
    model_names = list(results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("MODEL COMPARISON — TEST SET", fontsize=14,
                 fontweight="bold", color="#C9D1D9", y=1.02)

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [results[m]["test"][metric] for m in model_names]
        bars = ax.barh(model_names, vals,
                       color=PALETTE[:n_models], edgecolor="none", height=0.55)
        ax.set_title(label, fontsize=10, color="#58A6FF")
        ax.set_xlabel(label, fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", ha="left", fontsize=8,
                    color="#C9D1D9")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "01_model_comparison", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – ROC + PR curves for best model
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_pr(model, X_test: np.ndarray, y_test: np.ndarray,
                model_name: str = "best_model",
                out_dir: str = "outputs") -> str:
    y_prob = model.predict_proba(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ROC & PR CURVES  —  {model_name.upper()}",
                 fontsize=12, fontweight="bold", color="#C9D1D9")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    from sklearn.metrics import auc as sk_auc
    auc_val = sk_auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#00C9FF", lw=2, label=f"AUC = {auc_val:.4f}")
    ax1.plot([0, 1], [0, 1], ":", color="#8B949E", lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.08, color="#00C9FF")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve", color="#58A6FF")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # PR
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = (prec * np.diff(np.concatenate([[0], rec]))).sum()
    ax2.plot(rec, prec, color="#FF6B6B", lw=2, label=f"AP = {ap:.4f}")
    ax2.fill_between(rec, prec, alpha=0.08, color="#FF6B6B")
    ax2.axhline(y_test.mean(), color="#8B949E", ls=":", lw=1,
                label=f"Baseline = {y_test.mean():.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", color="#58A6FF")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "02_roc_pr_curves", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Per-profile CTR prediction quality
# ─────────────────────────────────────────────────────────────────────────────
def plot_profile_analysis(model,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          meta_test: pd.DataFrame,
                          out_dir: str = "outputs") -> str:
    if meta_test is None or "_profile" not in meta_test.columns:
        print("  [Eval] No profile metadata — skipping profile plot.")
        return ""

    y_prob = model.predict_proba(X_test)
    meta = meta_test.copy()
    meta["y_true"] = y_test
    meta["y_prob"] = y_prob

    profiles  = meta["_profile"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PER-PROFILE CTR ANALYSIS", fontsize=12,
                 fontweight="bold", color="#C9D1D9")

    # actual vs predicted CTR per profile
    grp = meta.groupby("_profile").agg(
        actual_ctr=("y_true", "mean"),
        pred_ctr=("y_prob", "mean"),
        n=("y_true", "count")
    ).reset_index()

    x = np.arange(len(grp))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, grp["actual_ctr"] * 100, w,
           label="Actual CTR",    color="#00C9FF", alpha=0.85)
    ax.bar(x + w/2, grp["pred_ctr"] * 100,   w,
           label="Predicted CTR", color="#FF6B6B", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(grp["_profile"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("CTR (%)")
    ax.set_title("Actual vs Predicted CTR by User Profile",
                 color="#58A6FF", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # pCTR distribution violin per profile
    ax2 = axes[1]
    data_list = [meta[meta["_profile"] == p]["y_prob"].values
                 for p in grp["_profile"]]
    parts = ax2.violinplot(data_list, positions=x, widths=0.6,
                           showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], PALETTE):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("#FFD93D")
    ax2.set_xticks(x)
    ax2.set_xticklabels(grp["_profile"], rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Predicted Click Probability")
    ax2.set_title("pCTR Distribution by Profile", color="#58A6FF", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, "03_profile_analysis", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Bidder performance comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_bidder_performance(bidders: list,
                            out_dir: str = "outputs") -> str:
    stats = [b.stats() for b in bidders]
    df    = pd.DataFrame(stats)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("BIDDER STRATEGY PERFORMANCE", fontsize=13,
                 fontweight="bold", color="#C9D1D9")

    metrics_titles = [
        ("won",           "Impressions Won"),
        ("spent_$",       "Total Spend ($)"),
        ("ctr",           "Click-Through Rate"),
        ("cvr",           "Conversion Rate"),
        ("cpc_$",         "Cost Per Click ($)"),
        ("budget_used_%", "Budget Used (%)"),
    ]

    for ax, (col, title) in zip(axes.flat, metrics_titles):
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
        ax.bar(df["name"], df[col], color=colors, edgecolor="none")
        ax.set_title(title, color="#58A6FF", fontsize=10)
        ax.set_xticklabels(df["name"], rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(df[col]):
            ax.text(i, v * 1.02, f"{v:.3g}", ha="center",
                    fontsize=7, color="#C9D1D9")

    fig.tight_layout()
    return _save(fig, "04_bidder_performance", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Feature importances (LightGBM only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list,
                            top_n: int = 20,
                            out_dir: str = "outputs") -> str:
    if not hasattr(model, "feature_importances"):
        return ""

    imp_df = model.feature_importances(feature_names).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                   color=PALETTE[0], edgecolor="none")
    ax.set_title(f"TOP {top_n} FEATURE IMPORTANCES (LightGBM)",
                 color="#58A6FF", fontsize=11)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "05_feature_importance", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Calibration curve
# ─────────────────────────────────────────────────────────────────────────────
def plot_calibration(model, X_test: np.ndarray, y_test: np.ndarray,
                     model_name: str = "model",
                     out_dir: str = "outputs") -> str:
    y_prob = model.predict_proba(X_test)
    fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=15)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], ":", color="#8B949E", lw=1.5, label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="#00C9FF", lw=2,
            ms=6, label=model_name)
    ax.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.1, color="#00C9FF")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("MODEL CALIBRATION CURVE", color="#58A6FF", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "06_calibration_curve", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluation runner
# ─────────────────────────────────────────────────────────────────────────────
def run_full_evaluation(trained_models: dict,
                        results: dict,
                        best_name: str,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        meta_test,
                        pipeline,
                        bidders: list,
                        out_dir: str = "outputs") -> dict:
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    paths = {}
    paths["model_comparison"] = plot_model_comparison(results, out_dir)
    paths["roc_pr"] = plot_roc_pr(
        trained_models[best_name], X_test, y_test, best_name, out_dir
    )
    paths["profile_analysis"] = plot_profile_analysis(
        trained_models[best_name], X_test, y_test, meta_test, out_dir
    )
    paths["bidder_perf"] = plot_bidder_performance(bidders, out_dir)
    paths["feat_imp"] = plot_feature_importance(
        trained_models[best_name],
        pipeline.feature_names,
        out_dir=out_dir,
    )
    paths["calibration"] = plot_calibration(
        trained_models[best_name], X_test, y_test, best_name, out_dir
    )

    # metrics table
    rows = []
    for name, r in results.items():
        row = {"model": name}
        for split in ["val", "test"]:
            for k, v in r[split].items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{out_dir}/metrics_summary.csv", index=False)
    print(f"  [Eval] Metrics saved → {out_dir}/metrics_summary.csv")

    return {"paths": paths, "metrics": metrics_df}
