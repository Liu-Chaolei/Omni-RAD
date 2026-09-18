"""stage2_analysis.py — Stage 2 diagnostic analysis for the T-experiment plan.

Extends Stage 0 analysis with:
  D. Cross-stage comparison (T_oracle_posthoc vs T_oracle_aug)
  E. Multi-factor feature importance (Linear Regression, Random Forest)
  F. Oracle label export for Stage 4 PolicyNet training

Usage:
    python src/stage2_analysis.py \
        --sweep_dir ./results/stage2_oracle_sweep \
        --stage0_dir ./results/stage0_posthoc_sweep
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: pip install pandas")

from stage0_analysis import (
    load_sweep_csv,
    load_summary_csv,
    load_correlation_csv,
    strategy_comparison,
    oracle_distribution,
    gain_analysis,
    correlation_summary,
)


# =========================================================================
# D. Cross-stage comparison
# =========================================================================

def cross_stage_comparison(
    stage2_summary: pd.DataFrame,
    stage0_summary: pd.DataFrame,
) -> Dict[str, object]:
    """Compare T_oracle_aug (Stage 2) vs T_oracle_posthoc (Stage 0)."""

    result = {}

    t_aug = stage2_summary["T_oracle_main"].values.astype(float)
    t_post = stage0_summary["T_oracle_main"].values.astype(float)

    result["stage2_mean"] = float(t_aug.mean())
    result["stage2_std"] = float(t_aug.std())
    result["stage0_mean"] = float(t_post.mean())
    result["stage0_std"] = float(t_post.std())
    result["stage2_pct_999"] = float((t_aug == 999).mean())
    result["stage0_pct_999"] = float((t_post == 999).mean())

    # Paired comparison on matching (image_id, lambda) pairs
    merged = pd.merge(
        stage2_summary[["image_id", "lambda", "T_oracle_main"]],
        stage0_summary[["image_id", "lambda", "T_oracle_main"]],
        on=["image_id", "lambda"],
        suffixes=("_aug", "_posthoc"),
        how="inner",
    )
    if len(merged) > 0:
        diff = merged["T_oracle_main_aug"].values - merged["T_oracle_main_posthoc"].values
        result["paired_n"] = len(merged)
        result["paired_mean_diff"] = float(diff.mean())
        result["paired_std_diff"] = float(diff.std())
        result["paired_pct_changed"] = float((diff != 0).mean())
        result["paired_pct_aug_lower"] = float((diff < 0).mean())
        result["paired_corr"] = float(np.corrcoef(
            merged["T_oracle_main_aug"].values.astype(float),
            merged["T_oracle_main_posthoc"].values.astype(float),
        )[0, 1]) if len(merged) > 2 else float("nan")
    else:
        result["paired_n"] = 0
        result["paired_mean_diff"] = float("nan")
        result["paired_std_diff"] = float("nan")
        result["paired_pct_changed"] = float("nan")
        result["paired_pct_aug_lower"] = float("nan")
        result["paired_corr"] = float("nan")

    # JS divergence between distributions
    all_t_values = sorted(set(t_aug.astype(int).tolist() + t_post.astype(int).tolist()))
    hist_aug = np.array([np.sum(t_aug.astype(int) == t) for t in all_t_values], dtype=float)
    hist_post = np.array([np.sum(t_post.astype(int) == t) for t in all_t_values], dtype=float)
    hist_aug /= hist_aug.sum() + 1e-12
    hist_post /= hist_post.sum() + 1e-12
    m = 0.5 * (hist_aug + hist_post)
    kl_aug_m = np.sum(hist_aug * np.log((hist_aug + 1e-12) / (m + 1e-12)))
    kl_post_m = np.sum(hist_post * np.log((hist_post + 1e-12) / (m + 1e-12)))
    result["js_divergence"] = float(0.5 * kl_aug_m + 0.5 * kl_post_m)

    return result


# =========================================================================
# E. Multi-factor feature importance
# =========================================================================

def feature_importance_analysis(summary_df: pd.DataFrame) -> Dict[str, object]:
    """Analyze which decoder-side features best predict T_oracle_aug."""
    target_col = "T_oracle_main"
    if target_col not in summary_df.columns:
        return {"error": "T_oracle_main not found in summary"}

    # Identify feature columns (exclude metadata columns)
    exclude_cols = {
        "image_id", "lambda", "actual_bpp", "T_snr",
        "T_oracle_PSNR", "T_oracle_MS_SSIM", "T_oracle_LPIPS", "T_oracle_DISTS",
        "T_oracle_main", "best_score", "score_at_T_snr", "score_gap",
    }
    feature_cols = [c for c in summary_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if summary_df[c].dtype in [np.float64, np.float32, float, int]]

    if not feature_cols:
        return {"error": "No numeric feature columns found"}

    X = summary_df[feature_cols].values.astype(float)
    y = summary_df[target_col].values.astype(float)

    # Remove rows with NaN
    valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]

    if len(y) < 10:
        return {"error": f"Too few valid samples ({len(y)})"}

    result = {"n_samples": len(y), "n_features": len(feature_cols), "feature_names": feature_cols}

    # Linear Regression R²
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        lr = LinearRegression()
        lr.fit(X, y)
        result["linear_r2_train"] = float(lr.score(X, y))

        cv_scores = cross_val_score(lr, X, y, cv=min(5, len(y) // 5), scoring="r2")
        result["linear_r2_cv_mean"] = float(cv_scores.mean())
        result["linear_r2_cv_std"] = float(cv_scores.std())

        # Single-variable R² for T_snr comparison
        if "T_snr" in summary_df.columns:
            t_snr_vals = summary_df["T_snr"].values[valid_mask].astype(float).reshape(-1, 1)
            lr_snr = LinearRegression()
            lr_snr.fit(t_snr_vals, y)
            result["linear_r2_T_snr_only"] = float(lr_snr.score(t_snr_vals, y))
    except ImportError:
        result["linear_r2_train"] = "sklearn not available"

    # Random Forest feature importance
    try:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        result["rf_r2_train"] = float(rf.score(X, y))

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        result["rf_top_features"] = [
            {"feature": feature_cols[i], "importance": float(importances[i])}
            for i in sorted_idx[:15]
        ]

        cv_scores_rf = cross_val_score(rf, X, y, cv=min(5, len(y) // 5), scoring="r2")
        result["rf_r2_cv_mean"] = float(cv_scores_rf.mean())
        result["rf_r2_cv_std"] = float(cv_scores_rf.std())
    except ImportError:
        result["rf_r2_train"] = "sklearn not available"

    return result


# =========================================================================
# F. Oracle label export
# =========================================================================

def export_oracle_labels(summary_df: pd.DataFrame, out_dir: str) -> str:
    """Export oracle labels + features for Stage 4 PolicyNet training."""
    exclude_cols = {
        "T_oracle_PSNR", "T_oracle_MS_SSIM", "T_oracle_LPIPS", "T_oracle_DISTS",
        "best_score", "score_at_T_snr", "score_gap",
    }
    keep_cols = [c for c in summary_df.columns if c not in exclude_cols]
    export_df = summary_df[keep_cols].copy()
    export_df.rename(columns={"T_oracle_main": "T_oracle_aug"}, inplace=True)

    out_path = os.path.join(out_dir, "oracle_labels.csv")
    export_df.to_csv(out_path, index=False)
    return out_path


# =========================================================================
# G. Diagnostic conclusion
# =========================================================================

def stage2_conclusion(
    oracle_dist: Dict[str, object],
    gains: Dict[str, object],
    cross_stage: Optional[Dict[str, object]],
    feat_importance: Dict[str, object],
) -> str:
    """Generate Stage 2 diagnostic conclusion."""
    lines = []
    lines.append("=" * 70)
    lines.append("STAGE 2 DIAGNOSTIC CONCLUSION")
    lines.append("=" * 70)
    lines.append("")

    conc_999 = oracle_dist["concentration_at_999"]
    mean_psnr_gain = gains["psnr_gain_mean"]

    # Cross-stage comparison
    if cross_stage and cross_stage.get("paired_n", 0) > 0:
        s0_pct = cross_stage["stage0_pct_999"]
        s2_pct = cross_stage["stage2_pct_999"]
        js = cross_stage["js_divergence"]
        pct_changed = cross_stage["paired_pct_changed"]

        lines.append(f"[CROSS-STAGE] T_oracle concentration at 999:")
        lines.append(f"  Stage 0 (fixed-999 model): {s0_pct:.1%}")
        lines.append(f"  Stage 2 (T-aug model):     {s2_pct:.1%}")
        lines.append(f"  JS divergence: {js:.4f}")
        lines.append(f"  Paired samples changed: {pct_changed:.1%}")
        lines.append("")

        if s0_pct > 0.7 and s2_pct < s0_pct - 0.15:
            lines.append("  -> T-augmentation successfully reduced training bias.")
            lines.append("  -> T_oracle_aug is more dispersed and trustworthy.")
        elif s2_pct > 0.7:
            lines.append("  -> T_oracle_aug still concentrated at 999 even after T-aug.")
            lines.append("  -> Timestep adaptation space may be genuinely limited.")
            lines.append("  -> Consider: fixed T + uncertainty gate as alternative.")
        else:
            lines.append("  -> Both stages show dispersed oracle distributions.")
    lines.append("")

    # Feature importance
    lines.append(f"[FEATURE IMPORTANCE]")
    lr_r2 = feat_importance.get("linear_r2_cv_mean")
    rf_r2 = feat_importance.get("rf_r2_cv_mean")
    snr_r2 = feat_importance.get("linear_r2_T_snr_only")

    if isinstance(lr_r2, float):
        lines.append(f"  Linear Regression R² (CV): {lr_r2:.4f}")
    if isinstance(rf_r2, float):
        lines.append(f"  Random Forest R² (CV):     {rf_r2:.4f}")
    if isinstance(snr_r2, float):
        lines.append(f"  T_snr alone R²:            {snr_r2:.4f}")
        if isinstance(lr_r2, float) and lr_r2 > snr_r2 + 0.05:
            lines.append("  -> Multi-factor features explain T_oracle_aug better than T_snr alone.")
        elif isinstance(snr_r2, float) and snr_r2 < 0.1:
            lines.append("  -> T_snr has very low explanatory power for T_oracle_aug.")
    lines.append("")

    top_feats = feat_importance.get("rf_top_features", [])
    if top_feats:
        lines.append("  Top-5 RF features:")
        for i, f in enumerate(top_feats[:5]):
            lines.append(f"    {i+1}. {f['feature']}: {f['importance']:.4f}")
    lines.append("")

    # Gain assessment
    lines.append(f"[GAIN ASSESSMENT]")
    lines.append(f"  Mean PSNR gain (Oracle-T vs Fixed-999): {mean_psnr_gain:.4f} dB")
    lines.append(f"  Pct positive: {gains['pct_positive_psnr_gain']:.1%}")
    if mean_psnr_gain > 0.2:
        lines.append("  -> Substantial gain; strong motivation for PolicyNet.")
    elif mean_psnr_gain > 0.05:
        lines.append("  -> Moderate gain; PolicyNet training worthwhile.")
    else:
        lines.append("  -> Small gain; consider fixed T + uncertainty gate.")
    lines.append("")

    # Overall recommendation
    lines.append("[RECOMMENDATION]")
    if isinstance(rf_r2, float) and rf_r2 > 0.3 and mean_psnr_gain > 0.05:
        lines.append("  -> Proceed to Stage 3 (E2E PolicyNet) and Stage 4 (Oracle-supervised).")
        lines.append("  -> oracle_labels.csv is ready for Stage 4 training.")
    elif mean_psnr_gain > 0.05:
        lines.append("  -> Proceed to Stage 3 (E2E PolicyNet) as primary route.")
        lines.append("  -> Feature predictability is limited; E2E may outperform supervised.")
    else:
        lines.append("  -> Timestep adaptation space is limited.")
        lines.append("  -> Consider alternative: fixed T + local uncertainty repair.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# =========================================================================
# Report writer
# =========================================================================

def write_stage2_report(
    sweep_dir: str,
    comparison: pd.DataFrame,
    oracle_dist: Dict[str, object],
    gains: Dict[str, object],
    corr_summary: Dict[str, float],
    cross_stage: Optional[Dict[str, object]],
    feat_importance: Dict[str, object],
    conclusion: str,
) -> str:
    """Write full Stage 2 diagnostic report."""
    report_path = os.path.join(sweep_dir, "stage2_report.txt")
    lines = []

    lines.append("=" * 70)
    lines.append("STAGE 2: T-AUGMENTATION MODEL ORACLE-T SWEEP REPORT")
    lines.append("=" * 70)
    lines.append("")

    # A. Strategy comparison
    lines.append("-" * 70)
    lines.append("A. STRATEGY COMPARISON TABLE")
    lines.append("-" * 70)
    lines.append(comparison.to_string(float_format="%.4f"))
    lines.append("")

    # B. Oracle distribution
    lines.append("-" * 70)
    lines.append("B. T_ORACLE_AUG DISTRIBUTION")
    lines.append("-" * 70)
    lines.append(f"  Total samples: {oracle_dist['total_samples']}")
    lines.append(f"  Mean T_oracle: {oracle_dist['mean_T_oracle']:.1f}")
    lines.append(f"  Std T_oracle:  {oracle_dist['std_T_oracle']:.1f}")
    lines.append(f"  Median T_oracle: {oracle_dist['median_T_oracle']:.0f}")
    lines.append(f"  Concentration at 999: {oracle_dist['concentration_at_999']:.1%}")
    lines.append("")
    lines.append("  Value counts:")
    for t_val, count in sorted(oracle_dist["value_counts"].items()):
        pct = count / oracle_dist["total_samples"] * 100
        bar = "#" * int(pct / 2)
        lines.append(f"    T={int(t_val):4d}: {count:4d} ({pct:5.1f}%) {bar}")
    lines.append("")

    # C. Gain analysis
    lines.append("-" * 70)
    lines.append("C. GAIN ANALYSIS (Oracle-T vs Fixed-999)")
    lines.append("-" * 70)
    lines.append(f"  N samples: {gains['n_samples']}")
    lines.append(f"  PSNR gain mean: {gains['psnr_gain_mean']:.4f} dB")
    lines.append(f"  PSNR gain std:  {gains['psnr_gain_std']:.4f} dB")
    lines.append(f"  PSNR gain max:  {gains['psnr_gain_max']:.4f} dB")
    lines.append(f"  Pct positive PSNR gain: {gains['pct_positive_psnr_gain']:.1%}")
    lines.append(f"  LPIPS gain mean: {gains['lpips_gain_mean']:.4f}")
    lines.append(f"  Pct positive LPIPS gain: {gains['pct_positive_lpips_gain']:.1%}")
    lines.append("")

    # D. Cross-stage comparison
    lines.append("-" * 70)
    lines.append("D. CROSS-STAGE COMPARISON (Stage 0 vs Stage 2)")
    lines.append("-" * 70)
    if cross_stage:
        lines.append(f"  Stage 0 mean T_oracle: {cross_stage['stage0_mean']:.1f}")
        lines.append(f"  Stage 2 mean T_oracle: {cross_stage['stage2_mean']:.1f}")
        lines.append(f"  Stage 0 pct at 999: {cross_stage['stage0_pct_999']:.1%}")
        lines.append(f"  Stage 2 pct at 999: {cross_stage['stage2_pct_999']:.1%}")
        lines.append(f"  JS divergence: {cross_stage['js_divergence']:.4f}")
        if cross_stage.get("paired_n", 0) > 0:
            lines.append(f"  Paired samples: {cross_stage['paired_n']}")
            lines.append(f"  Paired mean diff (aug - posthoc): {cross_stage['paired_mean_diff']:.1f}")
            lines.append(f"  Paired pct changed: {cross_stage['paired_pct_changed']:.1%}")
            lines.append(f"  Paired correlation: {cross_stage['paired_corr']:.4f}")
    else:
        lines.append("  (Stage 0 results not available for comparison)")
    lines.append("")

    # E. Feature importance
    lines.append("-" * 70)
    lines.append("E. MULTI-FACTOR FEATURE IMPORTANCE")
    lines.append("-" * 70)
    if "error" in feat_importance:
        lines.append(f"  Error: {feat_importance['error']}")
    else:
        lines.append(f"  N samples: {feat_importance.get('n_samples', '?')}")
        lines.append(f"  N features: {feat_importance.get('n_features', '?')}")
        lr_r2 = feat_importance.get("linear_r2_cv_mean")
        if isinstance(lr_r2, float):
            lines.append(f"  Linear Regression R² (CV): {lr_r2:.4f} +/- {feat_importance.get('linear_r2_cv_std', 0):.4f}")
        rf_r2 = feat_importance.get("rf_r2_cv_mean")
        if isinstance(rf_r2, float):
            lines.append(f"  Random Forest R² (CV): {rf_r2:.4f} +/- {feat_importance.get('rf_r2_cv_std', 0):.4f}")
        snr_r2 = feat_importance.get("linear_r2_T_snr_only")
        if isinstance(snr_r2, float):
            lines.append(f"  T_snr alone R²: {snr_r2:.4f}")
        lines.append("")
        top_feats = feat_importance.get("rf_top_features", [])
        if top_feats:
            lines.append("  Random Forest top features:")
            for i, f in enumerate(top_feats[:15]):
                lines.append(f"    {i+1:2d}. {f['feature']:30s} importance={f['importance']:.4f}")
    lines.append("")

    # F. Correlation summary
    lines.append("-" * 70)
    lines.append("F. KEY CORRELATIONS (feature vs T_oracle_aug)")
    lines.append("-" * 70)
    if corr_summary:
        for key, val in corr_summary.items():
            lines.append(f"  {key}: {val:.4f}")
    else:
        lines.append("  (correlation_table.csv not found)")
    lines.append("")

    # G. Conclusion
    lines.append(conclusion)

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)
    return report_path


def make_stage2_plots(
    sweep_dir: str,
    sweep_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stage0_summary: Optional[pd.DataFrame],
) -> None:
    """Generate Stage 2 diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return

    plot_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    t_candidates = sorted(sweep_df["T"].unique())

    # Plot 1: T_oracle_aug histogram
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    t_oracle = summary_df["T_oracle_main"].values
    ax.hist(t_oracle, bins=len(t_candidates), edgecolor="black", alpha=0.7, label="T_oracle_aug")
    ax.set_xlabel("T_oracle")
    ax.set_ylabel("Count")
    ax.set_title("Stage 2: Distribution of T_oracle_aug (T-aug model)")
    ax.axvline(x=999, color="red", linestyle="--", alpha=0.5, label="T=999")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "t_oracle_aug_histogram.png"))
    plt.close(fig)

    # Plot 2: Cross-stage comparison histogram
    if stage0_summary is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
        t_post = stage0_summary["T_oracle_main"].values
        axes[0].hist(t_post, bins=len(t_candidates), edgecolor="black", alpha=0.7, color="C0")
        axes[0].set_xlabel("T_oracle_posthoc")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Stage 0: Fixed-999 model")
        axes[0].axvline(x=999, color="red", linestyle="--", alpha=0.5)

        axes[1].hist(t_oracle, bins=len(t_candidates), edgecolor="black", alpha=0.7, color="C1")
        axes[1].set_xlabel("T_oracle_aug")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Stage 2: T-aug model")
        axes[1].axvline(x=999, color="red", linestyle="--", alpha=0.5)

        fig.suptitle("Cross-Stage: T_oracle distribution comparison")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "cross_stage_histogram.png"))
        plt.close(fig)

    # Plot 3: Per-T average metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    mean_by_t = sweep_df.groupby("T")[["PSNR", "LPIPS"]].mean()
    axes[0].plot(mean_by_t.index, mean_by_t["PSNR"].values, "o-", markersize=6)
    axes[0].set_xlabel("Timestep T")
    axes[0].set_ylabel("Mean PSNR (dB)")
    axes[0].set_title("Average PSNR vs Timestep")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(mean_by_t.index, mean_by_t["LPIPS"].values, "o-", markersize=6, color="orange")
    axes[1].set_xlabel("Timestep T")
    axes[1].set_ylabel("Mean LPIPS (lower is better)")
    axes[1].set_title("Average LPIPS vs Timestep")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "metrics_vs_timestep.png"))
    plt.close(fig)

    # Plot 4: Feature importance bar chart
    try:
        from sklearn.ensemble import RandomForestRegressor

        exclude_cols = {
            "image_id", "lambda", "actual_bpp", "T_snr",
            "T_oracle_PSNR", "T_oracle_MS_SSIM", "T_oracle_LPIPS", "T_oracle_DISTS",
            "T_oracle_main", "best_score", "score_at_T_snr", "score_gap",
        }
        feature_cols = [c for c in summary_df.columns if c not in exclude_cols
                        and summary_df[c].dtype in [np.float64, np.float32, float, int]]
        if feature_cols:
            X = summary_df[feature_cols].values.astype(float)
            y = summary_df["T_oracle_main"].values.astype(float)
            valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
            X, y = X[valid], y[valid]
            if len(y) > 10:
                rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                importances = rf.feature_importances_
                sorted_idx = np.argsort(importances)[::-1][:15]

                fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
                names = [feature_cols[i] for i in sorted_idx]
                vals = [importances[i] for i in sorted_idx]
                ax.barh(range(len(names)), vals, align="center")
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Feature Importance")
                ax.set_title("Random Forest: Top-15 features for predicting T_oracle_aug")
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, "feature_importance.png"))
                plt.close(fig)
    except ImportError:
        pass

    # Plot 5: PSNR gain distribution
    gains_psnr = []
    for _, row in summary_df.iterrows():
        img_id = row["image_id"]
        lmbda = row["lambda"]
        t_oracle_val = int(row["T_oracle_main"])
        fixed_row = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == 999)
        ]
        oracle_row = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == t_oracle_val)
        ]
        if not fixed_row.empty and not oracle_row.empty:
            gains_psnr.append(float(oracle_row.iloc[0]["PSNR"] - fixed_row.iloc[0]["PSNR"]))

    if gains_psnr:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax.hist(gains_psnr, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_xlabel("PSNR gain (Oracle-T minus Fixed-999) [dB]")
        ax.set_ylabel("Count")
        ax.set_title("Stage 2: Per-image PSNR gain from Oracle-T_aug")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "psnr_gain_histogram.png"))
        plt.close(fig)

    print(f"  Plots saved to: {plot_dir}")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 diagnostic analysis of T-aug model Oracle-T sweep."
    )
    parser.add_argument(
        "--sweep_dir", type=str, required=True,
        help="Directory containing Stage 2 oracle_timestep_sweep.csv and oracle_timestep_summary.csv",
    )
    parser.add_argument(
        "--stage0_dir", type=str, default=None,
        help="Directory containing Stage 0 results for cross-stage comparison",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    print(f"Loading Stage 2 data from: {args.sweep_dir}")
    sweep_df = load_sweep_csv(args.sweep_dir)
    summary_df = load_summary_csv(args.sweep_dir)
    corr_df = load_correlation_csv(args.sweep_dir)

    print(f"  Sweep rows: {len(sweep_df)}")
    print(f"  Summary rows: {len(summary_df)}")
    print(f"  T candidates: {sorted(sweep_df['T'].unique())}")

    # Load Stage 0 for cross-comparison
    stage0_summary = None
    if args.stage0_dir and os.path.isdir(args.stage0_dir):
        try:
            stage0_summary = load_summary_csv(args.stage0_dir)
            print(f"  Stage 0 summary rows: {len(stage0_summary)}")
        except FileNotFoundError:
            print("  [WARNING] Stage 0 summary not found, skipping cross-stage comparison")

    print("\nA. Computing strategy comparison...")
    comparison = strategy_comparison(sweep_df, summary_df)
    print(comparison.to_string(float_format="%.4f"))

    print("\nB. Analyzing T_oracle_aug distribution...")
    oracle_dist = oracle_distribution(summary_df)
    print(f"  Concentration at 999: {oracle_dist['concentration_at_999']:.1%}")
    print(f"  Mean T_oracle: {oracle_dist['mean_T_oracle']:.1f}")

    print("\nC. Computing gain analysis...")
    gains = gain_analysis(sweep_df, summary_df)
    print(f"  Mean PSNR gain: {gains['psnr_gain_mean']:.4f} dB")
    print(f"  Pct positive: {gains['pct_positive_psnr_gain']:.1%}")

    print("\nD. Cross-stage comparison...")
    cross_stage = None
    if stage0_summary is not None:
        cross_stage = cross_stage_comparison(summary_df, stage0_summary)
        print(f"  Stage 0 pct@999: {cross_stage['stage0_pct_999']:.1%}")
        print(f"  Stage 2 pct@999: {cross_stage['stage2_pct_999']:.1%}")
        print(f"  JS divergence: {cross_stage['js_divergence']:.4f}")
    else:
        print("  (skipped — no Stage 0 data)")

    print("\nE. Multi-factor feature importance...")
    feat_importance = feature_importance_analysis(summary_df)
    if "error" not in feat_importance:
        lr_r2 = feat_importance.get("linear_r2_cv_mean")
        rf_r2 = feat_importance.get("rf_r2_cv_mean")
        snr_r2 = feat_importance.get("linear_r2_T_snr_only")
        if isinstance(lr_r2, float):
            print(f"  Linear R² (CV): {lr_r2:.4f}")
        if isinstance(rf_r2, float):
            print(f"  RF R² (CV): {rf_r2:.4f}")
        if isinstance(snr_r2, float):
            print(f"  T_snr alone R²: {snr_r2:.4f}")
    else:
        print(f"  Error: {feat_importance['error']}")

    print("\nF. Exporting oracle labels...")
    corr_summary_dict = correlation_summary(corr_df)
    labels_path = export_oracle_labels(summary_df, args.sweep_dir)
    print(f"  Exported to: {labels_path}")

    print("\nG. Generating conclusion...")
    conclusion = stage2_conclusion(oracle_dist, gains, cross_stage, feat_importance)
    print(conclusion)

    print("\nWriting report...")
    report_path = write_stage2_report(
        args.sweep_dir, comparison, oracle_dist, gains,
        corr_summary_dict, cross_stage, feat_importance, conclusion,
    )
    print(f"  Report saved to: {report_path}")

    if not args.no_plots:
        print("\nGenerating plots...")
        make_stage2_plots(args.sweep_dir, sweep_df, summary_df, stage0_summary)

    print("\nStage 2 analysis complete.")


if __name__ == "__main__":
    main()
