"""stage0_analysis.py — Stage 0 diagnostic analysis for the T-experiment plan.

Reads the CSV outputs from experiment_oracle_timestep.py and produces:
  A. Strategy comparison table (Fixed-999 vs Fixed-best-global vs SNR-T vs Oracle-T)
  B. T_oracle_posthoc distribution analysis
  C. Gain analysis (Oracle-T vs Fixed-999)
  D. Correlation summary
  E. Diagnostic conclusions

Usage:
    python src/stage0_analysis.py --sweep_dir ./results/stage0_posthoc_sweep
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


def load_sweep_csv(sweep_dir: str) -> pd.DataFrame:
    path = os.path.join(sweep_dir, "oracle_timestep_sweep.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sweep CSV not found: {path}")
    return pd.read_csv(path)


def load_summary_csv(sweep_dir: str) -> pd.DataFrame:
    path = os.path.join(sweep_dir, "oracle_timestep_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return pd.read_csv(path)


def load_correlation_csv(sweep_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(sweep_dir, "correlation_table.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def strategy_comparison(sweep_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """A. Compute per-strategy average metrics.

    Strategies:
      - Fixed-999: use T=999 for all images
      - Fixed-best-global: find the single T that maximizes average PSNR
      - SNR-T: use T_snr from summary (nearest candidate match)
      - Oracle-T: use per-(image, lambda) best T (LPIPS-based from summary)
    """
    metrics = ["PSNR", "MS-SSIM", "LPIPS", "DISTS"]
    t_candidates = sorted(sweep_df["T"].unique())

    # Fixed-999
    fixed_999 = sweep_df[sweep_df["T"] == 999][metrics].mean()

    # Fixed-best-global: find T that maximizes mean PSNR across all (image, lambda)
    mean_by_t = sweep_df.groupby("T")[metrics].mean()
    best_global_t = int(mean_by_t["PSNR"].idxmax())
    fixed_best = mean_by_t.loc[best_global_t]

    # SNR-T: for each (image, lambda), find the sweep row closest to T_snr
    snr_rows = []
    for _, row in summary_df.iterrows():
        img_id = row["image_id"]
        lmbda = row["lambda"]
        t_snr = int(row["T_snr"])
        nearest_t = min(t_candidates, key=lambda t: abs(t - t_snr))
        match = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == nearest_t)
        ]
        if not match.empty:
            snr_rows.append(match.iloc[0][metrics])
    snr_t_avg = pd.DataFrame(snr_rows).mean() if snr_rows else pd.Series(
        {m: float("nan") for m in metrics}
    )

    # Oracle-T: for each (image, lambda), use T_oracle_main from summary
    oracle_rows = []
    for _, row in summary_df.iterrows():
        img_id = row["image_id"]
        lmbda = row["lambda"]
        t_oracle = int(row["T_oracle_main"])
        match = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == t_oracle)
        ]
        if not match.empty:
            oracle_rows.append(match.iloc[0][metrics])
    oracle_avg = pd.DataFrame(oracle_rows).mean() if oracle_rows else pd.Series(
        {m: float("nan") for m in metrics}
    )

    result = pd.DataFrame(
        {
            "Fixed-999": fixed_999,
            f"Fixed-best-global (T={best_global_t})": fixed_best,
            "SNR-T (per-image)": snr_t_avg,
            "Oracle-T (per-image)": oracle_avg,
        }
    ).T
    result.index.name = "Strategy"
    return result


def oracle_distribution(summary_df: pd.DataFrame) -> Dict[str, object]:
    """B. Analyze T_oracle_posthoc distribution."""
    t_oracle = summary_df["T_oracle_main"].values.astype(float)
    total = len(t_oracle)

    concentration_999 = float(np.sum(t_oracle == 999)) / total if total > 0 else 0.0

    # Per-lambda-bin distribution
    lambda_bins = {}
    if "lambda" in summary_df.columns:
        log_lam = np.log(summary_df["lambda"].values.astype(float))
        edges = np.linspace(log_lam.min(), log_lam.max(), 5)
        bin_idx = np.digitize(log_lam, edges[1:-1])
        for bi in range(4):
            mask = bin_idx == bi
            if mask.sum() > 0:
                lam_lo = np.exp(edges[bi])
                lam_hi = np.exp(edges[bi + 1]) if bi + 1 < len(edges) else np.inf
                bin_vals = t_oracle[mask]
                lambda_bins[f"lambda[{lam_lo:.2f},{lam_hi:.2f})"] = {
                    "n": int(mask.sum()),
                    "mean_T_oracle": float(bin_vals.mean()),
                    "std_T_oracle": float(bin_vals.std()),
                    "pct_999": float(np.sum(bin_vals == 999)) / mask.sum(),
                }

    return {
        "total_samples": total,
        "mean_T_oracle": float(t_oracle.mean()),
        "std_T_oracle": float(t_oracle.std()),
        "median_T_oracle": float(np.median(t_oracle)),
        "concentration_at_999": concentration_999,
        "value_counts": dict(
            zip(*np.unique(t_oracle.astype(int), return_counts=True))
        ),
        "per_lambda_bin": lambda_bins,
    }


def gain_analysis(
    sweep_df: pd.DataFrame, summary_df: pd.DataFrame
) -> Dict[str, object]:
    """C. Oracle-T vs Fixed-999 per-image gain analysis."""
    metrics = ["PSNR", "LPIPS", "DISTS"]
    t_candidates = sorted(sweep_df["T"].unique())

    gains_psnr = []
    gains_lpips = []
    top_gains = []

    for _, row in summary_df.iterrows():
        img_id = row["image_id"]
        lmbda = row["lambda"]
        t_oracle = int(row["T_oracle_main"])

        fixed_row = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == 999)
        ]
        oracle_row = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == t_oracle)
        ]
        if fixed_row.empty or oracle_row.empty:
            continue

        psnr_gain = float(oracle_row.iloc[0]["PSNR"] - fixed_row.iloc[0]["PSNR"])
        lpips_gain = float(fixed_row.iloc[0]["LPIPS"] - oracle_row.iloc[0]["LPIPS"])
        gains_psnr.append(psnr_gain)
        gains_lpips.append(lpips_gain)
        top_gains.append({
            "image_id": img_id,
            "lambda": float(lmbda),
            "T_oracle": t_oracle,
            "psnr_gain": psnr_gain,
            "lpips_gain": lpips_gain,
        })

    gains_psnr = np.array(gains_psnr)
    gains_lpips = np.array(gains_lpips)

    top_gains.sort(key=lambda x: x["psnr_gain"], reverse=True)

    return {
        "n_samples": len(gains_psnr),
        "psnr_gain_mean": float(gains_psnr.mean()) if len(gains_psnr) > 0 else 0.0,
        "psnr_gain_std": float(gains_psnr.std()) if len(gains_psnr) > 0 else 0.0,
        "psnr_gain_max": float(gains_psnr.max()) if len(gains_psnr) > 0 else 0.0,
        "pct_positive_psnr_gain": float((gains_psnr > 0).mean()) if len(gains_psnr) > 0 else 0.0,
        "lpips_gain_mean": float(gains_lpips.mean()) if len(gains_lpips) > 0 else 0.0,
        "pct_positive_lpips_gain": float((gains_lpips > 0).mean()) if len(gains_lpips) > 0 else 0.0,
        "top10_gains": top_gains[:10],
    }


def correlation_summary(corr_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """D. Extract ALL feature correlations from correlation_table.csv."""
    if corr_df is None:
        return {}

    all_group = corr_df[corr_df["group"] == "all"]
    result = {}
    for _, row in all_group.iterrows():
        feat = row["feature"]
        p = row.get("pearson_corr", float("nan"))
        s = row.get("spearman_corr", float("nan"))
        if pd.notna(p):
            result[f"pearson({feat}, T_oracle)"] = float(p)
        if pd.notna(s):
            result[f"spearman({feat}, T_oracle)"] = float(s)
    return result


def advanced_analysis(summary_df: pd.DataFrame) -> Dict[str, object]:
    """F. Multi-variate and non-linear analysis of T_oracle predictability.

    Includes:
      - Random Forest / Linear Regression R² with all features
      - Per-lambda-bin correlation analysis
      - Interaction term analysis
      - Top feature ranking
    """
    target_col = "T_oracle_main"
    if target_col not in summary_df.columns:
        return {"error": f"{target_col} not found"}

    # Identify numeric feature columns
    exclude_cols = {
        "image_id", "lambda", "actual_bpp", "T_snr",
        "T_oracle_PSNR", "T_oracle_MS_SSIM", "T_oracle_LPIPS", "T_oracle_DISTS",
        "T_oracle_main", "best_score", "score_at_T_snr", "score_gap",
    }
    feature_cols = [c for c in summary_df.columns
                    if c not in exclude_cols
                    and summary_df[c].dtype in [np.float64, np.float32, float, int]]

    if not feature_cols:
        return {"error": "No numeric feature columns found"}

    X = summary_df[feature_cols].values.astype(float)
    y = summary_df[target_col].values.astype(float)
    valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]

    if len(y) < 10:
        return {"error": f"Too few valid samples ({len(y)})"}

    result = {
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
    }

    # --- Linear Regression ---
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        lr = LinearRegression()
        lr.fit(X, y)
        result["linear_r2_train"] = float(lr.score(X, y))

        n_splits = min(5, max(2, len(y) // 10))
        cv_scores = cross_val_score(lr, X, y, cv=n_splits, scoring="r2")
        result["linear_r2_cv_mean"] = float(cv_scores.mean())
        result["linear_r2_cv_std"] = float(cv_scores.std())

        # T_snr alone baseline
        if "T_snr" in summary_df.columns:
            t_snr_vals = summary_df["T_snr"].values[valid_mask].astype(float).reshape(-1, 1)
            lr_snr = LinearRegression()
            lr_snr.fit(t_snr_vals, y)
            result["linear_r2_T_snr_only"] = float(lr_snr.score(t_snr_vals, y))

        # Top linear coefficients
        coef_importance = np.abs(lr.coef_) * X.std(axis=0)
        sorted_idx = np.argsort(coef_importance)[::-1]
        result["linear_top_features"] = [
            {"feature": feature_cols[i], "standardized_coef": float(coef_importance[i])}
            for i in sorted_idx[:10]
        ]
    except ImportError:
        result["linear_r2_train"] = "sklearn not available"

    # --- Random Forest ---
    try:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        result["rf_r2_train"] = float(rf.score(X, y))

        cv_scores_rf = cross_val_score(rf, X, y, cv=n_splits, scoring="r2")
        result["rf_r2_cv_mean"] = float(cv_scores_rf.mean())
        result["rf_r2_cv_std"] = float(cv_scores_rf.std())

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        result["rf_top_features"] = [
            {"feature": feature_cols[i], "importance": float(importances[i])}
            for i in sorted_idx[:15]
        ]
    except ImportError:
        result["rf_r2_train"] = "sklearn not available"

    # --- Per-lambda-bin correlation ---
    if "lambda" in summary_df.columns:
        lam_valid = summary_df["lambda"].values[valid_mask]
        log_lam = np.log(lam_valid + 1e-8)
        n_bins = min(4, max(2, len(y) // 20))
        edges = np.linspace(log_lam.min(), log_lam.max(), n_bins + 1)
        bin_idx = np.digitize(log_lam, edges[1:-1])

        per_bin = {}
        for bi in range(n_bins):
            mask = bin_idx == bi
            if mask.sum() < 5:
                continue
            lam_lo = np.exp(edges[bi])
            lam_hi = np.exp(edges[bi + 1]) if bi + 1 < len(edges) else np.inf
            bin_y = y[mask]
            bin_label = f"lambda[{lam_lo:.2f},{lam_hi:.2f})"

            bin_corrs = {}
            for fi, feat_name in enumerate(feature_cols):
                feat_vals = X[mask, fi]
                if feat_vals.std() > 1e-12 and bin_y.std() > 1e-12:
                    r = float(np.corrcoef(feat_vals, bin_y)[0, 1])
                    bin_corrs[feat_name] = r

            # Sort by absolute correlation
            top_in_bin = sorted(bin_corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            per_bin[bin_label] = {
                "n": int(mask.sum()),
                "T_oracle_mean": float(bin_y.mean()),
                "T_oracle_std": float(bin_y.std()),
                "top_corr_features": top_in_bin,
            }
        result["per_lambda_bin"] = per_bin

    # --- Interaction terms ---
    interaction_results = []
    key_pairs = [
        ("latent_energy", "log_lambda"),
        ("latent_energy", "scales_mean"),
        ("latent_energy", "actual_bpp"),
        ("scales_std", "log_lambda"),
        ("res1_norm", "latent_energy"),
        ("delta_energy", "latent_energy"),
        ("cos_sample_x0", "latent_energy"),
    ]
    for f1, f2 in key_pairs:
        if f1 in feature_cols and f2 in feature_cols:
            i1 = feature_cols.index(f1)
            i2 = feature_cols.index(f2)
            interaction = X[:, i1] * X[:, i2]
            if interaction.std() > 1e-12:
                r = float(np.corrcoef(interaction, y)[0, 1])
                interaction_results.append({"pair": f"{f1} × {f2}", "pearson": r})
    interaction_results.sort(key=lambda x: abs(x["pearson"]), reverse=True)
    result["interaction_terms"] = interaction_results

    return result


def diagnostic_conclusion(
    oracle_dist: Dict[str, object],
    gains: Dict[str, object],
) -> str:
    """E. Generate diagnostic conclusion text."""
    lines = []
    lines.append("=" * 70)
    lines.append("STAGE 0 DIAGNOSTIC CONCLUSION")
    lines.append("=" * 70)

    conc_999 = oracle_dist["concentration_at_999"]
    mean_psnr_gain = gains["psnr_gain_mean"]
    pct_positive = gains["pct_positive_psnr_gain"]

    lines.append("")
    lines.append(f"T_oracle concentration at 999: {conc_999:.1%}")
    lines.append(f"Mean PSNR gain (Oracle vs Fixed-999): {mean_psnr_gain:.4f} dB")
    lines.append(f"Fraction of samples with positive PSNR gain: {pct_positive:.1%}")
    lines.append("")

    if conc_999 > 0.8:
        lines.append("[DIAGNOSIS] T_oracle_posthoc is heavily concentrated at 999.")
        lines.append("  -> The model has over-adapted to T=999 during training.")
        lines.append("  -> This does NOT mean timestep adaptation has no value.")
        lines.append("  -> Proceed to Stage 1 (T-augmentation training) to remove bias.")
        lines.append("  -> Do NOT use T_oracle_posthoc as PolicyNet supervision labels.")
    elif conc_999 > 0.5:
        lines.append("[DIAGNOSIS] T_oracle_posthoc is moderately concentrated at 999.")
        lines.append("  -> Training bias exists but is not extreme.")
        lines.append("  -> T-augmentation (Stage 1) is still recommended.")
        if mean_psnr_gain > 0.1:
            lines.append("  -> Meaningful gain exists even with biased model.")
    else:
        lines.append("[DIAGNOSIS] T_oracle_posthoc is dispersed across candidates.")
        lines.append("  -> Timestep adaptation has clear potential.")
        if mean_psnr_gain > 0.1:
            lines.append("  -> Strong gain signal; can proceed directly to Stage 3/4.")
        else:
            lines.append("  -> Gains are modest; T-augmentation may still help.")

    lines.append("")
    if mean_psnr_gain < 0.05:
        lines.append("[GAIN ASSESSMENT] Oracle-T gain over Fixed-999 is very small (<0.05 dB).")
        lines.append("  -> Timestep adaptation space may be limited for this model/task.")
        lines.append("  -> Consider: fixed T + uncertainty gate as alternative.")
    elif mean_psnr_gain < 0.2:
        lines.append("[GAIN ASSESSMENT] Oracle-T gain is moderate (0.05-0.2 dB).")
        lines.append("  -> Worthwhile to pursue, especially after T-augmentation.")
    else:
        lines.append(f"[GAIN ASSESSMENT] Oracle-T gain is substantial ({mean_psnr_gain:.3f} dB).")
        lines.append("  -> Strong motivation for learned timestep policy.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def write_report(
    sweep_dir: str,
    comparison: pd.DataFrame,
    oracle_dist: Dict[str, object],
    gains: Dict[str, object],
    corr_summary: Dict[str, float],
    conclusion: str,
    adv_analysis: Optional[Dict[str, object]] = None,
) -> str:
    """Write full Stage 0 diagnostic report to text file."""
    report_path = os.path.join(sweep_dir, "stage0_report.txt")
    lines = []

    lines.append("=" * 70)
    lines.append("STAGE 0: POSTHOC T SWEEP DIAGNOSTIC REPORT")
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
    lines.append("B. T_ORACLE_POSTHOC DISTRIBUTION")
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
    if oracle_dist["per_lambda_bin"]:
        lines.append("  Per-lambda-bin:")
        for bin_name, stats in oracle_dist["per_lambda_bin"].items():
            lines.append(
                f"    {bin_name}: n={stats['n']}, "
                f"mean_T={stats['mean_T_oracle']:.1f}, "
                f"std={stats['std_T_oracle']:.1f}, "
                f"pct_999={stats['pct_999']:.1%}"
            )
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
    if gains["top10_gains"]:
        lines.append("  Top-10 PSNR gains:")
        for i, g in enumerate(gains["top10_gains"]):
            lines.append(
                f"    {i+1:2d}. {g['image_id']} (lambda={g['lambda']:.2f}, "
                f"T_oracle={g['T_oracle']}) "
                f"PSNR+{g['psnr_gain']:.3f}dB, LPIPS-{g['lpips_gain']:.4f}"
            )
    lines.append("")

    # D. Full correlation table (sorted by |pearson|)
    lines.append("-" * 70)
    lines.append("D. ALL FEATURE CORRELATIONS (feature vs T_oracle)")
    lines.append("-" * 70)
    if corr_summary:
        # Group pearson and spearman together per feature
        features_seen = {}
        for key, val in corr_summary.items():
            if key.startswith("pearson("):
                feat = key[len("pearson("):-len(", T_oracle)")]
                features_seen.setdefault(feat, {})["pearson"] = val
            elif key.startswith("spearman("):
                feat = key[len("spearman("):-len(", T_oracle)")]
                features_seen.setdefault(feat, {})["spearman"] = val
        # Sort by |pearson|
        sorted_feats = sorted(features_seen.items(),
                              key=lambda kv: abs(kv[1].get("pearson", 0)), reverse=True)
        lines.append(f"  {'Feature':<35s} {'Pearson':>8s} {'Spearman':>9s}")
        lines.append(f"  {'-'*35} {'-'*8} {'-'*9}")
        for feat, vals in sorted_feats:
            p = vals.get("pearson", float("nan"))
            s = vals.get("spearman", float("nan"))
            lines.append(f"  {feat:<35s} {p:>8.4f} {s:>9.4f}")
    else:
        lines.append("  (correlation_table.csv not found)")
    lines.append("")

    # E. Advanced multi-variate analysis
    if adv_analysis and "error" not in adv_analysis:
        lines.append("-" * 70)
        lines.append("E. MULTI-VARIATE FEATURE IMPORTANCE ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"  N samples: {adv_analysis.get('n_samples', '?')}")
        lines.append(f"  N features: {adv_analysis.get('n_features', '?')}")
        lines.append("")

        lr_r2 = adv_analysis.get("linear_r2_cv_mean")
        rf_r2 = adv_analysis.get("rf_r2_cv_mean")
        snr_r2 = adv_analysis.get("linear_r2_T_snr_only")
        if isinstance(lr_r2, float):
            lines.append(f"  Linear Regression R² (CV): {lr_r2:.4f} ± {adv_analysis.get('linear_r2_cv_std', 0):.4f}")
        if isinstance(rf_r2, float):
            lines.append(f"  Random Forest R² (CV):     {rf_r2:.4f} ± {adv_analysis.get('rf_r2_cv_std', 0):.4f}")
        if isinstance(snr_r2, float):
            lines.append(f"  T_snr alone R²:            {snr_r2:.4f}")
            if isinstance(rf_r2, float):
                lines.append(f"  Multi-feature gain over T_snr: {rf_r2 - snr_r2:+.4f}")
        lines.append("")

        # RF top features
        rf_feats = adv_analysis.get("rf_top_features", [])
        if rf_feats:
            lines.append("  Random Forest top-15 features:")
            for i, f in enumerate(rf_feats[:15]):
                lines.append(f"    {i+1:2d}. {f['feature']:<35s} importance={f['importance']:.4f}")
            lines.append("")

        # Linear top features
        lr_feats = adv_analysis.get("linear_top_features", [])
        if lr_feats:
            lines.append("  Linear Regression top-10 (standardized |coef|):")
            for i, f in enumerate(lr_feats[:10]):
                lines.append(f"    {i+1:2d}. {f['feature']:<35s} |coef|={f['standardized_coef']:.4f}")
            lines.append("")

        # Per-lambda-bin analysis
        per_bin = adv_analysis.get("per_lambda_bin", {})
        if per_bin:
            lines.append("  Per-lambda-bin top correlations:")
            for bin_label, bin_info in per_bin.items():
                lines.append(f"    {bin_label} (n={bin_info['n']}, "
                             f"T_mean={bin_info['T_oracle_mean']:.1f}, "
                             f"T_std={bin_info['T_oracle_std']:.1f}):")
                for feat, corr_val in bin_info.get("top_corr_features", []):
                    lines.append(f"      {feat:<30s} r={corr_val:.4f}")
            lines.append("")

        # Interaction terms
        interactions = adv_analysis.get("interaction_terms", [])
        if interactions:
            lines.append("  Interaction term correlations with T_oracle:")
            for it in interactions:
                lines.append(f"    {it['pair']:<40s} pearson={it['pearson']:.4f}")
            lines.append("")
    elif adv_analysis and "error" in adv_analysis:
        lines.append("-" * 70)
        lines.append("E. MULTI-VARIATE ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"  Error: {adv_analysis['error']}")
        lines.append("")

    # F. Conclusion
    lines.append(conclusion)

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)
    return report_path


def make_plots(sweep_dir: str, sweep_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Generate Stage 0 diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return

    plot_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: T_oracle histogram
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    t_oracle = summary_df["T_oracle_main"].values
    t_candidates = sorted(sweep_df["T"].unique())
    ax.hist(t_oracle, bins=len(t_candidates), edgecolor="black", alpha=0.7)
    ax.set_xlabel("T_oracle_posthoc")
    ax.set_ylabel("Count")
    ax.set_title("Stage 0: Distribution of T_oracle_posthoc (fixed-999 model)")
    ax.axvline(x=999, color="red", linestyle="--", label="T=999")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "t_oracle_histogram.png"))
    plt.close(fig)

    # Plot 2: Per-T average PSNR
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    mean_by_t = sweep_df.groupby("T")["PSNR"].mean()
    ax.plot(mean_by_t.index, mean_by_t.values, "o-", markersize=6)
    ax.set_xlabel("Timestep T")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("Stage 0: Average PSNR vs Timestep (all images, all lambdas)")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "psnr_vs_timestep.png"))
    plt.close(fig)

    # Plot 3: Per-T average LPIPS
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    mean_by_t = sweep_df.groupby("T")["LPIPS"].mean()
    ax.plot(mean_by_t.index, mean_by_t.values, "o-", markersize=6, color="orange")
    ax.set_xlabel("Timestep T")
    ax.set_ylabel("Mean LPIPS (lower is better)")
    ax.set_title("Stage 0: Average LPIPS vs Timestep (all images, all lambdas)")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "lpips_vs_timestep.png"))
    plt.close(fig)

    # Plot 4: PSNR gain distribution
    gains_psnr = []
    for _, row in summary_df.iterrows():
        img_id = row["image_id"]
        lmbda = row["lambda"]
        t_oracle = int(row["T_oracle_main"])
        fixed_row = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == 999)
        ]
        oracle_row = sweep_df[
            (sweep_df["image_id"] == img_id)
            & (sweep_df["lambda"] == lmbda)
            & (sweep_df["T"] == t_oracle)
        ]
        if not fixed_row.empty and not oracle_row.empty:
            gains_psnr.append(
                float(oracle_row.iloc[0]["PSNR"] - fixed_row.iloc[0]["PSNR"])
            )

    if gains_psnr:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax.hist(gains_psnr, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_xlabel("PSNR gain (Oracle-T minus Fixed-999) [dB]")
        ax.set_ylabel("Count")
        ax.set_title("Stage 0: Per-image PSNR gain from Oracle-T")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "psnr_gain_histogram.png"))
        plt.close(fig)

    print(f"  Plots saved to: {plot_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 0 diagnostic analysis of posthoc T sweep results."
    )
    parser.add_argument(
        "--sweep_dir", type=str, required=True,
        help="Directory containing oracle_timestep_sweep.csv and oracle_timestep_summary.csv",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.sweep_dir}")
    sweep_df = load_sweep_csv(args.sweep_dir)
    summary_df = load_summary_csv(args.sweep_dir)
    corr_df = load_correlation_csv(args.sweep_dir)

    print(f"  Sweep rows: {len(sweep_df)}")
    print(f"  Summary rows: {len(summary_df)}")
    print(f"  T candidates: {sorted(sweep_df['T'].unique())}")

    print("\nComputing strategy comparison...")
    comparison = strategy_comparison(sweep_df, summary_df)
    print(comparison.to_string(float_format="%.4f"))

    print("\nAnalyzing T_oracle distribution...")
    oracle_dist = oracle_distribution(summary_df)
    print(f"  Concentration at 999: {oracle_dist['concentration_at_999']:.1%}")
    print(f"  Mean T_oracle: {oracle_dist['mean_T_oracle']:.1f}")

    print("\nComputing gain analysis...")
    gains = gain_analysis(sweep_df, summary_df)
    print(f"  Mean PSNR gain: {gains['psnr_gain_mean']:.4f} dB")
    print(f"  Pct positive: {gains['pct_positive_psnr_gain']:.1%}")

    print("\nExtracting correlation summary (all features)...")
    corr_summary = correlation_summary(corr_df)
    n_corr = len([k for k in corr_summary if k.startswith("pearson")])
    print(f"  {n_corr} features analyzed")

    print("\nRunning multi-variate feature importance analysis...")
    adv_analysis = advanced_analysis(summary_df)
    if "error" not in adv_analysis:
        lr_r2 = adv_analysis.get("linear_r2_cv_mean")
        rf_r2 = adv_analysis.get("rf_r2_cv_mean")
        snr_r2 = adv_analysis.get("linear_r2_T_snr_only")
        if isinstance(lr_r2, float):
            print(f"  Linear R² (CV): {lr_r2:.4f}")
        if isinstance(rf_r2, float):
            print(f"  RF R² (CV): {rf_r2:.4f}")
        if isinstance(snr_r2, float):
            print(f"  T_snr alone R²: {snr_r2:.4f}")
    else:
        print(f"  Error: {adv_analysis['error']}")

    print("\nGenerating diagnostic conclusion...")
    conclusion = diagnostic_conclusion(oracle_dist, gains)
    print(conclusion)

    print("\nWriting report...")
    report_path = write_report(
        args.sweep_dir, comparison, oracle_dist, gains, corr_summary, conclusion,
        adv_analysis=adv_analysis,
    )
    print(f"  Report saved to: {report_path}")

    if not args.no_plots:
        print("\nGenerating plots...")
        make_plots(args.sweep_dir, sweep_df, summary_df)

    print("\nStage 0 analysis complete.")


if __name__ == "__main__":
    main()
