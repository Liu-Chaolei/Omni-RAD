"""Quantify how much of T_opt's variance is driven by image content vs λ.

Consumes the CSV outputs of experiment_timesteps.py (per_image.csv,
sweep.csv) and produces:
  - R² comparison: logλ → T_calc  vs  logλ → T_opt
  - Variance decomposition: σ(T_opt | fixed λ) vs σ(T_opt | fixed image)
  - Two-way ANOVA on T_opt with η²(image) and η²(λ)
  - Optional: image-complexity metrics correlated with T_opt (per λ)
  - Distortion-vs-T curves for the most/least complex images

Usage:
    python src/experiment_timesteps_content.py \
        --per_image_csv  results/timesteps/per_image.csv \
        --sweep_csv      results/timesteps/sweep.csv \
        --img_dir        /path/to/Kodak24/HR \
        --out_dir        results/timesteps/content
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def r2_comparison(df: pd.DataFrame) -> dict:
    log_lam = np.log(df["lambda"].to_numpy(dtype=np.float64))
    t_calc = df["T_calc"].to_numpy(dtype=np.float64)
    t_opt = df["T_opt"].to_numpy(dtype=np.float64)

    r_calc, _ = stats.pearsonr(log_lam, t_calc)
    r_opt, _ = stats.pearsonr(log_lam, t_opt)

    return {
        "n_samples": int(len(df)),
        "pearson_logLam_Tcalc": float(r_calc),
        "pearson_logLam_Topt": float(r_opt),
        "R2_logLam_explains_Tcalc": float(r_calc ** 2),
        "R2_logLam_explains_Topt": float(r_opt ** 2),
        "interpretation": (
            f"logλ explains {r_calc**2*100:.1f}% of the variance in T_calc "
            f"but only {r_opt**2*100:.1f}% of the variance in T_opt — "
            f"the remaining {(1-r_opt**2)*100:.1f}% must come from image "
            f"content or noise."
        ),
    }


def variance_decomposition(df: pd.DataFrame) -> dict:
    sigma_image_at_fixed_lam = df.groupby("lambda")["T_opt"].std(ddof=1)
    sigma_lam_at_fixed_image = df.groupby("image")["T_opt"].std(ddof=1)

    return {
        "mean_sigma_T_opt_across_images_given_lambda": float(
            sigma_image_at_fixed_lam.mean()
        ),
        "mean_sigma_T_opt_across_lambdas_given_image": float(
            sigma_lam_at_fixed_image.mean()
        ),
        "ratio_image_over_lambda": float(
            sigma_image_at_fixed_lam.mean()
            / max(sigma_lam_at_fixed_image.mean(), 1e-9)
        ),
        "per_lambda_sigma_image": {
            f"{k:.3f}": float(v) for k, v in sigma_image_at_fixed_lam.items()
        },
        "per_image_sigma_lambda": {
            str(k): float(v) for k, v in sigma_lam_at_fixed_image.items()
        },
    }


def two_way_anova(df: pd.DataFrame) -> dict:
    # Balanced two-way ANOVA without interaction; computed manually so the
    # script has no statsmodels dependency. Sum-of-squares formulas:
    #   SS_total = Σ (y - ȳ)²
    #   SS_A     = n_b * Σ_a (ȳ_a - ȳ)²        (image)
    #   SS_B     = n_a * Σ_b (ȳ_b - ȳ)²        (lambda)
    #   SS_res   = SS_total - SS_A - SS_B
    y = df["T_opt"].to_numpy(dtype=np.float64)
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())

    image_means = df.groupby("image")["T_opt"].mean()
    lam_means = df.groupby("lambda")["T_opt"].mean()
    n_a = len(image_means)
    n_b = len(lam_means)

    counts_image = df.groupby("image").size()
    counts_lam = df.groupby("lambda").size()
    balanced = (counts_image.nunique() == 1) and (counts_lam.nunique() == 1)

    ss_image = float(
        (counts_image.values * (image_means - grand) ** 2).sum()
    )
    ss_lam = float((counts_lam.values * (lam_means - grand) ** 2).sum())
    ss_res = max(ss_total - ss_image - ss_lam, 0.0)

    df_image = n_a - 1
    df_lam = n_b - 1
    df_res = max(len(y) - n_a - n_b + 1, 1)

    ms_image = ss_image / max(df_image, 1)
    ms_lam = ss_lam / max(df_lam, 1)
    ms_res = ss_res / max(df_res, 1)

    f_image = ms_image / ms_res if ms_res > 0 else float("inf")
    f_lam = ms_lam / ms_res if ms_res > 0 else float("inf")

    p_image = (
        float(1 - stats.f.cdf(f_image, df_image, df_res))
        if np.isfinite(f_image)
        else 0.0
    )
    p_lam = (
        float(1 - stats.f.cdf(f_lam, df_lam, df_res))
        if np.isfinite(f_lam)
        else 0.0
    )

    return {
        "balanced_design": bool(balanced),
        "n_images": int(n_a),
        "n_lambdas": int(n_b),
        "SS_total": ss_total,
        "SS_image": ss_image,
        "SS_lambda": ss_lam,
        "SS_residual": ss_res,
        "eta2_image": ss_image / ss_total if ss_total > 0 else 0.0,
        "eta2_lambda": ss_lam / ss_total if ss_total > 0 else 0.0,
        "eta2_residual": ss_res / ss_total if ss_total > 0 else 0.0,
        "F_image": float(f_image),
        "F_lambda": float(f_lam),
        "p_image": p_image,
        "p_lambda": p_lam,
        "interpretation": (
            f"η²(image)={ss_image/ss_total*100:.1f}% vs "
            f"η²(λ)={ss_lam/ss_total*100:.1f}%: "
            f"per-image content explains "
            f"{(ss_image/max(ss_lam,1e-9)):.2f}× as much variance in T_opt "
            f"as the bitrate factor."
        ),
    }


def _image_complexity(img_path: str) -> Optional[dict]:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(img_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float64)
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None

    # Laplacian variance via 3x3 kernel (no cv2 dependency).
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    pad = np.pad(gray, 1, mode="reflect")
    lap = (
        k[0, 1] * pad[0:-2, 1:-1]
        + k[1, 0] * pad[1:-1, 0:-2]
        + k[1, 1] * pad[1:-1, 1:-1]
        + k[1, 2] * pad[1:-1, 2:]
        + k[2, 1] * pad[2:, 1:-1]
    )
    lap_var = float(lap.var())

    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=False)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())

    # JPEG Q=75 byte size as a proxy for compressibility.
    try:
        import io
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
        jpeg_bytes = float(buf.tell()) / (rgb.shape[0] * rgb.shape[1])
    except Exception:
        jpeg_bytes = float("nan")

    return {
        "laplacian_var": lap_var,
        "entropy_8bit": entropy,
        "jpeg_q75_bpp": jpeg_bytes * 8.0,
    }


def complexity_correlations(
    df: pd.DataFrame,
    img_dir: str,
    out_csv: str,
) -> dict:
    metrics: dict[str, dict] = {}
    for image in sorted(df["image"].unique()):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            cand = os.path.join(img_dir, str(image) + ext)
            if os.path.isfile(cand):
                m = _image_complexity(cand)
                if m is not None:
                    metrics[image] = m
                break

    if not metrics:
        return {"available": False, "reason": "no images matched"}

    metric_df = pd.DataFrame.from_dict(metrics, orient="index")
    metric_df.index.name = "image"
    metric_df.to_csv(out_csv)

    rows = []
    for lam in sorted(df["lambda"].unique()):
        sub = df[df["lambda"] == lam].copy()
        sub = sub.merge(metric_df, left_on="image", right_index=True, how="inner")
        if len(sub) < 4:
            continue
        for col in metric_df.columns:
            x = sub[col].to_numpy(dtype=np.float64)
            y = sub["T_opt"].to_numpy(dtype=np.float64)
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                continue
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            rows.append({
                "lambda": float(lam),
                "metric": col,
                "n": int(len(sub)),
                "pearson": float(r_p),
                "pearson_p": float(p_p),
                "spearman": float(r_s),
                "spearman_p": float(p_s),
            })

    if not rows:
        return {"available": False, "reason": "insufficient samples per λ"}

    corr_df = pd.DataFrame(rows)
    corr_csv = out_csv.replace(".csv", "_corr.csv")
    corr_df.to_csv(corr_csv, index=False)

    pooled = []
    full = df.merge(metric_df, left_on="image", right_index=True, how="inner")
    for col in metric_df.columns:
        x = full[col].to_numpy(dtype=np.float64)
        y = full["T_opt"].to_numpy(dtype=np.float64)
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        r_p, _ = stats.pearsonr(x, y)
        r_s, _ = stats.spearmanr(x, y)
        pooled.append({
            "metric": col,
            "pearson_pooled": float(r_p),
            "spearman_pooled": float(r_s),
        })

    return {
        "available": True,
        "metrics_csv": out_csv,
        "per_lambda_corr_csv": corr_csv,
        "pooled_correlations": pooled,
        "n_images_with_metrics": int(len(metrics)),
    }


def plot_extreme_curves(
    sweep: pd.DataFrame,
    df: pd.DataFrame,
    out_path: str,
) -> Optional[str]:
    image_topt_var = df.groupby("image")["T_opt"].var().sort_values()
    if len(image_topt_var) < 2:
        return None

    low_image = image_topt_var.index[0]
    high_image = image_topt_var.index[-1]

    lam_counts = sweep.groupby("lambda").size()
    target_lam = float(lam_counts.idxmax())

    sub = sweep[
        (sweep["lambda"] == target_lam)
        & (sweep["image"].isin([low_image, high_image]))
    ].copy()
    if sub.empty:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for img_name, grp in sub.groupby("image"):
        grp_sorted = grp.sort_values("T")
        argmin_t = grp_sorted.loc[grp_sorted["distortion"].idxmin(), "T"]
        ax.plot(
            grp_sorted["T"], grp_sorted["distortion"],
            marker="o", markersize=3, linewidth=1.2,
            label=f"{img_name}  (argmin T={int(argmin_t)})",
        )
    ax.set_xlabel("Timestep T")
    ax.set_ylabel("Distortion")
    ax.set_title(
        f"Distortion vs T at λ={target_lam:.3f}\n"
        f"low-var image: {low_image} | high-var image: {high_image}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_image_csv)
    df["lambda"] = df["lambda"].astype(float)

    summary: dict = {
        "input": {
            "per_image_csv": args.per_image_csv,
            "sweep_csv": args.sweep_csv,
            "img_dir": args.img_dir,
        }
    }

    print("=" * 72)
    print("  R² comparison: logλ vs T_calc / T_opt")
    print("=" * 72)
    summary["r2_comparison"] = r2_comparison(df)
    print(json.dumps(summary["r2_comparison"], indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print("  Variance decomposition: σ(T_opt | fixed λ) vs σ(T_opt | fixed image)")
    print("=" * 72)
    summary["variance_decomposition"] = variance_decomposition(df)
    print(
        f"  mean σ across images (fixed λ): "
        f"{summary['variance_decomposition']['mean_sigma_T_opt_across_images_given_lambda']:.3f}"
    )
    print(
        f"  mean σ across λ (fixed image):  "
        f"{summary['variance_decomposition']['mean_sigma_T_opt_across_lambdas_given_image']:.3f}"
    )
    print(
        f"  ratio image/λ:                  "
        f"{summary['variance_decomposition']['ratio_image_over_lambda']:.3f}"
    )

    print("\n" + "=" * 72)
    print("  Two-way ANOVA on T_opt (image + λ)")
    print("=" * 72)
    summary["anova"] = two_way_anova(df)
    print(json.dumps(summary["anova"], indent=2, ensure_ascii=False))

    if args.img_dir and os.path.isdir(args.img_dir):
        print("\n" + "=" * 72)
        print("  Image-complexity correlations with T_opt")
        print("=" * 72)
        out_csv = os.path.join(args.out_dir, "image_complexity.csv")
        summary["complexity"] = complexity_correlations(df, args.img_dir, out_csv)
        print(json.dumps(summary["complexity"], indent=2, ensure_ascii=False))
    else:
        summary["complexity"] = {"available": False, "reason": "img_dir not provided"}

    if os.path.isfile(args.sweep_csv):
        print("\n" + "=" * 72)
        print("  Distortion-vs-T curve for low/high T_opt-variance images")
        print("=" * 72)
        sweep = pd.read_csv(args.sweep_csv)
        sweep["lambda"] = sweep["lambda"].astype(float)
        plot_path = os.path.join(args.out_dir, "extreme_images_curve.png")
        produced = plot_extreme_curves(sweep, df, plot_path)
        summary["extreme_curve_plot"] = produced
        print(f"  Plot → {produced}")
    else:
        summary["extreme_curve_plot"] = None

    out_json = os.path.join(args.out_dir, "content_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary → {out_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variance decomposition of T_opt: image content vs λ.",
    )
    parser.add_argument("--per_image_csv", type=str, required=True)
    parser.add_argument("--sweep_csv", type=str, required=True)
    parser.add_argument(
        "--img_dir", type=str, default="",
        help="Optional dir containing the source images; enables complexity stats.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/timesteps/content",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
