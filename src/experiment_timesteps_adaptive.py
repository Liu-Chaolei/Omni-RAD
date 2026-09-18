"""
experiment_timesteps_adaptive.py — T 是内容自适应的 (T is content-adaptive)

Implements the design described in src/experiment_timesteps_adaptive.md:

  Goal:
    Show that the analytical starting timestep T computed by
    DynamicTimestepModule (SNR-based, derived from the entropy-model σ)
    is *content-adaptive* — i.e. complex images need stronger generation
    (different T) than simple images.

  Procedure (per image × per λ):
    1. Compute image complexity metrics on the original RGB image
       (independent of λ):
         • Canny edge density
         • FFT high-frequency energy ratio
         • DCT high-frequency energy ratio
    2. Run encoder + LatentCodec once to obtain T_calc (and bpp).

  Reported statistics (across all (image, λ) samples and per-λ):
    • std / variance / range of T_calc
    • Pearson  / Spearman corr(T_calc, complexity)  for each metric
    • Per-λ Pearson / Spearman correlations
    • Per-image rank correlation (across λ) — sanity check

Run:
    CUDA_VISIBLE_DEVICES=0 python src/experiment_timesteps_adaptive.py \\
        --base_config_file ./configs/base.yaml \\
        --test_config_file ./configs/test.yaml \\
        --out_dir          ./results/experiment_timesteps_adaptive
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from torch_ema import ExponentialMovingAverage
from torchvision import transforms


# ---------------------------------------------------------------------------
# Small helpers (mirrored from experiment_timesteps.py)
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def preprocess_image(image_path: str, transform) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return safe_corr(rx, ry)


# ---------------------------------------------------------------------------
# Image complexity metrics (operate on the original PIL image / RGB ndarray)
# ---------------------------------------------------------------------------

def _to_gray_uint8(img_pil: Image.Image) -> np.ndarray:
    """RGB PIL → grayscale uint8 ndarray, shape [H, W]."""
    gray = img_pil.convert("L")
    return np.asarray(gray, dtype=np.uint8)


def compute_canny_edge_density(img_pil: Image.Image,
                               low: int = 100, high: int = 200) -> float:
    """Fraction of pixels classified as edges by Canny.

    Falls back to a Sobel-magnitude threshold if cv2 is unavailable.
    """
    gray = _to_gray_uint8(img_pil)
    try:
        import cv2  # type: ignore

        edges = cv2.Canny(gray, low, high)
        return float((edges > 0).mean())
    except ImportError:
        gx = np.zeros_like(gray, dtype=np.float64)
        gy = np.zeros_like(gray, dtype=np.float64)
        gx[:, 1:-1] = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
        gy[1:-1, :] = gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64)
        mag = np.sqrt(gx * gx + gy * gy)
        thr = mag.mean() + mag.std()
        return float((mag > thr).mean())


def compute_fft_high_freq_ratio(img_pil: Image.Image,
                                cutoff: float = 0.25) -> float:
    """Fraction of FFT spectral energy lying outside the centred low-pass
    disk of radius ``cutoff * min(H, W) / 2``.

    cutoff ∈ (0, 1).  cutoff=0.25 → keep top 75% of frequencies as 'high'.
    """
    gray = _to_gray_uint8(img_pil).astype(np.float64) / 255.0
    H, W = gray.shape

    F2 = np.fft.fft2(gray - gray.mean())
    F2 = np.fft.fftshift(F2)
    power = np.abs(F2) ** 2

    yy, xx = np.indices((H, W), dtype=np.float64)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rr = np.sqrt(((yy - cy) / max(H, 1)) ** 2 + ((xx - cx) / max(W, 1)) ** 2)
    radius = cutoff * 0.5
    high_mask = rr > radius

    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[high_mask].sum() / total)


def compute_dct_high_freq_ratio(img_pil: Image.Image,
                                cutoff: float = 0.25) -> float:
    """Fraction of DCT-II energy outside the top-left low-frequency block.

    Block size = ``cutoff * H`` × ``cutoff * W``.
    cutoff ∈ (0, 1).  cutoff=0.25 → keep first 25% in each dimension as 'low'.
    """
    gray = _to_gray_uint8(img_pil).astype(np.float64) / 255.0
    gray = gray - gray.mean()

    try:
        from scipy.fft import dctn  # type: ignore

        D = dctn(gray, type=2, norm="ortho")
    except ImportError:
        # 1-D DCT-II via FFT, applied along each axis.
        def _dct1d(x: np.ndarray, axis: int) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            N = x.shape[axis]
            v = np.concatenate([x, np.flip(x, axis=axis)], axis=axis)
            V = np.fft.fft(v, axis=axis)
            k = np.arange(N)
            shape = [1] * x.ndim
            shape[axis] = N
            phase = np.exp(-1j * np.pi * k / (2 * N)).reshape(shape)
            X = (V[(slice(None),) * axis + (slice(0, N),)] * phase).real
            scale = np.full(N, np.sqrt(2.0 / N))
            scale[0] = np.sqrt(1.0 / N)
            return X * scale.reshape(shape)

        D = _dct1d(_dct1d(gray, axis=0), axis=1)

    H, W = D.shape
    h_low = max(1, int(round(cutoff * H)))
    w_low = max(1, int(round(cutoff * W)))

    power = D ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    low_energy = power[:h_low, :w_low].sum()
    return float((total - low_energy) / total)


def compute_complexity_metrics(img_pil: Image.Image,
                               fft_cutoff: float = 0.25,
                               dct_cutoff: float = 0.25) -> dict:
    return {
        "canny_density":  compute_canny_edge_density(img_pil),
        "fft_hf_ratio":   compute_fft_high_freq_ratio(img_pil, cutoff=fft_cutoff),
        "dct_hf_ratio":   compute_dct_high_freq_ratio(img_pil, cutoff=dct_cutoff),
    }


# ---------------------------------------------------------------------------
# Codec forward — only T_calc + bpp are needed for this experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def codec_forward_T(net,
                    img_padded: torch.Tensor,
                    ori_h: int, ori_w: int,
                    lmbda_tensor: torch.Tensor) -> Tuple[float, float]:
    """Run encoder + LatentCodec once, return (T_calc, bpp)."""
    latent2 = net.aux_codec((img_padded + 1) / 2).detach()
    lq_latent = net.vae.encode(img_padded).latent_dist.mode() * net.vae.config.scaling_factor

    _, rate_out, _, T_calc = net.codec(
        lq_latent, latent2, ori_h, ori_w, lmbda_tensor
    )

    return float(T_calc[0].item()), float(rate_out.quantized_total_bpp.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(base_config: dict, test_config: dict, stage: int, out_dir: str,
         fft_cutoff: float, dct_cutoff: float) -> None:
    if base_config.get("global_seed") is not None:
        set_seed(base_config["global_seed"])

    os.makedirs(out_dir, exist_ok=True)

    from StableCodec_variable2_step import StableCodec

    net = StableCodec(
        sd_path=test_config["model"]["sd_path"],
        config=test_config["model"],
        stage=stage,
    )

    if test_config["model"].get("codec_path") is not None and test_config.get("use_ema", False):
        ckpt = torch.load(test_config["model"]["codec_path"], map_location="cpu")
        ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
        ema_net.load_state_dict(ckpt["ema_state_dict"])
        ema_net.copy_to(net.parameters())
        del ckpt, ema_net

    net.cuda().eval()

    if test_config.get("enable_xformers_memory_efficient_attention", False):
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    device = next(net.parameters()).device

    # ---- Image set ----
    images = sorted(
        glob.glob(base_config["test_dataset"] + "/*.png")
        + glob.glob(base_config["test_dataset"] + "/*.jpg")
    )
    print(f"\nFound {len(images)} images in {base_config['test_dataset']}\n")

    # ---- λ values ----
    lambda_min = float(test_config["model"].get("lambda_min", 0.5))
    lambda_max = float(test_config["model"].get("lambda_max", 32.0))
    num_lambdas = int(test_config.get("num_lambdas", 6))

    if test_config.get("lambda_list") is not None:
        lambda_list = [float(v) for v in test_config["lambda_list"]]
    else:
        lambda_list = np.exp(np.linspace(
            np.log(lambda_min), np.log(lambda_max), num_lambdas
        )).tolist()

    print(f"Evaluating {len(lambda_list)} lambda values: "
          f"{[f'{v:.3f}' for v in lambda_list]}\n")

    # ---- Pre-compute complexity metrics for every image (λ-independent) ----
    print("Computing image complexity metrics ...")
    img_complexity: dict = {}
    for img_path in images:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        img_pil = Image.open(img_path).convert("RGB")
        img_complexity[fname] = compute_complexity_metrics(
            img_pil, fft_cutoff=fft_cutoff, dct_cutoff=dct_cutoff,
        )
    print(f"  done ({len(img_complexity)} images).\n")

    # ---- Per-image CSV ----
    per_image_csv = os.path.join(out_dir, "per_image.csv")
    f_csv = open(per_image_csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow([
        "lambda", "image", "T_calc", "bpp",
        "canny_density", "fft_hf_ratio", "dct_hf_ratio",
    ])

    all_records: list = []

    for lmbda_val in lambda_list:
        tag = f"lambda={lmbda_val:.3f}"
        print(f"\n{'=' * 72}\n  Evaluating {tag}\n{'=' * 72}")

        for img_path in images:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
            ori_h, ori_w = img.shape[2:]

            stride_h, stride_w = base_config.get("model_stride", [64, 64])
            pad_h = (math.ceil(ori_h / stride_h)) * stride_h - ori_h
            pad_w = (math.ceil(ori_w / stride_w)) * stride_w - ori_w
            img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode="reflect")
            _, _, H, W = img_padded.shape

            lmbda_tensor = torch.tensor([lmbda_val], dtype=torch.float32, device=device)

            try:
                T_calc, bpp = codec_forward_T(net, img_padded, H, W, lmbda_tensor)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  CUDA OOM (codec) on {fname}, skipping.")
                    torch.cuda.empty_cache()
                    continue
                raise

            cx = img_complexity[fname]
            writer.writerow([
                f"{lmbda_val:.6f}", fname,
                f"{T_calc:.4f}", f"{bpp:.6f}",
                f"{cx['canny_density']:.6f}",
                f"{cx['fft_hf_ratio']:.6f}",
                f"{cx['dct_hf_ratio']:.6f}",
            ])
            f_csv.flush()

            all_records.append({
                "lambda":         lmbda_val,
                "image":          fname,
                "T_calc":         T_calc,
                "bpp":            bpp,
                **cx,
            })

            print(f"  {fname:<24s}  λ={lmbda_val:7.3f}  "
                  f"T={T_calc:7.2f}  bpp={bpp:.4f}  "
                  f"canny={cx['canny_density']:.4f}  "
                  f"fft={cx['fft_hf_ratio']:.4f}  "
                  f"dct={cx['dct_hf_ratio']:.4f}")

    f_csv.close()

    if not all_records:
        print("  [WARNING] no records produced — aborting.")
        return

    _aggregate_and_report(all_records, lambda_list, out_dir,
                          fft_cutoff=fft_cutoff, dct_cutoff=dct_cutoff)


# ---------------------------------------------------------------------------
# Aggregation, summary JSON, plots
# ---------------------------------------------------------------------------

def _aggregate_and_report(all_records: list,
                          lambda_list: list,
                          out_dir: str,
                          fft_cutoff: float,
                          dct_cutoff: float) -> None:
    metrics = ("canny_density", "fft_hf_ratio", "dct_hf_ratio")

    T_all   = np.asarray([r["T_calc"] for r in all_records], dtype=np.float64)
    lam_all = np.asarray([r["lambda"] for r in all_records], dtype=np.float64)
    bpp_all = np.asarray([r["bpp"]    for r in all_records], dtype=np.float64)
    log_lam = np.log(np.clip(lam_all, 1e-12, None))

    metric_arrays = {
        m: np.asarray([r[m] for r in all_records], dtype=np.float64)
        for m in metrics
    }

    # ---- Overall T statistics ----
    T_stats = {
        "n":     int(T_all.size),
        "mean":  float(T_all.mean()),
        "std":   float(T_all.std(ddof=0)),
        "var":   float(T_all.var(ddof=0)),
        "min":   float(T_all.min()),
        "max":   float(T_all.max()),
        "range": float(T_all.max() - T_all.min()),
        "p05":   float(np.percentile(T_all,  5)),
        "p95":   float(np.percentile(T_all, 95)),
    }

    # ---- Overall correlations T_calc vs each complexity metric ----
    overall_corr = {}
    for m in metrics:
        v = metric_arrays[m]
        overall_corr[m] = {
            "pearson":  safe_corr(T_all, v),
            "spearman": spearman_corr(T_all, v),
        }
    overall_corr["log_lambda"] = {
        "pearson":  safe_corr(log_lam, T_all),
        "spearman": spearman_corr(log_lam, T_all),
    }
    overall_corr["bpp"] = {
        "pearson":  safe_corr(bpp_all, T_all),
        "spearman": spearman_corr(bpp_all, T_all),
    }

    # ---- Per-λ statistics & correlations (controls for λ) ----
    per_lambda: dict = {}
    for lv in sorted(set(lam_all.tolist())):
        m = lam_all == lv
        if m.sum() < 2:
            continue
        T_lv = T_all[m]
        entry = {
            "n":      int(m.sum()),
            "T_mean": float(T_lv.mean()),
            "T_std":  float(T_lv.std(ddof=0)),
            "T_var":  float(T_lv.var(ddof=0)),
            "T_min":  float(T_lv.min()),
            "T_max":  float(T_lv.max()),
        }
        for met in metrics:
            v = metric_arrays[met][m]
            entry[f"pearson_{met}"]  = safe_corr(T_lv, v)
            entry[f"spearman_{met}"] = spearman_corr(T_lv, v)
        per_lambda[f"{lv:.6f}"] = entry

    # ---- Per-image rank correlation across λ (sanity) ----
    per_image: dict = {}
    img_set = sorted({r["image"] for r in all_records})
    for img in img_set:
        idx = [i for i, r in enumerate(all_records) if r["image"] == img]
        if len(idx) < 2:
            continue
        T_im = T_all[idx]
        lam_im = lam_all[idx]
        per_image[img] = {
            "n":              len(idx),
            "T_mean":         float(T_im.mean()),
            "T_std":          float(T_im.std(ddof=0)),
            "spearman_lambda": spearman_corr(lam_im, T_im),
        }

    summary = {
        "n_samples":    len(all_records),
        "n_images":     len(img_set),
        "n_lambdas":    len(set(lam_all.tolist())),
        "fft_cutoff":   fft_cutoff,
        "dct_cutoff":   dct_cutoff,
        "T_stats":      T_stats,
        "overall_corr": overall_corr,
        "per_lambda":   per_lambda,
        "per_image":    per_image,
    }

    print(f"\n\n{'=' * 72}")
    print(f"  T content-adaptivity summary (n={len(all_records)})")
    print(f"{'=' * 72}")
    print(f"  T_calc  mean={T_stats['mean']:7.2f}   std={T_stats['std']:6.2f}   "
          f"var={T_stats['var']:8.2f}   range=[{T_stats['min']:.1f}, {T_stats['max']:.1f}]")
    print()
    print(f"  Overall correlations (T_calc vs ...):")
    for k, v in overall_corr.items():
        print(f"    {k:<14s}  pearson={v['pearson']:+.4f}  spearman={v['spearman']:+.4f}")
    print()
    if per_lambda:
        print(f"  Per-λ  T-std and correlations (controls for λ):")
        for k, v in per_lambda.items():
            print(f"    λ={float(k):8.3f}  n={v['n']:3d}  "
                  f"T_mean={v['T_mean']:7.2f}  T_std={v['T_std']:6.2f}   "
                  f"canny p={v['pearson_canny_density']:+.3f}/s={v['spearman_canny_density']:+.3f}   "
                  f"fft p={v['pearson_fft_hf_ratio']:+.3f}/s={v['spearman_fft_hf_ratio']:+.3f}   "
                  f"dct p={v['pearson_dct_hf_ratio']:+.3f}/s={v['spearman_dct_hf_ratio']:+.3f}")
    print(f"{'=' * 72}\n")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Per-image CSV  →  {os.path.join(out_dir, 'per_image.csv')}")
    print(f"  Summary JSON   →  {summary_path}")

    _make_plots(all_records, T_all, lam_all, log_lam, metric_arrays,
                per_lambda, out_dir)


def _make_plots(all_records, T_all, lam_all, log_lam, metric_arrays,
                per_lambda, out_dir) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    metrics = list(metric_arrays.keys())

    # Scatter T_calc vs each complexity metric, coloured by log λ.
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, met in zip(axes, metrics):
        v = metric_arrays[met]
        sc = ax.scatter(v, T_all, c=log_lam, cmap="viridis",
                        s=24, alpha=0.75, edgecolor="k", linewidth=0.3)
        p = safe_corr(v, T_all)
        s = spearman_corr(v, T_all)
        ax.set_xlabel(met)
        ax.set_ylabel("T_calc")
        ax.set_title(f"T_calc vs {met}\n(Pearson={p:+.3f}, Spearman={s:+.3f})")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("log λ")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "T_vs_complexity.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  Plot           →  {p1}")

    # Per-λ Spearman correlations as grouped bar.
    if per_lambda:
        lams_sorted = sorted(per_lambda.keys(), key=float)
        x = np.arange(len(lams_sorted))
        width = 0.27

        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(lams_sorted)), 4.5))
        for i, met in enumerate(metrics):
            ys = [per_lambda[k][f"spearman_{met}"] for k in lams_sorted]
            ax.bar(x + (i - 1) * width, ys, width, label=met)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(k):.2f}" for k in lams_sorted], rotation=45)
        ax.set_xlabel("λ")
        ax.set_ylabel("Spearman corr(T_calc, complexity)")
        ax.set_title("Per-λ rank correlation between T_calc and complexity")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        p2 = os.path.join(out_dir, "per_lambda_spearman.png")
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  Plot           →  {p2}")

    # T distribution per λ — boxplot.
    if per_lambda:
        lams_sorted = sorted(per_lambda.keys(), key=float)
        data = [T_all[lam_all == float(k)] for k in lams_sorted]
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(lams_sorted)), 4.5))
        ax.boxplot(data, labels=[f"{float(k):.2f}" for k in lams_sorted],
                   showmeans=True)
        ax.set_xlabel("λ")
        ax.set_ylabel("T_calc")
        ax.set_title("Distribution of T_calc across images, per λ")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        p3 = os.path.join(out_dir, "T_distribution_per_lambda.png")
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"  Plot           →  {p3}")


# CUDA_VISIBLE_DEVICES=0 python src/experiment_timesteps_adaptive.py --stage 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate that T_calc is content-adaptive — correlate "
                    "T_calc with image complexity metrics on CLIC.")
    parser.add_argument("--stage",            type=int, default=1)
    parser.add_argument("--base_config_file", type=str, default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str, default="./configs/test.yaml")
    parser.add_argument("--out_dir",          type=str,
                        default="./results/experiment_timesteps_adaptive")
    parser.add_argument("--fft_cutoff", type=float, default=0.25,
                        help="normalised radius below which FFT energy is "
                             "considered low-frequency (∈ (0, 1])")
    parser.add_argument("--dct_cutoff", type=float, default=0.25,
                        help="fraction of each axis kept as DCT low-frequency "
                             "block (∈ (0, 1])")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config_file)
    test_config = load_yaml(args.test_config_file)

    main(base_config, test_config, stage=args.stage, out_dir=args.out_dir,
         fft_cutoff=args.fft_cutoff, dct_cutoff=args.dct_cutoff)
