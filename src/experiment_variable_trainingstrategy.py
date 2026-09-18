"""
训练策略消融实验 (Training Strategy Ablation Study)

比较不同训练策略对 variable-rate StableCodec 码率-质量性能的影响：
  S1: Discrete-8       — 8个离散λ点
  S2: Discrete-16      — 16个离散λ点
  S3: Scalar-Uniform   — 标量对数均匀采样
  S4: Pixelwise-Tensor — 像素级λ张量采样（Ours）
  S5: Mixed            — 前半离散 + 后半像素级

核心评测：
  1. 密集λ扫描的 R-D 曲线对比（BPP vs PSNR / LPIPS）
  2. 插值泛化性：在"未见"λ点上的泛化间隙（generalization gap）
  3. 平滑性分析：|d(LPIPS)/d(log λ)| 一阶导数突变检测
  4. BD-Rate 量化全局差异

用法 (从 repo 根目录运行):
    python src/experiment_variable_trainingstrategy.py \\
        --sd_path      /path/to/sd-turbo_256 \\
        --elic_path    /path/to/elic_official.pth \\
        --img_dir      /path/to/Kodak24/HR \\
        --out_dir      results/training_strategy \\
        --strategies   '{"S1_Discrete8": "ckpt/s1.pth.tar",
                         "S4_PixelwiseTensor": "ckpt/s4.pth.tar"}'
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from StableCodec_variable2 import StableCodec

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


# ---------------------------------------------------------------------------
# Strategy metadata (display style)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyMeta:
    display_name: str
    color: str
    linestyle: str
    marker: str


STRATEGY_META: dict[str, StrategyMeta] = {
    "S1_Discrete8":       StrategyMeta("Discrete-8",             "#E74C3C", "--", "o"),
    "S2_Discrete16":      StrategyMeta("Discrete-16",            "#E67E22", "-.", "s"),
    "S3_ScalarUniform":   StrategyMeta("Scalar-Uniform",         "#3498DB", ":",  "^"),
    "S4_PixelwiseTensor": StrategyMeta("Pixelwise-Tensor (Ours)","#27AE60", "-",  "D"),
    "S5_Mixed":           StrategyMeta("Mixed (Disc→Pixel)",     "#9B59B6", "-",  "v"),
}

DEFAULT_SEEN_LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


# ---------------------------------------------------------------------------
# Lambda set utilities
# ---------------------------------------------------------------------------

def compute_log_midpoints(lambdas: list[float]) -> list[float]:
    """Geometric mean of adjacent λ values (log-space midpoint)."""
    midpoints = []
    for i in range(len(lambdas) - 1):
        mid = float(np.exp((np.log(lambdas[i]) + np.log(lambdas[i + 1])) / 2))
        midpoints.append(mid)
    return midpoints


def generate_dense_lambdas(
    lam_min: float, lam_max: float, n_points: int = 50,
) -> list[float]:
    return np.exp(
        np.linspace(np.log(lam_min), np.log(lam_max), n_points)
    ).tolist()


def merge_lambda_sets(*sets: list[float]) -> list[float]:
    merged = set()
    for s in sets:
        for v in s:
            merged.add(round(v, 8))
    return sorted(merged)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(
    img_dir: str, n_images: int, device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    paths = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:n_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    return imgs.to(device), [p.stem for p in paths]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_01 = (pred.clamp(-1, 1) + 1) / 2
    target_01 = (target.clamp(-1, 1) + 1) / 2
    mse = ((pred_01 - target_01) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_bd_rate(
    bpp_ref: np.ndarray, psnr_ref: np.ndarray,
    bpp_test: np.ndarray, psnr_test: np.ndarray,
) -> float:
    """BD-Rate (%): negative means test uses less BPP at same PSNR."""
    if len(bpp_ref) < 4 or len(bpp_test) < 4:
        return float("nan")
    log_bpp_ref = np.log10(np.clip(bpp_ref, 1e-8, None))
    log_bpp_test = np.log10(np.clip(bpp_test, 1e-8, None))
    p_ref = np.polyfit(psnr_ref, log_bpp_ref, min(3, len(psnr_ref) - 1))
    p_test = np.polyfit(psnr_test, log_bpp_test, min(3, len(psnr_test) - 1))
    psnr_lo = max(psnr_ref.min(), psnr_test.min())
    psnr_hi = min(psnr_ref.max(), psnr_test.max())
    if psnr_hi <= psnr_lo:
        return float("nan")
    int_ref = np.polyint(p_ref)
    int_test = np.polyint(p_test)
    avg_ref = (np.polyval(int_ref, psnr_hi) - np.polyval(int_ref, psnr_lo)) / (psnr_hi - psnr_lo)
    avg_test = (np.polyval(int_test, psnr_hi) - np.polyval(int_test, psnr_lo)) / (psnr_hi - psnr_lo)
    return float((10 ** (avg_test - avg_ref) - 1) * 100)


# ---------------------------------------------------------------------------
# Weight loading (swap strategy without rebuilding the whole model)
# ---------------------------------------------------------------------------

def load_strategy_weights(
    model: StableCodec, checkpoint_path: str, device: torch.device,
) -> None:
    sd = torch.load(checkpoint_path, map_location=device)

    # Codec (shape-safe)
    _sd_codec = model.codec.state_dict()
    codec_ckpt = sd.get("state_dict_codec", {})
    for k in list(codec_ckpt.keys()):
        if k in _sd_codec and codec_ckpt[k].shape == _sd_codec[k].shape:
            _sd_codec[k] = codec_ckpt[k]
    model.codec.load_state_dict(_sd_codec)

    # VAE LoRA
    _sd_vae = model.vae.state_dict()
    ckpt_vae = sd.get("state_dict_vae", {})
    for k in list(ckpt_vae.keys()):
        if k in _sd_vae and ckpt_vae[k].shape == _sd_vae[k].shape:
            _sd_vae[k] = ckpt_vae[k]
    model.vae.load_state_dict(_sd_vae)

    # UNet LoRA + conv_in
    _sd_unet = model.unet.state_dict()
    ckpt_unet = sd.get("state_dict_unet", {})
    for k in list(ckpt_unet.keys()):
        if k in _sd_unet and ckpt_unet[k].shape == _sd_unet[k].shape:
            _sd_unet[k] = ckpt_unet[k]
    model.unet.load_state_dict(_sd_unet)

    # unet_lora_proj
    if "state_dict_lora_proj" in sd:
        model.unet_lora_proj.load_state_dict(sd["state_dict_lora_proj"])


# ---------------------------------------------------------------------------
# Per-strategy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_at_lambdas(
    model: StableCodec,
    images: torch.Tensor,
    lambdas: list[float],
    device: torch.device,
    lpips_fn=None,
) -> dict[float, dict[str, float]]:
    """Evaluate model at each λ, averaging metrics over all images.

    Returns: {λ: {"psnr": float, "lpips": float, "bpp": float}}
    """
    N = images.shape[0]
    results: dict[float, dict[str, float]] = {}

    for lam in lambdas:
        psnr_acc: list[float] = []
        lpips_acc: list[float] = []
        bpp_acc: list[float] = []

        for i in range(N):
            img = images[i:i + 1]
            _, _, h, w = img.shape
            lmbda_t = torch.tensor([lam], dtype=torch.float32, device=device)
            output, rate_out = model(img, [1], h, w, lmbda=lmbda_t)

            psnr_acc.append(compute_psnr(output, img))
            bpp_acc.append(rate_out.per_image_bpp[0].item())

            if lpips_fn is not None:
                lp = lpips_fn(output.clamp(-1, 1), img).item()
                lpips_acc.append(lp)

        results[lam] = {
            "psnr": float(np.mean(psnr_acc)),
            "lpips": float(np.mean(lpips_acc)) if lpips_acc else 0.0,
            "bpp": float(np.mean(bpp_acc)),
        }
        print(f"    λ={lam:8.3f}  PSNR={results[lam]['psnr']:.2f}dB  "
              f"LPIPS={results[lam]['lpips']:.4f}  BPP={results[lam]['bpp']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Generalization gap & smoothness
# ---------------------------------------------------------------------------

def compute_generalization_gap(
    results: dict[float, dict[str, float]],
    seen_lambdas: list[float],
    unseen_lambdas: list[float],
    metric: str = "lpips",
) -> dict:
    """Gap = actual_at_midpoint - interpolated_from_neighbors.

    Positive gap means worse than expected (for LPIPS, where lower is better).
    """
    seen_vals = [results[lam][metric] for lam in seen_lambdas]
    unseen_vals = [results[lam][metric] for lam in unseen_lambdas]
    expected = [(seen_vals[i] + seen_vals[i + 1]) / 2
                for i in range(len(unseen_lambdas))]
    gaps = [a - e for a, e in zip(unseen_vals, expected)]
    return {
        "mean_gap": float(np.mean(gaps)),
        "max_gap": float(np.max(np.abs(gaps))),
        "gap_per_point": gaps,
    }


def compute_smoothness(
    results: dict[float, dict[str, float]],
    dense_lambdas: list[float],
    metric: str = "lpips",
) -> tuple[np.ndarray, np.ndarray]:
    """|d(metric)/d(log λ)| — peaks indicate generalization failure."""
    vals = np.array([results[lam][metric] for lam in dense_lambdas])
    log_lams = np.log(np.array(dense_lambdas))
    grad = np.abs(np.diff(vals) / np.diff(log_lams))
    return np.array(dense_lambdas[1:]), grad


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _get_meta(name: str) -> StrategyMeta:
    return STRATEGY_META.get(name, StrategyMeta(name, "gray", "-", "x"))


def plot_rd_curves(
    all_results: dict[str, dict[float, dict[str, float]]],
    seen_lambdas: list[float],
    out_path: str,
) -> None:
    """BPP vs LPIPS for all strategies. Full range + low-rate zoom."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for sname, res in all_results.items():
        meta = _get_meta(sname)
        lams = sorted(res.keys())
        bpp_vals = [res[l]["bpp"] for l in lams]
        lpips_vals = [res[l]["lpips"] for l in lams]
        for ax in axes:
            ax.plot(bpp_vals, lpips_vals, color=meta.color,
                    linestyle=meta.linestyle, linewidth=2,
                    marker=meta.marker, markersize=4,
                    label=meta.display_name)

    axes[1].set_yscale("log")

    for ax in axes:
        ax.set_xlabel("Bitrate (BPP)", fontsize=12)
        ax.set_ylabel("LPIPS \u2193", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Rate-Quality Curves (full range)", fontsize=13)
    axes[1].set_title("Rate-Quality Curves (log LPIPS)", fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rd_curves_psnr(
    all_results: dict[str, dict[float, dict[str, float]]],
    out_path: str,
) -> None:
    """BPP vs PSNR for all strategies."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for sname, res in all_results.items():
        meta = _get_meta(sname)
        lams = sorted(res.keys())
        bpp_vals = [res[l]["bpp"] for l in lams]
        psnr_vals = [res[l]["psnr"] for l in lams]
        ax.plot(bpp_vals, psnr_vals, color=meta.color,
                linestyle=meta.linestyle, linewidth=2,
                marker=meta.marker, markersize=4,
                label=meta.display_name)

    ax.set_xlabel("Bitrate (BPP)", fontsize=12)
    ax.set_ylabel("PSNR (dB) \u2191", fontsize=12)
    ax.set_title("Rate-Quality Curves (PSNR)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_smoothness_analysis(
    all_results: dict[str, dict[float, dict[str, float]]],
    dense_lambdas: list[float],
    seen_lambdas: list[float],
    out_path: str,
) -> None:
    """Top: LPIPS vs λ;  Bottom: |d(LPIPS)/d(log λ)|."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for sname, res in all_results.items():
        meta = _get_meta(sname)
        lpips_vals = [res[l]["lpips"] for l in dense_lambdas]
        axes[0].semilogx(dense_lambdas, lpips_vals,
                         color=meta.color, linewidth=2,
                         label=meta.display_name)
        lams_grad, grad = compute_smoothness(res, dense_lambdas, "lpips")
        axes[1].semilogx(lams_grad, grad,
                         color=meta.color, linewidth=1.5,
                         label=meta.display_name)

    for lam in seen_lambdas:
        for ax in axes:
            ax.axvline(x=lam, color="gray", alpha=0.25,
                       linewidth=1, linestyle="--")

    axes[0].set_ylabel("LPIPS \u2193", fontsize=12)
    axes[1].set_ylabel("|d(LPIPS)/d(log \u03bb)| \u2193\n(lower = smoother)",
                       fontsize=11)
    axes[1].set_xlabel("Rate parameter \u03bb (log scale)", fontsize=12)

    axes[0].set_title("Rate-Quality Curves", fontsize=13)
    axes[1].set_title("Local Smoothness (derivative magnitude)\n"
                      "Peaks near training points = generalization failure",
                      fontsize=12)

    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_gap_chart(
    gap_per_strategy: dict[str, dict],
    seen_lambdas: list[float],
    out_path: str,
) -> None:
    """Bar chart: generalization gap per interval, grouped by strategy."""
    n_gaps = len(seen_lambdas) - 1
    interval_labels = [
        f"{seen_lambdas[i]:.1f}\n~{seen_lambdas[i+1]:.1f}"
        for i in range(n_gaps)
    ]

    x = np.arange(n_gaps)
    strategies = list(gap_per_strategy.keys())
    n_strats = len(strategies)
    width = 0.8 / max(n_strats, 1)

    fig, ax = plt.subplots(figsize=(max(12, 2 * n_gaps), 5))

    for si, sname in enumerate(strategies):
        meta = _get_meta(sname)
        gaps = gap_per_strategy[sname]["gap_per_point"]
        bars = ax.bar(x + si * width, gaps, width,
                      label=meta.display_name, color=meta.color, alpha=0.8)
        max_idx = int(np.argmax(np.abs(gaps)))
        ax.text(x[max_idx] + si * width, gaps[max_idx],
                f"{gaps[max_idx]:.4f}", ha="center", va="bottom",
                fontsize=7, color=meta.color)

    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(interval_labels, fontsize=9)
    ax.set_xlabel("Rate Interval (between training \u03bb points)", fontsize=12)
    ax.set_ylabel("Generalization Gap (LPIPS \u2191 = worse)", fontsize=12)
    ax.set_title("Interpolation Quality at Unseen Rate Points\n"
                 "(midpoint of each training interval)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_vs_bpp_psnr(
    all_results: dict[str, dict[float, dict[str, float]]],
    out_path: str,
) -> None:
    """λ (x, log) vs BPP (left y) and PSNR (right y) for each strategy."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for sname, res in all_results.items():
        meta = _get_meta(sname)
        lams = sorted(res.keys())
        bpp_vals = [res[l]["bpp"] for l in lams]
        psnr_vals = [res[l]["psnr"] for l in lams]
        ax1.semilogx(lams, bpp_vals, color=meta.color,
                     linestyle=meta.linestyle, linewidth=2,
                     label=f"{meta.display_name} (BPP)")
        ax2.semilogx(lams, psnr_vals, color=meta.color,
                     linestyle=":", linewidth=1.5, alpha=0.6)

    ax1.set_xlabel("\u03bb (log scale)", fontsize=12)
    ax1.set_ylabel("BPP \u2193", fontsize=12)
    ax2.set_ylabel("PSNR (dB) \u2191", fontsize=12, color="gray")
    ax1.set_title("\u03bb \u2192 BPP / PSNR mapping per strategy", fontsize=13)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def write_results(
    all_results: dict[str, dict[float, dict[str, float]]],
    gap_per_strategy: dict[str, dict],
    bd_rates: dict[str, float],
    seen_lambdas: list[float],
    unseen_lambdas: list[float],
    out_path: str,
) -> None:
    ref_name = list(all_results.keys())[0]

    with open(out_path, "w") as f:
        f.write("Training Strategy Ablation Study — Numeric Results\n")
        f.write("=" * 70 + "\n\n")

        # BD-Rate table
        f.write("BD-Rate vs reference ({}):\n".format(ref_name))
        for sname, rate in bd_rates.items():
            meta = _get_meta(sname)
            f.write(f"  {meta.display_name:30s}  BD-Rate = {rate:+.2f}%\n")
        f.write("\n")

        # Per-strategy seen/unseen metrics
        for sname, res in all_results.items():
            meta = _get_meta(sname)
            f.write("-" * 60 + "\n")
            f.write(f"Strategy: {meta.display_name}\n\n")

            f.write("  Seen λ points:\n")
            for lam in seen_lambdas:
                if lam in res:
                    r = res[lam]
                    f.write(f"    λ={lam:8.3f}  PSNR={r['psnr']:.2f}  "
                            f"LPIPS={r['lpips']:.4f}  BPP={r['bpp']:.4f}\n")

            f.write("  Unseen λ midpoints:\n")
            for lam in unseen_lambdas:
                if lam in res:
                    r = res[lam]
                    f.write(f"    λ={lam:8.3f}  PSNR={r['psnr']:.2f}  "
                            f"LPIPS={r['lpips']:.4f}  BPP={r['bpp']:.4f}\n")

            if sname in gap_per_strategy:
                g = gap_per_strategy[sname]
                f.write(f"  Generalization gap (LPIPS):  "
                        f"mean={g['mean_gap']:.5f}  max={g['max_gap']:.5f}\n")
                f.write(f"    per-interval: {[f'{v:.5f}' for v in g['gap_per_point']]}\n")
            f.write("\n")

        # Compact table
        f.write("=" * 70 + "\n")
        f.write("Compact Summary Table\n\n")
        header = (f"{'Strategy':30s} {'BD-Rate':>8s} {'Gap_mean':>9s} "
                  f"{'Gap_max':>8s} {'PSNR_mean':>10s} {'LPIPS_mean':>10s}\n")
        f.write(header)
        f.write("-" * len(header) + "\n")
        for sname, res in all_results.items():
            meta = _get_meta(sname)
            all_psnr = np.mean([res[l]["psnr"] for l in sorted(res.keys())])
            all_lpips = np.mean([res[l]["lpips"] for l in sorted(res.keys())])
            gap_mean = gap_per_strategy.get(sname, {}).get("mean_gap", float("nan"))
            gap_max = gap_per_strategy.get(sname, {}).get("max_gap", float("nan"))
            bd = bd_rates.get(sname, float("nan"))
            f.write(f"{meta.display_name:30s} {bd:>+8.2f}% {gap_mean:>9.5f} "
                    f"{gap_max:>8.5f} {all_psnr:>10.2f} {all_lpips:>10.4f}\n")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    strategies: dict[str, str] = json.loads(args.strategies)
    if not strategies:
        raise ValueError("--strategies must map at least one strategy name to "
                         "a checkpoint path")

    # Lambda sets
    seen_lambdas = [float(x) for x in args.seen_lambdas.split(",")]
    unseen_lambdas = compute_log_midpoints(seen_lambdas)
    dense_lambdas = generate_dense_lambdas(
        args.lambda_min, args.lambda_max, args.n_dense)
    extrapolation_lambdas = [float(x) for x in args.extrapolation_lambdas.split(",")]
    all_lambdas = merge_lambda_sets(
        dense_lambdas, seen_lambdas, unseen_lambdas, extrapolation_lambdas)

    print(f"Seen λ:          {seen_lambdas}")
    print(f"Unseen midpoints: {[f'{v:.3f}' for v in unseen_lambdas]}")
    print(f"Dense sweep:     {len(dense_lambdas)} points in "
          f"[{args.lambda_min}, {args.lambda_max}]")
    print(f"Total eval λ:    {len(all_lambdas)}\n")

    # LPIPS network
    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
        print("LPIPS (AlexNet) loaded.\n")
    else:
        print("WARNING: lpips not installed — LPIPS metrics will be 0.\n")

    # Load images
    images, names = load_images(args.img_dir, args.n_images, device)
    N = images.shape[0]
    print(f"Loaded {N} images from {args.img_dir}\n")

    # Base config (shared by all strategies)
    config = {
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "latent_tiled_size": 96,
        "latent_tiled_overlap": 32,
        "lora_rank_vae": 16,
        "lora_rank_unet": 32,
        "vae_encoder_tiled_size": 1024,
        "vae_decoder_tiled_size": 160,
        "timesteps": 999,
        "pos_prompt": (
            "A high-resolution, 8K, ultra-realistic image with sharp focus, "
            "vibrant colors, and natural lighting."
        ),
        "elic_path": args.elic_path,
    }

    # Build model once with the first strategy's checkpoint
    strategy_names = list(strategies.keys())
    first_name = strategy_names[0]
    first_ckpt = strategies[first_name]
    config["codec_path"] = first_ckpt

    print("=" * 60)
    print("Building StableCodec (variable-rate) ...")
    model = StableCodec(sd_path=args.sd_path, config=config)
    model.set_eval()
    model.to(device)
    print("Model ready.\n")

    # Evaluate all strategies
    all_results: dict[str, dict[float, dict[str, float]]] = {}

    for si, sname in enumerate(strategy_names):
        ckpt_path = strategies[sname]
        meta = _get_meta(sname)
        print("=" * 60)
        print(f"[{si+1}/{len(strategy_names)}] Evaluating: {meta.display_name}")
        print(f"  Checkpoint: {ckpt_path}")

        if si > 0:
            print("  Loading weights ...")
            load_strategy_weights(model, ckpt_path, device)
            model.set_eval()

        all_results[sname] = evaluate_at_lambdas(
            model, images, all_lambdas, device, lpips_fn)
        print()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    # Generalization gap (LPIPS)
    gap_per_strategy: dict[str, dict] = {}
    for sname in strategy_names:
        has_all = all(lam in all_results[sname] for lam in seen_lambdas + unseen_lambdas)
        if has_all:
            gap_per_strategy[sname] = compute_generalization_gap(
                all_results[sname], seen_lambdas, unseen_lambdas, "lpips")
            g = gap_per_strategy[sname]
            meta = _get_meta(sname)
            print(f"  {meta.display_name:30s}  gap_mean={g['mean_gap']:.5f}  "
                  f"gap_max={g['max_gap']:.5f}")

    # BD-Rate (reference = first strategy)
    ref_name = strategy_names[0]
    ref_res = all_results[ref_name]
    ref_lams = sorted(ref_res.keys())
    ref_bpp = np.array([ref_res[l]["bpp"] for l in ref_lams])
    ref_psnr = np.array([ref_res[l]["psnr"] for l in ref_lams])

    bd_rates: dict[str, float] = {}
    for sname in strategy_names:
        if sname == ref_name:
            bd_rates[sname] = 0.0
            continue
        res = all_results[sname]
        lams = sorted(res.keys())
        bpp = np.array([res[l]["bpp"] for l in lams])
        psnr = np.array([res[l]["psnr"] for l in lams])
        bd_rates[sname] = compute_bd_rate(ref_bpp, ref_psnr, bpp, psnr)

    print("\nBD-Rate vs {}:".format(_get_meta(ref_name).display_name))
    for sname, bd in bd_rates.items():
        print(f"  {_get_meta(sname).display_name:30s}  {bd:+.2f}%")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    print("\nGenerating visualizations ...")

    # Only include dense lambdas (not extrapolation) for smooth plots
    dense_only = {
        sname: {l: v for l, v in res.items() if l in dense_lambdas}
        for sname, res in all_results.items()
    }

    plot_rd_curves(
        all_results, seen_lambdas,
        os.path.join(args.out_dir, "rd_curves_lpips.png"))

    plot_rd_curves_psnr(
        all_results,
        os.path.join(args.out_dir, "rd_curves_psnr.png"))

    dense_sorted = sorted(dense_lambdas)
    valid_dense = {}
    for sname, res in all_results.items():
        valid = [l for l in dense_sorted if l in res]
        if len(valid) >= 3:
            valid_dense[sname] = {l: res[l] for l in valid}

    if valid_dense:
        plot_smoothness_analysis(
            valid_dense,
            [l for l in dense_sorted
             if all(l in valid_dense[s] for s in valid_dense)],
            seen_lambdas,
            os.path.join(args.out_dir, "smoothness_analysis.png"))

    if gap_per_strategy:
        plot_generalization_gap_chart(
            gap_per_strategy, seen_lambdas,
            os.path.join(args.out_dir, "generalization_gap.png"))

    plot_lambda_vs_bpp_psnr(
        all_results,
        os.path.join(args.out_dir, "lambda_vs_bpp_psnr.png"))

    # Results file
    write_results(
        all_results, gap_per_strategy, bd_rates,
        seen_lambdas, unseen_lambdas,
        os.path.join(args.out_dir, "results.txt"))

    # Save raw data as JSON
    json_data = {}
    for sname, res in all_results.items():
        json_data[sname] = {str(k): v for k, v in res.items()}
    with open(os.path.join(args.out_dir, "raw_results.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nAll results saved to {args.out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training Strategy Ablation: compare discrete vs continuous "
                    "vs pixelwise λ training on variable-rate StableCodec.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--sd_path", required=True,
                        help="Path to SD-Turbo (256ch variant)")
    parser.add_argument("--elic_path", required=True,
                        help="Path to ELIC pretrained weights")
    parser.add_argument("--img_dir", required=True,
                        help="Directory of test images (e.g. Kodak24/HR)")
    parser.add_argument("--out_dir", default="results/training_strategy",
                        help="Output directory")
    parser.add_argument(
        "--strategies", required=True,
        help='JSON mapping strategy name → checkpoint path, e.g.:\n'
             '\'{"S1_Discrete8": "ckpt/s1.pth.tar", '
             '"S4_PixelwiseTensor": "ckpt/s4.pth.tar"}\'')
    parser.add_argument("--n_images", type=int, default=5,
                        help="Number of test images to use")
    parser.add_argument("--n_dense", type=int, default=50,
                        help="Number of dense λ sweep points")
    parser.add_argument("--lambda_min", type=float, default=0.1,
                        help="Lambda min for model (match training config)")
    parser.add_argument("--lambda_max", type=float, default=128.0,
                        help="Lambda max for model (match training config)")
    parser.add_argument(
        "--seen_lambdas", default="0.5,1,2,4,8,16,32,64",
        help="Comma-separated 'seen' λ values (Discrete-8 training points)")
    parser.add_argument(
        "--extrapolation_lambdas", default="0.08,0.09,140,160",
        help="Comma-separated λ values outside training range")
    args = parser.parse_args()

    run_experiment(args)
