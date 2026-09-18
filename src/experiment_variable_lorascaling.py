"""
LoRA Scaling Factor Visualization (LoRA Scaling Factor Analysis)

证明：模型训练后，自发地学会了让 α(λ) 随码率单调变化——
这不是人为约束的结果，而是率失真优化目标自然涌现的行为。

做法：
  1. 加载训练好的 variable-rate StableCodec 模型
  2. 扫描 λ，计算全局 α(λ) = 1 + tanh(unet_lora_proj(film_embed(λ)))
  3. 静态分析各 LoRA 层的权重幅度 ||ΔW_i|| = ||B @ A|| × scaling
  4. 计算有效逐层 scaling = α(λ) × ||ΔW_i||（外积矩阵）
  5. 可选：hook g_a / g_s 获取 lT 质量 SSIM(y, y_hat)
  6. 统计检验 α(λ) 单调性 + 可视化

期望结果：
  - α(λ) 随 λ 增大（低码率）而增大 → LoRA 贡献更大
  - bottleneck / decoder 层的 ||ΔW|| 显著大于 encoder 浅层
  - α(λ) 与 lT 质量（SSIM）呈负相关

用法 (从 repo 根目录运行):
    python src/experiment_variable_lorascaling.py \\
        --sd_path      /path/to/sd-turbo_256 \\
        --elic_path    /path/to/elic_official.pth \\
        --codec_path   /path/to/variable_rate_checkpoint.pth.tar \\
        --out_dir      results/lora_scaling

    # 可选：提供图像目录以计算 lT 质量
    python src/experiment_variable_lorascaling.py \\
        --sd_path      /path/to/sd-turbo_256 \\
        --elic_path    /path/to/elic_official.pth \\
        --codec_path   /path/to/variable_rate_checkpoint.pth.tar \\
        --img_dir      /path/to/Kodak24/HR \\
        --out_dir      results/lora_scaling
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from StableCodec_variable2 import StableCodec


# ---------------------------------------------------------------------------
# Alpha sweep (no images needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_global_alpha(
    model: nn.Module,
    lambda_range: tuple[float, float],
    n_points: int = 200,
    device: torch.device = torch.device("cuda"),
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep λ log-uniformly and compute α(λ) = 1 + tanh(proj(embed(λ))).

    Returns:
        (lambdas, alphas) — 1D arrays of shape (n_points,).
    """
    lambdas = np.exp(
        np.linspace(
            np.log(lambda_range[0]),
            np.log(lambda_range[1]),
            n_points,
        )
    )
    alphas = np.empty(n_points)

    for i, lam in enumerate(lambdas):
        lam_t = torch.tensor([lam], dtype=torch.float32, device=device)
        film_embed = model.codec.film_embed(lam_t)
        delta_s = model.unet_lora_proj(film_embed)
        alpha = (1.0 + torch.tanh(delta_s)).item()
        alphas[i] = alpha

    return lambdas, alphas


# ---------------------------------------------------------------------------
# Static LoRA weight magnitude analysis
# ---------------------------------------------------------------------------

def compute_lora_weight_magnitudes(
    model: nn.Module,
) -> dict[str, float]:
    """Compute ||ΔW_i|| = ||lora_B @ lora_A|| × scaling for each LoRA layer.

    Returns:
        {layer_name: magnitude} dict.
    """
    magnitudes: dict[str, float] = {}

    for layer_name_str in model.unet_lora_layers:
        parts = layer_name_str.split(".")
        mod = model.unet
        for p in parts:
            mod = getattr(mod, p)

        for adapter_name in mod.lora_A:
            A = mod.lora_A[adapter_name].weight
            B = mod.lora_B[adapter_name].weight
            scaling = mod.scaling[adapter_name]
            delta_w = (B @ A) * scaling
            mag = delta_w.norm().item()
            magnitudes[layer_name_str] = mag
            break

    return magnitudes


def group_layers_by_unet_position(
    layer_names: list[str],
) -> dict[str, list[str]]:
    """Group UNet layers into encoder / middle / decoder."""
    groups: dict[str, list[str]] = {
        "encoder": [],
        "middle": [],
        "decoder": [],
    }
    for name in layer_names:
        if "down_blocks" in name:
            groups["encoder"].append(name)
        elif "mid_block" in name:
            groups["middle"].append(name)
        elif "up_blocks" in name:
            groups["decoder"].append(name)
        else:
            groups["decoder"].append(name)
    return groups


# ---------------------------------------------------------------------------
# lT quality via hooks
# ---------------------------------------------------------------------------

_GA_OUTPUT: list[torch.Tensor] = []
_GS_INPUT: list[torch.Tensor] = []


def _ga_hook(module: nn.Module, inp, out):
    if isinstance(out, tuple):
        out = out[0]
    _GA_OUTPUT.clear()
    _GA_OUTPUT.append(out.detach())


def _gs_hook(module: nn.Module, inp, out):
    y_hat = inp[0].detach()
    _GS_INPUT.clear()
    _GS_INPUT.append(y_hat)


@torch.no_grad()
def collect_lt_quality(
    model: nn.Module,
    images: torch.Tensor,
    lambdas: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute mean SSIM(y, y_hat) over images for each λ.

    Returns:
        quality — 1D array of shape (len(lambdas),).
    """
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

    h_ga = model.codec.g_a.register_forward_hook(_ga_hook)
    h_gs = model.codec.g_s.register_forward_hook(_gs_hook)

    N = images.shape[0]
    quality = np.empty(len(lambdas))

    try:
        for li, lam in enumerate(lambdas):
            lam_t = torch.tensor([lam], dtype=torch.float32, device=device)
            batch_ssim: list[float] = []

            for idx in range(N):
                img = images[idx : idx + 1]
                _, _, H, W = img.shape

                _GA_OUTPUT.clear()
                _GS_INPUT.clear()
                _ = model(img, [1], H, W, lmbda=lam_t)

                if _GA_OUTPUT and _GS_INPUT:
                    y = _GA_OUTPUT[0]
                    y_hat = _GS_INPUT[0]
                    if y.shape != y_hat.shape:
                        min_h = min(y.shape[2], y_hat.shape[2])
                        min_w = min(y.shape[3], y_hat.shape[3])
                        y = y[:, :, :min_h, :min_w]
                        y_hat = y_hat[:, :, :min_h, :min_w]
                    dr = y.max() - y.min()
                    if dr < 1e-6:
                        dr = torch.tensor(1.0, device=device)
                    s = ssim_fn(y_hat, y, data_range=dr)
                    batch_ssim.append(s.item())

            quality[li] = float(np.mean(batch_ssim)) if batch_ssim else 0.0
    finally:
        h_ga.remove()
        h_gs.remove()

    return quality


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
# Statistical tests
# ---------------------------------------------------------------------------

def monotonicity_test(
    lambdas: np.ndarray, alphas: np.ndarray,
) -> dict[str, float]:
    corr, p_val = spearmanr(lambdas, alphas)

    diffs = np.diff(alphas)
    if corr > 0:
        monotone_ratio = float((diffs > 0).mean())
    else:
        monotone_ratio = float((diffs < 0).mean())

    lo_mask = lambdas < np.percentile(lambdas, 20)
    hi_mask = lambdas > np.percentile(lambdas, 80)
    delta_alpha = float(alphas[hi_mask].mean() - alphas[lo_mask].mean())

    return {
        "spearman_r": float(corr),
        "p_value": float(p_val),
        "monotone_ratio": monotone_ratio,
        "delta_alpha": delta_alpha,
        "alpha_min": float(alphas.min()),
        "alpha_max": float(alphas.max()),
        "alpha_range": float(alphas.max() - alphas.min()),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_alpha_curve(
    lambdas: np.ndarray,
    alphas: np.ndarray,
    stats: dict[str, float],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lambdas, alphas, linewidth=2.5, color="#2D6A4F")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="α=1.0 (neutral)")

    lo_region = lambdas[lambdas < np.percentile(lambdas, 15)]
    hi_region = lambdas[lambdas > np.percentile(lambdas, 85)]
    if len(lo_region) > 0:
        ax.axvspan(lambdas[0], lo_region[-1], alpha=0.06, color="blue",
                   label="High bitrate region")
    if len(hi_region) > 0:
        ax.axvspan(hi_region[0], lambdas[-1], alpha=0.06, color="red",
                   label="Low bitrate region")

    ax.text(
        0.05, 0.92,
        f"Spearman r = {stats['spearman_r']:.4f}\n"
        f"p = {stats['p_value']:.2e}\n"
        f"Monotone ratio = {stats['monotone_ratio']:.1%}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        verticalalignment="top",
    )

    ax.set_xlabel("Rate parameter λ (log scale)", fontsize=12)
    ax.set_ylabel("LoRA Scaling Factor α(λ)", fontsize=12)
    ax.set_title("Rate-Adaptive LoRA Scaling Factor α(λ)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weight_magnitude_bar(
    magnitudes: dict[str, float],
    groups: dict[str, list[str]],
    out_path: str,
) -> None:
    colors_map = {
        "encoder": "#4C9BE8",
        "middle": "#2D6A4F",
        "decoder": "#E07B39",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    group_means = []
    group_stds = []
    group_names = []
    group_colors = []

    for gname in ["encoder", "middle", "decoder"]:
        layers = groups[gname]
        if not layers:
            continue
        vals = [magnitudes.get(l, 0.0) for l in layers]
        group_means.append(np.mean(vals))
        group_stds.append(np.std(vals))
        group_names.append(f"{gname}\n({len(layers)} layers)")
        group_colors.append(colors_map[gname])

    x_pos = np.arange(len(group_names))
    ax.bar(x_pos, group_means, yerr=group_stds, color=group_colors,
           edgecolor="k", capsize=8, width=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_names, fontsize=11)
    ax.set_ylabel("Mean ||ΔW|| (Frobenius norm × scaling)", fontsize=11)
    ax.set_title("Static LoRA Weight Magnitude by UNet Position", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_effective_scaling_heatmap(
    lambdas: np.ndarray,
    alphas: np.ndarray,
    magnitudes: dict[str, float],
    layer_order: list[str],
    groups: dict[str, list[str]],
    out_path: str,
) -> None:
    mag_vec = np.array([magnitudes.get(l, 0.0) for l in layer_order])
    effective = np.outer(mag_vec, alphas)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        effective, aspect="auto", cmap="RdYlBu_r",
        extent=[np.log10(lambdas[0]), np.log10(lambdas[-1]),
                len(layer_order), 0],
    )
    plt.colorbar(im, ax=ax, label="Effective LoRA Scaling  α(λ) × ||ΔW_i||")

    lambda_ticks = [0.1, 0.5, 1, 5, 10, 50, 128]
    valid_ticks = [t for t in lambda_ticks
                   if lambdas[0] <= t <= lambdas[-1]]
    ax.set_xticks([np.log10(t) for t in valid_ticks])
    ax.set_xticklabels([str(t) for t in valid_ticks])
    ax.set_xlabel("Rate parameter λ", fontsize=12)
    ax.set_ylabel("UNet Layer (shallow → deep → shallow)", fontsize=12)

    n_enc = sum(1 for l in layer_order if l in groups.get("encoder", []))
    n_mid = sum(1 for l in layer_order if l in groups.get("middle", []))
    if n_enc > 0:
        ax.axhline(y=n_enc, color="white", linewidth=2)
    if n_mid > 0:
        ax.axhline(y=n_enc + n_mid, color="white", linewidth=2)

    if n_enc > 0:
        ax.text(-0.06, n_enc / 2, "Enc", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10, color="#4C9BE8",
                fontweight="bold")
    if n_mid > 0:
        ax.text(-0.06, n_enc + n_mid / 2, "Mid",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10, color="#2D6A4F",
                fontweight="bold")
    n_dec = len(layer_order) - n_enc - n_mid
    if n_dec > 0:
        ax.text(-0.06, n_enc + n_mid + n_dec / 2, "Dec",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10, color="#E07B39",
                fontweight="bold")

    ax.set_title(
        "Per-Layer Effective LoRA Scaling\n"
        "α(λ) × ||ΔW_i|| across UNet Depth and Rate",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_main_figure(
    lambdas: np.ndarray,
    alphas: np.ndarray,
    stats: dict[str, float],
    lt_quality: np.ndarray | None,
    magnitudes: dict[str, float],
    groups: dict[str, list[str]],
    out_path: str,
) -> None:
    """Main figure: (a) α(λ) curve, (b) α vs lT quality or weight magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- (a) Global α(λ) curve ---
    ax1 = axes[0]
    ax1.semilogx(lambdas, alphas, color="#2D6A4F", linewidth=2.5,
                 label="α(λ) = 1 + tanh(proj(embed(λ)))")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                label="α=1.0 (neutral)")

    lo_region = lambdas[lambdas < np.percentile(lambdas, 15)]
    hi_region = lambdas[lambdas > np.percentile(lambdas, 85)]
    if len(lo_region) > 0:
        ax1.axvspan(lambdas[0], lo_region[-1], alpha=0.06, color="blue")
    if len(hi_region) > 0:
        ax1.axvspan(hi_region[0], lambdas[-1], alpha=0.06, color="red")

    ax1.text(
        0.05, 0.92,
        f"Spearman r = {stats['spearman_r']:.4f}\n"
        f"p = {stats['p_value']:.2e}",
        transform=ax1.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        verticalalignment="top",
    )
    ax1.set_xlabel("Rate parameter λ (log scale)", fontsize=12)
    ax1.set_ylabel("LoRA Scaling Factor α(λ)", fontsize=12)
    ax1.set_title("(a) Rate-Adaptive LoRA Scaling", fontsize=13)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # --- (b) α vs lT quality or weight magnitude ---
    ax2 = axes[1]

    if lt_quality is not None:
        ax2_twin = ax2.twinx()
        l1, = ax2.semilogx(lambdas, alphas, color="#2D6A4F", linewidth=2.5,
                           label="α(λ)")
        l2, = ax2_twin.semilogx(lambdas, lt_quality, color="#C0392B",
                                linewidth=2.5, linestyle="-.",
                                label="lT Quality (SSIM)")
        corr = float(np.corrcoef(alphas, lt_quality)[0, 1])
        ax2.text(
            0.05, 0.92,
            f"Pearson r = {corr:.3f}",
            transform=ax2.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            verticalalignment="top",
        )
        ax2.set_ylabel("LoRA Scaling Factor α(λ)", fontsize=12)
        ax2_twin.set_ylabel("lT Quality (SSIM)", fontsize=12)
        lines = [l1, l2]
        ax2.legend(lines, [l.get_label() for l in lines],
                   loc="center right", fontsize=10)
        ax2.set_title(
            "(b) Rate-Denoising Coupling:\nα(λ) vs. Latent Token Quality",
            fontsize=13,
        )
    else:
        colors_map = {
            "encoder": "#4C9BE8",
            "middle": "#2D6A4F",
            "decoder": "#E07B39",
        }
        labels_map = {
            "encoder": "Encoder",
            "middle": "Bottleneck",
            "decoder": "Decoder",
        }
        x_idx = 0
        x_positions = []
        x_labels_list = []
        for gname in ["encoder", "middle", "decoder"]:
            layers = groups[gname]
            if not layers:
                continue
            vals = [magnitudes.get(l, 0.0) for l in layers]
            ax2.bar(x_idx, np.mean(vals), yerr=np.std(vals),
                    color=colors_map[gname], edgecolor="k", capsize=6,
                    width=0.6, label=labels_map[gname])
            x_positions.append(x_idx)
            x_labels_list.append(labels_map[gname])
            x_idx += 1
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels_list, fontsize=11)
        ax2.set_ylabel("Mean ||ΔW|| (weight magnitude)", fontsize=12)
        ax2.set_title("(b) Static LoRA Weight Magnitude\nby UNet Position",
                      fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

    ax2.set_xlabel("Rate parameter λ (log scale)", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Build model ---
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
        "codec_path": args.codec_path,
        "elic_path": args.elic_path,
    }

    print("=" * 60)
    print("Building StableCodec (variable-rate) ...")
    model = StableCodec(sd_path=args.sd_path, config=config)
    model.set_eval()
    model.to(device)
    print("Model ready.\n")

    lambda_range = (args.lambda_min, args.lambda_max)

    # =====================================================================
    # Step 1: Global α(λ) sweep (fast, no images)
    # =====================================================================
    print("Step 1: Sweeping λ for global α(λ) ...")
    lambdas, alphas = collect_global_alpha(
        model, lambda_range, n_points=args.n_points, device=device,
    )
    print(f"  α range: [{alphas.min():.6f}, {alphas.max():.6f}]")
    print(f"  α at λ_min={lambdas[0]:.4f}: {alphas[0]:.6f}")
    print(f"  α at λ_max={lambdas[-1]:.4f}: {alphas[-1]:.6f}")
    print()

    # =====================================================================
    # Step 2: Static LoRA weight magnitudes
    # =====================================================================
    print("Step 2: Computing per-layer LoRA weight magnitudes ...")
    magnitudes = compute_lora_weight_magnitudes(model)
    groups = group_layers_by_unet_position(list(magnitudes.keys()))

    for gname in ["encoder", "middle", "decoder"]:
        layers = groups[gname]
        if layers:
            vals = [magnitudes[l] for l in layers]
            print(f"  {gname:8s}: {len(layers):3d} layers, "
                  f"mean ||ΔW|| = {np.mean(vals):.6f}, "
                  f"std = {np.std(vals):.6f}")
    print()

    # =====================================================================
    # Step 3: Statistical tests
    # =====================================================================
    print("Step 3: Statistical tests on α(λ) monotonicity ...")
    stats = monotonicity_test(lambdas, alphas)
    print(f"  Spearman r       = {stats['spearman_r']:.4f}")
    print(f"  p-value          = {stats['p_value']:.2e}")
    print(f"  Monotone ratio   = {stats['monotone_ratio']:.1%}")
    print(f"  Δα (high-low λ)  = {stats['delta_alpha']:.6f}")
    print(f"  α range          = [{stats['alpha_min']:.6f}, "
          f"{stats['alpha_max']:.6f}]")
    print()

    # =====================================================================
    # Step 4 (optional): lT quality via SSIM
    # =====================================================================
    lt_quality = None
    if args.img_dir:
        print("Step 4: Computing lT quality (SSIM) ...")
        images, names = load_images(args.img_dir, args.n_images, device)
        print(f"  Loaded {images.shape[0]} images from {args.img_dir}")

        lt_lambdas = np.exp(
            np.linspace(
                np.log(lambda_range[0]),
                np.log(lambda_range[1]),
                min(args.n_points, 50),
            )
        )
        lt_quality_raw = collect_lt_quality(model, images, lt_lambdas, device)

        lt_quality = np.interp(lambdas, lt_lambdas, lt_quality_raw)

        corr_alpha_lt = float(np.corrcoef(alphas, lt_quality)[0, 1])
        print(f"  lT SSIM range: [{lt_quality.min():.4f}, "
              f"{lt_quality.max():.4f}]")
        print(f"  Pearson r(α, lT SSIM) = {corr_alpha_lt:.4f}")
        print()
    else:
        print("Step 4: Skipped (no --img_dir provided)\n")

    # =====================================================================
    # Step 5: Visualizations
    # =====================================================================
    print("Step 5: Generating visualizations ...")

    plot_alpha_curve(
        lambdas, alphas, stats,
        os.path.join(args.out_dir, "alpha_curve.pdf"),
    )
    print("  Saved alpha_curve.pdf")

    plot_weight_magnitude_bar(
        magnitudes, groups,
        os.path.join(args.out_dir, "weight_magnitude_bar.pdf"),
    )
    print("  Saved weight_magnitude_bar.pdf")

    layer_order = (
        groups.get("encoder", [])
        + groups.get("middle", [])
        + groups.get("decoder", [])
    )
    if layer_order:
        plot_effective_scaling_heatmap(
            lambdas, alphas, magnitudes, layer_order, groups,
            os.path.join(args.out_dir, "effective_scaling_heatmap.pdf"),
        )
        print("  Saved effective_scaling_heatmap.pdf")

    plot_main_figure(
        lambdas, alphas, stats, lt_quality, magnitudes, groups,
        os.path.join(args.out_dir, "scaling_factor_main.pdf"),
    )
    print("  Saved scaling_factor_main.pdf")

    # =====================================================================
    # Step 6: Numeric results
    # =====================================================================
    results_path = os.path.join(args.out_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("LoRA Scaling Factor Analysis — Numeric Results\n")
        f.write("=" * 55 + "\n\n")

        f.write("Global α(λ) Statistics:\n")
        f.write(f"  Spearman r       = {stats['spearman_r']:.4f}\n")
        f.write(f"  p-value          = {stats['p_value']:.2e}\n")
        f.write(f"  Monotone ratio   = {stats['monotone_ratio']:.1%}\n")
        f.write(f"  Δα (high−low λ)  = {stats['delta_alpha']:.6f}\n")
        f.write(f"  α range          = [{stats['alpha_min']:.6f}, "
                f"{stats['alpha_max']:.6f}]\n\n")

        f.write("α(λ) samples:\n")
        sample_indices = np.linspace(0, len(lambdas) - 1, 10, dtype=int)
        for idx in sample_indices:
            f.write(f"  λ={lambdas[idx]:10.4f}  α={alphas[idx]:.6f}\n")
        f.write("\n")

        f.write("Per-group LoRA weight magnitudes:\n")
        for gname in ["encoder", "middle", "decoder"]:
            layers = groups[gname]
            if layers:
                vals = [magnitudes[l] for l in layers]
                f.write(f"  {gname:8s}: n={len(layers):3d}  "
                        f"mean={np.mean(vals):.6f}  "
                        f"std={np.std(vals):.6f}  "
                        f"max={np.max(vals):.6f}\n")
        f.write("\n")

        f.write("Top-10 LoRA layers by ||ΔW||:\n")
        sorted_layers = sorted(magnitudes.items(), key=lambda x: -x[1])
        for name, mag in sorted_layers[:10]:
            f.write(f"  {name:55s}  ||ΔW|| = {mag:.6f}\n")
        f.write("\n")

        if lt_quality is not None:
            corr_val = float(np.corrcoef(alphas, lt_quality)[0, 1])
            f.write("lT Quality (SSIM) Analysis:\n")
            f.write(f"  Pearson r(α, lT SSIM) = {corr_val:.4f}\n")
            f.write(f"  lT SSIM range         = [{lt_quality.min():.4f}, "
                    f"{lt_quality.max():.4f}]\n\n")

            f.write("lT quality samples:\n")
            for idx in sample_indices:
                f.write(f"  λ={lambdas[idx]:10.4f}  "
                        f"α={alphas[idx]:.6f}  "
                        f"SSIM={lt_quality[idx]:.4f}\n")

    print(f"  Saved {results_path}")
    print(f"\nAll results saved to {args.out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA Scaling Factor Analysis: prove α(λ) emerges as a "
                    "monotonic rate-adaptive strategy from R-D optimization.",
    )
    parser.add_argument("--sd_path", required=True,
                        help="Path to SD-Turbo (256ch variant)")
    parser.add_argument("--elic_path", required=True,
                        help="Path to ELIC pretrained weights")
    parser.add_argument("--codec_path", required=True,
                        help="Path to variable-rate StableCodec checkpoint")
    parser.add_argument("--img_dir", default=None,
                        help="Directory of test images for lT quality "
                             "(optional; e.g. Kodak24/HR)")
    parser.add_argument("--out_dir", default="results/lora_scaling",
                        help="Output directory for plots and results")
    parser.add_argument("--n_points", type=int, default=200,
                        help="Number of λ points to sweep")
    parser.add_argument("--n_images", type=int, default=5,
                        help="Number of images for lT quality")
    parser.add_argument("--lambda_min", type=float, default=0.1,
                        help="Lambda min (match training config)")
    parser.add_argument("--lambda_max", type=float, default=128.0,
                        help="Lambda max (match training config)")
    args = parser.parse_args()

    run_experiment(args)
