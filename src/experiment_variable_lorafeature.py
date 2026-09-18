"""
LoRA 残差能量图实验 (LoRA Residual Energy Map)

证明：低码率(高λ)下 LoRA 在"发力"无中生有(纹理幻觉器)，
      高码率(低λ)下 LoRA 在"摸鱼"(仅做微调)。

做法：
  1. 加载训练好的 variable-rate StableCodec 模型
  2. 在 UNet decoder 高分辨率层的 LoRA 模块注入 hook，
     捕获每层 ΔF_LoRA = lora_B(lora_A(x)) * scaling
  3. 对同一张图像，分别注入不同 λ 值 (从低λ/高码率 到 高λ/低码率)
  4. 计算 ΔF_LoRA 在通道维度的 L2 范数，得到 2D 能量热力图
  5. 可视化对比

期望结果：
  - 高λ (极低码率): 热力图大面积高亮（LoRA 充当"纹理幻觉器"）
  - 低λ (高码率): 热力图整体暗淡（LoRA 仅做保守微调）

用法 (从 repo 根目录运行):
    python src/experiment_variable_lorafeature.py \\
        --sd_path      /path/to/sd-turbo_256 \\
        --elic_path    /path/to/elic_official.pth \\
        --codec_path   /path/to/variable_rate_checkpoint.pth.tar \\
        --img_dir      /path/to/Kodak24/HR \\
        --out_dir      results/lora_energy
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from StableCodec_variable2 import StableCodec


# ---------------------------------------------------------------------------
# LoRA delta capture
# ---------------------------------------------------------------------------

LORA_DELTAS: dict[str, torch.Tensor] = {}


def make_capturing_lora_fwd(layer_name: str):
    """Return a patched LoRA forward that stores ΔF_LoRA in LORA_DELTAS."""
    def fwd(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            return self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        delta_total = torch.zeros_like(result)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x_cast = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                delta = lora_B(lora_A(dropout(x_cast))) * scaling
            else:
                x_cast = dropout(x_cast)
                delta = self._apply_dora(
                    x_cast, lora_A, lora_B, scaling, active_adapter
                )
            delta_total = delta_total + delta
            result = result + delta

        LORA_DELTAS[layer_name] = delta_total.detach()
        result = result.to(torch_result_dtype)
        return result

    return fwd


def patch_unet_lora_layers(
    unet: nn.Module, target_block: str
) -> list[str]:
    """Replace forward on LoRA layers in target_block with capturing version."""
    patched = []
    for name, module in unet.named_modules():
        if not name.startswith(target_block):
            continue
        if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
            cap_fwd = make_capturing_lora_fwd(name)
            module.forward = cap_fwd.__get__(module, module.__class__)
            patched.append(name)
    return patched


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(
    img_dir: str, n_images: int, device: torch.device
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
# Energy map computation
# ---------------------------------------------------------------------------

def compute_energy_map(
    deltas: dict[str, torch.Tensor], mode: str = "l2"
) -> np.ndarray:
    """Aggregate per-layer LoRA deltas into a single 2D energy map.

    All deltas are resized to the largest spatial resolution found,
    then summed.  Returns shape [H, W].
    """
    if not deltas:
        raise ValueError("No LoRA deltas captured — check target_block name")

    max_h = max(d.shape[2] for d in deltas.values())
    max_w = max(d.shape[3] for d in deltas.values())
    ref_device = next(iter(deltas.values())).device

    energy = torch.zeros(1, 1, max_h, max_w, device=ref_device)
    for delta in deltas.values():
        if mode == "l2":
            e = delta.norm(dim=1, keepdim=True)
        else:
            e = delta.abs().mean(dim=1, keepdim=True)
        if e.shape[2] != max_h or e.shape[3] != max_w:
            e = F_torch.interpolate(
                e, (max_h, max_w), mode="bilinear", align_corners=False
            )
        energy = energy + e

    return energy[0, 0].cpu().numpy()


def compute_single_layer_energy(
    delta: torch.Tensor, mode: str = "l2"
) -> np.ndarray:
    if mode == "l2":
        e = delta.norm(dim=1, keepdim=True)
    else:
        e = delta.abs().mean(dim=1, keepdim=True)
    return e[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_heatmap_grid(
    img_np: np.ndarray,
    energies: dict[float, np.ndarray],
    name: str,
    target_block: str,
    out_path: str,
) -> None:
    """One row: original | heatmap@λ1 | heatmap@λ2 | ... """
    lambdas = sorted(energies.keys())
    n_cols = len(lambdas) + 1

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    all_vals = list(energies.values())
    vmin = min(e.min() for e in all_vals)
    vmax = max(e.max() for e in all_vals)

    im = None
    for j, lam in enumerate(lambdas):
        im = axes[j + 1].imshow(
            energies[lam], cmap="jet", vmin=vmin, vmax=vmax
        )
        axes[j + 1].set_title(f"λ={lam}", fontsize=12)
        axes[j + 1].axis("off")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="LoRA Energy")
    fig.suptitle(
        f"LoRA Residual Energy Map — {name}\n(Block: {target_block})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_heatmap(
    img_np: np.ndarray,
    energies: dict[float, np.ndarray],
    name: str,
    target_block: str,
    out_path: str,
    alpha: float = 0.5,
) -> None:
    """Overlay heatmap on original image for each λ."""
    lambdas = sorted(energies.keys())
    n_cols = len(lambdas)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    all_vals = list(energies.values())
    vmax = max(e.max() for e in all_vals)

    for j, lam in enumerate(lambdas):
        axes[j].imshow(img_np)
        e_resized = np.array(
            Image.fromarray(energies[lam]).resize(
                (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
            )
        )
        im = axes[j].imshow(
            e_resized, cmap="jet", alpha=alpha, vmin=0, vmax=vmax
        )
        axes[j].set_title(f"λ={lam}", fontsize=13)
        axes[j].axis("off")

    fig.colorbar(im, ax=axes, shrink=0.8, label="LoRA Energy")
    fig.suptitle(
        f"LoRA Energy Overlay — {name}\n(Block: {target_block})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_energy_bar(
    lambdas: list[float],
    mean_energies: dict[float, list[float]],
    n_images: int,
    target_block: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(mean_energies[lam]) for lam in lambdas]
    stds = [np.std(mean_energies[lam]) for lam in lambdas]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(lambdas)))

    bars = ax.bar(
        [str(lam) for lam in lambdas], means, yerr=stds,
        color=colors, edgecolor="k", capsize=5,
    )
    ax.set_xlabel("λ (rate-distortion tradeoff)", fontsize=12)
    ax.set_ylabel("Mean LoRA Energy", fontsize=12)
    ax.set_title(
        f"Mean LoRA Residual Energy vs λ\n"
        f"(averaged over {n_images} images, block: {target_block})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer_breakdown(
    lambdas_pair: tuple[float, float],
    per_layer_stats: dict[float, dict[str, float]],
    target_block: str,
    out_path: str,
) -> None:
    """Bar chart: per-layer LoRA energy for the two extreme λ values."""
    lam_lo, lam_hi = lambdas_pair
    layer_names = sorted(per_layer_stats[lam_lo].keys())
    x_pos = np.arange(len(layer_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(layer_names) * 0.6), 5))
    ax.bar(
        x_pos - width / 2,
        [per_layer_stats[lam_lo].get(n, 0) for n in layer_names],
        width, label=f"λ={lam_lo} (high bitrate)", color="steelblue",
    )
    ax.bar(
        x_pos + width / 2,
        [per_layer_stats[lam_hi].get(n, 0) for n in layer_names],
        width, label=f"λ={lam_hi} (low bitrate)", color="firebrick",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean |ΔF_LoRA|", fontsize=11)
    ax.set_title(
        f"Per-Layer LoRA Energy Breakdown ({target_block})", fontsize=13
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_energy_curve(
    lambdas: list[float],
    mean_energies: dict[float, list[float]],
    n_images: int,
    target_block: str,
    out_path: str,
) -> None:
    """Line plot of mean LoRA energy vs log(λ)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [np.mean(mean_energies[lam]) for lam in lambdas]
    stds = [np.std(mean_energies[lam]) for lam in lambdas]

    ax.errorbar(
        lambdas, means, yerr=stds, marker="o", linewidth=2,
        capsize=4, color="darkred", markerfacecolor="gold",
    )
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)", fontsize=12)
    ax.set_ylabel("Mean LoRA Energy", fontsize=12)
    ax.set_title(
        f"LoRA Residual Energy vs λ\n"
        f"({n_images} images, block: {target_block})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
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

    # --- Patch LoRA layers for delta capture ---
    target_block = args.target_block
    patched_layers = patch_unet_lora_layers(model.unet, target_block)
    print(f"Patched {len(patched_layers)} LoRA layers in '{target_block}':")
    for layer in patched_layers:
        print(f"  {layer}")
    if not patched_layers:
        print("WARNING: No LoRA layers found in target block. "
              "Available up_blocks:")
        for name, _ in model.unet.named_modules():
            if "up_blocks" in name and "lora_A" in name:
                print(f"  {name}")
        return
    print()

    # --- Load images ---
    images, names = load_images(args.img_dir, args.n_images, device)
    N = images.shape[0]
    print(f"Loaded {N} images from {args.img_dir}\n")

    # --- Lambda values ---
    lambdas = sorted([float(x) for x in args.lambdas.split(",")])
    print(f"Testing λ values: {lambdas}\n")

    # --- Run forward and collect energy maps ---
    all_energies: dict[int, dict[float, np.ndarray]] = {}
    mean_energies: dict[float, list[float]] = {lam: [] for lam in lambdas}

    for img_idx in range(N):
        img = images[img_idx : img_idx + 1]
        B, C, H, W = img.shape
        pos_tag_prompt = [1]

        all_energies[img_idx] = {}

        for lam in lambdas:
            LORA_DELTAS.clear()
            lmbda = torch.tensor([lam], dtype=torch.float32, device=device)

            _ = model(img, pos_tag_prompt, H, W, lmbda=lmbda)

            energy = compute_energy_map(LORA_DELTAS, mode=args.energy_mode)
            all_energies[img_idx][lam] = energy
            mean_energies[lam].append(float(energy.mean()))

            print(
                f"  [{names[img_idx]}] λ={lam:6.2f}  "
                f"mean_energy={energy.mean():.6f}  "
                f"max_energy={energy.max():.6f}"
            )
        print()

    # --- Visualization ---
    print("Generating visualizations ...")

    # 1. Per-image heatmap grids
    for img_idx in range(N):
        img_np = (
            (images[img_idx].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            * 255
        ).clip(0, 255).astype(np.uint8)

        plot_heatmap_grid(
            img_np, all_energies[img_idx], names[img_idx], target_block,
            os.path.join(args.out_dir, f"lora_energy_{names[img_idx]}.png"),
        )
        plot_overlay_heatmap(
            img_np, all_energies[img_idx], names[img_idx], target_block,
            os.path.join(args.out_dir, f"lora_overlay_{names[img_idx]}.png"),
        )
        print(f"  Saved heatmaps for {names[img_idx]}")

    # 2. Mean energy vs λ (bar chart + line plot)
    plot_mean_energy_bar(
        lambdas, mean_energies, N, target_block,
        os.path.join(args.out_dir, "mean_energy_bar.png"),
    )
    plot_energy_curve(
        lambdas, mean_energies, N, target_block,
        os.path.join(args.out_dir, "mean_energy_curve.png"),
    )

    # 3. Per-layer energy breakdown (extreme λ pair)
    per_layer_stats: dict[float, dict[str, float]] = {}
    for lam in [lambdas[0], lambdas[-1]]:
        LORA_DELTAS.clear()
        lmbda = torch.tensor([lam], dtype=torch.float32, device=device)
        _ = model(images[0:1], [1], images.shape[2], images.shape[3], lmbda=lmbda)

        per_layer_stats[lam] = {}
        for layer_name, delta in LORA_DELTAS.items():
            short = layer_name.replace(target_block + ".", "")
            per_layer_stats[lam][short] = float(delta.abs().mean().item())

    plot_per_layer_breakdown(
        (lambdas[0], lambdas[-1]),
        per_layer_stats,
        target_block,
        os.path.join(args.out_dir, "per_layer_breakdown.png"),
    )

    # 4. Save numeric results
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write("LoRA Residual Energy Map — Numeric Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target block: {target_block}\n")
        f.write(f"Energy mode:  {args.energy_mode}\n")
        f.write(f"Images:       {N}\n\n")

        f.write("Mean energy per λ:\n")
        for lam in lambdas:
            m = np.mean(mean_energies[lam])
            s = np.std(mean_energies[lam])
            f.write(f"  λ={lam:8.2f}  mean={m:.6f}  std={s:.6f}\n")

        f.write(f"\nEnergy ratio (λ_high / λ_low): "
                f"{np.mean(mean_energies[lambdas[-1]]) / max(np.mean(mean_energies[lambdas[0]]), 1e-12):.2f}x\n")

        f.write("\nPer-layer breakdown (first image):\n")
        for layer_name in sorted(per_layer_stats[lambdas[0]].keys()):
            e_lo = per_layer_stats[lambdas[0]].get(layer_name, 0)
            e_hi = per_layer_stats[lambdas[-1]].get(layer_name, 0)
            ratio = e_hi / max(e_lo, 1e-12)
            f.write(
                f"  {layer_name:50s}  "
                f"λ={lambdas[0]:5.1f}: {e_lo:.6f}  "
                f"λ={lambdas[-1]:5.1f}: {e_hi:.6f}  "
                f"ratio={ratio:.2f}x\n"
            )

    print(f"\nAll results saved to {args.out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA Residual Energy Map: prove LoRA acts as texture "
                    "hallucinator at low bitrate and refiner at high bitrate.",
    )
    parser.add_argument("--sd_path", required=True,
                        help="Path to SD-Turbo (256ch variant)")
    parser.add_argument("--elic_path", required=True,
                        help="Path to ELIC pretrained weights")
    parser.add_argument("--codec_path", required=True,
                        help="Path to variable-rate StableCodec checkpoint")
    parser.add_argument("--img_dir", required=True,
                        help="Directory of test images (e.g. Kodak24/HR)")
    parser.add_argument("--out_dir", default="results/lora_energy",
                        help="Output directory for plots and results")
    parser.add_argument("--n_images", type=int, default=5,
                        help="Number of images to process")
    parser.add_argument("--lambdas", default="0.5,2,8,32",
                        help="Comma-separated λ values to test")
    parser.add_argument("--lambda_min", type=float, default=0.1,
                        help="Lambda min for model (match training config)")
    parser.add_argument("--lambda_max", type=float, default=128.0,
                        help="Lambda max for model (match training config)")
    parser.add_argument("--target_block", default="up_blocks.3",
                        help="UNet block to analyze (e.g. up_blocks.2, up_blocks.3)")
    parser.add_argument("--energy_mode", default="l2",
                        choices=["l2", "mean_abs"],
                        help="How to aggregate channel dim: L2 norm or mean abs")
    args = parser.parse_args()

    run_experiment(args)
