"""
特征频谱差异实验 (Frequency Spectrum Analysis of Feature Maps)

证明：低码率(高λ)下 UNet 自主生成了高频纹理（频谱四周能量高），
      高码率(低λ)下 UNet 不需要凭空造高频（隐变量本身包含结构）。

做法：
  1. 加载训练好的 variable-rate StableCodec 模型
  2. 在 UNet 目标层注入 forward hook，捕获输出特征 F_out
  3. 对 F_out 做 2D FFT，获取幅度谱 (Magnitude Spectrum)
  4. 计算径向平均得到 1D 频率-能量衰减曲线
  5. 对比不同 λ 下的频谱差异

期望结果：
  - 高λ (极低码率): 高频区域（频谱四周）拥有更多能量
  - 低λ (高码率): 高频能量相对较少

用法 (从 repo 根目录运行):
    python src/experiment_variable_spectrum.py \\
        --sd_path      /path/to/sd-turbo_256 \\
        --elic_path    /path/to/elic_official.pth \\
        --codec_path   /path/to/variable_rate_checkpoint.pth.tar \\
        --img_dir      /path/to/Kodak24/HR \\
        --out_dir      results/spectrum
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
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from StableCodec_variable2 import StableCodec


# ---------------------------------------------------------------------------
# Feature capture via forward hooks
# ---------------------------------------------------------------------------

FEATURE_MAPS: dict[str, torch.Tensor] = {}


def _make_hook(name: str):
    def hook_fn(module: nn.Module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        FEATURE_MAPS[name] = out.detach()
    return hook_fn


def install_hooks(
    unet: nn.Module, target_block: str,
) -> list[torch.utils.hooks.RemovableHook]:
    handles = []
    for name, module in unet.named_modules():
        if name == target_block:
            handles.append(module.register_forward_hook(_make_hook(name)))
        elif name.startswith(target_block + ".") and isinstance(
            module, (nn.Conv2d, nn.Linear)
        ):
            pass
    if not handles:
        for name, module in unet.named_modules():
            if name == target_block:
                handles.append(module.register_forward_hook(_make_hook(name)))
    return handles


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
# FFT spectrum analysis
# ---------------------------------------------------------------------------

def compute_magnitude_spectrum(feat: torch.Tensor) -> np.ndarray:
    """Compute log magnitude spectrum averaged over batch and channels.

    Args:
        feat: (B, C, H, W) feature tensor.

    Returns:
        2D log-magnitude spectrum (H, W), DC at center.
    """
    fft2 = torch.fft.fft2(feat, dim=(-2, -1))
    fft2_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
    mag = fft2_shifted.abs()
    mag_avg = mag.mean(dim=(0, 1))
    return torch.log1p(mag_avg).cpu().numpy()


def radial_profile(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute azimuthally averaged 1D radial profile of a 2D spectrum.

    Returns:
        (frequencies, energies) — 1D arrays.
    """
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    radial_sum = np.zeros(max_r + 1)
    radial_count = np.zeros(max_r + 1)
    mask = r <= max_r
    for ri in range(max_r + 1):
        ring = spectrum[r == ri]
        radial_sum[ri] = ring.sum()
        radial_count[ri] = len(ring)
    radial_count[radial_count == 0] = 1
    radial_mean = radial_sum / radial_count
    freqs = np.arange(max_r + 1) / max_r
    return freqs, radial_mean


def compute_hf_energy_ratio(
    spectrum: np.ndarray, cutoff: float = 0.5,
) -> float:
    """Ratio of high-frequency energy (radius > cutoff * max_r) to total."""
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = min(cy, cx)
    total = spectrum.sum()
    hf = spectrum[r > cutoff * max_r].sum()
    return float(hf / max(total, 1e-12))


def compute_spectral_centroid(
    freqs: np.ndarray, energies: np.ndarray,
) -> float:
    """Weighted mean frequency (higher = more HF content)."""
    total = energies.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * energies) / total)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_spectrum_grid(
    img_np: np.ndarray,
    spectra: dict[float, np.ndarray],
    name: str,
    target_block: str,
    out_path: str,
) -> None:
    """One row: original | 2D spectrum@λ1 | spectrum@λ2 | ..."""
    lambdas = sorted(spectra.keys())
    n_cols = len(lambdas) + 1

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    all_vals = list(spectra.values())
    vmin = min(e.min() for e in all_vals)
    vmax = max(e.max() for e in all_vals)

    im = None
    for j, lam in enumerate(lambdas):
        im = axes[j + 1].imshow(
            spectra[lam], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[j + 1].set_title(f"λ={lam}", fontsize=12)
        axes[j + 1].axis("off")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8,
                 label="Log Magnitude")
    fig.suptitle(
        f"Feature Frequency Spectrum — {name}\n({target_block})",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_radial_curves(
    radial_per_lam: dict[float, tuple[np.ndarray, np.ndarray]],
    name: str,
    target_block: str,
    out_path: str,
) -> None:
    """Overlay 1D radial frequency-energy curves for all λ."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.coolwarm
    lambdas = sorted(radial_per_lam.keys())
    colors = cmap(np.linspace(0, 1, len(lambdas)))

    for lam, color in zip(lambdas, colors):
        freqs, energies = radial_per_lam[lam]
        ax.plot(freqs, energies, linewidth=2, color=color, label=f"λ={lam}")

    ax.set_xlabel("Normalized Frequency (0=DC, 1=Nyquist)", fontsize=12)
    ax.set_ylabel("Mean Log Magnitude", fontsize=12)
    ax.set_title(
        f"Radial Frequency-Energy Curve — {name}\n({target_block})",
        fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_avg_radial_curves(
    lambdas: list[float],
    avg_radials: dict[float, tuple[np.ndarray, np.ndarray]],
    n_images: int,
    target_block: str,
    out_path: str,
) -> None:
    """Average radial curves across all images."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(lambdas)))

    for lam, color in zip(lambdas, colors):
        freqs, energies = avg_radials[lam]
        ax.plot(freqs, energies, linewidth=2, color=color, label=f"λ={lam}")

    ax.set_xlabel("Normalized Frequency (0=DC, 1=Nyquist)", fontsize=12)
    ax.set_ylabel("Mean Log Magnitude", fontsize=12)
    ax.set_title(
        f"Average Radial Spectrum vs λ\n"
        f"({n_images} images, {target_block})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hf_energy_bar(
    lambdas: list[float],
    hf_ratios: dict[float, list[float]],
    n_images: int,
    target_block: str,
    out_path: str,
) -> None:
    """Bar chart of high-frequency energy ratio vs λ."""
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(hf_ratios[lam]) for lam in lambdas]
    stds = [np.std(hf_ratios[lam]) for lam in lambdas]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(lambdas)))

    ax.bar(
        [str(lam) for lam in lambdas], means, yerr=stds,
        color=colors, edgecolor="k", capsize=5)
    ax.set_xlabel("λ (rate-distortion tradeoff)", fontsize=12)
    ax.set_ylabel("High-Frequency Energy Ratio", fontsize=12)
    ax.set_title(
        f"HF Energy Ratio vs λ\n"
        f"(averaged over {n_images} images, {target_block})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hf_energy_curve(
    lambdas: list[float],
    hf_ratios: dict[float, list[float]],
    n_images: int,
    target_block: str,
    out_path: str,
) -> None:
    """Line plot of HF energy ratio vs log(λ)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [np.mean(hf_ratios[lam]) for lam in lambdas]
    stds = [np.std(hf_ratios[lam]) for lam in lambdas]

    ax.errorbar(
        lambdas, means, yerr=stds, marker="o", linewidth=2,
        capsize=4, color="darkred", markerfacecolor="gold")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)", fontsize=12)
    ax.set_ylabel("HF Energy Ratio", fontsize=12)
    ax.set_title(
        f"High-Frequency Energy Ratio vs λ\n"
        f"({n_images} images, {target_block})", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_centroid_curve(
    lambdas: list[float],
    centroids: dict[float, list[float]],
    n_images: int,
    target_block: str,
    out_path: str,
) -> None:
    """Line plot of spectral centroid vs log(λ)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [np.mean(centroids[lam]) for lam in lambdas]
    stds = [np.std(centroids[lam]) for lam in lambdas]

    ax.errorbar(
        lambdas, means, yerr=stds, marker="s", linewidth=2,
        capsize=4, color="darkblue", markerfacecolor="cyan")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)", fontsize=12)
    ax.set_ylabel("Spectral Centroid (normalized freq)", fontsize=12)
    ax.set_title(
        f"Spectral Centroid vs λ\n"
        f"({n_images} images, {target_block})", fontsize=13)
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

    # --- Install hooks ---
    handles = install_hooks(model.unet, args.target_block)
    print(f"Installed {len(handles)} hook(s) on '{args.target_block}'")
    if not handles:
        print("WARNING: No hooks installed. Available blocks:")
        for name, _ in model.unet.named_modules():
            if "up_blocks" in name and "." not in name.split("up_blocks.")[-1]:
                print(f"  {name}")
        return
    print()

    # --- Load images ---
    images, names = load_images(args.img_dir, args.n_images, device)
    N = images.shape[0]
    print(f"Loaded {N} images from {args.img_dir}\n")

    lambdas = sorted([float(x) for x in args.lambdas.split(",")])
    print(f"Testing λ values: {lambdas}\n")

    # --- Run forward and collect spectra ---
    all_spectra: dict[int, dict[float, np.ndarray]] = {}
    all_radials: dict[int, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    hf_ratios: dict[float, list[float]] = {lam: [] for lam in lambdas}
    centroids: dict[float, list[float]] = {lam: [] for lam in lambdas}
    # For averaging radial curves
    radial_accum: dict[float, list[np.ndarray]] = {lam: [] for lam in lambdas}

    for img_idx in range(N):
        img = images[img_idx:img_idx + 1]
        B, C, H, W = img.shape
        all_spectra[img_idx] = {}
        all_radials[img_idx] = {}

        for lam in lambdas:
            FEATURE_MAPS.clear()
            lmbda = torch.tensor([lam], dtype=torch.float32, device=device)
            _ = model(img, [1], H, W, lmbda=lmbda)

            if not FEATURE_MAPS:
                print(f"  WARNING: No features captured for λ={lam}")
                continue

            feat = next(iter(FEATURE_MAPS.values()))
            if feat.ndim == 3:
                sq = int(feat.shape[1] ** 0.5)
                feat = feat.view(feat.shape[0], sq, sq, feat.shape[2])
                feat = feat.permute(0, 3, 1, 2)

            spectrum = compute_magnitude_spectrum(feat)
            freqs, radial_e = radial_profile(spectrum)
            hf_r = compute_hf_energy_ratio(spectrum, cutoff=args.hf_cutoff)
            sc = compute_spectral_centroid(freqs, radial_e)

            all_spectra[img_idx][lam] = spectrum
            all_radials[img_idx][lam] = (freqs, radial_e)
            hf_ratios[lam].append(hf_r)
            centroids[lam].append(sc)
            radial_accum[lam].append(radial_e)

            print(
                f"  [{names[img_idx]}] λ={lam:6.2f}  "
                f"HF_ratio={hf_r:.4f}  centroid={sc:.4f}  "
                f"spectrum_shape={spectrum.shape}")
        print()

    # --- Compute average radial curves ---
    avg_radials: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lam in lambdas:
        if radial_accum[lam]:
            min_len = min(len(r) for r in radial_accum[lam])
            stacked = np.stack([r[:min_len] for r in radial_accum[lam]])
            avg_e = stacked.mean(axis=0)
            avg_f = np.linspace(0, 1, min_len)
            avg_radials[lam] = (avg_f, avg_e)

    # --- Visualizations ---
    print("Generating visualizations ...")

    for img_idx in range(N):
        img_np = (
            (images[img_idx].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        ).clip(0, 255).astype(np.uint8)

        if all_spectra[img_idx]:
            plot_spectrum_grid(
                img_np, all_spectra[img_idx], names[img_idx],
                args.target_block,
                os.path.join(args.out_dir,
                             f"spectrum_grid_{names[img_idx]}.png"))

        if all_radials[img_idx]:
            plot_radial_curves(
                all_radials[img_idx], names[img_idx], args.target_block,
                os.path.join(args.out_dir,
                             f"radial_curve_{names[img_idx]}.png"))

        print(f"  Saved plots for {names[img_idx]}")

    # Aggregate plots
    if avg_radials:
        plot_avg_radial_curves(
            lambdas, avg_radials, N, args.target_block,
            os.path.join(args.out_dir, "avg_radial_curves.png"))

    plot_hf_energy_bar(
        lambdas, hf_ratios, N, args.target_block,
        os.path.join(args.out_dir, "hf_energy_bar.png"))

    plot_hf_energy_curve(
        lambdas, hf_ratios, N, args.target_block,
        os.path.join(args.out_dir, "hf_energy_curve.png"))

    plot_spectral_centroid_curve(
        lambdas, centroids, N, args.target_block,
        os.path.join(args.out_dir, "spectral_centroid_curve.png"))

    # --- Numeric results ---
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write("Feature Frequency Spectrum Analysis — Numeric Results\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Target block: {args.target_block}\n")
        f.write(f"HF cutoff:    {args.hf_cutoff}\n")
        f.write(f"Images:       {N}\n\n")

        f.write("HF Energy Ratio per λ:\n")
        for lam in lambdas:
            m = np.mean(hf_ratios[lam])
            s = np.std(hf_ratios[lam])
            f.write(f"  λ={lam:8.2f}  mean={m:.6f}  std={s:.6f}\n")

        f.write(f"\nHF ratio (λ_high / λ_low): "
                f"{np.mean(hf_ratios[lambdas[-1]]) / max(np.mean(hf_ratios[lambdas[0]]), 1e-12):.2f}x\n")

        f.write("\nSpectral Centroid per λ:\n")
        for lam in lambdas:
            m = np.mean(centroids[lam])
            s = np.std(centroids[lam])
            f.write(f"  λ={lam:8.2f}  mean={m:.6f}  std={s:.6f}\n")

        f.write(f"\nCentroid ratio (λ_high / λ_low): "
                f"{np.mean(centroids[lambdas[-1]]) / max(np.mean(centroids[lambdas[0]]), 1e-12):.2f}x\n")

    # --- Cleanup hooks ---
    for h in handles:
        h.remove()

    print(f"\nAll results saved to {args.out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Frequency Spectrum Analysis: prove UNet generates "
                    "more HF texture at low bitrate (high λ).",
    )
    parser.add_argument("--sd_path", required=True,
                        help="Path to SD-Turbo (256ch variant)")
    parser.add_argument("--elic_path", required=True,
                        help="Path to ELIC pretrained weights")
    parser.add_argument("--codec_path", required=True,
                        help="Path to variable-rate StableCodec checkpoint")
    parser.add_argument("--img_dir", required=True,
                        help="Directory of test images (e.g. Kodak24/HR)")
    parser.add_argument("--out_dir", default="results/spectrum",
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
                        help="UNet block to capture (e.g. up_blocks.2, up_blocks.3)")
    parser.add_argument("--hf_cutoff", type=float, default=0.5,
                        help="Frequency cutoff for HF energy ratio (0-1)")
    args = parser.parse_args()

    run_experiment(args)
