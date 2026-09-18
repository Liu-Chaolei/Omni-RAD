"""
实验三：统计冗余可压缩性验证
============================
验证论点：256通道之间存在可利用的统计相关性，反而更易压缩。

三种分析方法：
  1. PCA有效秩 — 衡量信息的真实维度（有效秩/总维度 越低 → 冗余越多）
  2. 通道间平均绝对相关系数 — 越高 → 通道间冗余越多
  3. 直接压缩效率 — 通过完整 forward 获取 BPP

Usage:
    python src/experiment_redundancy.py \
        --config_4ch configs/stage1.yaml \
        --config_256ch configs/stage2.yaml \
        --output_dir ./results/experiment3

    # 仅跑统计分析（跳过 BPP，不需要完整模型 forward）
    python src/experiment_redundancy.py --config_4ch configs/ch4.yaml --config_256ch configs/ch256.yaml --skip_bpp
"""

import argparse
import csv
import gc
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# 1. Config & Model loading (reuses profile_model.py logic)
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and merge base + stage config."""
    with open(config_path, "r") as f:
        stage_config = yaml.safe_load(f)

    config_dir = os.path.dirname(config_path)
    base_path = os.path.join(config_dir, "base.yaml")
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
        # Save base model dict before top-level overwrite
        base_model = base_config.get("model", {})
        base_config.update(stage_config)
        # Re-merge nested 'model' dict: base keys + stage overrides
        if "model" in stage_config:
            stage_model = stage_config["model"]
            base_config["model"] = {**base_model, **stage_model}
        return base_config

    return stage_config


def build_model(config_path: str):
    """Build StableCodec from a config file (eval mode, on CUDA)."""
    config = load_config(config_path)
    model_config = config["model"]

    from StableCodec_ori import StableCodec
    model = StableCodec(
        sd_path=model_config["sd_path"],
        lmbda=config.get("lmbda", model_config.get("lambda_ref", 0.5)),
        config=model_config,
    )
    model.set_eval()
    model.cuda()
    return model, config


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------

def load_test_images(
    dataset_path: str,
    model_stride: Tuple[int, int] = (64, 64),
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    """Load all images from dataset_path, normalize to [-1, 1], pad to stride multiples.

    Returns a list of tensors, each shaped [1, 3, H', W'] (padded).
    Also returns original (H, W) per image for BPP calculation.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
    img_paths = sorted(
        p for p in Path(dataset_path).iterdir()
        if p.suffix.lower() in exts
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {dataset_path}")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    images = []
    original_sizes = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        original_sizes.append((img.height, img.width))
        tensor = preprocess(img).unsqueeze(0)  # [1, 3, H, W]

        # Pad to model_stride multiples
        _, _, h, w = tensor.shape
        pad_h = (model_stride[0] - h % model_stride[0]) % model_stride[0]
        pad_w = (model_stride[1] - w % model_stride[1]) % model_stride[1]
        if pad_h > 0 or pad_w > 0:
            tensor = nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        images.append(tensor)

    print(f"[Data] Loaded {len(images)} images from {dataset_path}")
    return images, original_sizes


# ---------------------------------------------------------------------------
# 3. Latent variable collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latents(
    model,
    images: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Collect VAE encoder latents for all images.

    Returns:
        latents: List of tensors, each shaped [1, C, H_lat_i, W_lat_i] (on CPU).
                 Spatial dimensions may differ across images; do NOT torch.cat here,
                 as images can have varying sizes which yield different latent H/W.
    """
    latents = []
    for img in images:
        latent = model.vae.encode(img.cuda()).latent_dist.mode()
        latents.append(latent.cpu())
    return latents


# ---------------------------------------------------------------------------
# 4. Analysis methods
# ---------------------------------------------------------------------------

def _flatten_latents(latents) -> np.ndarray:
    """Flatten latent tensors -> [total_pixels, C] as NumPy array.

    Accepts either:
      - a List of [1, C, H_i, W_i] tensors (variable spatial sizes across images), or
      - a single [N, C, H, W] Tensor (uniform spatial size).

    Images may have different resolutions, so we flatten each latent individually
    along its own spatial dimensions, then concatenate across images.
    This guarantees the channel axis C is the only axis that must be consistent.
    """
    if isinstance(latents, (list, tuple)):
        # Variable spatial sizes: flatten each [1, C, H_i, W_i] -> [H_i*W_i, C]
        flat_parts = [
            t.permute(0, 2, 3, 1).contiguous().reshape(-1, t.shape[1]).numpy()
            for t in latents
        ]
        flat = np.concatenate(flat_parts, axis=0)
    else:
        # Uniform spatial size: single tensor [N, C, H, W] -> [N*H*W, C]
        flat = latents.permute(0, 2, 3, 1).contiguous().reshape(-1, latents.shape[1]).numpy()

    n_obs, n_channels = flat.shape
    if n_obs < n_channels:
        warnings.warn(
            f"Fewer observations ({n_obs}) than channels ({n_channels}); "
            "covariance matrix is rank-deficient. Results may be unreliable."
        )
    return flat


def compute_effective_rank(flat: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute PCA effective rank of latent channels.

    Args:
        flat: [N*H*W, C] flattened latent array (from _flatten_latents)

    Returns:
        effective_rank: exp(Shannon entropy of normalized eigenvalues)
        eigenvalues: sorted normalized eigenvalues (descending)
    """
    # Covariance matrix of channels: [C, C]
    cov = np.cov(flat, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.abs(eigenvalues)

    # Normalize
    eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-12)

    # Effective rank = exp(-sum(p * log(p)))
    log_eig = np.log(eigenvalues + 1e-12)
    entropy = -np.sum(eigenvalues * log_eig)
    effective_rank = np.exp(entropy)

    # Sort descending for visualization
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    return effective_rank, eigenvalues_sorted


def compute_channel_correlation(flat: np.ndarray) -> float:
    """Compute mean absolute off-diagonal correlation coefficient across channels.

    Args:
        flat: [N*H*W, C] flattened latent array (from _flatten_latents)

    Returns:
        mean_abs_corr: scalar
    """
    corr_matrix = np.corrcoef(flat, rowvar=False)
    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    return float(np.mean(np.abs(off_diag)))


@torch.no_grad()
def measure_bpp(
    model,
    images: List[torch.Tensor],
    original_sizes: List[Tuple[int, int]],
) -> Dict[str, float]:
    """Measure average BPP through the full forward pass.

    Returns dict with mean/min/max BPP values.
    """
    bpp_list = []
    for img, (ori_h, ori_w) in zip(images, original_sizes):
        pos_prompt = ["a photo"]
        _, rate_output = model(img.cuda(), pos_prompt, ori_h, ori_w)
        bpp_list.append(rate_output.quantized_total_bpp.item())

    return {
        "mean_bpp": float(np.mean(bpp_list)),
        "min_bpp": float(np.min(bpp_list)),
        "max_bpp": float(np.max(bpp_list)),
        "std_bpp": float(np.std(bpp_list)),
    }


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def plot_redundancy_analysis(
    eigs_4: np.ndarray,
    eigs_256: np.ndarray,
    rank_4: float,
    rank_256: float,
    corr_4: float,
    corr_256: float,
    output_paths: List[str],
    bpp_4: Optional[float] = None,
    bpp_256: Optional[float] = None,
) -> None:
    """Generate redundancy analysis plots and save to multiple formats."""
    num_cols = 3 if bpp_4 is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5))

    # --- Plot 1: Eigenvalue decay curves ---
    ax = axes[0]
    ax.plot(
        np.arange(1, len(eigs_4) + 1), eigs_4,
        "o-", color="steelblue", markersize=4, label=f"4ch (rank={rank_4:.1f})",
    )
    # For 256ch, show top-50 for readability
    top_n = min(50, len(eigs_256))
    ax.plot(
        np.arange(1, top_n + 1), eigs_256[:top_n],
        "s-", color="coral", markersize=3, label=f"256ch top-{top_n} (rank={rank_256:.1f})",
    )
    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Normalized Eigenvalue")
    ax.set_title("Eigenvalue Decay (Information Compactness)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Relative effective rank bar chart ---
    ax = axes[1]
    rel_rank_4 = rank_4 / 4.0
    rel_rank_256 = rank_256 / 256.0
    bars = ax.bar(
        ["4ch", "256ch"],
        [rel_rank_4, rel_rank_256],
        color=["steelblue", "coral"],
        width=0.5,
    )
    # Annotate bars
    for bar, val in zip(bars, [rel_rank_4, rel_rank_256]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Effective Rank / Total Channels")
    ax.set_title("Relative Effective Rank\n(lower = more redundancy)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 3: BPP comparison (optional) ---
    if bpp_4 is not None:
        ax = axes[2]
        bars = ax.bar(
            ["4ch", "256ch"],
            [bpp_4, bpp_256],
            color=["steelblue", "coral"],
            width=0.5,
        )
        for bar, val in zip(bars, [bpp_4, bpp_256]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10,
            )
        ax.set_ylabel("Bits Per Pixel (BPP)")
        ax.set_title("Compression Efficiency\n(lower = better compression)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for path in output_paths:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Report printing & CSV export
# ---------------------------------------------------------------------------

def print_analysis_report(
    stats_4: dict,
    stats_256: dict,
) -> None:
    """Print formatted comparison report to terminal."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  实验三：统计冗余可压缩性验证 — 结果汇总")
    print(sep)

    header = f"{'Metric':<35} {'4ch':>15} {'256ch':>15}"
    print(header)
    print("-" * 72)

    rows = [
        ("Total Channels", f"{stats_4['channels']}", f"{stats_256['channels']}"),
        ("Effective Rank", f"{stats_4['effective_rank']:.2f}", f"{stats_256['effective_rank']:.2f}"),
        ("Relative Rank (rank/channels)", f"{stats_4['relative_rank']:.4f}", f"{stats_256['relative_rank']:.4f}"),
        ("Mean |Channel Correlation|", f"{stats_4['mean_abs_corr']:.4f}", f"{stats_256['mean_abs_corr']:.4f}"),
    ]

    if "mean_bpp" in stats_4:
        rows.append(("Mean BPP", f"{stats_4['mean_bpp']:.4f}", f"{stats_256['mean_bpp']:.4f}"))

    for label, v4, v256 in rows:
        print(f"{label:<35} {v4:>15} {v256:>15}")

    print(sep)
    print("\n[解读]")
    print(f"  - 256ch 的相对有效秩 ({stats_256['relative_rank']:.4f}) "
          f"{'<' if stats_256['relative_rank'] < stats_4['relative_rank'] else '>='} "
          f"4ch ({stats_4['relative_rank']:.4f})")
    print(f"    → 256ch 中{'存在更多冗余' if stats_256['relative_rank'] < stats_4['relative_rank'] else '冗余不显著'}，"
          f"g_a 可利用的统计相关性{'更强' if stats_256['relative_rank'] < stats_4['relative_rank'] else '较弱'}")
    print(f"  - 256ch 的通道间相关性 ({stats_256['mean_abs_corr']:.4f}) "
          f"{'>' if stats_256['mean_abs_corr'] > stats_4['mean_abs_corr'] else '<='} "
          f"4ch ({stats_4['mean_abs_corr']:.4f})")

    if "mean_bpp" in stats_4 and "mean_bpp" in stats_256:
        print(f"  - 256ch 的 BPP ({stats_256['mean_bpp']:.4f}) "
              f"{'<' if stats_256['mean_bpp'] < stats_4['mean_bpp'] else '>='} "
              f"4ch ({stats_4['mean_bpp']:.4f})")
    print()


def save_csv(stats_4: dict, stats_256: dict, output_path: str) -> None:
    """Save analysis results to CSV."""
    # Union of keys from both dicts (preserving insertion order)
    fieldnames = list(dict.fromkeys(list(stats_4.keys()) + list(stats_256.keys())))
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config"] + fieldnames, extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerow({"config": "4ch", **stats_4})
        writer.writerow({"config": "256ch", **stats_256})
    print(f"[CSV] Saved to {output_path}")


# ---------------------------------------------------------------------------
# 7. Free GPU memory
# ---------------------------------------------------------------------------

def free_model(model) -> None:
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 8. Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args) -> None:
    """Run the full redundancy analysis experiment."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config to get test dataset path
    config = load_config(args.config_4ch)
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = config.get("test_dataset")
        if dataset_path is None:
            raise ValueError("test_dataset not found in config. "
                             "Please set it in base.yaml or pass --dataset.")

    model_stride = tuple(config.get("model_stride", [64, 64]))

    # Load test images once (shared across both models)
    print("\n" + "=" * 60)
    print("  Loading test images ...")
    print("=" * 60)
    images, original_sizes = load_test_images(dataset_path, model_stride)

    # ================================================================
    # Phase 1: 4-channel model
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 1: Analyzing 4-channel model")
    print("=" * 60)
    model_4ch, config_4ch = build_model(args.config_4ch)
    latent_channels_4 = config_4ch["model"].get("latent_channels", 4)
    print(f"[Config] latent_channels = {latent_channels_4}")

    # Collect latents
    print("[4ch] Collecting VAE encoder latents ...")
    latents_4ch = collect_latents(model_4ch, images)
    print(f"[4ch] Collected {len(latents_4ch)} latents, "
          f"channels={latents_4ch[0].shape[1]}, "
          f"spatial sizes vary (e.g. first: {tuple(latents_4ch[0].shape[2:])})")

    # Measure BPP (optional)
    bpp_stats_4 = {}
    if not args.skip_bpp:
        print("[4ch] Measuring BPP through full forward ...")
        bpp_stats_4 = measure_bpp(model_4ch, images, original_sizes)
        print(f"[4ch] Mean BPP: {bpp_stats_4['mean_bpp']:.4f}")

    # Free 4ch model GPU memory
    free_model(model_4ch)

    # Analyze latents (CPU computation) — flatten once, reuse for both analyses
    print("[4ch] Flattening latents for analysis ...")
    flat_4ch = _flatten_latents(latents_4ch)

    print("[4ch] Computing PCA effective rank ...")
    rank_4, eigs_4 = compute_effective_rank(flat_4ch)
    print(f"[4ch] Effective rank: {rank_4:.2f} / {latent_channels_4}")

    print("[4ch] Computing channel correlation ...")
    corr_4 = compute_channel_correlation(flat_4ch)
    print(f"[4ch] Mean |correlation|: {corr_4:.4f}")

    stats_4 = {
        "channels": latent_channels_4,
        "effective_rank": rank_4,
        "relative_rank": rank_4 / latent_channels_4,
        "mean_abs_corr": corr_4,
        **bpp_stats_4,
    }

    del latents_4ch, flat_4ch
    gc.collect()

    # ================================================================
    # Phase 2: 256-channel model
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Analyzing 256-channel model")
    print("=" * 60)
    model_256ch, config_256ch = build_model(args.config_256ch)
    latent_channels_256 = config_256ch["model"].get("latent_channels", 256)
    print(f"[Config] latent_channels = {latent_channels_256}")

    # Collect latents
    print("[256ch] Collecting VAE encoder latents ...")
    latents_256ch = collect_latents(model_256ch, images)
    print(f"[256ch] Collected {len(latents_256ch)} latents, "
          f"channels={latents_256ch[0].shape[1]}, "
          f"spatial sizes vary (e.g. first: {tuple(latents_256ch[0].shape[2:])})")

    # Measure BPP (optional)
    bpp_stats_256 = {}
    if not args.skip_bpp:
        print("[256ch] Measuring BPP through full forward ...")
        bpp_stats_256 = measure_bpp(model_256ch, images, original_sizes)
        print(f"[256ch] Mean BPP: {bpp_stats_256['mean_bpp']:.4f}")

    # Free 256ch model GPU memory
    free_model(model_256ch)

    # Analyze latents (CPU computation) — flatten once, reuse for both analyses
    print("[256ch] Flattening latents for analysis ...")
    flat_256ch = _flatten_latents(latents_256ch)

    print("[256ch] Computing PCA effective rank ...")
    rank_256, eigs_256 = compute_effective_rank(flat_256ch)
    print(f"[256ch] Effective rank: {rank_256:.2f} / {latent_channels_256}")

    print("[256ch] Computing channel correlation ...")
    corr_256 = compute_channel_correlation(flat_256ch)
    print(f"[256ch] Mean |correlation|: {corr_256:.4f}")

    stats_256 = {
        "channels": latent_channels_256,
        "effective_rank": rank_256,
        "relative_rank": rank_256 / latent_channels_256,
        "mean_abs_corr": corr_256,
        **bpp_stats_256,
    }

    del latents_256ch, flat_256ch
    gc.collect()

    # ================================================================
    # Phase 3: Report & Visualization
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 3: Generating report")
    print("=" * 60)

    # Terminal report
    print_analysis_report(stats_4, stats_256)

    # CSV export
    csv_path = str(output_dir / "redundancy_stats.csv")
    save_csv(stats_4, stats_256, csv_path)

    # Visualization (save both PDF and PNG in one render)
    pdf_path = str(output_dir / "redundancy_analysis.pdf")
    png_path = str(output_dir / "redundancy_analysis.png")
    plot_redundancy_analysis(
        eigs_4=eigs_4,
        eigs_256=eigs_256,
        rank_4=rank_4,
        rank_256=rank_256,
        corr_4=corr_4,
        corr_256=corr_256,
        output_paths=[pdf_path, png_path],
        bpp_4=bpp_stats_4.get("mean_bpp"),
        bpp_256=bpp_stats_256.get("mean_bpp"),
    )

    print("\n[Done] All results saved to:", output_dir)


# ---------------------------------------------------------------------------
# 9. CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="实验三：统计冗余可压缩性验证 — 4ch vs 256ch VAE latent analysis",
    )
    parser.add_argument(
        "--config_4ch", type=str, required=True,
        help="Config YAML for 4-channel model (e.g., ../configs/stage1.yaml)",
    )
    parser.add_argument(
        "--config_256ch", type=str, required=True,
        help="Config YAML for 256-channel model (e.g., ../configs/stage2.yaml)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/experiment3",
        help="Directory to save results (default: ./results/experiment3)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override test dataset path (default: from base.yaml test_dataset)",
    )
    parser.add_argument(
        "--skip_bpp", action="store_true",
        help="Skip BPP measurement (only run PCA + correlation analysis)",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()