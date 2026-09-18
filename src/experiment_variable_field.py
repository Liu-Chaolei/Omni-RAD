"""
自注意力感受野散布实验 (Self-Attention Receptive Field Spread)

证明：低码率(高λ)下 UNet 自注意力在"四处张望"寻找上下文线索来脑补（全局/分散），
      高码率(低λ)下 UNet 自注意力在"专注局部"精确还原（局部/集中）。

做法：
  1. 加载训练好的 variable-rate StableCodec 模型
  2. 在 UNet Spatial Transformer 的 self-attention (attn1) 层注入自定义 Processor，
     捕获指定 Query 点的 Attention 权重行
  3. 对同一张图像，分别注入不同 λ 值
  4. 计算 Attention 散布指标：Entropy, Effective Radius, Top-k Concentration
  5. 可视化对比

期望结果：
  - 高λ (极低码率): Attention 权重分布广泛且分散（全局上下文聚合）
  - 低λ (高码率): Attention 权重高度集中在 Query 点邻域（局部精确还原）

用法 (从 repo 根目录运行):
    python src/experiment_variable_field.py \\
        --sd_path      /path/to/sd-turbo_256 \\
        --elic_path    /path/to/elic_official.pth \\
        --codec_path   /path/to/variable_rate_checkpoint.pth.tar \\
        --img_dir      /path/to/Kodak24/HR \\
        --out_dir      results/attn_field
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
import torch.nn.functional as F_torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from StableCodec_variable2 import StableCodec

# ---------------------------------------------------------------------------
# Global attention capture storage
# ---------------------------------------------------------------------------
# {layer_name: {query_token_index: attn_row_tensor[N]}}
ATTN_MAPS: dict[str, dict[int, torch.Tensor]] = {}

# ---------------------------------------------------------------------------
# Custom Attention Processor
# ---------------------------------------------------------------------------

class CaptureAttnProcessor:
    """Drop-in replacement for AttnProcessor2_0 that captures attention rows
    for specified query token indices while preserving normal forward."""

    def __init__(self, query_indices: list[int], layer_name: str):
        self.query_indices = query_indices
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_len, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # --- Capture attention rows for query indices ---
        if self.query_indices:
            scale_factor = head_dim ** -0.5
            captured = {}
            for qi in self.query_indices:
                if qi < seq_len:
                    q_row = query[:, :, qi:qi+1, :]
                    scores = torch.matmul(
                        q_row, key.transpose(-2, -1)) * scale_factor
                    attn_w = torch.softmax(scores, dim=-1)
                    captured[qi] = attn_w.mean(1).mean(0).squeeze(0).detach().cpu()
            ATTN_MAPS[self.layer_name] = captured

        # --- Normal SDP attention for output ---
        hidden_states = F_torch.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Install / restore attention processors
# ---------------------------------------------------------------------------

def install_capture_processors(
    unet: nn.Module, target_block: str, query_indices: list[int],
) -> dict[str, object]:
    originals = {}
    for name, module in unet.named_modules():
        if not name.startswith(target_block):
            continue
        if not name.endswith(".attn1"):
            continue
        originals[name] = module.processor
        module.set_processor(CaptureAttnProcessor(query_indices, name))
    return originals


def restore_processors(unet: nn.Module, originals: dict[str, object]) -> None:
    for name, module in unet.named_modules():
        if name in originals:
            module.set_processor(originals[name])


# ---------------------------------------------------------------------------
# Query point utilities
# ---------------------------------------------------------------------------

def auto_query_points(sh: int, sw: int) -> list[tuple[int, int]]:
    ch, cw = sh // 2, sw // 2
    qh, qw = sh // 4, sw // 4
    return [
        (ch, cw),
        (qh, qw),
        (qh, sw - qw - 1),
        (sh - qh - 1, qw),
        (sh - qh - 1, sw - qw - 1),
    ]


def parse_query_points(s: str) -> list[tuple[int, int]]:
    points = []
    for pair in s.split(";"):
        h, w = pair.strip().split(",")
        points.append((int(h), int(w)))
    return points


def image_to_token_hw(
    img_h: int, img_w: int, img_size: int, spatial_size: int,
) -> tuple[int, int]:
    scale = spatial_size / img_size
    return (
        min(int(img_h * scale), spatial_size - 1),
        min(int(img_w * scale), spatial_size - 1),
    )


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

def compute_entropy(attn_row: np.ndarray) -> float:
    p = attn_row.clip(1e-12, None)
    return float(-np.sum(p * np.log2(p)))


def compute_effective_radius(
    attn_row: np.ndarray, qh: int, qw: int, sh: int, sw: int,
) -> float:
    coords_h, coords_w = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    dist = np.sqrt(
        (coords_h.flatten() - qh) ** 2 + (coords_w.flatten() - qw) ** 2
    ).astype(np.float64)
    return float(np.sum(attn_row * dist))


def compute_topk_concentration(attn_row: np.ndarray, k: int = 16) -> float:
    return float(np.sort(attn_row)[-k:].sum())


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _collect_vrange(maps_per_lam, query_indices, sh, sw):
    vals = []
    for lam_maps in maps_per_lam.values():
        for qi in query_indices:
            if qi in lam_maps:
                vals.append(lam_maps[qi].max())
    return max(vals) if vals else 1.0


def plot_attn_heatmap_grid(
    img_np, attn_per_lam, query_points_hw, spatial_h, spatial_w,
    name, target_block, out_path,
):
    lambdas = sorted(attn_per_lam.keys())
    n_rows = len(query_points_hw)
    n_cols = len(lambdas) + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    qi_list = [qh * spatial_w + qw for qh, qw in query_points_hw]
    vmax = _collect_vrange(attn_per_lam, qi_list, spatial_h, spatial_w)
    img_h, img_w = img_np.shape[:2]
    im = None

    for ri, (qh, qw) in enumerate(query_points_hw):
        axes[ri, 0].imshow(img_np)
        mh = int(qh / spatial_h * img_h)
        mw = int(qw / spatial_w * img_w)
        axes[ri, 0].plot(mw, mh, "r+", markersize=15, markeredgewidth=3)
        axes[ri, 0].set_title(f"Query ({qh},{qw})", fontsize=10)
        axes[ri, 0].axis("off")

        ti = qh * spatial_w + qw
        for ci, lam in enumerate(lambdas):
            a2d = attn_per_lam[lam].get(ti, np.zeros((spatial_h, spatial_w)))
            im = axes[ri, ci + 1].imshow(
                a2d, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
            axes[ri, ci + 1].plot(qw, qh, "c+", markersize=10, markeredgewidth=2)
            axes[ri, ci + 1].set_title(f"λ={lam}", fontsize=10)
            axes[ri, ci + 1].axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Attention Weight")
    fig.suptitle(
        f"Self-Attention Receptive Field — {name}\n({target_block})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_attn_overlay(
    img_np, attn_per_lam, query_idx, query_hw,
    spatial_h, spatial_w, name, target_block, out_path, alpha=0.55,
):
    lambdas = sorted(attn_per_lam.keys())
    n_cols = len(lambdas)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    all_m = [attn_per_lam[l].get(query_idx, np.zeros((spatial_h, spatial_w)))
             for l in lambdas]
    vmax = max(m.max() for m in all_m) if all_m else 1.0
    qh, qw = query_hw
    img_h, img_w = img_np.shape[:2]
    mh, mw = int(qh / spatial_h * img_h), int(qw / spatial_w * img_w)
    im = None

    for j, lam in enumerate(lambdas):
        axes[j].imshow(img_np)
        a2d = attn_per_lam[lam].get(
            query_idx, np.zeros((spatial_h, spatial_w)))
        a_u8 = (a2d / max(vmax, 1e-12) * 255).astype(np.uint8)
        a_rs = np.array(Image.fromarray(a_u8).resize(
            (img_w, img_h), Image.BILINEAR)) / 255.0 * vmax
        im = axes[j].imshow(a_rs, cmap="hot", alpha=alpha, vmin=0, vmax=vmax)
        axes[j].plot(mw, mh, "c+", markersize=15, markeredgewidth=3)
        axes[j].set_title(f"λ={lam}", fontsize=13)
        axes[j].axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Attention Weight")
    fig.suptitle(
        f"Attention Overlay — {name} (query {query_hw})\n({target_block})",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_vs_lambda(
    lambdas, entropy_per_lam, n_images, target_block, out_path,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [np.mean(entropy_per_lam[l]) for l in lambdas]
    stds = [np.std(entropy_per_lam[l]) for l in lambdas]
    ax.errorbar(lambdas, means, yerr=stds, marker="o", linewidth=2,
                capsize=4, color="darkblue", markerfacecolor="gold")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)", fontsize=12)
    ax.set_ylabel("Attention Entropy (bits)", fontsize=12)
    ax.set_title(f"Self-Attention Entropy vs λ\n"
                 f"({n_images} images, {target_block})", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_radius_vs_lambda(
    lambdas, radius_per_lam, n_images, target_block, out_path,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [np.mean(radius_per_lam[l]) for l in lambdas]
    stds = [np.std(radius_per_lam[l]) for l in lambdas]
    ax.errorbar(lambdas, means, yerr=stds, marker="s", linewidth=2,
                capsize=4, color="darkred", markerfacecolor="cyan")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)", fontsize=12)
    ax.set_ylabel("Effective Radius (tokens)", fontsize=12)
    ax.set_title(f"Effective Receptive Field Radius vs λ\n"
                 f"({n_images} images, {target_block})", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multi_layer_comparison(
    lambdas_pair, layer_entropy, target_block, out_path,
):
    lam_lo, lam_hi = lambdas_pair
    layer_names = sorted(layer_entropy[lam_lo].keys())
    if not layer_names:
        return
    x_pos = np.arange(len(layer_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(layer_names) * 1.5), 5))
    ax.bar(x_pos - width / 2,
           [layer_entropy[lam_lo].get(n, 0) for n in layer_names],
           width, label=f"λ={lam_lo} (high bitrate)", color="steelblue")
    ax.bar(x_pos + width / 2,
           [layer_entropy[lam_hi].get(n, 0) for n in layer_names],
           width, label=f"λ={lam_hi} (low bitrate)", color="firebrick")
    ax.set_xticks(x_pos)
    short = [n.replace(target_block + ".", "") for n in layer_names]
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Attention Entropy (bits)", fontsize=11)
    ax.set_title(f"Per-Layer Attention Entropy ({target_block})", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Spatial resolution for target block
# ---------------------------------------------------------------------------

BLOCK_SPATIAL = {
    "up_blocks.1": 16,
    "up_blocks.2": 32,
    "up_blocks.3": 64,
}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    spatial = BLOCK_SPATIAL.get(args.target_block, 32)
    print(f"Target block: {args.target_block}  spatial: {spatial}x{spatial}")

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

    # --- Query points ---
    if args.query_points:
        img_pts = parse_query_points(args.query_points)
        qpts_hw = [image_to_token_hw(h, w, 512, spatial) for h, w in img_pts]
    else:
        qpts_hw = auto_query_points(spatial, spatial)
    query_indices = [qh * spatial + qw for qh, qw in qpts_hw]
    print(f"Query points (token h,w): {qpts_hw}")
    print(f"Query token indices: {query_indices}\n")

    # --- Install capture processors ---
    originals = install_capture_processors(
        model.unet, args.target_block, query_indices)
    print(f"Installed CaptureAttnProcessor on {len(originals)} attn1 layers:")
    for n in originals:
        print(f"  {n}")
    if not originals:
        print("WARNING: No attn1 layers found. Check target_block name.")
        return
    print()

    # --- Load images ---
    images, names = load_images(args.img_dir, args.n_images, device)
    N = images.shape[0]
    print(f"Loaded {N} images from {args.img_dir}\n")

    lambdas = sorted([float(x) for x in args.lambdas.split(",")])
    print(f"Testing λ values: {lambdas}\n")

    # --- Collect attention maps and metrics ---
    # Per-image, per-lambda: {img_idx: {lam: {token_idx: 2d_map}}}
    all_attn: dict[int, dict[float, dict[int, np.ndarray]]] = {}
    entropy_per_lam: dict[float, list[float]] = {l: [] for l in lambdas}
    radius_per_lam: dict[float, list[float]] = {l: [] for l in lambdas}
    topk_per_lam: dict[float, list[float]] = {l: [] for l in lambdas}

    for img_idx in range(N):
        img = images[img_idx:img_idx+1]
        B, C, H, W = img.shape
        all_attn[img_idx] = {}

        for lam in lambdas:
            ATTN_MAPS.clear()
            lmbda = torch.tensor([lam], dtype=torch.float32, device=device)
            _ = model(img, [1], H, W, lmbda=lmbda)

            # Aggregate across layers: average attention rows
            agg: dict[int, np.ndarray] = {}
            n_layers = 0
            for layer_name, layer_maps in ATTN_MAPS.items():
                n_layers += 1
                for qi, row_t in layer_maps.items():
                    row_np = row_t.numpy().astype(np.float64)
                    row_np = row_np / max(row_np.sum(), 1e-12)
                    if qi not in agg:
                        agg[qi] = np.zeros_like(row_np)
                    agg[qi] = agg[qi] + row_np
            for qi in agg:
                agg[qi] = agg[qi] / max(n_layers, 1)
                agg[qi] = agg[qi] / max(agg[qi].sum(), 1e-12)

            # Reshape to 2D
            agg_2d: dict[int, np.ndarray] = {}
            for qi, row in agg.items():
                agg_2d[qi] = row.reshape(spatial, spatial)
            all_attn[img_idx][lam] = agg_2d

            # Compute metrics (average over query points)
            ents, rads, topks = [], [], []
            for qi_idx, (qh, qw) in enumerate(qpts_hw):
                ti = qh * spatial + qw
                if ti in agg:
                    ents.append(compute_entropy(agg[ti]))
                    rads.append(compute_effective_radius(
                        agg[ti], qh, qw, spatial, spatial))
                    topks.append(compute_topk_concentration(agg[ti]))

            avg_ent = float(np.mean(ents)) if ents else 0.0
            avg_rad = float(np.mean(rads)) if rads else 0.0
            avg_topk = float(np.mean(topks)) if topks else 0.0
            entropy_per_lam[lam].append(avg_ent)
            radius_per_lam[lam].append(avg_rad)
            topk_per_lam[lam].append(avg_topk)

            print(f"  [{names[img_idx]}] λ={lam:6.2f}  "
                  f"entropy={avg_ent:.3f}  radius={avg_rad:.3f}  "
                  f"topk16={avg_topk:.4f}")
        print()

    # --- Visualizations ---
    print("Generating visualizations ...")

    for img_idx in range(N):
        img_np = (
            (images[img_idx].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        ).clip(0, 255).astype(np.uint8)

        plot_attn_heatmap_grid(
            img_np, all_attn[img_idx], qpts_hw, spatial, spatial,
            names[img_idx], args.target_block,
            os.path.join(args.out_dir, f"attn_grid_{names[img_idx]}.png"))

        # Overlay for center query point
        center_qi = qpts_hw[0][0] * spatial + qpts_hw[0][1]
        plot_attn_overlay(
            img_np, all_attn[img_idx], center_qi, qpts_hw[0],
            spatial, spatial, names[img_idx], args.target_block,
            os.path.join(args.out_dir, f"attn_overlay_{names[img_idx]}.png"))
        print(f"  Saved plots for {names[img_idx]}")

    # Entropy & radius vs λ
    plot_entropy_vs_lambda(
        lambdas, entropy_per_lam, N, args.target_block,
        os.path.join(args.out_dir, "entropy_vs_lambda.png"))
    plot_radius_vs_lambda(
        lambdas, radius_per_lam, N, args.target_block,
        os.path.join(args.out_dir, "radius_vs_lambda.png"))

    # Multi-layer comparison (re-run extreme λ pair on first image)
    layer_entropy: dict[float, dict[str, float]] = {}
    for lam in [lambdas[0], lambdas[-1]]:
        ATTN_MAPS.clear()
        lmbda = torch.tensor([lam], dtype=torch.float32, device=device)
        _ = model(images[0:1], [1], images.shape[2], images.shape[3],
                  lmbda=lmbda)
        layer_entropy[lam] = {}
        for layer_name, layer_maps in ATTN_MAPS.items():
            ents = []
            for qi, row_t in layer_maps.items():
                row_np = row_t.numpy().astype(np.float64)
                row_np = row_np / max(row_np.sum(), 1e-12)
                ents.append(compute_entropy(row_np))
            layer_entropy[lam][layer_name] = float(np.mean(ents)) if ents else 0.0

    plot_multi_layer_comparison(
        (lambdas[0], lambdas[-1]), layer_entropy, args.target_block,
        os.path.join(args.out_dir, "multi_layer_entropy.png"))

    # --- Numeric results ---
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write("Self-Attention Receptive Field Spread — Numeric Results\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Target block:  {args.target_block}\n")
        f.write(f"Spatial size:  {spatial}x{spatial}\n")
        f.write(f"Query points:  {qpts_hw}\n")
        f.write(f"Images:        {N}\n\n")

        f.write("Per-λ metrics (averaged over query points and images):\n")
        f.write(f"{'λ':>8s}  {'Entropy':>10s}  {'Eff.Radius':>12s}  "
                f"{'Top16 Conc.':>12s}\n")
        f.write("-" * 50 + "\n")
        for lam in lambdas:
            me = np.mean(entropy_per_lam[lam])
            mr = np.mean(radius_per_lam[lam])
            mt = np.mean(topk_per_lam[lam])
            f.write(f"{lam:8.2f}  {me:10.4f}  {mr:12.4f}  {mt:12.4f}\n")

        e_lo = np.mean(entropy_per_lam[lambdas[0]])
        e_hi = np.mean(entropy_per_lam[lambdas[-1]])
        r_lo = np.mean(radius_per_lam[lambdas[0]])
        r_hi = np.mean(radius_per_lam[lambdas[-1]])
        f.write(f"\nEntropy ratio (λ_high/λ_low): "
                f"{e_hi / max(e_lo, 1e-12):.3f}x\n")
        f.write(f"Radius ratio  (λ_high/λ_low): "
                f"{r_hi / max(r_lo, 1e-12):.3f}x\n")

        f.write("\nPer-layer entropy (first image, extreme λ pair):\n")
        for ln in sorted(layer_entropy.get(lambdas[0], {}).keys()):
            el = layer_entropy[lambdas[0]].get(ln, 0)
            eh = layer_entropy[lambdas[-1]].get(ln, 0)
            short = ln.replace(args.target_block + ".", "")
            f.write(f"  {short:40s}  λ={lambdas[0]:5.1f}: {el:.4f}  "
                    f"λ={lambdas[-1]:5.1f}: {eh:.4f}\n")

    # --- Restore processors ---
    restore_processors(model.unet, originals)

    print(f"\nAll results saved to {args.out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-Attention Receptive Field Spread: prove attention "
                    "shifts from global (low bitrate) to local (high bitrate).",
    )
    parser.add_argument("--sd_path", required=True,
                        help="Path to SD-Turbo (256ch variant)")
    parser.add_argument("--elic_path", required=True,
                        help="Path to ELIC pretrained weights")
    parser.add_argument("--codec_path", required=True,
                        help="Path to variable-rate StableCodec checkpoint")
    parser.add_argument("--img_dir", required=True,
                        help="Directory of test images (e.g. Kodak24/HR)")
    parser.add_argument("--out_dir", default="results/attn_field",
                        help="Output directory for plots and results")
    parser.add_argument("--n_images", type=int, default=5,
                        help="Number of images to process")
    parser.add_argument("--lambdas", default="0.5,2,8,32",
                        help="Comma-separated λ values to test")
    parser.add_argument("--lambda_min", type=float, default=0.1,
                        help="Lambda min for model (match training config)")
    parser.add_argument("--lambda_max", type=float, default=128.0,
                        help="Lambda max for model (match training config)")
    parser.add_argument("--target_block", default="up_blocks.2",
                        help="UNet block to analyze (up_blocks.1/2/3)")
    parser.add_argument("--query_points", default=None,
                        help="Manual query points in image coords: "
                             "'h1,w1;h2,w2;...' (512x512 space). "
                             "If omitted, 5 auto points are used.")
    args = parser.parse_args()

    run_experiment(args)
