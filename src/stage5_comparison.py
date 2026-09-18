"""stage5_comparison.py — Stage 5: Unified comparison of all timestep strategies.

Runs the T-aug model with different timestep strategies on a test set and
produces a comprehensive comparison table with per-strategy metrics.

Strategies evaluated:
  1. Fixed-999: always use T=999
  2. Fixed-best-global: use the single best T from Stage 2 oracle sweep
  3. SNR-T: original DynamicTimestepModule
  4. PolicyNet-E2E: end-to-end trained PolicyNet (Stage 3)
  5. PolicyNet-Oracle: Oracle-supervised PolicyNet (Stage 4)
  6. Oracle-T-aug: per-image oracle from Stage 2 (upper bound)

Usage:
    python src/stage5_comparison.py \
        --base_config ./configs/base.yaml \
        --test_config ./configs/test.yaml \
        --codec_path /path/to/t_aug_checkpoint.pth.tar \
        --policynet_e2e_path ./results/stage3/checkpoint.pth.tar \
        --policynet_oracle_path ./results/stage4_policynet_all/policynet_oracle.pth \
        --oracle_csv ./results/stage2_oracle_sweep/oracle_timestep_summary.csv \
        --out_dir ./results/stage5_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: pip install pandas")

try:
    import pyiqa
except ImportError:
    raise ImportError("pyiqa is required: pip install pyiqa")


# =========================================================================
# Utilities
# =========================================================================

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def preprocess_image(path: str, device: torch.device) -> Tuple[torch.Tensor, int, int]:
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    tensor = tensor * 2.0 - 1.0
    _, _, h, w = tensor.shape
    stride_h, stride_w = 64, 64
    pad_h = (math.ceil(h / stride_h) * stride_h) - h
    pad_w = (math.ceil(w / stride_w) * stride_w) - w
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def to_iqa(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# =========================================================================
# Strategy definitions
# =========================================================================

class TimestepStrategy:
    """Base class for timestep strategies."""
    name: str = "base"

    def get_timestep(self, ctx: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class FixedStrategy(TimestepStrategy):
    def __init__(self, t_value: int = 999):
        self.t_value = t_value
        self.name = f"Fixed-{t_value}"

    def get_timestep(self, ctx: Dict[str, Any]) -> torch.Tensor:
        B = ctx["B"]
        device = ctx["device"]
        return torch.full((B,), float(self.t_value), device=device)


class SNRTStrategy(TimestepStrategy):
    name = "SNR-T"

    def get_timestep(self, ctx: Dict[str, Any]) -> torch.Tensor:
        return ctx["T_snr"]


class PolicyNetStrategy(TimestepStrategy):
    def __init__(self, name: str, policynet_ckpt: dict, device: torch.device):
        self.name = name
        self._build_model(policynet_ckpt, device)

    def _build_model(self, ckpt: dict, device: torch.device):
        from timestep_policy_net import TimestepPolicyNet, POLICY_FEATURE_DIM

        in_dim = ckpt.get("in_dim", POLICY_FEATURE_DIM)
        hidden = ckpt.get("hidden", 128)
        t_min = ckpt.get("t_min", 870.0)
        t_max = ckpt.get("t_max", 999.0)

        self.model = TimestepPolicyNet(
            in_dim=in_dim, hidden=hidden, t_min=t_min, t_max=t_max, delta_max=30.0
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.feature_cols = ckpt.get("feature_cols", None)
        self.norm_stats = ckpt.get("norm_stats", None)

    def get_timestep(self, ctx: Dict[str, Any]) -> torch.Tensor:
        features = ctx["features"]
        T_snr = ctx.get("T_snr", torch.tensor(999.0, device=features.device))
        if T_snr.dim() == 0:
            T_snr = T_snr.expand(features.shape[0])
        if self.norm_stats and self.feature_cols:
            mean = torch.tensor(
                [self.norm_stats[c][0] for c in self.feature_cols],
                device=features.device, dtype=features.dtype
            )
            std = torch.tensor(
                [self.norm_stats[c][1] for c in self.feature_cols],
                device=features.device, dtype=features.dtype
            )
            features = (features - mean.unsqueeze(0)) / std.unsqueeze(0)
        return self.model(features, T_snr)


class OracleStrategy(TimestepStrategy):
    name = "Oracle-T-aug"

    def __init__(self, oracle_df: pd.DataFrame):
        self._oracle_map = {}
        for _, row in oracle_df.iterrows():
            key = (str(row["image_id"]), float(row["lambda"]))
            t_col = "T_oracle_aug" if "T_oracle_aug" in oracle_df.columns else "T_oracle_main"
            self._oracle_map[key] = float(row[t_col])

    def get_timestep(self, ctx: Dict[str, Any]) -> torch.Tensor:
        image_id = ctx["image_id"]
        lmbda_val = ctx["lmbda_val"]
        key = (image_id, lmbda_val)
        t_val = self._oracle_map.get(key, 999.0)
        B = ctx["B"]
        return torch.full((B,), t_val, device=ctx["device"])


# =========================================================================
# Model loading and codec forward
# =========================================================================

def load_model(test_config: dict, codec_path: str, device: torch.device):
    """Load StableCodec model (uses the step variant with DynamicTimestepModule)."""
    from StableCodec_variable2_step import StableCodec
    from latent_codec_variable2_step import LAMBDA_MIN, LAMBDA_MAX

    model_cfg = test_config["model"]
    net = StableCodec(sd_path=model_cfg["sd_path"], config=model_cfg).to(device)
    net.eval()

    from diffusers.utils.import_utils import is_xformers_available
    if test_config.get("enable_xformers_memory_efficient_attention", False):
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()

    sd = torch.load(codec_path, map_location="cpu")
    _load_partial(net.codec, sd.get("state_dict_codec", {}), "codec")
    _load_partial(net.unet, sd.get("state_dict_unet", {}), "unet")
    _load_partial(net.vae, sd.get("state_dict_vae", {}), "vae")
    if "state_dict_lora_proj" in sd:
        net.unet_lora_proj.load_state_dict(sd["state_dict_lora_proj"])

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _load_partial(module, state_dict, name):
    own = module.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    module.load_state_dict(filtered, strict=False)


# =========================================================================
# Core evaluation: encode once, decode with different strategies
# =========================================================================

def codec_forward_capture(net, img_padded, ori_h, ori_w, lmbda_tensor):
    """Run encoder once and capture intermediate state for multi-strategy decode."""
    device = img_padded.device
    B = img_padded.shape[0]

    latent2 = net.aux_codec((img_padded + 1.0) / 2.0).detach()
    lq_latent = net.vae.encode(img_padded).latent_dist.mode() * net.vae.config.scaling_factor

    # Run codec forward to get lq_latent_hat, res1, scales_all, T_snr
    # We monkey-patch to capture scales_all
    scales_capture = [None]
    orig_dt_forward = net.codec.dynamic_timestep.forward

    def _capture_scales(scales_all):
        scales_capture[0] = scales_all.detach()
        return orig_dt_forward(scales_all)

    net.codec.dynamic_timestep.forward = _capture_scales
    lq_latent_hat, rate_out, res1, T_snr = net.codec(
        lq_latent, latent2, ori_h, ori_w, lmbda_tensor
    )
    net.codec.dynamic_timestep.forward = orig_dt_forward

    # Compute film_embed and lora_scale
    film_embed = net.codec.film_embed(lmbda_tensor)
    delta_s = net.unet_lora_proj(film_embed)
    lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)

    # Build features for PolicyNet strategies
    from timestep_policy_net import build_timestep_features, SNRTimestepFeature
    scales_all = scales_capture[0]

    # Recompute snr_compress from scales_all
    sigma_quant = 1.0 / 12.0
    signal_var = scales_all.mean(dim=[1, 2, 3]) ** 2
    snr_compress = signal_var / sigma_quant

    bpp_val = rate_out.quantized_total_bpp.detach()
    sample = lq_latent_hat[:, :256] if lq_latent_hat.shape[1] >= 256 else lq_latent_hat

    features = build_timestep_features(
        lmbda=lmbda_tensor, bpp=bpp_val,
        scales_all=scales_all, y_hat=lq_latent_hat.detach(),
        sample=sample.detach(), res1=res1.detach(),
        T_snr=T_snr.detach(), snr_compress=snr_compress,
    )

    pos_caption_enc = torch.cat(
        [net.pos_caption_enc for _ in range(B)], dim=0
    ).to(device)

    return {
        "lq_latent_hat": lq_latent_hat,
        "res1": res1,
        "T_snr": T_snr,
        "scales_all": scales_all,
        "rate_out": rate_out,
        "film_embed": film_embed,
        "lora_scale": lora_scale,
        "pos_caption_enc": pos_caption_enc,
        "features": features,
        "bpp": float(bpp_val.mean().item()),
    }


def decode_with_timestep(net, ctx: dict, timestep: torch.Tensor) -> torch.Tensor:
    """Decode at a given timestep using cached encoder output."""
    device = ctx["lq_latent_hat"].device
    B = ctx["lq_latent_hat"].shape[0]
    t_long = timestep.long().clamp(0, 999).to(device)

    model_pred = net.unet(
        ctx["lq_latent_hat"], t_long,
        encoder_hidden_states=ctx["pos_caption_enc"]
    ).sample * ctx["lora_scale"]

    # DDPM step (integer version for evaluation)
    ac = net.sched.alphas_cumprod
    alpha_prod_t = ac[t_long].view(-1, 1, 1, 1)
    sqrt_alpha_t = alpha_prod_t.sqrt()
    sqrt_one_minus = (1.0 - alpha_prod_t).sqrt()
    sample = ctx["lq_latent_hat"][:, :256]
    x0_pred = (sample - sqrt_one_minus * model_pred) / sqrt_alpha_t

    x_denoised = x0_pred + ctx["res1"]
    output = net.vae.decode(x_denoised / net.vae.config.scaling_factor).sample.clamp(-1, 1)
    return output


# =========================================================================
# Per-image evaluation
# =========================================================================

def evaluate_one_image(
    net,
    img_padded: torch.Tensor,
    gt: torch.Tensor,
    ori_h: int,
    ori_w: int,
    lmbda_val: float,
    image_id: str,
    strategies: List[TimestepStrategy],
    metrics_bundle: dict,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Evaluate one image with all strategies. Returns list of result dicts."""
    B = img_padded.shape[0]
    lmbda_tensor = torch.full((B,), lmbda_val, device=device)

    with torch.no_grad():
        ctx = codec_forward_capture(net, img_padded, ori_h, ori_w, lmbda_tensor)

    results = []
    for strategy in strategies:
        strategy_ctx = {
            "B": B,
            "device": device,
            "T_snr": ctx["T_snr"],
            "features": ctx["features"],
            "image_id": image_id,
            "lmbda_val": lmbda_val,
        }

        with torch.no_grad():
            timestep = strategy.get_timestep(strategy_ctx)
            output = decode_with_timestep(net, ctx, timestep)

        output_crop = output[..., :ori_h, :ori_w]
        gt_crop = gt[..., :ori_h, :ori_w]

        rec_iqa = to_iqa(output_crop)
        gt_iqa = to_iqa(gt_crop)

        psnr_v = float(metrics_bundle["psnr"](rec_iqa, gt_iqa).item())
        lpips_v = float(metrics_bundle["lpips"](rec_iqa, gt_iqa).item())
        dists_v = float(metrics_bundle["dists"](rec_iqa, gt_iqa).item())

        results.append({
            "image_id": image_id,
            "lambda": lmbda_val,
            "strategy": strategy.name,
            "T_used": float(timestep.mean().item()),
            "bpp": ctx["bpp"],
            "PSNR": psnr_v,
            "LPIPS": lpips_v,
            "DISTS": dists_v,
        })

    return results


# =========================================================================
# Report generation
# =========================================================================

def generate_report(all_results: List[Dict[str, Any]], out_dir: str) -> str:
    """Generate comparison report from all per-image results."""
    df = pd.DataFrame(all_results)

    # Strategy-level averages
    summary = df.groupby("strategy").agg({
        "PSNR": "mean",
        "LPIPS": "mean",
        "DISTS": "mean",
        "bpp": "mean",
        "T_used": ["mean", "std"],
    }).round(4)
    summary.columns = ["PSNR", "LPIPS", "DISTS", "bpp", "T_mean", "T_std"]
    summary = summary.sort_values("PSNR", ascending=False)

    # Per-lambda breakdown
    per_lambda = df.groupby(["strategy", "lambda"]).agg({
        "PSNR": "mean", "LPIPS": "mean", "bpp": "mean", "T_used": "mean",
    }).round(4)

    # Write report
    lines = []
    lines.append("=" * 78)
    lines.append("STAGE 5: TIMESTEP STRATEGY COMPARISON REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("-" * 78)
    lines.append("A. OVERALL COMPARISON (averaged over all images and lambdas)")
    lines.append("-" * 78)
    lines.append(summary.to_string())
    lines.append("")

    # Delta vs Fixed-999
    if "Fixed-999" in summary.index:
        base_psnr = summary.loc["Fixed-999", "PSNR"]
        base_lpips = summary.loc["Fixed-999", "LPIPS"]
        lines.append("-" * 78)
        lines.append("B. GAIN vs Fixed-999")
        lines.append("-" * 78)
        for strat in summary.index:
            dp = summary.loc[strat, "PSNR"] - base_psnr
            dl = base_lpips - summary.loc[strat, "LPIPS"]
            lines.append(f"  {strat:35s} PSNR: {dp:+.4f} dB  LPIPS: {dl:+.4f}")
        lines.append("")

    # Per-lambda table
    lines.append("-" * 78)
    lines.append("C. PER-LAMBDA BREAKDOWN")
    lines.append("-" * 78)
    lines.append(per_lambda.to_string())
    lines.append("")

    # T_pred distribution per strategy
    lines.append("-" * 78)
    lines.append("D. TIMESTEP DISTRIBUTION PER STRATEGY")
    lines.append("-" * 78)
    for strat in df["strategy"].unique():
        t_vals = df[df["strategy"] == strat]["T_used"].values
        lines.append(f"  {strat:35s} mean={t_vals.mean():.1f} std={t_vals.std():.1f} "
                     f"min={t_vals.min():.0f} max={t_vals.max():.0f}")
    lines.append("")

    # Conclusion
    lines.append("-" * 78)
    lines.append("E. CONCLUSIONS")
    lines.append("-" * 78)
    best_strat = summary.index[0]
    lines.append(f"  Best strategy by PSNR: {best_strat} ({summary.loc[best_strat, 'PSNR']:.4f} dB)")
    best_lpips_strat = summary.sort_values("LPIPS").index[0]
    lines.append(f"  Best strategy by LPIPS: {best_lpips_strat} ({summary.loc[best_lpips_strat, 'LPIPS']:.4f})")

    # E2E vs Oracle comparison
    e2e_strats = [s for s in summary.index if "E2E" in s]
    oracle_strats = [s for s in summary.index if "Oracle" in s and "Oracle-T" not in s]
    if e2e_strats and oracle_strats:
        best_e2e = summary.loc[e2e_strats, "PSNR"].max()
        best_oracle = summary.loc[oracle_strats, "PSNR"].max()
        if best_e2e >= best_oracle - 0.05:
            lines.append("  -> E2E PolicyNet matches or exceeds Oracle-supervised; prefer simpler E2E route.")
        else:
            lines.append(f"  -> Oracle-supervised gains {best_oracle - best_e2e:.3f} dB over E2E; worth keeping.")
    lines.append("")
    lines.append("=" * 78)

    report_text = "\n".join(lines)
    report_path = os.path.join(out_dir, "stage5_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save raw CSV
    csv_path = os.path.join(out_dir, "stage5_results.csv")
    df.to_csv(csv_path, index=False)

    # Save summary CSV
    summary_path = os.path.join(out_dir, "stage5_summary.csv")
    summary.to_csv(summary_path)

    return report_path


def make_plots(all_results: List[Dict[str, Any]], out_dir: str) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(all_results)

    # Plot 1: RD curves (PSNR vs bpp) per strategy
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat].groupby("lambda").agg({"bpp": "mean", "PSNR": "mean"})
        sub = sub.sort_values("bpp")
        ax.plot(sub["bpp"], sub["PSNR"], "o-", markersize=5, label=strat)
    ax.set_xlabel("BPP")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-Distortion Comparison (PSNR)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "rd_curve_psnr.png"))
    plt.close(fig)

    # Plot 2: RD curves (LPIPS vs bpp)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat].groupby("lambda").agg({"bpp": "mean", "LPIPS": "mean"})
        sub = sub.sort_values("bpp")
        ax.plot(sub["bpp"], sub["LPIPS"], "o-", markersize=5, label=strat)
    ax.set_xlabel("BPP")
    ax.set_ylabel("LPIPS (lower is better)")
    ax.set_title("Rate-Distortion Comparison (LPIPS)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "rd_curve_lpips.png"))
    plt.close(fig)

    # Plot 3: Bar chart of mean PSNR per strategy
    summary = df.groupby("strategy")["PSNR"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.barh(range(len(summary)), summary.values, align="center")
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean PSNR (dB)")
    ax.set_title("Strategy Comparison: Mean PSNR")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "strategy_psnr_bar.png"))
    plt.close(fig)

    # Plot 4: T_used distribution boxplot
    strategies = df["strategy"].unique()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    data = [df[df["strategy"] == s]["T_used"].values for s in strategies]
    ax.boxplot(data, labels=strategies, showfliers=False, vert=True)
    ax.set_ylabel("T_used")
    ax.set_title("Timestep Distribution per Strategy")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "timestep_boxplot.png"))
    plt.close(fig)

    print(f"  Plots saved to: {plot_dir}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Unified comparison of all timestep strategies."
    )
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--test_config", type=str, required=True)
    parser.add_argument("--codec_path", type=str, required=True,
                        help="T-aug model checkpoint")
    parser.add_argument("--policynet_e2e_path", type=str, default=None,
                        help="Stage 3 E2E PolicyNet checkpoint (full model .pth.tar)")
    parser.add_argument("--policynet_oracle_path", type=str, default=None,
                        help="Stage 4 Oracle PolicyNet checkpoint (.pth)")
    parser.add_argument("--oracle_csv", type=str, default=None,
                        help="Stage 2 oracle_timestep_summary.csv for Oracle-T upper bound")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_lambdas", type=int, default=6)
    parser.add_argument("--lambda_min", type=float, default=0.2)
    parser.add_argument("--lambda_max", type=float, default=128.0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load configs
    base_config = load_yaml(args.base_config)
    test_config = load_yaml(args.test_config)
    train_config = {**base_config, **test_config}

    # Load model
    print("Loading model...")
    net = load_model(train_config, args.codec_path, device)

    # Build metrics
    print("Building metrics...")
    metrics_bundle = {
        "psnr": pyiqa.create_metric("psnr", device=device, as_loss=False),
        "lpips": pyiqa.create_metric("lpips", device=device, as_loss=False),
        "dists": pyiqa.create_metric("dists", device=device, as_loss=False),
    }

    # Build strategies
    print("Building strategies...")
    strategies: List[TimestepStrategy] = [
        FixedStrategy(999),
        SNRTStrategy(),
    ]

    # Find best global T from oracle CSV if available
    if args.oracle_csv and os.path.isfile(args.oracle_csv):
        oracle_df = pd.read_csv(args.oracle_csv)
        # Oracle upper bound strategy
        strategies.append(OracleStrategy(oracle_df))
        # Fixed-best-global: find the T with best average across all samples
        # (requires sweep CSV, not just summary; use T_oracle mode as proxy)
        t_col = "T_oracle_aug" if "T_oracle_aug" in oracle_df.columns else "T_oracle_main"
        if t_col in oracle_df.columns:
            from collections import Counter
            t_counts = Counter(oracle_df[t_col].astype(int).tolist())
            best_global_t = t_counts.most_common(1)[0][0]
            strategies.insert(1, FixedStrategy(best_global_t))
            print(f"  Fixed-best-global: T={best_global_t}")
    else:
        oracle_df = None

    # PolicyNet-E2E (loaded from full model checkpoint's codec state_dict)
    if args.policynet_e2e_path and os.path.isfile(args.policynet_e2e_path):
        print(f"  Loading PolicyNet-E2E from: {args.policynet_e2e_path}")
        e2e_sd = torch.load(args.policynet_e2e_path, map_location="cpu")
        # Extract PolicyNet weights from codec state_dict
        codec_sd = e2e_sd.get("state_dict_codec", {})
        policy_keys = {k: v for k, v in codec_sd.items() if "timestep_policy" in k}
        if policy_keys:
            from timestep_policy_net import TimestepPolicyNet, POLICY_FEATURE_DIM
            policy_sd = {k.replace("timestep_policy.", ""): v for k, v in policy_keys.items()}
            e2e_ckpt = {
                "state_dict": policy_sd,
                "in_dim": POLICY_FEATURE_DIM,
                "hidden": 128,
                "t_min": 870.0,
                "t_max": 999.0,
            }
            strategies.append(PolicyNetStrategy("PolicyNet-E2E", e2e_ckpt, device))
        else:
            print("  [WARNING] No timestep_policy keys found in E2E checkpoint")

    # PolicyNet-Oracle (standalone checkpoint from Stage 4)
    if args.policynet_oracle_path and os.path.isfile(args.policynet_oracle_path):
        print(f"  Loading PolicyNet-Oracle from: {args.policynet_oracle_path}")
        oracle_ckpt = torch.load(args.policynet_oracle_path, map_location="cpu")
        strategies.append(PolicyNetStrategy("PolicyNet-Oracle", oracle_ckpt, device))

    print(f"  Strategies: {[s.name for s in strategies]}")

    # Collect test images
    test_dir = base_config.get("test_dataset")
    if not test_dir:
        raise ValueError("base_config['test_dataset'] is required")
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in Path(test_dir).iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
    if args.max_images:
        images = images[:args.max_images]
    print(f"  Test images: {len(images)}")

    # Lambda grid
    if args.num_lambdas <= 1:
        lambda_grid = [args.lambda_min]
    else:
        lambda_grid = np.geomspace(args.lambda_min, args.lambda_max, args.num_lambdas).tolist()
    print(f"  Lambda grid: {[f'{v:.2f}' for v in lambda_grid]}")

    # Run evaluation
    print("\nRunning evaluation...")
    all_results: List[Dict[str, Any]] = []

    for img_idx, img_path in enumerate(images):
        image_id = img_path.stem
        img_padded, ori_h, ori_w = preprocess_image(str(img_path), device)
        gt = img_padded  # ground truth is the padded input (crop at eval)

        for lmbda_val in lambda_grid:
            results = evaluate_one_image(
                net, img_padded, gt, ori_h, ori_w,
                lmbda_val, image_id, strategies, metrics_bundle, device,
            )
            all_results.extend(results)

        if (img_idx + 1) % 5 == 0 or img_idx == 0:
            print(f"  [{img_idx+1}/{len(images)}] {image_id}")

    # Generate report
    print("\nGenerating report...")
    report_path = generate_report(all_results, args.out_dir)
    print(f"  Report: {report_path}")

    if not args.no_plots:
        print("Generating plots...")
        make_plots(all_results, args.out_dir)

    print("\nStage 5 comparison complete.")
    print(f"  Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
