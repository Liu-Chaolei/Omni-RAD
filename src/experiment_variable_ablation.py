"""
experiment_variable_ablation.py — Ablation study for λ-FiLM injection points.

Tests 10 meaningful variants (M0–M9) plus εSD fine-grained sub-ablation
across the 4 injection points in StableCodec variable-rate conditioning:
  [1] ga-FiLM  (AnalysisTransform)
  [2] gs-FiLM  (SynthesisTransform)
  [3] εSD-LoRA (UNet model_pred scaling)
  [4] DAux-FiLM (AuxDecoder)

Design doc: experiment_variable_ablation.txt

Usage (from repo root):
    python src/experiment_variable_ablation.py \
        --sd_path      /path/to/sd-turbo_256 \
        --elic_path    /path/to/elic_official.pth \
        --codec_path   /path/to/variable_rate_checkpoint.pth.tar \
        --img_dir      /path/to/Kodak24/HR \
        --out_dir      results/ablation
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
# Ablation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationConfig:
    use_ga_film: bool = True
    use_gs_film: bool = True
    use_daux_film: bool = True
    esd_lora_mode: str = "adaptive"  # "none" | "fixed" | "random" | "adaptive"

    @property
    def label(self) -> str:
        parts = []
        if self.use_ga_film:   parts.append("ga")
        if self.use_gs_film:   parts.append("gs")
        if self.use_daux_film: parts.append("DAux")
        if self.esd_lora_mode != "none":
            parts.append(f"εSD({self.esd_lora_mode})")
        return "+".join(parts) if parts else "baseline"

    @property
    def short_desc(self) -> str:
        flags = [self.use_ga_film, self.use_gs_film, self.use_daux_film]
        flag_str = "".join("Y" if f else "N" for f in flags)
        return f"[{flag_str}|{self.esd_lora_mode[0]}]"


VARIANTS: Dict[str, AblationConfig] = {
    "M0": AblationConfig(False, False, False, "none"),
    "M1": AblationConfig(True,  False, False, "none"),
    "M2": AblationConfig(False, True,  False, "none"),
    "M3": AblationConfig(False, False, True,  "none"),
    "M4": AblationConfig(False, False, False, "adaptive"),
    "M5": AblationConfig(True,  True,  False, "none"),
    "M6": AblationConfig(False, False, True,  "adaptive"),
    "M7": AblationConfig(False, True,  True,  "adaptive"),
    "M8": AblationConfig(True,  False, True,  "adaptive"),
    "M9": AblationConfig(True,  True,  True,  "adaptive"),
    # εSD fine-grained
    "M4_fixed":  AblationConfig(False, False, False, "fixed"),
    "M4_random": AblationConfig(False, False, False, "random"),
}

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "M0":  "纯基线，四点均无条件化",
    "M1":  "仅 ga-FiLM（编码分析变换）",
    "M2":  "仅 gs-FiLM（解码综合变换）",
    "M3":  "仅 DAux-FiLM（辅助解码器）",
    "M4":  "仅 εSD-LoRA（自适应）",
    "M5":  "完整信息通路 (ga+gs)",
    "M6":  "完整生成通路 (DAux+εSD)",
    "M7":  "缺编码端 ga，其余完整",
    "M8":  "缺解码端 gs，其余完整",
    "M9":  "完整 λ-FiLM",
    "M4_fixed":  "εSD 固定 alpha=1",
    "M4_random": "εSD 随机 alpha",
}

NEEDS_RETRAINING = {"M0", "M5", "M6"}


# ---------------------------------------------------------------------------
# εSD replacement modules
# ---------------------------------------------------------------------------

class _ConstantProj(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("val", torch.tensor([[value]], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.val.expand(x.shape[0], 1)


class _RandomProj(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand(x.shape[0], 1, device=x.device) * 2 - 1


def _identity_film(h: torch.Tensor, film_embed: torch.Tensor) -> torch.Tensor:
    return h


# ---------------------------------------------------------------------------
# Apply / restore ablation
# ---------------------------------------------------------------------------

def apply_ablation(
    model: nn.Module,
    config: AblationConfig,
    device: torch.device,
) -> dict:
    saved = {}
    codec = model.codec

    if not config.use_ga_film:
        ga = codec.g_a
        saved["ga_films"] = []
        for name in ("film1", "film2", "film3"):
            layer = getattr(ga, name)
            saved["ga_films"].append((name, layer.forward))
            layer.forward = _identity_film

    if not config.use_gs_film:
        gs = codec.g_s
        saved["gs_films"] = []
        for name in ("film1", "film2", "film3"):
            layer = getattr(gs, name)
            saved["gs_films"].append((name, layer.forward))
            layer.forward = _identity_film

    if not config.use_daux_film:
        aux = codec.aux
        saved["daux_films"] = []
        for name in ("film1", "film2", "film3"):
            layer = getattr(aux, name)
            saved["daux_films"].append((name, layer.forward))
            layer.forward = _identity_film

    if config.esd_lora_mode != "adaptive":
        saved["unet_lora_proj"] = model.unet_lora_proj
        if config.esd_lora_mode == "none":
            model.unet_lora_proj = _ConstantProj(0.0).to(device)
        elif config.esd_lora_mode == "fixed":
            lmbda_mid = math.sqrt(codec.lambda_min * codec.lambda_max)
            with torch.no_grad():
                film_mid = codec.film_embed(
                    torch.tensor([lmbda_mid], device=device)
                )
                fixed_val = saved["unet_lora_proj"](film_mid).item()
            model.unet_lora_proj = _ConstantProj(fixed_val).to(device)
        elif config.esd_lora_mode == "random":
            model.unet_lora_proj = _RandomProj().to(device)

    return saved


def restore_ablation(model: nn.Module, saved: dict) -> None:
    codec = model.codec

    if "ga_films" in saved:
        ga = codec.g_a
        for name, orig_fwd in saved["ga_films"]:
            getattr(ga, name).forward = orig_fwd

    if "gs_films" in saved:
        gs = codec.g_s
        for name, orig_fwd in saved["gs_films"]:
            getattr(gs, name).forward = orig_fwd

    if "daux_films" in saved:
        aux = codec.aux
        for name, orig_fwd in saved["daux_films"]:
            getattr(aux, name).forward = orig_fwd

    if "unet_lora_proj" in saved:
        model.unet_lora_proj = saved["unet_lora_proj"]


# ---------------------------------------------------------------------------
# Image loading and metrics
# ---------------------------------------------------------------------------

def load_images(img_dir: str, n_images: int = 0) -> List[Tuple[str, torch.Tensor]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if n_images > 0:
        paths = paths[:n_images]

    tf = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        images.append((p.name, tf(img).unsqueeze(0)))
    return images


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1) / 2
    y01 = (y.clamp(-1, 1) + 1) / 2
    mse = (x01 - y01).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


try:
    import lpips as _lpips_mod
    _lpips_fn = None

    def compute_lpips(x: torch.Tensor, y: torch.Tensor) -> float:
        global _lpips_fn
        if _lpips_fn is None:
            _lpips_fn = _lpips_mod.LPIPS(net="alex").to(x.device)
            _lpips_fn.eval()
        with torch.no_grad():
            return _lpips_fn(x.clamp(-1, 1), y.clamp(-1, 1)).item()
except ImportError:
    def compute_lpips(x: torch.Tensor, y: torch.Tensor) -> float:
        return float("nan")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_rd_curves(
    results: dict,
    out_dir: str,
    metric: str = "psnr",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cmap = plt.cm.tab10

    for i, (vname, vdata) in enumerate(sorted(results.items())):
        bpps = [pt["bpp"] for pt in vdata]
        vals = [pt[metric] for pt in vdata]
        if all(math.isnan(v) for v in vals):
            continue
        color = cmap(i % 10)
        marker = "o" if "_" not in vname else "s"
        ax.plot(bpps, vals, "-", color=color, label=vname,
                marker=marker, markersize=5)

    ax.set_xlabel("BPP")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation RD Curves ({metric.upper()})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ablation_rd_{metric}.png"), dpi=200)
    plt.close(fig)


def plot_bars(
    results: dict,
    out_dir: str,
    lmbda_val: float,
    metric: str = "psnr",
) -> None:
    names, vals = [], []
    for vname in sorted(results.keys()):
        pts = [p for p in results[vname] if abs(p["lmbda"] - lmbda_val) < 1e-6]
        if pts:
            names.append(vname)
            vals.append(pts[0][metric])

    if not names:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    colors = ["#2ecc71" if n == "M9" else "#3498db" for n in names]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation at \u03bb={lmbda_val} ({metric.upper()})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"ablation_bar_{metric}_lmbda{lmbda_val}.png"),
        dpi=200,
    )
    plt.close(fig)


def plot_esd_comparison(results: dict, out_dir: str) -> None:
    esd_variants = ["M5", "M4_fixed", "M4_random", "M9"]
    esd_labels = ["No LoRA (M5)", "Fixed LoRA", "Random alpha",
                  "\u03bb-adaptive (M9)"]
    available = [v for v in esd_variants if v in results]
    if len(available) < 2:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for vname, label in zip(esd_variants, esd_labels):
        if vname not in results:
            continue
        bpps = [pt["bpp"] for pt in results[vname]]
        lpips_vals = [pt["lpips"] for pt in results[vname]]
        if all(math.isnan(v) for v in lpips_vals):
            continue
        ax.plot(bpps, lpips_vals, "-o", label=label, markersize=5)

    ax.set_xlabel("BPP")
    ax.set_ylabel("LPIPS \u2193")
    ax.set_title("\u03b5SD-LoRA Fine-Grained Ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_esd_comparison.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def write_results_table(results: dict, out_path: str) -> None:
    lines = ["=" * 80, "Ablation Experiment Results", "=" * 80, ""]

    lines.append(
        f"{'Variant':<12} {'Config':<12} "
        f"{'lmbda':>6} {'BPP':>8} {'PSNR':>8} {'LPIPS':>8}"
    )
    lines.append("-" * 60)

    for vname in sorted(results.keys()):
        for pt in results[vname]:
            cfg_str = VARIANTS[vname].short_desc if vname in VARIANTS else ""
            lpips_str = (
                f"{pt['lpips']:.4f}"
                if not math.isnan(pt["lpips"]) else "N/A"
            )
            lines.append(
                f"{vname:<12} {cfg_str:<12} {pt['lmbda']:>6.1f} "
                f"{pt['bpp']:>8.4f} {pt['psnr']:>8.2f} {lpips_str:>8}"
            )
        lines.append("")

    esd_keys = [k for k in results if k.startswith("M4_") or k in ("M5", "M9")]
    if esd_keys:
        lines.extend([
            "", "=" * 60,
            "eSD Fine-Grained Ablation",
            "=" * 60, "",
        ])
        lines.append(
            f"{'Config':<20} {'lmbda':>6} {'BPP':>8} {'PSNR':>8} {'LPIPS':>8}"
        )
        lines.append("-" * 55)
        for vname in ["M5", "M4_fixed", "M4_random", "M4", "M9"]:
            if vname not in results:
                continue
            for pt in results[vname]:
                lpips_str = (
                    f"{pt['lpips']:.4f}"
                    if not math.isnan(pt["lpips"]) else "N/A"
                )
                lines.append(
                    f"{vname:<20} {pt['lmbda']:>6.1f} "
                    f"{pt['bpp']:>8.4f} {pt['psnr']:>8.2f} {lpips_str:>8}"
                )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(
    args: argparse.Namespace,
    device: torch.device,
    codec_path: str,
) -> nn.Module:
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
        "codec_path": codec_path,
        "elic_path": args.elic_path,
    }
    model = StableCodec(sd_path=args.sd_path, config=config)
    model.set_eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    lambdas = [float(x) for x in args.lambdas.split(",")]
    variant_names = (
        args.variants.split(",") if args.variants
        else list(VARIANTS.keys())
    )

    retrained_ckpts: Dict[str, str] = {}
    if args.retrained_ckpts:
        for pair in args.retrained_ckpts.split(","):
            name, path = pair.split(":")
            retrained_ckpts[name.strip()] = path.strip()

    images = load_images(args.img_dir, args.n_images)
    print(f"Loaded {len(images)} images from {args.img_dir}")

    results: Dict[str, List[dict]] = {}
    current_ckpt = None
    model = None

    for vname in variant_names:
        if vname not in VARIANTS:
            print(f"[SKIP] Unknown variant: {vname}")
            continue

        config = VARIANTS[vname]
        print(f"\n{'=' * 60}")
        print(f"Variant {vname}: {VARIANT_DESCRIPTIONS.get(vname, '')}")
        print(f"  Config: {config}")
        print(f"{'=' * 60}")

        if vname in NEEDS_RETRAINING and vname in retrained_ckpts:
            ckpt = retrained_ckpts[vname]
        elif vname in NEEDS_RETRAINING:
            print(
                f"  [WARN] {vname} needs retraining but no checkpoint "
                f"provided — using M9 weights (inference-time ablation)."
            )
            ckpt = args.codec_path
        else:
            ckpt = args.codec_path

        if model is None or ckpt != current_ckpt:
            del model
            torch.cuda.empty_cache()
            print(f"  Building model from {ckpt} ...")
            model = _build_model(args, device, ckpt)
            current_ckpt = ckpt

        results[vname] = []

        for lmbda_val in lambdas:
            psnr_acc, lpips_acc, bpp_acc = [], [], []
            lmbda_t = torch.tensor([lmbda_val], device=device)

            saved = apply_ablation(model, config, device)

            for img_name, img_tensor in images:
                img = img_tensor.to(device)
                B, _, H, W = img.shape

                output, rate_out = model(
                    img,
                    pos_prompt=[0],
                    ori_h=H,
                    ori_w=W,
                    lmbda=lmbda_t,
                )

                psnr_acc.append(compute_psnr(img, output))
                lpips_acc.append(compute_lpips(img, output))
                bpp_acc.append(rate_out.quantized_total_bpp.item())

            restore_ablation(model, saved)

            avg_psnr = float(np.mean(psnr_acc))
            avg_lpips = float(np.nanmean(lpips_acc))
            avg_bpp = float(np.mean(bpp_acc))

            results[vname].append({
                "lmbda": lmbda_val,
                "bpp": avg_bpp,
                "psnr": avg_psnr,
                "lpips": avg_lpips,
            })

            print(
                f"  \u03bb={lmbda_val:6.1f}  BPP={avg_bpp:.4f}  "
                f"PSNR={avg_psnr:.2f}  LPIPS={avg_lpips:.4f}"
            )

    with open(os.path.join(args.out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_rd_curves(results, args.out_dir, metric="psnr")
    plot_rd_curves(results, args.out_dir, metric="lpips")
    for lv in lambdas:
        plot_bars(results, args.out_dir, lv, metric="psnr")
    plot_esd_comparison(results, args.out_dir)
    write_results_table(
        results, os.path.join(args.out_dir, "ablation_results.txt")
    )

    print(f"\nDone. Results in {args.out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Variable-rate ablation experiment"
    )
    p.add_argument("--sd_path", required=True,
                   help="Path to SD-Turbo model")
    p.add_argument("--elic_path", required=True,
                   help="Path to ELIC checkpoint")
    p.add_argument("--codec_path", required=True,
                   help="Path to M9 (full) checkpoint")
    p.add_argument("--img_dir", required=True,
                   help="Image directory (e.g. Kodak)")
    p.add_argument("--out_dir", default="results/ablation",
                   help="Output directory")
    p.add_argument("--n_images", type=int, default=0,
                   help="Max images (0 = all)")
    p.add_argument("--lambdas", default="0.5,1,2,4,8,16,32",
                   help="Comma-separated lambda values")
    p.add_argument("--variants", default=None,
                   help="Comma-separated variant names (default: all)")
    p.add_argument("--retrained_ckpts", default=None,
                   help="variant:path pairs, e.g. M0:/path,M5:/path")
    p.add_argument("--lambda_min", type=float, default=0.1)
    p.add_argument("--lambda_max", type=float, default=128.0)
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
