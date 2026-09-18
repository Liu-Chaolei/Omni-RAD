"""
SNR验证实验：验证256通道VAE潜变量经过 g_a 融合 ELIC aux_codec 输出后，
得到的压缩特征 y 在加噪后保持更高的有效信噪比，从而使一步去噪更准确。

融合流程与 StableCodec_ori.py / latent_codec_ori.py 完全一致：
    aux_latent = aux_codec((img + 1) / 2)          # ELIC g_a, 输出 320ch
    vae_latent  = vae.encode(img).mode() * scale    # VAE encoder, 输出 4ch 或 256ch
    y           = codec.g_a(vae_latent, aux_latent) # AnalysisTransform_4/256, 输出 320ch

SNR 分析在 y 上进行，因为 y 是真正被熵编码和传输的表示。

对应 analysis.txt 中的实验设计。

用法（从repo根目录运行）：
    python src/experiment_snr.py \\
        --sd_path      /data/ssd/liuchaolei/models/Diffusion/sd-turbo \\
        --sd_path_256  /data/ssd/liuchaolei/models/Diffusion/sd-turbo_256 \\
        --elic_path    /data/ssd/liuchaolei/models/Image-Coder/ELIC/elic_official.pth \\
        --codec_path_4   /data/ssd/liuchaolei/models/Image-Coder/StableCodec/checkpoints/stablecodec_4ch.pth.tar \\
        --codec_path_256 /data/ssd/liuchaolei/models/Image-Coder/StableCodec/checkpoints/stablecodec_ft2.pkl \\
        --img_dir      /data/ssd/liuchaolei/image_datasets/Kodak24/HR \\
        --out_dir      results/snr_experiment
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
# ELIC 和 LatentCodec 从项目内导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ELIC.model.elic_official import ELIC
from latent_codec_ori import LatentCodec
from model import make_1step_sched_cuda


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SNRResult(NamedTuple):
    snr_db: float
    denoise_psnr: float


# ---------------------------------------------------------------------------
# Helpers: image loading
# ---------------------------------------------------------------------------

def load_images(img_dir: str, n_images: int, device: torch.device) -> torch.Tensor:
    """Load up to n_images PNG/JPG images, resize to 512×512, normalize to [-1,1]."""
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
    return imgs.to(device)


def get_alpha_bar(sched: DDPMScheduler, t: int) -> float:
    """Return ᾱ_t from the scheduler's cumulative product table."""
    return sched.alphas_cumprod[t].item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB between two tensors of arbitrary range ([-1,1] here)."""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(4.0 / mse)   # signal range = 2, peak² = 4


# ---------------------------------------------------------------------------
# Helpers: model builders
# ---------------------------------------------------------------------------

def build_aux_codec(elic_path: str, device: torch.device):
    """
    加载 ELIC 预训练模型，返回其分析变换 g_a（冻结）。
    输出：aux_codec，接收 [0,1] 范围的图像，输出 320ch 特征 (B,320,H/16,W/16)。
    """
    model = ELIC()
    checkpoint = torch.load(elic_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    aux_codec = model.g_a
    aux_codec = aux_codec.to(device)
    aux_codec.eval()
    aux_codec.requires_grad_(False)
    return aux_codec


def build_vae(sd_path: str, device: torch.device,
              codec_path: str | None = None,
              ignore_mismatch: bool = False) -> AutoencoderKL:
    """
    加载 AutoencoderKL。
    - ignore_mismatch=False → 标准 4ch VAE
    - ignore_mismatch=True  → 允许 256ch conv_out 的 VAE，再从 codec_path 注入 VAE 权重
    """
    vae = AutoencoderKL.from_pretrained(
        sd_path,
        subfolder="vae",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=ignore_mismatch,
    ).to(device)

    if codec_path is not None:
        sd = torch.load(codec_path, map_location="cpu")
        vae_sd = vae.state_dict()
        ckpt_vae = sd.get("state_dict_vae", {})
        for k, v in ckpt_vae.items():
            if k in vae_sd and v.shape == vae_sd[k].shape:
                vae_sd[k] = v
        vae.load_state_dict(vae_sd)
        del sd, vae_sd, ckpt_vae

    vae.eval()
    vae.requires_grad_(False)
    return vae


def build_codec(codec_path: str, latent_channels: int, device: torch.device) -> LatentCodec:
    """
    构建 LatentCodec（只需要 g_a 权重），加载检查点中的 state_dict_codec。
    latent_channels=4   → AnalysisTransform_4
    latent_channels=256 → AnalysisTransform_256
    """
    # lambda 值对 g_a 无影响，设为 1 即可
    codec = LatentCodec(lambda_rate=1, latent_channels=latent_channels)

    sd = torch.load(codec_path, map_location="cpu")
    # 检查点可能是 .pth.tar（含 state_dict_codec）或直接是 state_dict
    if isinstance(sd, dict) and "state_dict_codec" in sd:
        codec_sd = sd["state_dict_codec"]
    else:
        codec_sd = sd

    # 只加载能匹配的键（忽略形状不符的条目，避免其他组件干扰）
    model_sd = codec.state_dict()
    filtered = {k: v for k, v in codec_sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    model_sd.update(filtered)
    codec.load_state_dict(model_sd)

    codec = codec.to(device)
    codec.eval()
    codec.requires_grad_(False)
    return codec


# ---------------------------------------------------------------------------
# Core: compute fused feature y = g_a(vae_latent, aux_latent)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_fused_y(
    vae: AutoencoderKL,
    aux_codec,
    codec: LatentCodec,
    img: torch.Tensor,          # (1, 3, H, W), range [-1, 1]
) -> torch.Tensor:
    """
    完整复现 StableCodec_ori.py 的前向流程，得到融合特征 y。

    aux_latent = aux_codec((img + 1) / 2)            # ELIC g_a, [0,1] 输入
    vae_latent  = vae.encode(img).latent_dist.mode() * scale
    y           = codec.g_a(vae_latent, aux_latent)  # AnalysisTransform_4/256
    """
    img_01 = (img + 1.0) / 2.0                               # [-1,1] → [0,1]
    aux_latent = aux_codec(img_01)                            # (1, 320, H/16, W/16)

    vae_latent = vae.encode(img).latent_dist.mode()           # (1, C, H/8, W/8)
    vae_latent = vae_latent * vae.config.scaling_factor        # scale to ~unit variance

    y = codec.g_a(vae_latent, aux_latent)                     # (1, 320, H/32, W/32)
    return y


# ---------------------------------------------------------------------------
# Experiment 1 & 2 measurement functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_fused_snr(
    vae_4ch: AutoencoderKL,
    vae_256ch: AutoencoderKL,
    aux_codec,
    codec_4ch: LatentCodec,
    codec_256ch: LatentCodec,
    images: torch.Tensor,
    sched: DDPMScheduler,
    noise_timestep: int,
) -> dict[str, SNRResult]:
    """
    测量 4ch 和 256ch 两个变体在 g_a 融合后的特征 y 上的有效 SNR。

    因为 y 是 320ch 确定性特征（不是随机变量），SNR 定义为：
        SNR = (ᾱ_t · E[y²]) / ((1 − ᾱ_t) · E[ε²])
    其中 ε ~ N(0,I) 是加噪的噪声项。
    """
    results: dict[str, SNRResult] = {}
    alpha_bar = get_alpha_bar(sched, noise_timestep)

    configs = [
        ("4ch",   vae_4ch,   codec_4ch),
        ("256ch", vae_256ch, codec_256ch),
    ]

    for name, vae, codec in configs:
        snr_list: list[float] = []
        psnr_list: list[float] = []

        for img in images:
            img = img.unsqueeze(0)

            # 1. 得到融合特征 y
            y = compute_fused_y(vae, aux_codec, codec, img)  # (1, 320, h, w)

            # 2. 按噪声调度加噪
            noise = torch.randn_like(y)
            # y_noisy = sqrt(ᾱ_t)·y + sqrt(1−ᾱ_t)·ε

            # 3. 有效 SNR
            signal_power = (alpha_bar * y.pow(2)).mean().item()
            noise_power  = ((1 - alpha_bar) * noise.pow(2)).mean().item()
            snr_db = 10 * np.log10(signal_power / noise_power + 1e-12)
            snr_list.append(snr_db)

            # 4. 理论重建上界 PSNR（y 直接经 g_s 解码，无噪声时的图像质量）
            #    利用 LatentCodec 的合成变换 g_s 解码 y，再经 VAE decoder 还原图像
            y_hat = codec.g_s(y)          # (1, 320, H/8, W/8)
            # AnalysisTransform_256 的 pre1 将 256ch 下采样到 128ch（stride=2）
            # 因此 g_s 输出的空间尺寸与 VAE latent 一致 (H/8, W/8)
            # 取 aux decoder 输出作为 VAE latent 近似（形状匹配 vae decoder 期望输入）
            vae_latent_recon = codec.aux(y_hat)   # (1, C_vae, H/8, W/8)
            # 解码回图像
            recon = vae.decode(
                vae_latent_recon / vae.config.scaling_factor
            ).sample.clamp(-1, 1)
            psnr_list.append(compute_psnr(recon, img))

        results[name] = SNRResult(
            snr_db=float(np.mean(snr_list)),
            denoise_psnr=float(np.mean(psnr_list)),
        )
        print(
            f"{name:5s} | Fused-y SNR @ t={noise_timestep}: {results[name].snr_db:6.2f} dB"
            f" | Recon PSNR: {results[name].denoise_psnr:.2f} dB"
        )

    return results


@torch.no_grad()
def plot_snr_vs_timestep(
    vae_4ch: AutoencoderKL,
    vae_256ch: AutoencoderKL,
    aux_codec,
    codec_4ch: LatentCodec,
    codec_256ch: LatentCodec,
    image: torch.Tensor,
    sched: DDPMScheduler,
    timesteps: list[int],
    out_path: str,
) -> None:
    """
    绘制不同噪声步骤下融合特征 y 的 SNR 曲线，展示 256ch 在整个噪声调度过程中保持更高 SNR。
    """
    snr_4ch, snr_256ch = [], []
    img = image.unsqueeze(0)

    # 预计算一次 y（与 t 无关）
    y_4   = compute_fused_y(vae_4ch,   aux_codec, codec_4ch,   img)
    y_256 = compute_fused_y(vae_256ch, aux_codec, codec_256ch, img)

    for t in timesteps:
        alpha_bar = get_alpha_bar(sched, t)
        for y, snr_list in [(y_4, snr_4ch), (y_256, snr_256ch)]:
            noise = torch.randn_like(y)
            signal_power = (alpha_bar * y.pow(2)).mean().item()
            noise_power  = ((1 - alpha_bar) * noise.pow(2)).mean().item()
            snr_list.append(10 * np.log10(signal_power / noise_power + 1e-12))

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, snr_4ch,   label="4ch VAE (fused y)",   marker="o", linewidth=2)
    plt.plot(timesteps, snr_256ch, label="256ch VAE (fused y)", marker="s", linewidth=2)
    plt.xlabel("Noise Timestep T")
    plt.ylabel("Effective SNR of Fused Feature y (dB)")
    plt.title("Fused-Feature SNR across Noise Schedule\n(y = g_a(vae_latent, aux_latent))")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"SNR curve saved to {out_path}")


@torch.no_grad()
def plot_signal_power_comparison(
    vae_4ch: AutoencoderKL,
    vae_256ch: AutoencoderKL,
    aux_codec,
    codec_4ch: LatentCodec,
    codec_256ch: LatentCodec,
    images: torch.Tensor,
    out_path: str,
) -> None:
    """
    额外分析：比较两个变体融合特征 y 的信号功率分布。
    更高的信号功率 E[y²] 意味着在相同噪声调度下具有更高的 SNR。
    绘制每张图像的 E[y²] 散点图和均值。
    """
    power_4ch, power_256ch = [], []

    for img in images:
        img = img.unsqueeze(0)
        y_4   = compute_fused_y(vae_4ch,   aux_codec, codec_4ch,   img)
        y_256 = compute_fused_y(vae_256ch, aux_codec, codec_256ch, img)
        power_4ch.append(y_4.pow(2).mean().item())
        power_256ch.append(y_256.pow(2).mean().item())

    n = len(images)
    x = np.arange(n)

    plt.figure(figsize=(max(8, n // 2), 5))
    plt.bar(x - 0.2, power_4ch,   width=0.4, label=f"4ch   (mean={np.mean(power_4ch):.4f})",   alpha=0.75)
    plt.bar(x + 0.2, power_256ch, width=0.4, label=f"256ch (mean={np.mean(power_256ch):.4f})", alpha=0.75)
    plt.axhline(np.mean(power_4ch),   color="C0", linestyle="--", linewidth=1)
    plt.axhline(np.mean(power_256ch), color="C1", linestyle="--", linewidth=1)
    plt.xlabel("Image Index")
    plt.ylabel("E[y²] — Signal Power of Fused Feature")
    plt.title("Signal Power Comparison: 4ch vs 256ch VAE\n(y = g_a(vae_latent, aux_latent))")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Signal power comparison saved to {out_path}")
    print(f"  4ch   mean E[y²] = {np.mean(power_4ch):.6f}")
    print(f"  256ch mean E[y²] = {np.mean(power_256ch):.6f}")
    ratio = np.mean(power_256ch) / (np.mean(power_4ch) + 1e-12)
    print(f"  256ch / 4ch ratio = {ratio:.3f}×")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SNR experiment on fused feature y = g_a(vae_latent, aux_latent)"
    )
    p.add_argument("--sd_path",        required=True,
                   help="Path to standard sd-turbo model directory (4ch VAE)")
    p.add_argument("--sd_path_256",    required=True,
                   help="Path to sd-turbo model directory with 256ch VAE modification")
    p.add_argument("--elic_path",      required=True,
                   help="Path to pretrained ELIC checkpoint (.pth)")
    p.add_argument("--codec_path_4",   required=True,
                   help="StableCodec checkpoint (.pth.tar) for 4ch LatentCodec weights")
    p.add_argument("--codec_path_256", required=True,
                   help="StableCodec checkpoint (.pkl/.pth.tar) for 256ch LatentCodec weights")
    p.add_argument("--img_dir",        required=True,  help="Directory of test images")
    p.add_argument("--out_dir",        default="results/snr_experiment")
    p.add_argument("--n_images",       type=int, default=24)
    p.add_argument("--noise_timestep", type=int, default=999)
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 加载 DDPMScheduler ───────────────────────────────────────────────────
    print("Loading scheduler ...")
    sched = make_1step_sched_cuda(args.sd_path)
    sched.alphas_cumprod = sched.alphas_cumprod.to(device)

    # ── 加载图像 ─────────────────────────────────────────────────────────────
    print("Loading images ...")
    images = load_images(args.img_dir, args.n_images, device)
    print(f"  Loaded {len(images)} images")

    # ── 加载 ELIC aux_codec ───────────────────────────────────────────────────
    print("\nLoading ELIC aux_codec ...")
    aux_codec = build_aux_codec(args.elic_path, device)
    print("  ELIC g_a loaded and frozen.")

    # ── 加载 VAE（4ch 和 256ch）─────────────────────────────────────────────
    print("\nLoading 4ch VAE ...")
    vae_4ch = build_vae(args.sd_path, device, codec_path=None, ignore_mismatch=False)

    print("Loading 256ch VAE ...")
    vae_256ch = build_vae(
        args.sd_path_256, device,
        codec_path=args.codec_path_256,
        ignore_mismatch=True,
    )

    # ── 加载 LatentCodec（4ch 和 256ch）─────────────────────────────────────
    print("\nLoading 4ch LatentCodec ...")
    codec_4ch = build_codec(args.codec_path_4, latent_channels=4, device=device)

    print("Loading 256ch LatentCodec ...")
    codec_256ch = build_codec(args.codec_path_256, latent_channels=256, device=device)

    # ── Experiment 1: 固定 timestep 的 SNR 比较 ──────────────────────────────
    print(f"\n[Experiment 1] Fused-feature SNR comparison at t={args.noise_timestep}")
    results = measure_fused_snr(
        vae_4ch, vae_256ch,
        aux_codec,
        codec_4ch, codec_256ch,
        images, sched, args.noise_timestep,
    )

    summary_path = os.path.join(args.out_dir, "snr_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Experiment: SNR on fused feature y = g_a(vae_latent, aux_latent)\n")
        f.write(f"Noise timestep: {args.noise_timestep}\n")
        f.write(f"Number of images: {len(images)}\n\n")
        for name, r in results.items():
            f.write(f"{name}: SNR={r.snr_db:.4f} dB, Recon PSNR={r.denoise_psnr:.4f} dB\n")
        delta_snr = results["256ch"].snr_db - results["4ch"].snr_db
        f.write(f"\nΔSNR (256ch - 4ch): {delta_snr:+.4f} dB\n")
    print(f"Summary written to {summary_path}")

    # ── Experiment 2: SNR vs timestep 曲线 ──────────────────────────────────
    print("\n[Experiment 2] Fused-feature SNR vs timestep curve ...")
    timesteps = list(range(0, 1000, 50))
    plot_snr_vs_timestep(
        vae_4ch, vae_256ch,
        aux_codec,
        codec_4ch, codec_256ch,
        images[0], sched, timesteps,
        out_path=os.path.join(args.out_dir, "snr_vs_timestep.pdf"),
    )

    # ── Experiment 3: 信号功率分布比较 ──────────────────────────────────────
    print("\n[Experiment 3] Signal power distribution of fused feature y ...")
    plot_signal_power_comparison(
        vae_4ch, vae_256ch,
        aux_codec,
        codec_4ch, codec_256ch,
        images,
        out_path=os.path.join(args.out_dir, "signal_power_comparison.pdf"),
    )

    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
