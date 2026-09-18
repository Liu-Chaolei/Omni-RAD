"""
流形扭曲测试：验证256通道VAE对量化噪声的鲁棒性优于4通道VAE。

理论基础（见 experiment_quantization_robustness.txt）：
    在极低码率压缩中，隐空间必须经历严重量化。4ch隐空间将图像高维特征极度压缩，
    微小的量化误差经解码器放大后导致巨大的结构性失真。256ch隐空间中特征更稀疏、
    正交，相同量化噪声对最终重建流形的破坏更小。

实验做法（方向一：在完整 StableCodec 推理链路中注入噪声）：
    1. 走完整编码端：aux_latent = aux_codec((img+1)/2)
                     lq_latent  = vae.encode(img) * scale_factor
    2. 在 lq_latent 处注入噪声：lq_latent_noisy = lq_latent + noise
    3. 走完整解码端（严格按照 StableCodec_ori.py / latent_codec_ori.py，跳过熵编码
       的 likelihood 计算，但保留所有结构）：
         g_a(lq_latent_noisy, aux_latent)
           → h_a → ste_round(z) → h_s → base
           → 4-pass checkerboard + LRP → y_hat
           → g_s(y_hat) = lq_latent_hat，aux(y_hat) = res1
         lq_latent_hat → UNet → scheduler.step → x_denoised + res1
         x_denoised → vae.decode → output_image
    4. 逐渐增大噪声 σ 或量化步长 Δ，绘制 PSNR 曲线

注意：编解码流程严格参照 StableCodec_ori.py 和 latent_codec_ori.py，
      除了不走熵编码的 likelihood / rate 计算之外，不做任何修改。

用法（从 repo 根目录运行）：
    python src/experiment_quantization_robustness.py \\
        --sd_path        /data/ssd/liuchaolei/models/Diffusion/sd-turbo \\
        --sd_path_256    /data/ssd/liuchaolei/models/Diffusion/sd-turbo_256 \\
        --elic_path      /data/ssd/liuchaolei/models/Image-Coder/ELIC/elic_official.pth \\
        --codec_path_4   /data/ssd/liuchaolei/models/Image-Coder/StableCodec/checkpoints/stablecodec_4ch.pth.tar \\
        --codec_path_256 /data/ssd/liuchaolei/models/Image-Coder/StableCodec/checkpoints/stablecodec_ft2.pkl \\
        --img_dir        /data/ssd/liuchaolei/image_datasets/Kodak24/HR \\
        --out_dir        results/quantization_robustness
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
import torch.nn.functional as F
from compressai.ops import quantize_ste as ste_round
from diffusers import AutoencoderKL, UNet2DConditionModel
from model import make_1step_sched_cuda, my_lora_fwd
from peft import LoraConfig
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ELIC.model.elic_official import ELIC
from latent_codec_ori import LatentCodec


# ---------------------------------------------------------------------------
# Helpers: image loading & metrics
# ---------------------------------------------------------------------------

def load_images(img_dir: str, n_images: int, device: torch.device) -> torch.Tensor:
    """加载最多 n_images 张图像，resize 到 512×512，归一化到 [-1, 1]。"""
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


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算两个 [-1,1] 范围张量之间的 PSNR（dB），信号峰值为 2。"""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(4.0 / mse)  # peak² = (max - min)² = 4


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_aux_codec(elic_path: str, device: torch.device):
    """加载 ELIC g_a（冻结），接收 [0,1] 图像，输出 320ch 特征。"""
    model = ELIC()
    checkpoint = torch.load(elic_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    aux_codec = model.g_a.to(device)
    aux_codec.eval()
    aux_codec.requires_grad_(False)
    return aux_codec


def build_vae(
    sd_path: str,
    device: torch.device,
    codec_path: str | None = None,
    ignore_mismatch: bool = False,
) -> AutoencoderKL:
    """
    加载 AutoencoderKL，注入 VAE LoRA，与 StableCodec_ori.py 的 __init__ 完全一致。
    ignore_mismatch=True 用于 256ch VAE。
    """
    vae = AutoencoderKL.from_pretrained(
        sd_path,
        subfolder="vae",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=ignore_mismatch,
    ).to(device)

    # LoRA（与 StableCodec_ori.py 一致）
    target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
    vae_lora_config = LoraConfig(r=16, init_lora_weights="gaussian", target_modules=target_modules_vae)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

    # my_lora_fwd patch（与 StableCodec_ori.py 一致）
    vae_lora_layers = [
        name[: -len(".base_layer")]
        for name, _ in vae.named_modules()
        if "base_layer" in name
    ]
    for name, module in vae.named_modules():
        if name in vae_lora_layers:
            module.forward = my_lora_fwd.__get__(module, module.__class__)

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


def build_unet(
    sd_path: str,
    device: torch.device,
    codec_path: str | None = None,
) -> UNet2DConditionModel:
    """
    加载 UNet，注入 LoRA 并替换 conv_in（320→320），
    与 StableCodec_ori.py 的 __init__ 完全一致。
    """
    unet = UNet2DConditionModel.from_pretrained(
        sd_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    ).to(device)

    # LoRA（与 StableCodec_ori.py 一致）
    target_modules_unet = [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
        "conv_shortcut", "conv_out", "proj_in", "proj_out",
        "ff.net.2", "ff.net.0.proj",
    ]
    unet_lora_config = LoraConfig(
        r=32, init_lora_weights="gaussian", target_modules=target_modules_unet
    )
    unet.add_adapter(unet_lora_config)

    # my_lora_fwd patch（与 StableCodec_ori.py 一致）
    unet_lora_layers = [
        name[: -len(".base_layer")]
        for name, _ in unet.named_modules()
        if "base_layer" in name
    ]
    for name, module in unet.named_modules():
        if name in unet_lora_layers:
            module.forward = my_lora_fwd.__get__(module, module.__class__)

    # 替换 conv_in：4ch→320ch（与 StableCodec_ori.py 一致）
    unet.conv_in = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device)

    if codec_path is not None:
        sd = torch.load(codec_path, map_location="cpu")
        unet_sd = unet.state_dict()
        ckpt_unet = sd.get("state_dict_unet", {})
        for k, v in ckpt_unet.items():
            if k in unet_sd and v.shape == unet_sd[k].shape:
                unet_sd[k] = v
        unet.load_state_dict(unet_sd)
        del sd, unet_sd, ckpt_unet

    unet.eval()
    unet.requires_grad_(False)
    return unet


def build_codec(
    codec_path: str,
    latent_channels: int,
    device: torch.device,
) -> LatentCodec:
    """
    构建 LatentCodec，加载 state_dict_codec，与 StableCodec_ori.py 一致。
    """
    codec = LatentCodec(lambda_rate=1, latent_channels=latent_channels)

    sd = torch.load(codec_path, map_location="cpu")
    codec_ckpt = sd.get("state_dict_codec", sd)  # .pth.tar 或 .pkl

    model_sd = codec.state_dict()
    for k, v in codec_ckpt.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                model_sd[k] = v
            else:
                print(f"  [codec] shape mismatch skipped: {k} "
                      f"ckpt={tuple(v.shape)} model={tuple(model_sd[k].shape)}")
    codec.load_state_dict(model_sd)

    codec.to(device)
    codec.eval()
    codec.requires_grad_(False)
    return codec


def build_pos_caption(
    sd_path: str,
    pos_prompt: str,
    device: torch.device,
) -> torch.Tensor:
    """
    与 StableCodec_ori.py 的 set_prompt 完全一致，返回 pos_caption_enc (1, 77, 768)。
    """
    tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        sd_path, subfolder="text_encoder"
    ).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    tokens = tokenizer(
        pos_prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        enc = text_encoder(tokens)[0]  # (1, 77, 768)
    del tokenizer, text_encoder
    return enc


# ---------------------------------------------------------------------------
# Core: 完整推理路径（跳过熵编码 likelihood，保留所有其他结构）
# ---------------------------------------------------------------------------

@torch.no_grad()
def codec_forward_no_entropy(
    codec: LatentCodec,
    lq_latent: torch.Tensor,   # (1, C, H/8, W/8)，已乘 scaling_factor
    aux_latent: torch.Tensor,  # (1, 320, H/16, W/16)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    严格复现 latent_codec_ori.py LatentCodec.forward() 的解码路径，
    跳过 entropy_bottleneck / gaussian_conditional 的 likelihood 计算和
    TargetRateModule，其余结构（h_a, ste_round, h_s, checkerboard, LRP,
    g_s, aux）完全保留。

    Returns
    -------
    lq_latent_hat : (1, 320, H/8, W/8)  对应 forward() 中的 x_hat
    res1          : (1, C_vae, H/8, W/8) 对应 forward() 中的 res
    """
    # ── g_a ─────────────────────────────────────────────────────────────────
    y = codec.g_a(lq_latent, aux_latent)            # (1, 320, H/32, W/32)

    # ── HyperAnalysis + ste_round（与 forward 完全一致）─────────────────────
    z = codec.h_a(y)
    z_offset = codec.entropy_bottleneck._get_medians()
    z_hat = ste_round(z - z_offset) + z_offset      # (1, 160, H/64, W/64)

    # ── HyperSynthesis → base ─────────────────────────────────────────────
    base = codec.h_s(z_hat)                          # (1, 320, H/32, W/32)

    B, C, H, W = y.shape
    mask_0, mask_1, mask_2, mask_3 = codec.get_mask_four_parts(B, C, H, W, device=y.device)

    # ── 4-pass checkerboard + LRP（与 forward 完全一致）────────────────────
    means_0_supp, _ = codec.adapter_out[0](codec.g_c(codec.adapter_in[0](base))).chunk(2, 1)
    means_0 = means_0_supp * mask_0
    y_hat_0 = ste_round(y * mask_0 - means_0) + means_0
    lrp = codec.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
    y_hat_0 = y_hat_0 + 0.5 * torch.tanh(lrp)

    base = base * (1 - mask_0) + y_hat_0
    means_1_supp, _ = codec.adapter_out[1](codec.g_c(codec.adapter_in[1](base))).chunk(2, 1)
    means_1 = means_1_supp * mask_1
    y_hat_1 = ste_round(y * mask_1 - means_1) + means_1
    lrp = codec.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
    y_hat_1 = y_hat_1 + 0.5 * torch.tanh(lrp)

    base = base * (1 - mask_1) + y_hat_1
    means_2_supp, _ = codec.adapter_out[2](codec.g_c(codec.adapter_in[2](base))).chunk(2, 1)
    means_2 = means_2_supp * mask_2
    y_hat_2 = ste_round(y * mask_2 - means_2) + means_2
    lrp = codec.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
    y_hat_2 = y_hat_2 + 0.5 * torch.tanh(lrp)

    base = base * (1 - mask_2) + y_hat_2
    means_3_supp, _ = codec.adapter_out[3](codec.g_c(codec.adapter_in[3](base))).chunk(2, 1)
    means_3 = means_3_supp * mask_3
    y_hat_3 = ste_round(y * mask_3 - means_3) + means_3
    lrp = codec.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
    y_hat_3 = y_hat_3 + 0.5 * torch.tanh(lrp)

    y_hat = base * (1 - mask_3) + y_hat_3           # (1, 320, H/32, W/32)

    # ── g_s + aux（与 forward 完全一致）─────────────────────────────────────
    lq_latent_hat = codec.g_s(y_hat)                # (1, 320, H/8, W/8)
    res1 = codec.aux(y_hat)                          # (1, C_vae, H/8, W/8)

    return lq_latent_hat, res1


@torch.no_grad()
def full_pipeline(
    img: torch.Tensor,              # (1, 3, H, W), [-1, 1]
    vae: AutoencoderKL,
    aux_codec,
    codec: LatentCodec,
    unet: UNet2DConditionModel,
    sched,
    pos_caption_enc: torch.Tensor,  # (1, 77, 768)
    latent_channels: int,
    noise_fn,                        # callable(lq_latent) -> lq_latent_noisy
) -> torch.Tensor:
    """
    完整推理路径，严格参照 StableCodec_ori.py forward()，
    在 lq_latent 处注入噪声后走完整链路。

    StableCodec_ori.py forward() 对应步骤：
        latent2    = aux_codec((x + 1) / 2)
        lq_latent  = vae.encode(x).latent_dist.mode() * scale_factor
        ↓ [噪声注入]
        lq_latent_hat, _, res1 = codec(lq_latent_noisy, latent2)
        model_pred = unet(lq_latent_hat, timesteps, encoder_hidden_states)
        x_denoised = sched.step(model_pred, t, lq_latent_hat[:, :C]).prev_sample + res1
        output     = vae.decode(x_denoised / scale_factor).sample.clamp(-1, 1)
    """
    # ── Encoder（与 StableCodec_ori.py forward 完全一致）───────────────────
    img_01 = (img + 1.0) / 2.0
    aux_latent = aux_codec(img_01)                                          # (1, 320, H/16, W/16)
    lq_latent = vae.encode(img).latent_dist.mode() * vae.config.scaling_factor  # (1, C, H/8, W/8)

    # ── 噪声注入（在 vae_latent 处）────────────────────────────────────────
    lq_latent_noisy = noise_fn(lq_latent)

    # ── LatentCodec（跳过 likelihood，其余结构完全保留）────────────────────
    lq_latent_hat, res1 = codec_forward_no_entropy(codec, lq_latent_noisy, aux_latent)

    # ── One-Step Denoiser（与 StableCodec_ori.py forward 完全一致）─────────
    timesteps = torch.tensor([999], device=img.device).long()
    model_pred = unet(
        lq_latent_hat, timesteps, encoder_hidden_states=pos_caption_enc
    ).sample
    x_denoised = (
        sched.step(model_pred, timesteps, lq_latent_hat[:, :latent_channels],
                   return_dict=True).prev_sample
        + res1
    )

    # ── Decoder（与 StableCodec_ori.py forward 完全一致）───────────────────
    output_image = vae.decode(
        x_denoised / vae.config.scaling_factor
    ).sample.clamp(-1.0, 1.0)

    return output_image


# ---------------------------------------------------------------------------
# Noise injection functions
# ---------------------------------------------------------------------------

def inject_gaussian_noise(lq_latent: torch.Tensor, sigma: float) -> torch.Tensor:
    """Z' = Z + N(0, σ²)"""
    return lq_latent + sigma * torch.randn_like(lq_latent)


def inject_quantization(lq_latent: torch.Tensor, delta: float) -> torch.Tensor:
    """Z' = Round(Z / Δ) × Δ，均匀量化。delta=0 时直接返回原值。"""
    if delta <= 0.0:
        return lq_latent.clone()
    return torch.round(lq_latent / delta) * delta


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_gaussian_experiment(
    vae: AutoencoderKL,
    aux_codec,
    codec: LatentCodec,
    unet: UNet2DConditionModel,
    sched,
    pos_caption_enc: torch.Tensor,
    latent_channels: int,
    images: torch.Tensor,
    sigma_list: list[float],
    n_trials: int,
    variant_name: str,
) -> list[float]:
    """
    对所有图像、所有 sigma 值运行高斯噪声实验，返回每个 sigma 下的平均 PSNR。
    """
    psnr_means: list[float] = []
    for sigma in sigma_list:
        batch_psnr: list[float] = []
        for img in images:
            img_b = img.unsqueeze(0)
            trial_psnr: list[float] = []
            for _ in range(n_trials):
                recon = full_pipeline(
                    img=img_b,
                    vae=vae,
                    aux_codec=aux_codec,
                    codec=codec,
                    unet=unet,
                    sched=sched,
                    pos_caption_enc=pos_caption_enc,
                    latent_channels=latent_channels,
                    noise_fn=lambda z, s=sigma: inject_gaussian_noise(z, s),
                )
                trial_psnr.append(compute_psnr(recon, img_b))
            batch_psnr.append(float(np.mean(trial_psnr)))
        mean_psnr = float(np.mean(batch_psnr))
        psnr_means.append(mean_psnr)
        print(f"  [{variant_name}] σ={sigma:.4f} | PSNR={mean_psnr:.2f} dB")
    return psnr_means


@torch.no_grad()
def run_quantization_experiment(
    vae: AutoencoderKL,
    aux_codec,
    codec: LatentCodec,
    unet: UNet2DConditionModel,
    sched,
    pos_caption_enc: torch.Tensor,
    latent_channels: int,
    images: torch.Tensor,
    delta_list: list[float],
    variant_name: str,
) -> list[float]:
    """
    对所有图像、所有 delta 值运行均匀量化实验（确定性，无需多次 trial），
    返回每个 delta 下的平均 PSNR。
    """
    psnr_means: list[float] = []
    for delta in delta_list:
        batch_psnr: list[float] = []
        for img in images:
            img_b = img.unsqueeze(0)
            recon = full_pipeline(
                img=img_b,
                vae=vae,
                aux_codec=aux_codec,
                codec=codec,
                unet=unet,
                sched=sched,
                pos_caption_enc=pos_caption_enc,
                latent_channels=latent_channels,
                noise_fn=lambda z, d=delta: inject_quantization(z, d),
            )
            batch_psnr.append(compute_psnr(recon, img_b))
        mean_psnr = float(np.mean(batch_psnr))
        psnr_means.append(mean_psnr)
        print(f"  [{variant_name}] Δ={delta:.4f} | PSNR={mean_psnr:.2f} dB")
    return psnr_means


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_psnr_curve(
    x_vals: list[float],
    psnr_4ch: list[float],
    psnr_256ch: list[float],
    xlabel: str,
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(x_vals, psnr_4ch,   label="4ch VAE",   marker="o", linewidth=2, color="steelblue")
    plt.plot(x_vals, psnr_256ch, label="256ch VAE", marker="s", linewidth=2, color="darkorange")
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel("PSNR (dB)", fontsize=13)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {out_path}")


def save_summary(
    out_path: str,
    sigma_list: list[float],
    psnr_4ch_gauss: list[float],
    psnr_256ch_gauss: list[float],
    delta_list: list[float],
    psnr_4ch_quant: list[float],
    psnr_256ch_quant: list[float],
    n_images: int,
    n_trials: int,
) -> None:
    with open(out_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Quantization Robustness Experiment (Full Pipeline)\n")
        f.write("  Noise injected at lq_latent (VAE encoder output)\n")
        f.write(f"  # images : {n_images}\n")
        f.write(f"  # trials (Gaussian) : {n_trials}\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- Experiment A: Gaussian Noise (Z' = Z + N(0,σ²)) ---\n")
        f.write(f"{'sigma':>8} {'4ch PSNR':>12} {'256ch PSNR':>12} {'Δ PSNR':>10}\n")
        for sigma, p4, p256 in zip(sigma_list, psnr_4ch_gauss, psnr_256ch_gauss):
            f.write(f"{sigma:8.4f} {p4:12.4f} {p256:12.4f} {p256 - p4:+10.4f}\n")

        f.write("\n--- Experiment B: Uniform Quantization (Z' = Round(Z/Δ)×Δ) ---\n")
        f.write(f"{'delta':>8} {'4ch PSNR':>12} {'256ch PSNR':>12} {'Δ PSNR':>10}\n")
        for delta, p4, p256 in zip(delta_list, psnr_4ch_quant, psnr_256ch_quant):
            f.write(f"{delta:8.4f} {p4:12.4f} {p256:12.4f} {p256 - p4:+10.4f}\n")
    print(f"  Summary written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Quantization robustness: inject noise into VAE latent and run "
            "full StableCodec pipeline, compare 4ch vs 256ch."
        )
    )
    p.add_argument("--sd_path",        required=True,
                   help="Path to standard sd-turbo model directory (4ch VAE + UNet)")
    p.add_argument("--sd_path_256",    required=True,
                   help="Path to sd-turbo model directory with 256ch VAE modification")
    p.add_argument("--elic_path",      required=True,
                   help="Path to pretrained ELIC checkpoint (.pth)")
    p.add_argument("--codec_path_4",   required=True,
                   help="StableCodec checkpoint (.pth.tar) for 4ch variant")
    p.add_argument("--codec_path_256", required=True,
                   help="StableCodec checkpoint (.pkl) for 256ch variant")
    p.add_argument("--img_dir",        required=True,
                   help="Directory of test images (e.g. Kodak24/HR)")
    p.add_argument("--out_dir",        default="results/quantization_robustness")
    p.add_argument("--n_images",       type=int, default=24)
    p.add_argument("--n_trials",       type=int, default=3,
                   help="Noise trials for Gaussian experiment (results averaged)")
    p.add_argument("--pos_prompt",     type=str,
                   default="Enhance the low-quality image to high-quality",
                   help="Positive text prompt (must match training)")
    # Gaussian noise sweep
    p.add_argument("--sigma_min",      type=float, default=0.0)
    p.add_argument("--sigma_max",      type=float, default=2.0)
    p.add_argument("--sigma_steps",    type=int,   default=21)
    # Quantization step sweep
    p.add_argument("--delta_min",      type=float, default=0.0)
    p.add_argument("--delta_max",      type=float, default=2.0)
    p.add_argument("--delta_steps",    type=int,   default=21)
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 图像 ─────────────────────────────────────────────────────────────────
    print("Loading images ...")
    images = load_images(args.img_dir, args.n_images, device)
    print(f"  Loaded {len(images)} images (512×512)")

    # ── ELIC aux_codec（共用）────────────────────────────────────────────────
    print("\nLoading ELIC aux_codec ...")
    aux_codec = build_aux_codec(args.elic_path, device)

    # ── 4ch 模型组件 ─────────────────────────────────────────────────────────
    print("\n[4ch] Loading VAE ...")
    vae_4ch = build_vae(args.sd_path, device,
                        codec_path=args.codec_path_4, ignore_mismatch=False)

    print("[4ch] Loading UNet ...")
    unet_4ch = build_unet(args.sd_path, device, codec_path=args.codec_path_4)

    print("[4ch] Loading LatentCodec ...")
    codec_4ch = build_codec(args.codec_path_4, latent_channels=4, device=device)

    print("[4ch] Building scheduler & prompt ...")
    sched_4ch = make_1step_sched_cuda(args.sd_path)
    pos_enc_4ch = build_pos_caption(args.sd_path, args.pos_prompt, device)

    # ── 256ch 模型组件 ───────────────────────────────────────────────────────
    print("\n[256ch] Loading VAE ...")
    vae_256ch = build_vae(args.sd_path_256, device,
                          codec_path=args.codec_path_256, ignore_mismatch=True)

    print("[256ch] Loading UNet ...")
    unet_256ch = build_unet(args.sd_path_256, device, codec_path=args.codec_path_256)

    print("[256ch] Loading LatentCodec ...")
    codec_256ch = build_codec(args.codec_path_256, latent_channels=256, device=device)

    print("[256ch] Building scheduler & prompt ...")
    sched_256ch = make_1step_sched_cuda(args.sd_path_256)
    pos_enc_256ch = build_pos_caption(args.sd_path_256, args.pos_prompt, device)

    # ── Baseline：σ=0 / Δ=0 时的系统重建质量 ───────────────────────────────
    print("\n[Baseline] Full pipeline, no noise:")
    baseline_psnr: dict[str, list[float]] = {"4ch": [], "256ch": []}
    for img in images:
        img_b = img.unsqueeze(0)
        for name, vae, ac, codec, unet, sched, pos_enc, lc in [
            ("4ch",   vae_4ch,   aux_codec, codec_4ch,   unet_4ch,   sched_4ch,   pos_enc_4ch,   4),
            ("256ch", vae_256ch, aux_codec, codec_256ch, unet_256ch, sched_256ch, pos_enc_256ch, 256),
        ]:
            recon = full_pipeline(img_b, vae, ac, codec, unet, sched, pos_enc, lc,
                                  noise_fn=lambda z: z.clone())
            baseline_psnr[name].append(compute_psnr(recon, img_b))
    print(f"  4ch   baseline PSNR = {np.mean(baseline_psnr['4ch']):.2f} dB")
    print(f"  256ch baseline PSNR = {np.mean(baseline_psnr['256ch']):.2f} dB")

    # ── Experiment A：高斯噪声 ────────────────────────────────────────────────
    sigma_list = list(np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps))
    print(f"\n[Experiment A] Gaussian noise "
          f"σ ∈ [{args.sigma_min}, {args.sigma_max}], "
          f"{args.sigma_steps} steps, {args.n_trials} trials")

    print("  Running 4ch ...")
    psnr_4ch_gauss = run_gaussian_experiment(
        vae_4ch, aux_codec, codec_4ch, unet_4ch, sched_4ch, pos_enc_4ch, 4,
        images, sigma_list, args.n_trials, "4ch",
    )
    print("  Running 256ch ...")
    psnr_256ch_gauss = run_gaussian_experiment(
        vae_256ch, aux_codec, codec_256ch, unet_256ch, sched_256ch, pos_enc_256ch, 256,
        images, sigma_list, args.n_trials, "256ch",
    )
    plot_psnr_curve(
        sigma_list, psnr_4ch_gauss, psnr_256ch_gauss,
        xlabel="Gaussian Noise Std (σ)",
        title="Robustness to Gaussian Noise in VAE Latent Space\n"
              "Full StableCodec Pipeline  |  Z' = Z + N(0, σ²)",
        out_path=os.path.join(args.out_dir, "robustness_gaussian.pdf"),
    )

    # ── Experiment B：均匀量化 ────────────────────────────────────────────────
    delta_list = list(np.linspace(args.delta_min, args.delta_max, args.delta_steps))
    print(f"\n[Experiment B] Uniform quantization "
          f"Δ ∈ [{args.delta_min}, {args.delta_max}], "
          f"{args.delta_steps} steps")

    print("  Running 4ch ...")
    psnr_4ch_quant = run_quantization_experiment(
        vae_4ch, aux_codec, codec_4ch, unet_4ch, sched_4ch, pos_enc_4ch, 4,
        images, delta_list, "4ch",
    )
    print("  Running 256ch ...")
    psnr_256ch_quant = run_quantization_experiment(
        vae_256ch, aux_codec, codec_256ch, unet_256ch, sched_256ch, pos_enc_256ch, 256,
        images, delta_list, "256ch",
    )
    plot_psnr_curve(
        delta_list, psnr_4ch_quant, psnr_256ch_quant,
        xlabel="Quantization Step Size (Δ)",
        title="Robustness to Uniform Quantization in VAE Latent Space\n"
              "Full StableCodec Pipeline  |  Z' = Round(Z/Δ) × Δ",
        out_path=os.path.join(args.out_dir, "robustness_quantization.pdf"),
    )

    # ── 联合对比图 ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sigma_list, psnr_4ch_gauss,   label="4ch VAE",   marker="o", linewidth=2, color="steelblue")
    ax1.plot(sigma_list, psnr_256ch_gauss, label="256ch VAE", marker="s", linewidth=2, color="darkorange")
    ax1.set_xlabel("Gaussian Noise Std (σ)", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title("Gaussian Noise Robustness", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True)

    ax2.plot(delta_list, psnr_4ch_quant,   label="4ch VAE",   marker="o", linewidth=2, color="steelblue")
    ax2.plot(delta_list, psnr_256ch_quant, label="256ch VAE", marker="s", linewidth=2, color="darkorange")
    ax2.set_xlabel("Quantization Step Size (Δ)", fontsize=12)
    ax2.set_ylabel("PSNR (dB)", fontsize=12)
    ax2.set_title("Uniform Quantization Robustness", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True)

    fig.suptitle(
        "Manifold Distortion Test: 4ch vs 256ch VAE Latent Space\n"
        "(Noise injected at lq_latent; full StableCodec pipeline downstream)",
        fontsize=13,
    )
    fig.tight_layout()
    combined_path = os.path.join(args.out_dir, "robustness_combined.pdf")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"\n  Combined plot saved to {combined_path}")

    # ── 数值汇总 ──────────────────────────────────────────────────────────────
    save_summary(
        out_path=os.path.join(args.out_dir, "robustness_summary.txt"),
        sigma_list=sigma_list,
        psnr_4ch_gauss=psnr_4ch_gauss,
        psnr_256ch_gauss=psnr_256ch_gauss,
        delta_list=delta_list,
        psnr_4ch_quant=psnr_4ch_quant,
        psnr_256ch_quant=psnr_256ch_quant,
        n_images=len(images),
        n_trials=args.n_trials,
    )

    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
