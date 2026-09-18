"""
experiment_weight.py — Two-branch decoding weight analysis for variable-rate StableCodec
                        (StableCodec_variable2_step + latent_codec_variable2_step).

Implements the design described in src/experiment_weight.md:

  Records, for each λ ∈ lambda_list, per image:
    • scale            : 1 + tanh(unet_lora_proj(film_embed))   ∈ [0, 2]
    • aux_l2           : ||res1||_2                              (AuxDecoder branch)
    • unet_l2          : ||x0_pred_unet||_2                      (U-Net branch, post-DDPM)
    • model_pred_l2    : ||ε̂||_2                                 (raw U-Net noise output)
    • model_pred_s_l2  : ||ε̂ * scale||_2                         (scaled U-Net noise output)
    • aux_over_unet    : aux_l2 / unet_l2

  Goal:
    Show low-bitrate (large λ) drives the model toward the U-Net generative
    branch, and that the per-image scale parameter trends monotonically with λ.

  Expected:
    scale and aux_l2 / unet_l2 vary monotonically with λ.

Run:
    CUDA_VISIBLE_DEVICES=0 python src/experiment_weight.py \
        --base_config_file ./configs/base.yaml \
        --test_config_file ./configs/test.yaml \
        --out_dir          ./results/experiment_weight
"""

import os
import csv
import json
import math
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from torch_ema import ExponentialMovingAverage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def l2_per_image(t: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm of a [B, ...] tensor → [B]."""
    return t.flatten(1).float().norm(dim=1)


# ---------------------------------------------------------------------------
# Forward pass with intermediate-tensor extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_with_branch_stats(net, img_padded, ori_h, ori_w, lmbda_tensor):
    """Re-implements StableCodec_variable2_step.forward(), exposing the
    AUX-branch and U-Net-branch latent contributions to x_denoised plus the
    per-image lora_scale.

    Returns a dict of [B]-shaped tensors (kept on CPU floats) plus scalars.
    """
    B = img_padded.shape[0]
    device = img_padded.device

    # ---- Encoder ----
    latent2 = net.aux_codec((img_padded + 1) / 2).detach()
    pos_caption_enc = torch.cat(
        [net.pos_caption_enc for _ in range(B)], dim=0
    ).to(device)
    lq_latent = net.vae.encode(img_padded).latent_dist.mode() * net.vae.config.scaling_factor

    # ---- Latent codec (returns res1, T_star) ----
    lq_latent_hat, rate_out, res1, T_star = net.codec(
        lq_latent, latent2, ori_h, ori_w, lmbda_tensor
    )

    # ---- Injection Point 3: lora_scale ∈ [0, 2] ----
    film_embed = net.codec.film_embed(lmbda_tensor)            # [B, FILM_DIM]
    delta_s    = net.unet_lora_proj(film_embed)                # [B, 1]
    lora_scale = (1.0 + torch.tanh(delta_s))                   # [B, 1]
    lora_scale_b = lora_scale.view(B, 1, 1, 1)

    # ---- One-step U-Net denoise with per-image T* ----
    t_star_long = T_star.long().to(device)
    model_pred = net.unet(
        lq_latent_hat, t_star_long,
        encoder_hidden_states=pos_caption_enc,
    ).sample
    model_pred_scaled = model_pred * lora_scale_b

    # x0_pred from U-Net branch only (no res1)
    x_unet = net._batched_ddpm_step(
        model_pred_scaled, t_star_long, lq_latent_hat[:, :256]
    )

    return {
        'scale':            lora_scale.squeeze(-1).detach().cpu().float(),     # [B]
        'aux_l2':           l2_per_image(res1).cpu(),                          # [B]
        'unet_l2':          l2_per_image(x_unet).cpu(),                        # [B]
        'model_pred_l2':    l2_per_image(model_pred).cpu(),                    # [B]
        'model_pred_s_l2':  l2_per_image(model_pred_scaled).cpu(),             # [B]
        't_star':           T_star.detach().cpu().float(),                     # [B]
        'bpp':              float(rate_out.quantized_total_bpp.item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(base_config, test_config, stage, out_dir):
    if base_config.get('global_seed') is not None:
        set_seed(base_config['global_seed'])

    os.makedirs(out_dir, exist_ok=True)

    from StableCodec_variable2_step import StableCodec

    net = StableCodec(
        sd_path=test_config['model']['sd_path'],
        config=test_config['model'],
        stage=stage,
    )

    if test_config['model'].get('codec_path') is not None and test_config.get('use_ema', False):
        checkpoint = torch.load(test_config['model']['codec_path'], map_location='cpu')
        ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
        ema_net.load_state_dict(checkpoint['ema_state_dict'])
        ema_net.copy_to(net.parameters())
        del checkpoint, ema_net

    net.cuda().eval()

    if test_config.get('enable_xformers_memory_efficient_attention', False):
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    device = next(net.parameters()).device

    # ---- Image set ----
    images = sorted(
        glob.glob(base_config['test_dataset'] + '/*.png')
        + glob.glob(base_config['test_dataset'] + '/*.jpg')
    )
    print(f"\nFound {len(images)} images in {base_config['test_dataset']}\n")

    # ---- λ values to evaluate ----
    lambda_min  = float(test_config['model'].get('lambda_min', 0.5))
    lambda_max  = float(test_config['model'].get('lambda_max', 32.0))
    num_lambdas = int(test_config.get('num_lambdas', 8))

    if test_config.get('lambda_list') is not None:
        lambda_list = [float(v) for v in test_config['lambda_list']]
    else:
        lambda_list = np.exp(np.linspace(
            np.log(lambda_min), np.log(lambda_max), num_lambdas
        )).tolist()

    print(f"Evaluating {len(lambda_list)} lambda values: "
          f"{[f'{v:.3f}' for v in lambda_list]}\n")

    # ---- Per-image CSV ----
    per_image_csv = os.path.join(out_dir, 'per_image.csv')
    f_csv = open(per_image_csv, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow([
        'lambda', 'image', 'scale', 'aux_l2', 'unet_l2',
        'aux_over_unet', 'model_pred_l2', 'model_pred_s_l2',
        't_star', 'bpp',
    ])

    summary = {}

    for lmbda_val in lambda_list:
        tag = f"lambda={lmbda_val:.3f}"
        print(f"\n{'='*72}\n  Evaluating {tag}\n{'='*72}")

        rec = {
            'scale': [], 'aux_l2': [], 'unet_l2': [], 'aux_over_unet': [],
            'model_pred_l2': [], 'model_pred_s_l2': [], 't_star': [], 'bpp': [],
        }

        for img_path in images:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
            ori_h, ori_w = img.shape[2:]

            stride_h, stride_w = base_config.get('model_stride', [64, 64])
            pad_h = (math.ceil(ori_h / stride_h)) * stride_h - ori_h
            pad_w = (math.ceil(ori_w / stride_w)) * stride_w - ori_w
            img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')
            _, _, H, W = img_padded.shape

            lmbda_tensor = torch.tensor([lmbda_val], dtype=torch.float32, device=device)

            try:
                stats = forward_with_branch_stats(net, img_padded, H, W, lmbda_tensor)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f'  CUDA OOM on {fname}, skipping.')
                    torch.cuda.empty_cache()
                    continue
                raise

            scale_v = float(stats['scale'][0])
            aux_v   = float(stats['aux_l2'][0])
            unet_v  = float(stats['unet_l2'][0])
            mp_v    = float(stats['model_pred_l2'][0])
            mp_s_v  = float(stats['model_pred_s_l2'][0])
            tstar_v = float(stats['t_star'][0])
            bpp_v   = float(stats['bpp'])
            ratio_v = aux_v / unet_v if unet_v > 0 else float('nan')

            rec['scale'].append(scale_v)
            rec['aux_l2'].append(aux_v)
            rec['unet_l2'].append(unet_v)
            rec['aux_over_unet'].append(ratio_v)
            rec['model_pred_l2'].append(mp_v)
            rec['model_pred_s_l2'].append(mp_s_v)
            rec['t_star'].append(tstar_v)
            rec['bpp'].append(bpp_v)

            writer.writerow([
                f'{lmbda_val:.6f}', fname,
                f'{scale_v:.6f}', f'{aux_v:.6f}', f'{unet_v:.6f}',
                f'{ratio_v:.6f}', f'{mp_v:.6f}', f'{mp_s_v:.6f}',
                f'{tstar_v:.2f}', f'{bpp_v:.6f}',
            ])
            f_csv.flush()

            print(f'  {fname:<24s}  scale={scale_v:.4f}  '
                  f'aux={aux_v:8.2f}  unet={unet_v:8.2f}  '
                  f'aux/unet={ratio_v:6.4f}  T*={tstar_v:6.1f}  bpp={bpp_v:.4f}')

        if not rec['scale']:
            print(f'  [WARNING] no images succeeded for {tag}')
            continue

        agg = {k: float(np.mean(v)) for k, v in rec.items()}
        agg_std = {k: float(np.std(v)) for k, v in rec.items()}
        summary[f'{lmbda_val:.6f}'] = {
            'lambda': lmbda_val,
            'mean':   agg,
            'std':    agg_std,
            'n':      len(rec['scale']),
        }

        print(f'\n  [{tag}] mean: '
              f'scale={agg["scale"]:.4f}  '
              f'aux_l2={agg["aux_l2"]:.2f}  '
              f'unet_l2={agg["unet_l2"]:.2f}  '
              f'aux/unet={agg["aux_over_unet"]:.4f}  '
              f'bpp={agg["bpp"]:.4f}  T*={agg["t_star"]:.1f}')

    f_csv.close()

    # ---- Summary table ----
    print(f'\n\n{"="*100}')
    print(f'  Two-branch Decoding Weight Summary')
    print(f'{"="*100}')
    header = (f'  {"lambda":>10s}  {"scale":>8s}  {"aux_l2":>10s}  {"unet_l2":>10s}  '
              f'{"aux/unet":>10s}  {"mp_l2":>10s}  {"mp_s_l2":>10s}  '
              f'{"T*":>6s}  {"bpp":>8s}')
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for lmbda_val in lambda_list:
        key = f'{lmbda_val:.6f}'
        if key not in summary:
            continue
        m = summary[key]['mean']
        print(f'  {lmbda_val:10.3f}  {m["scale"]:8.4f}  '
              f'{m["aux_l2"]:10.2f}  {m["unet_l2"]:10.2f}  '
              f'{m["aux_over_unet"]:10.4f}  {m["model_pred_l2"]:10.2f}  '
              f'{m["model_pred_s_l2"]:10.2f}  '
              f'{m["t_star"]:6.1f}  {m["bpp"]:8.4f}')
    print(f'{"="*100}\n')

    # ---- Monotonicity diagnostic ----
    sorted_keys = sorted(summary.keys(), key=lambda k: summary[k]['lambda'])
    if len(sorted_keys) >= 2:
        scales  = [summary[k]['mean']['scale']         for k in sorted_keys]
        ratios  = [summary[k]['mean']['aux_over_unet'] for k in sorted_keys]
        lambdas = [summary[k]['lambda']                for k in sorted_keys]
        log_l = np.log(np.asarray(lambdas))
        corr_scale = float(np.corrcoef(log_l, np.asarray(scales))[0, 1])
        corr_ratio = float(np.corrcoef(log_l, np.asarray(ratios))[0, 1])
        print(f'  Pearson corr(log λ, scale)         = {corr_scale:+.4f}')
        print(f'  Pearson corr(log λ, aux/unet)      = {corr_ratio:+.4f}\n')
        summary['_corr_logλ_scale']    = corr_scale
        summary['_corr_logλ_aux_unet'] = corr_ratio

    # ---- Save aggregated summary ----
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Per-image CSV  →  {per_image_csv}')
    print(f'  Summary JSON   →  {os.path.join(out_dir, "summary.json")}')

    # ---- Optional plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sorted_keys = sorted(
            [k for k in summary if not k.startswith('_')],
            key=lambda k: summary[k]['lambda'],
        )
        if sorted_keys:
            xs    = [summary[k]['lambda']                 for k in sorted_keys]
            scs   = [summary[k]['mean']['scale']          for k in sorted_keys]
            ax_l  = [summary[k]['mean']['aux_l2']         for k in sorted_keys]
            un_l  = [summary[k]['mean']['unet_l2']        for k in sorted_keys]
            ratio = [summary[k]['mean']['aux_over_unet']  for k in sorted_keys]

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(xs, scs, 'o-')
            axes[0].set_xscale('log'); axes[0].set_xlabel('λ'); axes[0].set_ylabel('scale')
            axes[0].set_title('lora_scale vs λ'); axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 2)

            axes[1].plot(xs, ax_l, 'o-', label='AUX  L2')
            axes[1].plot(xs, un_l, 's-', label='U-Net L2')
            axes[1].set_xscale('log'); axes[1].set_yscale('log')
            axes[1].set_xlabel('λ'); axes[1].set_ylabel('L2 norm')
            axes[1].set_title('Branch L2 norms vs λ')
            axes[1].legend(); axes[1].grid(True, alpha=0.3)

            axes[2].plot(xs, ratio, 'o-')
            axes[2].set_xscale('log'); axes[2].set_xlabel('λ')
            axes[2].set_ylabel('AUX / U-Net'); axes[2].set_title('Branch ratio vs λ')
            axes[2].grid(True, alpha=0.3)

            fig.tight_layout()
            plot_path = os.path.join(out_dir, 'weight_vs_lambda.png')
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f'  Plot           →  {plot_path}')
    except ImportError:
        print('  (matplotlib not available — skipping plot)')


# CUDA_VISIBLE_DEVICES=0 python src/experiment_weight.py --stage 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-branch decoding weight experiment for variable-rate StableCodec.")
    parser.add_argument("--stage",            type=int, default=1)
    parser.add_argument("--base_config_file", type=str, default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str, default="./configs/test.yaml")
    parser.add_argument("--out_dir",          type=str,
                        default="./results/experiment_weight")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config_file)
    test_config = load_yaml(args.test_config_file)

    main(base_config, test_config, stage=args.stage, out_dir=args.out_dir)
