"""
experiment_timesteps.py — Correlation between SNR-derived T and empirical optimal T
                          for variable-rate StableCodec (variable2_step variant).

Implements the design described in src/experiment_timesteps.md:

  Goal:
    Validate that the analytical T computed by DynamicTimestepModule
    (SNR-based, derived from the entropy-model scales σ) is close to the
    *empirically optimal* T found by exhaustive search over candidate
    timesteps for the one-step UNet denoiser.

  Procedure (per image × per λ):
    1. Run encoder + codec exactly once to obtain
         • lq_latent_hat  (the noisy latent fed to the UNet)
         • res1           (the AuxDecoder skip)
         • scales_all     (entropy-model σ field)
         • T_calc         (SNR-derived T* from DynamicTimestepModule)
    2. Sweep T over a candidate grid (defaults span [t_min, t_max] of the
       DynamicTimestepModule, evenly spaced in integer steps).
    3. For each candidate T, run one UNet forward + DDPM one-step + VAE
       decode, then evaluate distortion (LPIPS by default, PSNR also logged).
    4. T_opt = argmin_T  distortion(T).

  Reported correlations (across all (image, λ) samples):
    • Pearson  corr(T_calc, T_opt)
    • Spearman corr(T_calc, T_opt)
    • Pearson  corr(log λ , T_opt)   (sanity check — should also be high)
    • Per-λ Pearson corr(T_calc, T_opt)

Run:
    CUDA_VISIBLE_DEVICES=0 python src/experiment_timesteps.py \
        --base_config_file ./configs/base.yaml \
        --test_config_file ./configs/test.yaml \
        --out_dir          ./results/experiment_timesteps \
        --metric           lpips \
        --t_step           5
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


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return safe_corr(rx, ry)


# ---------------------------------------------------------------------------
# Codec forward (single pass) — produces everything we need to vary T later
# ---------------------------------------------------------------------------

@torch.no_grad()
def codec_forward(net, img_padded, ori_h, ori_w, lmbda_tensor):
    """Run encoder + LatentCodec once, return all tensors needed for the
    downstream T-sweep without re-encoding.

    Returns dict with:
        lq_latent_hat  [B, 320, h, w]   gs output (the UNet input lT)
        res1           [B, 256, h, w]   AuxDecoder skip
        scales_all     [B, 320, h, w]   entropy-model σ field
        T_calc         [B]              SNR-derived T* (analytical)
        bpp            float            quantised total bpp
        film_embed     [B, FILM_DIM]    λ embedding (for Injection 3)
    """
    B = img_padded.shape[0]
    device = img_padded.device

    latent2 = net.aux_codec((img_padded + 1) / 2).detach()
    lq_latent = net.vae.encode(img_padded).latent_dist.mode() * net.vae.config.scaling_factor

    lq_latent_hat, rate_out, res1, T_calc = net.codec(
        lq_latent, latent2, ori_h, ori_w, lmbda_tensor
    )

    film_embed = net.codec.film_embed(lmbda_tensor)
    return {
        'lq_latent_hat': lq_latent_hat.detach(),
        'res1':          res1.detach(),
        'T_calc':        T_calc.detach(),
        'bpp':           float(rate_out.quantized_total_bpp.item()),
        'film_embed':    film_embed.detach(),
    }


# ---------------------------------------------------------------------------
# Decode at a given T — UNet forward + DDPM one-step + VAE decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_at_timestep(net, ctx, t_value: int, pos_caption_enc):
    """Run the UNet + DDPM step + VAE decode with timestep == t_value (scalar).

    ctx: dict from codec_forward.
    Returns reconstruction in [-1, 1], shape [B, 3, H, W].
    """
    device = ctx['lq_latent_hat'].device
    B = ctx['lq_latent_hat'].shape[0]

    delta_s = net.unet_lora_proj(ctx['film_embed'])               # [B, 1]
    lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)

    t_long = torch.full((B,), int(t_value), dtype=torch.long, device=device)

    model_pred = net.unet(
        ctx['lq_latent_hat'], t_long,
        encoder_hidden_states=pos_caption_enc,
    ).sample
    model_pred = model_pred * lora_scale

    x_denoised = (
        net._batched_ddpm_step(model_pred, t_long, ctx['lq_latent_hat'][:, :256])
        + ctx['res1']
    )
    output_image = net.vae.decode(
        x_denoised / net.vae.config.scaling_factor
    ).sample.clamp(-1, 1)
    return output_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(base_config, test_config, stage, out_dir, metric_name, t_step,
         t_min_override=None, t_max_override=None):
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
        ckpt = torch.load(test_config['model']['codec_path'], map_location='cpu')
        ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
        ema_net.load_state_dict(ckpt['ema_state_dict'])
        ema_net.copy_to(net.parameters())
        del ckpt, ema_net

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

    # ---- Quality metrics ----
    import pyiqa
    iqa_psnr = pyiqa.create_metric('psnr', device=device)
    if metric_name.lower() == 'lpips':
        import lpips
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.requires_grad_(False)

        def distortion_fn(x_hat_m11, gt_m11):
            return loss_fn_lpips(x_hat_m11, gt_m11).mean().item()
    elif metric_name.lower() == 'mse':
        def distortion_fn(x_hat_m11, gt_m11):
            return F.mse_loss(x_hat_m11, gt_m11).item()
    elif metric_name.lower() == 'psnr':
        # higher PSNR = better, so we negate so argmin == best
        def distortion_fn(x_hat_m11, gt_m11):
            x01 = ((x_hat_m11 + 1) / 2).clamp(0, 1)
            g01 = ((gt_m11    + 1) / 2).clamp(0, 1)
            return -float(iqa_psnr(x01, g01).mean().item())
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    # ---- Image set ----
    images = sorted(
        glob.glob(base_config['test_dataset'] + '/*.png')
        + glob.glob(base_config['test_dataset'] + '/*.jpg')
    )
    print(f"\nFound {len(images)} images in {base_config['test_dataset']}\n")

    # ---- λ values ----
    lambda_min = float(test_config['model'].get('lambda_min', 0.5))
    lambda_max = float(test_config['model'].get('lambda_max', 32.0))
    num_lambdas = int(test_config.get('num_lambdas', 6))

    if test_config.get('lambda_list') is not None:
        lambda_list = [float(v) for v in test_config['lambda_list']]
    else:
        lambda_list = np.exp(np.linspace(
            np.log(lambda_min), np.log(lambda_max), num_lambdas
        )).tolist()

    print(f"Evaluating {len(lambda_list)} lambda values: "
          f"{[f'{v:.3f}' for v in lambda_list]}\n")

    # ---- Candidate timestep grid ----
    dyn = net.codec.dynamic_timestep
    t_min_grid = int(t_min_override if t_min_override is not None else dyn.t_min)
    t_max_grid = int(t_max_override if t_max_override is not None else dyn.t_max)
    if t_max_grid <= t_min_grid:
        raise ValueError(
            f"t_max ({t_max_grid}) must be > t_min ({t_min_grid})")
    t_step = max(1, int(t_step))
    candidate_ts = list(range(t_min_grid, t_max_grid + 1, t_step))
    if candidate_ts[-1] != t_max_grid:
        candidate_ts.append(t_max_grid)
    print(f"T candidate grid: {len(candidate_ts)} values in "
          f"[{t_min_grid}, {t_max_grid}] step={t_step}\n")

    # ---- Per-image CSV ----
    per_image_csv = os.path.join(out_dir, 'per_image.csv')
    f_csv = open(per_image_csv, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow([
        'lambda', 'image', 'T_calc', 'T_opt', 'best_distortion',
        'distortion_at_T_calc', 'bpp',
    ])

    # ---- Per-(image, λ, T) sweep CSV ----
    sweep_csv = os.path.join(out_dir, 'sweep.csv')
    f_sweep = open(sweep_csv, 'w', newline='')
    sweep_writer = csv.writer(f_sweep)
    sweep_writer.writerow(['lambda', 'image', 'T', 'distortion', 'psnr'])

    # ---- Pre-compute caption encoding once ----
    pos_caption_enc_single = net.pos_caption_enc                  # [1, L, D]

    all_records = []

    for lmbda_val in lambda_list:
        tag = f"lambda={lmbda_val:.3f}"
        print(f"\n{'='*72}\n  Evaluating {tag}\n{'='*72}")

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
                ctx = codec_forward(net, img_padded, H, W, lmbda_tensor)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f'  CUDA OOM (codec) on {fname}, skipping.')
                    torch.cuda.empty_cache()
                    continue
                raise

            B = ctx['lq_latent_hat'].shape[0]
            pos_caption_enc = torch.cat(
                [pos_caption_enc_single for _ in range(B)], dim=0
            ).to(device)

            T_calc = float(ctx['T_calc'][0].item())

            distortions = []
            psnrs = []
            for t_val in candidate_ts:
                try:
                    x_hat = decode_at_timestep(net, ctx, t_val, pos_caption_enc)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f'  CUDA OOM (decode) at T={t_val} on {fname}, skipping T.')
                        torch.cuda.empty_cache()
                        distortions.append(float('inf'))
                        psnrs.append(float('nan'))
                        continue
                    raise

                x_hat_crop = x_hat[:, :, :ori_h, :ori_w]
                d_val = distortion_fn(x_hat_crop, img)
                # also compute PSNR for reference
                x01 = ((x_hat_crop + 1) / 2).clamp(0, 1)
                g01 = ((img        + 1) / 2).clamp(0, 1)
                p_val = float(iqa_psnr(x01, g01).mean().item())

                distortions.append(d_val)
                psnrs.append(p_val)
                sweep_writer.writerow([
                    f'{lmbda_val:.6f}', fname, t_val,
                    f'{d_val:.6f}', f'{p_val:.4f}',
                ])

            f_sweep.flush()

            d_arr = np.asarray(distortions, dtype=np.float64)
            best_idx = int(np.argmin(d_arr))
            T_opt = candidate_ts[best_idx]
            best_dist = float(d_arr[best_idx])

            # Distortion at the analytical T_calc — pick the nearest candidate
            calc_idx = int(np.argmin(np.abs(np.asarray(candidate_ts) - T_calc)))
            dist_at_calc = float(d_arr[calc_idx])

            writer.writerow([
                f'{lmbda_val:.6f}', fname,
                f'{T_calc:.2f}', T_opt,
                f'{best_dist:.6f}', f'{dist_at_calc:.6f}',
                f'{ctx["bpp"]:.6f}',
            ])
            f_csv.flush()

            all_records.append({
                'lambda':        lmbda_val,
                'image':         fname,
                'T_calc':        T_calc,
                'T_opt':         T_opt,
                'best_dist':     best_dist,
                'dist_at_calc':  dist_at_calc,
                'bpp':           ctx['bpp'],
            })

            print(f'  {fname:<24s}  λ={lmbda_val:7.3f}  '
                  f'T_calc={T_calc:6.1f}  T_opt={T_opt:4d}  '
                  f'd*={best_dist:.4f}  d(T_calc)={dist_at_calc:.4f}  '
                  f'gap={dist_at_calc - best_dist:+.4f}')

    f_csv.close()
    f_sweep.close()

    # ---- Aggregate & correlation analysis ----
    if not all_records:
        print('  [WARNING] no records produced — aborting.')
        return

    T_calc_all = np.asarray([r['T_calc']       for r in all_records])
    T_opt_all  = np.asarray([r['T_opt']        for r in all_records], dtype=np.float64)
    lam_all    = np.asarray([r['lambda']       for r in all_records])
    log_lam    = np.log(np.clip(lam_all, 1e-12, None))
    gap_all    = np.asarray([r['dist_at_calc'] - r['best_dist'] for r in all_records])

    pearson_calc_opt  = safe_corr(T_calc_all, T_opt_all)
    spearman_calc_opt = spearman_corr(T_calc_all, T_opt_all)
    pearson_logl_opt  = safe_corr(log_lam, T_opt_all)
    pearson_logl_calc = safe_corr(log_lam, T_calc_all)

    per_lambda_corr = {}
    for lv in sorted(set(lam_all.tolist())):
        m = lam_all == lv
        if m.sum() >= 2:
            per_lambda_corr[f'{lv:.6f}'] = {
                'n':        int(m.sum()),
                'pearson':  safe_corr(T_calc_all[m], T_opt_all[m]),
                'spearman': spearman_corr(T_calc_all[m], T_opt_all[m]),
                'mean_T_calc': float(T_calc_all[m].mean()),
                'mean_T_opt':  float(T_opt_all[m].mean()),
                'mean_gap':    float(gap_all[m].mean()),
            }

    summary = {
        'n_samples':              len(all_records),
        'metric':                 metric_name,
        't_grid':                 candidate_ts,
        'pearson_corr_calc_opt':  pearson_calc_opt,
        'spearman_corr_calc_opt': spearman_calc_opt,
        'pearson_corr_logL_opt':  pearson_logl_opt,
        'pearson_corr_logL_calc': pearson_logl_calc,
        'mean_gap':               float(gap_all.mean()),
        'median_gap':             float(np.median(gap_all)),
        'mean_abs_T_diff':        float(np.mean(np.abs(T_calc_all - T_opt_all))),
        'per_lambda':             per_lambda_corr,
    }

    print(f'\n\n{"="*72}')
    print(f'  Timestep correlation summary  (metric={metric_name}, n={len(all_records)})')
    print(f'{"="*72}')
    print(f'  Pearson  corr(T_calc, T_opt)  = {pearson_calc_opt:+.4f}')
    print(f'  Spearman corr(T_calc, T_opt)  = {spearman_calc_opt:+.4f}')
    print(f'  Pearson  corr(log λ, T_opt)   = {pearson_logl_opt:+.4f}')
    print(f'  Pearson  corr(log λ, T_calc)  = {pearson_logl_calc:+.4f}')
    print(f'  mean |T_calc - T_opt|         = {summary["mean_abs_T_diff"]:.2f}')
    print(f'  mean distortion gap (T_calc vs T_opt) = {summary["mean_gap"]:+.6f}')
    print(f'\n  Per-λ correlations:')
    for k, v in per_lambda_corr.items():
        print(f'    λ={float(k):8.3f}  n={v["n"]:3d}  '
              f'pearson={v["pearson"]:+.4f}  spearman={v["spearman"]:+.4f}  '
              f'mean_T_calc={v["mean_T_calc"]:6.1f}  mean_T_opt={v["mean_T_opt"]:6.1f}')
    print(f'{"="*72}\n')

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'  Per-image CSV  →  {per_image_csv}')
    print(f'  Sweep CSV      →  {sweep_csv}')
    print(f'  Summary JSON   →  {os.path.join(out_dir, "summary.json")}')

    # ---- Optional plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter T_calc vs T_opt, colour-coded by log λ
        sc = axes[0].scatter(T_calc_all, T_opt_all, c=log_lam, cmap='viridis',
                             s=24, alpha=0.75, edgecolor='k', linewidth=0.3)
        lo = min(T_calc_all.min(), T_opt_all.min()) - 5
        hi = max(T_calc_all.max(), T_opt_all.max()) + 5
        axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1, label='y = x')
        axes[0].set_xlabel('T_calc  (SNR-derived)')
        axes[0].set_ylabel('T_opt   (exhaustive)')
        axes[0].set_title(
            f'T_calc vs T_opt  '
            f'(Pearson={pearson_calc_opt:+.3f}, Spearman={spearman_calc_opt:+.3f})')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        cbar = fig.colorbar(sc, ax=axes[0])
        cbar.set_label('log λ')

        # Bar of per-λ Pearson
        if per_lambda_corr:
            lams_sorted = sorted(per_lambda_corr.keys(), key=float)
            pears = [per_lambda_corr[k]['pearson'] for k in lams_sorted]
            axes[1].bar(range(len(lams_sorted)), pears, color='steelblue')
            axes[1].set_xticks(range(len(lams_sorted)))
            axes[1].set_xticklabels([f'{float(k):.2f}' for k in lams_sorted],
                                    rotation=45)
            axes[1].set_ylabel('Pearson corr(T_calc, T_opt)')
            axes[1].set_xlabel('λ')
            axes[1].axhline(0, color='k', lw=0.5)
            axes[1].set_title('Per-λ Pearson correlation')
            axes[1].grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        plot_path = os.path.join(out_dir, 'timestep_correlation.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f'  Plot           →  {plot_path}')
    except ImportError:
        print('  (matplotlib not available — skipping plot)')


# CUDA_VISIBLE_DEVICES=0 python src/experiment_timesteps.py --stage 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correlation between SNR-derived T and empirical optimal T.")
    parser.add_argument("--stage",            type=int, default=1)
    parser.add_argument("--base_config_file", type=str, default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str, default="./configs/test.yaml")
    parser.add_argument("--out_dir",          type=str,
                        default="./results/experiment_timesteps")
    parser.add_argument("--metric",           type=str, default="lpips",
                        choices=["lpips", "mse", "psnr"],
                        help="distortion metric to minimise when picking T_opt")
    parser.add_argument("--t_step",           type=int, default=5,
                        help="step size for the candidate-T sweep")
    parser.add_argument("--t_min", type=int, default=None,
                        help="override DynamicTimestepModule.t_min for the sweep")
    parser.add_argument("--t_max", type=int, default=None,
                        help="override DynamicTimestepModule.t_max for the sweep")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config_file)
    test_config = load_yaml(args.test_config_file)

    main(base_config, test_config, stage=args.stage, out_dir=args.out_dir,
         metric_name=args.metric, t_step=args.t_step,
         t_min_override=args.t_min, t_max_override=args.t_max)
