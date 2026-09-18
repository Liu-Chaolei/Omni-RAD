# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope Restrictions

- Only work on files within the current directory
- Do not modify any content in the parent directories

## Project Overview

**StableCodec** is a diffusion-based extreme image compression system. It uses SD-Turbo (one-step diffusion) as the generative backbone, combined with a learned latent codec and a frozen ELIC auxiliary encoder, to achieve ultra-low bitrate coding (< 0.05 bpp) with high fidelity.

Paper: [StableCodec: Taming One-Step Diffusion for Extreme Image Compression](https://arxiv.org/abs/2506.21977)

## Environment Setup

```bash
conda create -n stablecodec python=3.10
conda activate stablecodec
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.1.2, diffusers 0.25.1, compressai 1.2.6, peft (LoRA), accelerate, transformers 4.46.3.

## Commands

All scripts must be run from the repo root so that `src/` imports resolve correctly.

### Training

```bash
# Stage 1 (MSE + LPIPS + CLIP loss, no GAN)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 src/trainv1.py \
    --stage 1 --MASTER_PORT 12355 --experiment_name <name>

# Stage 2 (adds GAN discriminator with DINO backbone)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 src/trainv1.py \
    --stage 2 --MASTER_PORT 12355 --experiment_name <name>
```

Config files are loaded automatically: `configs/base.yaml` + `configs/stage{N}.yaml` + `configs/val.yaml`. Key config fields to update before running: `model.sd_path`, `model.elic_path`, `model.codec_path`, `savepath`, dataset paths in `base.yaml`.

### Inference / Compression

```bash
bash compress_ori.sh
# or directly:
python src/compress_ori.py \
    --sd_path=<PATH_TO_SD_TURBO> \
    --elic_path=<PATH_TO_ELIC>/elic_official.pth \
    --img_path=<INPUT_DIR>/ \
    --rec_path=<OUTPUT_DIR>/rec/ \
    --bin_path=<OUTPUT_DIR>/bin/ \
    --codec_path=<PATH_TO_STABLECODEC>/stablecodec_ft2.pkl \
    # --color_fix  # recommended for high-res tiled inference
```

### Evaluation

```bash
python src/evaluate.py --recon_dir <rec_dir> --gt_dir <gt_dir>
# or: bash eval_folders.sh
```

### Analysis / Experiments

```bash
# SNR experiment: compare 4-ch vs 256-ch VAE fused latent SNR
python src/experiment_snr.py \
    --sd_path      /path/to/sd-turbo \
    --sd_path_256  /path/to/sd-turbo_256 \
    --elic_path    /path/to/elic_official.pth \
    --codec_path_4   /path/to/stablecodec_4ch.pth.tar \
    --codec_path_256 /path/to/stablecodec_ft2.pkl \
    --img_dir      /path/to/Kodak24/HR \
    --out_dir      results/snr_experiment

# Redundancy experiment: PCA rank, inter-channel correlation, BPP
python src/experiment_redundancy.py \
    --config_4ch configs/stage1.yaml \
    --config_256ch configs/stage2.yaml \
    --output_dir ./results/experiment3

# Quantization robustness experiment: manifold distortion test (analysis.txt)
# Injects Gaussian noise / uniform quantization into VAE latent space,
# compares PSNR-vs-noise curves for 4ch and 256ch VAE.
python src/experiment_quantization_robustness.py \
    --sd_path      /path/to/sd-turbo \
    --sd_path_256  /path/to/sd-turbo_256 \
    --codec_path_256 /path/to/stablecodec_ft2.pkl \
    --img_dir      /path/to/Kodak24/HR \
    --out_dir      results/quantization_robustness
```

## Architecture

### Full Forward Pipeline

```
Image x
  ├─► ELIC g_a (frozen) ─► aux_latent (B, 320, H/16, W/16)
  │
  └─► VAE encoder ─► lq_latent (B, C, H/8, W/8)
         C = 4   [latent_channels=4]
         C = 256 [latent_channels=256]
              │
              ▼
        LatentCodec.g_a (AnalysisTransform_4 or _256)
          pre1: Downsample(C→128) + pre2: Conv2d(320→64) → cat 192ch
          → BasicBlock → DS(256) → BasicBlock → DS(320) → BasicBlock
          ─► y  (B, 320, H/32, W/32)
              │
        HyperAnalysis(y) → z → EntropyBottleneck
        HyperSynthesis(z_hat) → base (B, 320, H/32, W/32)
        4-pass checkerboard context model (Adapter + SpatialContext + LRP)
          ─► y_hat (B, 320, H/32, W/32)
              │
        ┌─────┴──────────────┐
        ▼                    ▼
  g_s (SynthesisTransform)  aux (AuxDecoder)
  (B, 320, H/8, W/8)        (B, C, H/8, W/8) ── res1 skip ──┐
        │                                                      │
  UNet conv_in (320→320)                                       │
  UNet (+ LoRA) at timestep=999                                │
  sched.step(model_pred, t, lq_latent_hat[:, :C])             │
        └──────────── + res1 ────────────────────────────────-─┘
        ▼
  VAE decoder ─► output image
```

### Key Design Points

- **`latent_channels` switch**: In `configs/stage1.yaml` set `model.latent_channels: 4` for the original SD latent or `256` for the expanded VAE variant. This selects `AnalysisTransform_4`/`AuxDecoder_4` vs `AnalysisTransform_256`/`AuxDecoder_256` inside `LatentCodec`.
- **UNet `conv_in` replacement**: The standard 4ch→320ch conv is replaced with a 320ch→320ch conv (in `StableCodec_ori.py`) so `g_s`'s 320-ch output feeds directly into the UNet without a channel adapter.
- **`res1` skip connection**: `self.aux(y_hat)` produces a residual (B, C_vae, H/8, W/8) that is added to the denoised latent *before* VAE decoding — bypassing the UNet to preserve fine detail.
- **ELIC aux_codec is frozen**: Only `g_a` (analysis transform) of the pretrained ELIC model is used; `freeze_aux_encoder: True` in config.
- **4-pass checkerboard context**: Masks on a 2×2 spatial grid provide autoregressive spatial conditioning without sequential pixel-by-pixel decoding.

### Core Files

**`src/StableCodec_ori.py` — `StableCodec` (canonical model)**
- Wraps SD-Turbo: `tokenizer`, `text_encoder`, `vae`, `unet` with LoRA adapters on both (`lora_rank_vae=16`, `lora_rank_unet=32`)
- `self.codec` — `LatentCodec` (compresses the fused latent)
- `self.aux_codec` — ELIC `g_a`, frozen, maps image [0,1] → 320-ch feature
- Tiled VAE encoding/decoding via `VAEHook` for high-res images
- `save_model()` saves only LoRA weights + `conv_in` + `state_dict_codec`

**`src/latent_codec_ori.py` — `LatentCodec`**
- Dispatch on `latent_channels` selects the correct `AnalysisTransform` and `AuxDecoder`
- Entropy coding: `EntropyBottleneck` (hyper) + `GaussianConditional` (latent) from compressai
- Custom blocks: `InceptionDWConv2d`, `InceptionNeXt`, `GatedCNNBlock`, `BasicBlock`, `Downsample`, `Upsample`
- `TargetRateModule` computes rate loss weighted by `lambda_rate`
- Outputs `(x_hat, RateLossOutput, res)` where `res` is the AuxDecoder skip

**`src/trainv1.py` — Training loop**
- Two optimizers: main `AdamW` (LoRA + codec params) and `aux_optimizer` (entropy bottleneck `.quantiles`)
- Stage 1: MSE + LPIPS + CLIP loss + rate loss
- Stage 2: adds GAN discriminator (`vision_aided_loss.Discriminator` with DINO backbone)
- Training data: HDF5 (`base.yaml: hdf5_dataset`), validation from `ImageFolder`

### Config System

Layered YAML files in `configs/`:
- `base.yaml` — dataset paths, num_workers, patch size
- `stage1.yaml` / `stage2.yaml` — training hyperparams, loss weights, model paths, `latent_channels`
- `val.yaml` — validation frequency; `test.yaml` — inference settings

### Lambda / Bitrate Control

`lmbda` in stage configs controls the rate-distortion tradeoff:
- `stablecodec_ft2.pkl` → ~0.034 bpp (λ=2)
- `stablecodec_ft8.pkl` → ~0.018 bpp (λ=8)
- `stablecodec_ft32.pkl` → ~0.006 bpp (λ=32)

### Checkpoint Format

Saved by `StableCodec.save_model()`:
```python
{
    "state_dict_vae":   {...},  # only LoRA keys
    "state_dict_unet":  {...},  # LoRA keys + conv_in
    "state_dict_codec": {...},  # full LatentCodec weights
}
```
Shape-safe loading: mismatched keys are skipped (logged), so checkpoints from 4-ch and 256-ch variants can be partially loaded as warm-starts.

### Model Variants in `src/`

- `StableCodec_ori.py` + `latent_codec_ori.py` — **canonical model** used by `trainv1.py`
- `StableCodec_fusion*.py`, `StableCodec_variable*.py` — experimental variants
- `latent_codec.py`, `latent_codec_variable.py` — codec variants (no `ste_round` import difference from `_ori`)
- `experiment_snr.py` — validates 256-ch SNR hypothesis; see `analysis.txt` for theory
- `experiment_redundancy.py` — validates statistical compressibility of 256-ch latent
