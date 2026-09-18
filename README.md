# StableCodec

A high-fidelity neural image compression framework that leverages pre-trained diffusion models (SD-Turbo) to achieve extreme compression ratios while maintaining high perceptual quality. StableCodec uses one-step diffusion denoising as a powerful decoder, enabling it to reconstruct fine-grained details at very low bitrates where traditional codecs produce significant artifacts.

## Architecture

```
Encoder:   Input Image → VAE Encoder → Latent Representation
                                            ↓
Bottleneck:                    LatentCodec (Entropy Bottleneck)
                                            ↓
                                   Quantized Latent → Compressed Bitstream
                                            ↓
Decoder:   Compressed Bitstream → LatentCodec Decoder → Reconstructed Latent
                                            ↓
                              One-Step UNet Denoising (LoRA-enhanced)
                                            ↓
                                  VAE Decoder → Reconstructed Image
```

Key components:
- **SD-Turbo backbone**: Pre-trained VAE and UNet from Stability AI's SD-Turbo, fine-tuned with LoRA adapters
- **LatentCodec**: Hyper-prior based entropy coding module with InceptionNeXt blocks for latent compression
- **ELIC auxiliary encoder**: Provides additional structural guidance during compression
- **Tiled VAE**: Enables processing of high-resolution images (up to 8K) on limited VRAM by splitting into overlapping tiles

## Project Structure

```
├── configs/
│   ├── base.yaml          # Shared settings (patch size, workers, seed)
│   ├── stage0.yaml        # Stage 0: Initial latent codec training
│   ├── stage1.yaml        # Stage 1: Rate-distortion fine-tuning
│   ├── stage2.yaml        # Stage 2: GAN-based perceptual fine-tuning
│   ├── test.yaml          # Inference/testing configuration
│   └── val.yaml           # Validation settings
├── ELIC/
│   └── model/             # ELIC auxiliary encoder implementation
├── src/
│   ├── StableCodec.py     # Main model: integrates VAE, UNet, LoRA, LatentCodec
│   ├── latent_codec.py    # Entropy-constrained bottleneck for latent space
│   ├── model.py           # SD-Turbo scheduler and LoRA forward pass utilities
│   ├── train.py           # Multi-stage distributed training entry point
│   ├── test.py            # Testing with config-based evaluation
│   ├── inference.py       # Image compression and decompression
│   ├── evaluate.py        # Metric evaluation (PSNR, MS-SSIM, LPIPS, FID, KID)
│   ├── compress_utils.py  # Bitstream read/write utilities
│   ├── color_fix.py       # AdaIN-based color correction post-processing
│   ├── loss/              # Rate-distortion and perceptual loss functions
│   └── my_utils/          # Dataset loading, tiled VAE, training utilities
├── compress.sh            # Batch compression script
├── eval_folders.sh        # Batch evaluation script
├── requirements.txt       # Python dependencies
└── LICENSE                # MIT License
```

## Requirements

- Python 3.10+
- CUDA-compatible GPU (24GB+ VRAM recommended for training)
- Pre-trained model weights:
  - [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)
  - ELIC official checkpoint
  - CLIP ViT-B/32 (for CLIP loss during training)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Pre-trained Weights

Before training or inference, download the following:

| Weight | Description |
|--------|-------------|
| `sd-turbo` | Stability AI's SD-Turbo diffusion model |
| `elic_official.pth` | Pre-trained ELIC encoder weights |
| `clip-vit-base-patch32` | OpenAI CLIP model (for training loss only) |

Update the paths in the corresponding config files under `configs/`.

## Training

Training follows a three-stage progressive pipeline. All stages support Distributed Data Parallel (DDP).

### Stage 0 — Latent Codec Pre-training

Trains the LatentCodec module with high MSE weight for stable initialization:

```bash
torchrun --nproc_per_node=<NUM_GPUS> src/train.py --stage 0
```

### Stage 1 — Rate-Distortion Optimization

Fine-tunes the full pipeline with rate-distortion loss (MSE + LPIPS + CLIP):

```bash
torchrun --nproc_per_node=<NUM_GPUS> src/train.py --stage 1
```

### Stage 2 — GAN Fine-tuning

Adds adversarial loss for improved perceptual quality:

```bash
torchrun --nproc_per_node=<NUM_GPUS> src/train.py --stage 2
```

### Configuration

Each stage has a dedicated config file in `configs/`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `lambda` | Rate-distortion trade-off weight |
| `learning_rate` | Main optimizer learning rate |
| `lora_rank_unet` | LoRA rank for UNet (default: 32) |
| `lora_rank_vae` | LoRA rank for VAE encoder (default: 16) |
| `global_batch_size` | Total batch size across all GPUs |
| `patch_size` | Training crop size (default: 512x512) |
| `precision` | Training precision (`bf16` / `fp16` / `None`) |

## Inference

### Compress and Decompress

```bash
python src/inference.py \
    --sd_path <PATH_TO_SD_TURBO> \
    --elic_path <PATH_TO_ELIC_CHECKPOINT> \
    --codec_path <PATH_TO_STABLECODEC_CHECKPOINT> \
    --img_path <PATH_TO_INPUT_IMAGES> \
    --rec_path <PATH_TO_SAVE_RECONSTRUCTIONS> \
    --bin_path <PATH_TO_SAVE_BITSTREAMS>
```

Or use the provided script:

```bash
bash compress.sh
```

Optional flag `--color_fix` enables AdaIN-based color correction for the reconstructed images.

### Config-based Testing

```bash
python src/test.py
```

Edit `configs/test.yaml` to specify the model checkpoint and test dataset paths.

## Evaluation

Evaluate reconstruction quality against ground truth:

```bash
python src/evaluate.py \
    --recon_dir <PATH_TO_RECONSTRUCTIONS> \
    --gt_dir <PATH_TO_GROUND_TRUTH>
```

Supported metrics:

| Metric | Type | Description |
|--------|------|-------------|
| PSNR | Distortion | Peak Signal-to-Noise Ratio |
| MS-SSIM | Distortion | Multi-Scale Structural Similarity |
| LPIPS | Perceptual | Learned Perceptual Image Patch Similarity |
| DISTS | Perceptual | Deep Image Structure and Texture Similarity |
| FID | Distribution | Fréchet Inception Distance |
| KID | Distribution | Kernel Inception Distance |

## License

This project is released under the [MIT License](LICENSE).
