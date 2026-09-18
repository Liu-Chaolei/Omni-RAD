# Omni-RAD

Research code for **Omni-RAD: Variable-Rate One-Step Diffusion Image Compression with Adaptive Timestep Selection** (working English title of the accompanying Chinese manuscript).

Omni-RAD combines a 256-channel VAE representation, continuous λ-FiLM rate conditioning, and an entropy-scale SNR proxy that selects a per-image diffusion timestep. The decoder performs one denoising step and adds an auxiliary latent residual. This implementation builds on StableCodec and ELIC.

## Release status

This tree contains the A0 paper route. CFT, PolicyNet, GLC, oracle searches, and other exploratory variants have been removed. Historical experiments remain in Git history. The separate paper source directory is not modified or distributed here.

**Trained weights and the adapted 256-channel SD-Turbo backbone are not included.** Download links, final training configurations, and numerical reproduction of the paper results still need to be supplied. The checked-in YAML files are examples, not certified final paper settings. See [release validation](docs/RELEASE_VALIDATION.md) for checks performed and outstanding requirements.

## Structure

- `src/StableCodec_variable2_step.py`: Omni-RAD model and single-step decoder.
- `src/latent_codec_variable2_step.py`: FiLM transforms and entropy coding; `src/rate_control.py` holds weight-free rate/timestep control modules.
- `src/train.py`, `src/test.py`, `src/compress.py`, `src/evaluate.py`: training, estimated-rate testing, actual bitstream coding, and image metrics.
- `src/StableCodec_ori.py`, `src/latent_codec_ori.py`, `src/test_baseline.py`: optional fixed-rate 4/256-channel baseline evaluation.
- `src/my_utils/`, `src/vision_aided_loss/`, `ELIC/model/`: training utilities, DINO discriminator, and auxiliary encoder.
- `configs/`: portable example configurations. Generated outputs go into ignored `results/`.

## Installation

Use Python 3.10 and a CUDA-capable PyTorch environment. Run commands from the repository root:

```bash
conda create -n omnirad python=3.10
conda activate omnirad
pip install -r requirements.txt
```

The dependency set retains PyTorch 2.1.2 / torchvision 0.16.2 and diffusers 0.25.1. A clean installation has not yet been validated. Install xformers separately only if a compatible build is available; it is disabled by default.

## Weights and data

Set paths in your local copies of the YAML files (for example `configs/local/`). Required artifacts:

| Artifact | Purpose |
| --- | --- |
| Adapted SD-Turbo directory | Hugging Face layout with tokenizer, text encoder, scheduler, 256-channel VAE, and UNet producing 256 channels; ordinary 4-channel SD-Turbo is not a substitute. |
| ELIC checkpoint | Frozen auxiliary analysis transform. |
| Omni-RAD checkpoint | Codec, VAE/UNet adaptations, and λ prediction scaling (`state_dict_lora_proj`). |
| DINO / LPIPS weights | Downloaded by the respective libraries for training and metrics. |

Match the backbone architecture and λ embedding bounds to the checkpoint. Existing loading code supports initialization from partial checkpoints; warnings about skipped tensors must be resolved before reporting results.

Training uses an HDF5 file with one RGB H×W×3 image array per root key, at least 512×512 pixels. Validation uses the `valid/` subdirectory under `train_dataset`. Testing takes PNG/JPG images from `base.yaml:test_dataset`. Obtain datasets separately; do not commit datasets or model weights.

## Training

The paper route uses per-image log-uniform λ sampling and `mean(D_i / λ_i) + mean(bpp_i)`, followed by DINO adversarial fine-tuning. CLIP loss is disabled in the example configurations to match the manuscript's MSE + VGG-LPIPS objective.

```bash
accelerate launch --num_processes 1 src/train.py --stage 1 --config_dir configs --experiment_name rd
accelerate launch --num_processes 1 src/train.py --stage 2 --config_dir configs --experiment_name gan
```

Configure the initialization checkpoint for stage 1 and the trained stage-1 checkpoint for stage 2. `stage0.yaml` is an optional cold-start configuration of the same trainer; it is not a separate validated latent-only pretraining recipe. Set `max_train_steps`, batch size, and paths before launching. W&B and compilation are disabled by default.

## Testing and real compression

Forward testing reports **estimated bpp**, PSNR, VGG-LPIPS and timestep statistics, with a JSON summary. Set `lambda_list` and optionally `max_images` in the test config.

```bash
python src/test.py --base_config_file configs/base.yaml --test_config_file configs/test.yaml
python src/compress.py --config configs/test.yaml --img_path data/kodak --bin_path results/bits --rec_path results/rec --lmbda 2 --max_images 1
python src/compress.py --config configs/test.yaml --mode decode --bin_path results/bits --rec_path results/decoded
python src/evaluate.py --recon_dir results/rec --gt_dir data/kodak
```

The new `.ord` container stores original dimensions, entropy shape, float16 λ and two entropy streams. Encoding uses the serialized λ; decoding recomputes the timestep from entropy scales without the original image. Real bpp includes the header and is divided by original image area. This container is versioned and does not read legacy StableCodec bitstreams. Use one λ per output directory and the same checkpoint/backbone for encoding and decoding.

Evaluation matches image stems. It reports PSNR, SSIM, MS-SSIM, DISTS and AlexNet-LPIPS; this LPIPS differs from the VGG metric used in forward testing. FID/KID retain the inherited NeuralCompression patch evaluation procedure and are only reported for more than 50 images; these values should not be equated to paper curves without confirming the original protocol.

Optional baseline evaluation (requires its own trained checkpoint):

```bash
python src/test_baseline.py --method channel --base_config_file configs/base.yaml --test_config_file configs/ch4.yaml
python src/test_baseline.py --method channel --base_config_file configs/base.yaml --test_config_file configs/ch256.yaml
```

## Validation and attribution

```bash
python -m unittest discover -s tests -v
python -m compileall -q src ELIC
```

See [third-party notices](THIRD_PARTY_NOTICES.md) and the existing [license](LICENSE). Existing copyright notices are preserved; no new author name has been assigned. The notices record verified upstream file matches and remaining attribution limitations. Publication metadata and a citation will be added once finalized.
