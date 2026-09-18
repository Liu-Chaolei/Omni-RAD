# Repository Guidelines

## Project Structure & Module Organization

This repository contains StableCodec, a diffusion-based image compression system. Core Python code lives in `src/`: model definitions use `StableCodec*.py`, entropy bottlenecks use `latent_codec*.py`, and executable workflows use `train*.py`, `test*.py`, `compress_ori.py`, `inference.py`, and `evaluate.py`. Shared utilities are in `src/my_utils/`; GAN/perceptual loss helpers are in `src/vision_aided_loss/`. ELIC auxiliary encoder code is under `ELIC/model/`, and DCAE code is under `DCAE/`. YAML configurations are in `configs/`; generated metrics, logs, and sweep outputs belong in `results/`. Shell wrappers such as `compress.sh`, `eval_folders.sh`, `run_stage0_sweep.sh`, and `run_stage2_sweep.sh` should be run from the repository root.

## Build, Test, and Development Commands

Create the expected Python environment with:

```bash
conda create -n stablecodec python=3.10
conda activate stablecodec
pip install -r requirements.txt
```

Run training with config-driven stages, for example:

```bash
torchrun --nproc_per_node=4 src/train.py --stage 1
```

Common local workflows:

```bash
python src/test.py
python src/evaluate.py --recon_dir <rec_dir> --gt_dir <gt_dir>
bash compress_ori.sh
CUDA_VISIBLE_DEVICES=0 bash run_stage0_sweep.sh --max_images 24
```

Update checkpoint, dataset, and model paths in `configs/*.yaml` before running GPU workflows.

## Coding Style & Naming Conventions

Use Python 3.10-compatible code, 4-space indentation, and descriptive snake_case for functions, variables, and scripts. Keep module naming consistent with existing experiment variants, for example `StableCodec_variable2_step_A1.py` and `latent_codec_variable2_step.py`. Prefer config-driven parameters over hard-coded paths. Avoid committing generated caches such as `__pycache__/`, model checkpoints, bitstreams, or large datasets.

## Testing Guidelines

There is no dedicated pytest suite in this repository. Treat `src/test.py`, `src/evaluate.py`, compression scripts, and small sweep runs as regression checks. For risky model or codec changes, verify at least one short compression/evaluation path and record metrics such as PSNR, MS-SSIM, LPIPS, bpp, FID, or KID when relevant. Use small inputs or `--max_images` options for quick validation before launching full GPU jobs.

## Commit & Pull Request Guidelines

Recent history uses terse commit subjects, but contributors should use clear imperative summaries such as `fix latent codec checkpoint loading` or `add stage2 sweep report`. Pull requests should describe changed training/evaluation behavior, list edited configs, note required checkpoints or datasets, and include metric deltas or sample outputs for compression-quality changes. Add screenshots or reconstructed image examples when visual quality is affected.

## Security & Configuration Tips

Do not commit local absolute paths, API tokens, W&B credentials, pretrained weights, or dataset copies. Keep machine-specific paths in local config edits or command-line arguments, and document any new required external weights in `README.md` or the relevant config comments.
