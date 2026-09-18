# Release validation

## What changed

The public training/testing entries now select the A0 λ-FiLM + SNR-timestep model. Unrelated research branches, tracked bytecode, raw logs and the reference PDF were removed. The fixed-rate 4/256-channel baseline remains for optional evaluation. The separate manuscript directory was not edited.

Runtime fixes made during cleanup:

- Removed the unsupported CFT `rho` argument from the A0 training call.
- Corrected decoder latent slices and tile accumulation to 256 channels; tile weights broadcast across prediction channels.
- Added a versioned `.ord` container with original dimensions, float16 λ and entropy streams. Encoder and decoder use the same representable λ; the decoder recomputes T from entropy scales. Only one image per bitstream is supported.
- Estimated bpp now uses original image area. Evaluation pairs files by stem instead of sorted position.
- DINO-only training no longer imports the unused local CLIP implementation. Disabled CLIP loss no longer loads a CLIP vision model.
- Extracted rate/timestep modules without changing their state-dict keys; entropy/backbone checkpoints still use the existing dictionary layout.

## Checks completed

- All retained Python files compile; all 8 YAML configurations parse.
- No remaining imports of deleted top-level model modules.
- Training, testing, compression and evaluation `--help` work without model weights.
- Both shell wrappers pass `bash -n`.
- Five unit tests pass: container round-trip, malformed/truncated stream rejection, single-image constraint, SNR timestep limits/batch independence, FiLM identity initialization and checkpoint-bound restoration.
- Isolated CPU execution of the actual decoder methods with stub codec/UNet/VAE components validates 256-channel dimensions on both tiled and untiled paths. This is not an end-to-end model or entropy-coder test.
- CPU conditioning tests used an existing environment with PyTorch 2.13.0, not the recommended training environment. No dependency environment was modified.
- ELIC's three source files and the VAE hook match the recorded StableCodec upstream Git blobs. License copies are included.

## Still required for numerical reproduction

This machine has no available CUDA device, and its available PyTorch environment lacks diffusers, CompressAI and the other model dependencies. Full imports, clean dependency installation, a training step, and an actual image compression/evaluation run have **not** been validated. No PSNR/LPIPS/bpp parity with paper results is claimed.

Before publishing numerical reproduction claims, supply the adapted 256-channel SD-Turbo directory, ELIC weights, final Omni-RAD checkpoint, exact training configurations and dataset protocol. Run at least one stage-1 and stage-2 training step, plus a one-image encode/decode in separate processes. Test three λ values and an image large enough to trigger tiling; confirm finite metrics, original output sizes, and actual file bpp. Compare the unchanged forward path against the pre-cleanup A0 checkpoint on the same inputs.

The example λ range and checkpoint paths must be replaced with the final experiment settings. FID/KID currently retain an inherited patch-based protocol; confirm comparability to the manuscript before reporting them. No paper-result summary is fabricated from the removed exploratory logs.

## Repository handling

No commit, push, Git-history rewrite, or GitHub publication was performed. Deleted tracked files remain recoverable from the original commit; committing this cleanup does not remove those files from earlier Git history. New files must be included when committing.

The existing `AGENTS.md` is intentionally unchanged under the earlier instruction not to modify it; some historical workflow examples there no longer apply. README is the current user-facing workflow guide. The original MIT copyright notice is retained as an upstream notice, with no newly assigned author name.
