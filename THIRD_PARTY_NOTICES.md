# Third-party notices

This repository contains adapted research code. The root LICENSE was already present with the notice `Copyright (c) 2025 LuizScarlet`; it is preserved, not reassigned to the paper authors. No new copyright holder is asserted pending author confirmation.

| Component | Source / attribution | License and status |
| --- | --- | --- |
| Core codec and training | StableCodec, “Taming One-Step Diffusion for Extreme Image Compression” (ICCV 2025) | Local files describe StableCodec ancestry. Upstream [LuizScarlet/StableCodec](https://github.com/LuizScarlet/StableCodec), MIT; license copy in `licenses/StableCodec.txt`. |
| `ELIC/model/` | ELIC auxiliary encoder / checkerboard implementation | All three files are byte-identical to the StableCodec distribution at commit `b19401c3f5fc77bacf53c4ea07c08e4d3fc303b8`, whose root license is MIT. Preserve its notice; independent original ELIC provenance is not specified in these files. |
| `src/vision_aided_loss/` | [vision-aided-gan](https://github.com/nupurkmr9/vision-aided-gan), Nupur Kumari et al. | Upstream MIT text retained in `licenses/vision-aided-gan.txt`. This release limits the backbone to DINO; original discriminator/loss layers are retained. Exact fork revision is not recorded. |
| DINO weights/code loaded via torch.hub | [facebookresearch/dino](https://github.com/facebookresearch/dino) | Upstream Apache-2.0; fetched separately, no weights included. |
| Evaluation patch metrics | [NeuralCompression](https://github.com/facebookresearch/NeuralCompression), Meta Platforms | Adapted evaluation script; upstream MIT text in `licenses/NeuralCompression.txt`. Added stem matching and SSIM. |
| Tiled VAE helper | `src/my_utils/vaehook.py` (retains original comments) | Byte-identical to the same StableCodec commit; header credits LI YI (2023-03-02) under MIT. Original header retained. |
| SD-Turbo | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | Separately obtained model; consult its model card and license. This project's MIT file does not license model weights. |
| CompressAI, diffusers, transformers, PEFT, LPIPS and other packages | Respective upstream distributions | Installed dependencies retain their own licenses. |

No dataset, pretrained weight, external paper PDF, or paper source is bundled. Do not remove existing upstream notices when completing attribution. Verified upstream license copies do not imply that every local file has been traced to an upstream commit.

Source verification: Git blob hashes for the three ELIC files and VAE hook matched StableCodec commit `b19401c3f5fc77bacf53c4ea07c08e4d3fc303b8`. Upstream license texts were fetched during this cleanup.
