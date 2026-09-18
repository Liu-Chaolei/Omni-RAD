"""
testv1_variable.py — Testing script for variable-rate StableCodec checkpoints.

Evaluates the model at multiple λ values to produce a full RD curve.
Based on testv1.py, adapted for StableCodec_variable's lmbda interface.
"""

import os
import math
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import pyiqa
import lpips
from PIL import Image
from torchvision import transforms
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from torch_ema import ExponentialMovingAverage


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def main(base_config, test_config, stage):
    if base_config['global_seed'] is not None:
        set_seed(base_config['global_seed'])

    from StableCodec_variable import StableCodec

    net = StableCodec(
        sd_path=test_config['model']['sd_path'],
        config=test_config['model'],
        stage=stage,
    )

    if test_config['model']['codec_path'] is not None and test_config.get('use_ema', False):
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
    iqa_psnr = pyiqa.create_metric('psnr', device=device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_fn_lpips.requires_grad_(False)

    # Collect test images
    images = sorted(
        glob.glob(base_config['test_dataset'] + '/*.png')
        + glob.glob(base_config['test_dataset'] + '/*.jpg')
    )
    print(f"\nFound {len(images)} images in {base_config['test_dataset']}\n")

    # Lambda values to evaluate — log-spaced from lambda_min to lambda_max
    lambda_min = float(test_config['model'].get('lambda_min', 0.5))
    lambda_max = float(test_config['model'].get('lambda_max', 32.0))
    num_lambdas = test_config.get('num_lambdas', 6)
    lambda_list = np.exp(np.linspace(
        np.log(lambda_min), np.log(lambda_max), num_lambdas
    )).tolist()

    # Override with explicit list if provided
    if test_config.get('lambda_list') is not None:
        lambda_list = [float(v) for v in test_config['lambda_list']]

    print(f"Testing {len(lambda_list)} lambda values: "
          f"{[f'{v:.3f}' for v in lambda_list]}\n")

    all_results = {}

    for lmbda_val in lambda_list:
        tag = f"lambda={lmbda_val:.3f}"
        print(f"\n{'='*60}")
        print(f"  Evaluating {tag}")
        print(f"{'='*60}")

        rec_dir = os.path.join(test_config['rec_path'], f"lambda_{lmbda_val:.3f}")
        os.makedirs(rec_dir, exist_ok=True)

        psnr_meter = AverageMeter()
        lpips_meter = AverageMeter()
        bpp_list = []

        for img_path in images:
            print('[Processing]', img_path)
            fname = os.path.splitext(os.path.basename(img_path))[0]
            outf = os.path.join(rec_dir, fname + '.png')

            img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
            ori_h, ori_w = img.shape[2:]

            stride_h, stride_w = base_config.get('model_stride', [64, 64])
            pad_h = (math.ceil(ori_h / stride_h)) * stride_h - ori_h
            pad_w = (math.ceil(ori_w / stride_w)) * stride_w - ori_w
            img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')
            _, _, H, W = img_padded.shape

            with torch.no_grad():
                try:
                    pos_tag_prompt = [1]
                    lmbda_tensor = torch.tensor([lmbda_val],
                                                dtype=torch.float32,
                                                device=device)
                    x_hat, RateLossOutput = net(img_padded, pos_tag_prompt,
                                                H, W, lmbda=lmbda_tensor)

                    bpp = RateLossOutput.quantized_total_bpp.item()

                    x_hat_crop = x_hat[:, :, :ori_h, :ori_w]
                    img_crop = img

                    x_hat_01 = ((x_hat_crop + 1) / 2).clamp(0, 1)
                    gt_01 = ((img_crop + 1) / 2).clamp(0, 1)

                    cur_psnr = iqa_psnr(x_hat_01, gt_01).mean().item()
                    cur_lpips = loss_fn_lpips(x_hat_crop, img_crop).mean().item()

                    psnr_meter.update(cur_psnr)
                    lpips_meter.update(cur_lpips)
                    bpp_list.append(bpp)

                    print(f'  BPP: {bpp:.4f} | PSNR: {cur_psnr:.2f} | '
                          f'LPIPS: {cur_lpips:.4f}')

                    out_img = (x_hat_crop * 0.5 + 0.5).float().cpu().detach()
                    output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))
                    output_pil.save(outf)

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f'  CUDA OOM on {fname}, skipping.')
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        avg_bpp = np.mean(bpp_list) if bpp_list else 0.0
        avg_psnr = psnr_meter.avg
        avg_lpips = lpips_meter.avg
        all_results[lmbda_val] = {
            'bpp': avg_bpp, 'psnr': avg_psnr, 'lpips': avg_lpips,
        }

        print(f'\n  [{tag}] Avg BPP: {avg_bpp:.4f} | '
              f'PSNR: {avg_psnr:.2f} | LPIPS: {avg_lpips:.4f}')

    # Summary table
    print(f'\n\n{"="*70}')
    print(f'  Variable-Rate RD Summary')
    print(f'{"="*70}')
    print(f'  {"Lambda":>10s}  {"BPP":>8s}  {"PSNR":>8s}  {"LPIPS":>8s}')
    print(f'  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*8}')
    for lmbda_val in lambda_list:
        r = all_results.get(lmbda_val)
        if r:
            print(f'  {lmbda_val:10.3f}  {r["bpp"]:8.4f}  '
                  f'{r["psnr"]:8.2f}  {r["lpips"]:8.4f}')
    print(f'{"="*70}\n')


# CUDA_VISIBLE_DEVICES=3 python src/testv1_variable.py --stage 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing script for variable-rate StableCodec checkpoints.")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--base_config_file", type=str,
                        default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str,
                        default="./configs/test.yaml")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config_file)
    test_config = load_yaml(args.test_config_file)

    main(base_config, test_config, stage=args.stage)
