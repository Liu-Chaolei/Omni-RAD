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


def main(method, base_config, test_config, stage):
    if base_config['global_seed'] is not None:
        set_seed(base_config['global_seed'])

    if method == 'channel':
        from StableCodec_ori import StableCodec
    elif method == 'fusion1':
        from StableCodec_fusion import StableCodec
    elif method == 'fusion2':
        from StableCodec_fusion2 import StableCodec
    elif method == 'fusion3':
        from StableCodec_fusion3 import StableCodec
    elif method == 'variable':
        from StableCodec_variable import StableCodec
    elif method == 'variable_perbatch':
        from StableCodec_variable_perbatch import StableCodec

    # Use StableCodec_ori, same as trainv1.py
    net = StableCodec(
        sd_path=test_config['model']['sd_path'],
        lmbda=test_config.get('lmbda', 0.5),
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
    # NOTE: skip codec.update() — training validation does NOT call it
    # net.codec.update(force=True)

    if test_config.get('enable_xformers_memory_efficient_attention', False):
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Metrics — created once outside the loop
    device = next(net.parameters()).device
    iqa_psnr = pyiqa.create_metric('psnr', device=device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_fn_lpips.requires_grad_(False)

    psnr_meter = AverageMeter()
    lpips_meter = AverageMeter()
    bpp_list = []

    # Collect test images
    images = sorted(
        glob.glob(base_config['test_dataset'] + '/*.png')
        + glob.glob(base_config['test_dataset'] + '/*.jpg')
    )
    print(f"\nFound {len(images)} images in {base_config['test_dataset']}\n")
    os.makedirs(test_config['rec_path'], exist_ok=True)

    for img_path in images:
        print('[Processing]', img_path)
        fname = os.path.splitext(os.path.basename(img_path))[0]
        outf = os.path.join(test_config['rec_path'], fname + '.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        # Pad to multiple of 64 (model_stride)
        stride_h, stride_w = base_config.get('model_stride', [64, 64])
        pad_h = (math.ceil(ori_h / stride_h)) * stride_h - ori_h
        pad_w = (math.ceil(ori_w / stride_w)) * stride_w - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')
        _, _, H, W = img_padded.shape

        with torch.no_grad():
            try:
                pos_tag_prompt = [1]
                x_hat, RateLossOutput = net(img_padded, pos_tag_prompt, H, W)

                bpp_val = RateLossOutput.quantized_total_bpp.item()

                # Crop back to original size
                x_hat_crop = x_hat[:, :, :ori_h, :ori_w]
                img_crop = img  # original unpadded image

                # Convert to [0, 1] for metrics — same as trainv1.py validation
                x_hat_01 = ((x_hat_crop + 1) / 2).clamp(0, 1)
                gt_01 = ((img_crop + 1) / 2).clamp(0, 1)

                cur_psnr = iqa_psnr(x_hat_01, gt_01).mean().item()
                cur_lpips = loss_fn_lpips(x_hat_crop, img_crop).mean().item()

                psnr_meter.update(cur_psnr)
                lpips_meter.update(cur_lpips)
                bpp_list.append(bpp_val)

                print(f'  BPP: {bpp_val:.4f} | PSNR: {cur_psnr:.2f} | LPIPS: {cur_lpips:.4f}')

                # Save reconstructed image
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

    print(f'\n{"="*50}')
    print(f'Average BPP:   {np.mean(bpp_list):.4f}')
    print(f'Average PSNR:  {psnr_meter.avg:.2f}')
    print(f'Average LPIPS: {lpips_meter.avg:.4f}')
    print(f'{"="*50}\n')


# CUDA_VISIBLE_DEVICES=3 python src/testv1.py --method fusion3 --stage 4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for trainv1 checkpoints.")
    parser.add_argument("--method", type=str, default="channel")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--base_config_file", type=str, default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str, default="./configs/test.yaml")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config_file)
    test_config = load_yaml(args.test_config_file)

    main(args.method, base_config, test_config, stage=args.stage)