import os
import math
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import yaml
import argparse

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

from diffusers.utils.import_utils import is_xformers_available

from color_fix import adain_color_fix_quant
from StableCodec import StableCodec
from torch_ema import ExponentialMovingAverage
import lpips
import pyiqa


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


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


def main(base_config, test_config):
    if base_config['global_seed'] is not None:
        set_seed(base_config['global_seed'])

    net = StableCodec(config=test_config['model'])

    if test_config['model']['codec_path'] is not None and test_config['use_ema']:
        checkpoint = torch.load(test_config['model']['codec_path'], map_location='cpu')

        ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
        ema_net.load_state_dict(checkpoint['ema_state_dict'])
        ema_net.copy_to(net.parameters())

        del checkpoint, ema_net

    net.cuda().eval()
    net.codec.update(force=True)

    if test_config['enable_xformers_memory_efficient_attention']:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
      
    bpp = []
    images = glob.glob(test_config['test_dataset'] + '/*.png')
    print(f"\nFind {str(len(images))} images in {test_config['test_dataset']}\n")

    os.makedirs(test_config['rec_path'], exist_ok=True)

    for img_path in images:

        print('[Processing]', img_path)
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(test_config['rec_path'], fname+'.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            try:
                out = net(img_padded, freeze_aux=True, training=False)
                N, _, H, W = img_padded.size()
                num_pixels = N * H * W
                rate = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in out["likelihoods"].values()
                )
                out_img = out["x_hat"]
                out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
                out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(str(name))
                    print("CUDA out of memory. Continuing to next image.")
                    torch.cuda.empty_cache() 
                    continue
                else:
                    raise
        device = next(net.parameters()).device
        lpips_loss = AverageMeter()
        psnr_loss = AverageMeter()
        iqa_psnr = pyiqa.create_metric('psnr')
        loss_fn_alex = lpips.LPIPS(net='vgg')
        loss_fn_alex.net.requires_grad_(False)
        loss_fn_alex = loss_fn_alex.to(device)
        with torch.no_grad():
            psnr_loss.update(iqa_psnr(out["x_hat"], img_padded).mean().item())
            lpips_loss.update(loss_fn_alex(out["x_hat"], img_padded).mean().item())

        output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))

        bpp.append(rate.cpu())
        print('[BPP]', rate.cpu())

        if test_config['color_fix']:
            img = (img * 0.5 + 0.5).float().cpu().detach()
            im_lr_resize = transforms.ToPILImage()(img[0].clamp(0.0, 1.0))
            output_pil = adain_color_fix_quant(output_pil, im_lr_resize, 16)

        output_pil.save(outf)

    print(
        f"\tLPIPS: {lpips_loss.avg:.3f} |"
        f"\tPSNR: {psnr_loss.avg:.3f} |"
    )

    print('\n[Average BPP]', np.mean(bpp))


# CUDA_VISIBLE_DEVICES=3 python src/test.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument("--base_config_file", type=str, default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str, default="./configs/test.yaml")
    args = parser.parse_args()
    base_config = load_yaml(args.base_config_file)
    test_config = load_yaml(args.test_config_file)
    main(base_config, test_config)
