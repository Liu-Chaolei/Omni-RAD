"""
testv1_variable3_policy_attach.py

Evaluate a debug-step StableCodec checkpoint with an optional attached
PolicyNet delta_T correction.
"""

import argparse
import glob
import math
import os

import lpips
import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
import yaml
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torchvision import transforms


class AverageMeter:
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
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def main(base_config, test_config, stage):
    if base_config["global_seed"] is not None:
        set_seed(base_config["global_seed"])

    from StableCodec_variable2_step_policy_attach import StableCodec

    use_policy_delta = bool(test_config["model"].get("use_policy_delta", False))
    net = StableCodec(
        sd_path=test_config["model"]["sd_path"],
        config=test_config["model"],
        stage=stage,
    )
    net.cuda().eval()

    if test_config.get("enable_xformers_memory_efficient_attention", False):
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    device = next(net.parameters()).device
    iqa_psnr = pyiqa.create_metric("psnr", device=device)
    loss_fn_lpips = lpips.LPIPS(net="vgg").to(device)
    loss_fn_lpips.requires_grad_(False)

    images = sorted(
        glob.glob(base_config["test_dataset"] + "/*.png")
        + glob.glob(base_config["test_dataset"] + "/*.jpg")
    )
    print(f"\nFound {len(images)} images in {base_config['test_dataset']}")
    print(f"use_policy_delta={use_policy_delta}\n")

    lambda_min = float(test_config["model"].get("lambda_min", 0.5))
    lambda_max = float(test_config["model"].get("lambda_max", 32.0))
    num_lambdas = test_config.get("num_lambdas", 6)
    lambda_list = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num_lambdas)).tolist()
    if test_config.get("lambda_list") is not None:
        lambda_list = [float(v) for v in test_config["lambda_list"]]

    all_results = {}
    suffix = "policy" if use_policy_delta else "snr"

    for lmbda_val in lambda_list:
        tag = f"lambda={lmbda_val:.3f}"
        print(f"\n{'=' * 60}\n  Evaluating {tag}\n{'=' * 60}")
        rec_dir = os.path.join(test_config["rec_path"], suffix, f"lambda_{lmbda_val:.3f}")
        os.makedirs(rec_dir, exist_ok=True)

        psnr_meter = AverageMeter()
        lpips_meter = AverageMeter()
        bpp_meter = AverageMeter()
        t_snr_meter = AverageMeter()
        t_pred_meter = AverageMeter()
        t_used_meter = AverageMeter()
        delta_meter = AverageMeter()

        for img_path in images:
            print("[Processing]", img_path)
            fname = os.path.splitext(os.path.basename(img_path))[0]
            outf = os.path.join(rec_dir, fname + ".png")

            img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
            ori_h, ori_w = img.shape[2:]
            stride_h, stride_w = base_config.get("model_stride", [64, 64])
            pad_h = (math.ceil(ori_h / stride_h)) * stride_h - ori_h
            pad_w = (math.ceil(ori_w / stride_w)) * stride_w - ori_w
            img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode="reflect")
            _, _, H, W = img_padded.shape

            with torch.no_grad():
                try:
                    lmbda_tensor = torch.tensor([lmbda_val], dtype=torch.float32, device=device)
                    x_hat, rate_out, T_used, policy_info = net(
                        img_padded, [1], H, W,
                        lmbda=lmbda_tensor,
                        use_policy_delta=use_policy_delta,
                    )
                    x_hat_crop = x_hat[:, :, :ori_h, :ori_w]
                    x_hat_01 = ((x_hat_crop + 1) / 2).clamp(0, 1)
                    gt_01 = ((img + 1) / 2).clamp(0, 1)

                    cur_psnr = iqa_psnr(x_hat_01, gt_01).mean().item()
                    cur_lpips = loss_fn_lpips(x_hat_crop, img).mean().item()
                    cur_bpp = rate_out.quantized_total_bpp.item()
                    cur_t_snr = policy_info["T_snr"].mean().item()
                    cur_t_pred = policy_info["T_pred"].mean().item()
                    cur_t_used = T_used.mean().item()
                    cur_delta = policy_info["delta_T"].mean().item()

                    psnr_meter.update(cur_psnr)
                    lpips_meter.update(cur_lpips)
                    bpp_meter.update(cur_bpp)
                    t_snr_meter.update(cur_t_snr)
                    t_pred_meter.update(cur_t_pred)
                    t_used_meter.update(cur_t_used)
                    delta_meter.update(cur_delta)

                    print(
                        f"  BPP:{cur_bpp:.4f} PSNR:{cur_psnr:.2f} "
                        f"LPIPS:{cur_lpips:.4f} T_snr:{cur_t_snr:.1f} "
                        f"T_pred:{cur_t_pred:.1f} T_used:{cur_t_used:.1f} "
                        f"delta_T:{cur_delta:.1f}"
                    )

                    out_img = (x_hat_crop * 0.5 + 0.5).float().cpu().detach()
                    transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0)).save(outf)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  CUDA OOM on {fname}, skipping.")
                        torch.cuda.empty_cache()
                    else:
                        raise

        all_results[lmbda_val] = {
            "bpp": bpp_meter.avg,
            "psnr": psnr_meter.avg,
            "lpips": lpips_meter.avg,
            "t_snr": t_snr_meter.avg,
            "t_pred": t_pred_meter.avg,
            "t_used": t_used_meter.avg,
            "delta_t": delta_meter.avg,
        }
        print(
            f"\n  [{tag}] Avg BPP:{bpp_meter.avg:.4f} PSNR:{psnr_meter.avg:.2f} "
            f"LPIPS:{lpips_meter.avg:.4f} T_snr:{t_snr_meter.avg:.1f} "
            f"T_pred:{t_pred_meter.avg:.1f} T_used:{t_used_meter.avg:.1f} "
            f"delta_T:{delta_meter.avg:.1f}"
        )

    print(f"\n\n{'=' * 96}")
    print("  Attach PolicyNet RD Summary")
    print(f"{'=' * 96}")
    print(f"  {'Lambda':>10s} {'BPP':>8s} {'PSNR':>8s} {'LPIPS':>8s} {'T_snr':>7s} {'T_pred':>7s} {'T_used':>7s} {'dT':>7s}")
    for lmbda_val in lambda_list:
        r = all_results.get(lmbda_val)
        if r:
            print(
                f"  {lmbda_val:10.3f} {r['bpp']:8.4f} {r['psnr']:8.2f} "
                f"{r['lpips']:8.4f} {r['t_snr']:7.1f} {r['t_pred']:7.1f} "
                f"{r['t_used']:7.1f} {r['delta_t']:7.1f}"
            )
    print(f"{'=' * 96}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test attached PolicyNet delta_T correction.")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--base_config_file", type=str, default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str, default="./configs/test.yaml")
    args = parser.parse_args()

    main(load_yaml(args.base_config_file), load_yaml(args.test_config_file), args.stage)
