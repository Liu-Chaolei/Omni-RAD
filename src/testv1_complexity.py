"""Complexity measurement for the model tested by testv1.py.

Measures:
  * Total / trainable parameters (whole model + sub-modules)
  * Encoding speed (full encode path: aux_codec + vae.encode + codec.compress)
  * Decoding speed (full decode path: codec.decompress + unet + sched.step + vae.decode)

Usage (same CLI pattern as testv1.py):
    CUDA_VISIBLE_DEVICES=0 python src/testv1_complexity.py \
        --method channel --stage 1 \
        --base_config_file ./configs/base.yaml \
        --test_config_file ./configs/test.yaml \
        --warmup 3 --runs 10
"""

import argparse
import glob
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from accelerate.utils import set_seed
from torch_ema import ExponentialMovingAverage
from torchvision import transforms


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f} B"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return str(n)


def count_parameters(module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def cuda_time(fn, warmup: int, runs: int) -> dict:
    """Return avg/min/max elapsed ms for fn() using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "avg_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


# --------------------------------------------------------------------------
# Encode / decode closures (mirror StableCodec.compress / decompress)
# --------------------------------------------------------------------------

def build_encode_fn(net, x):
    """Closure that runs the full encoder path (image -> bitstream)."""

    def encode():
        with torch.no_grad():
            out = net.compress(x)
        return out

    return encode


def build_decode_fn(net, strings, shape, pos_prompt):
    """Closure that runs the full decoder path (bitstream -> image)."""

    def decode():
        with torch.no_grad():
            out = net.decompress(strings, shape, pos_prompt)
        return out

    return decode


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(method, base_config, test_config, stage, warmup, runs, max_images):
    if base_config.get('global_seed') is not None:
        set_seed(base_config['global_seed'])

    # Same method dispatch as testv1.py
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
    else:
        raise ValueError(f"Unknown method: {method}")

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
    # compress / decompress need quantized CDFs
    net.codec.update(force=True)

    # ----------------------------------------------------------------------
    # 1. Parameter counts
    # ----------------------------------------------------------------------
    modules = OrderedDict()
    modules["Total (net)"] = net
    modules["  VAE"] = net.vae
    modules["    VAE.encoder"] = net.vae.encoder
    modules["    VAE.decoder"] = net.vae.decoder
    modules["  UNet"] = net.unet
    modules["  LatentCodec"] = net.codec
    if hasattr(net, "aux_codec"):
        modules["  ELIC aux_codec"] = net.aux_codec

    print("\n" + "=" * 78)
    print(f"  Parameter Report  (method={method}, stage={stage})")
    print("=" * 78)
    print(f"{'Module':<32} {'Total':>18} {'Trainable':>18}")
    print("-" * 78)
    for name, m in modules.items():
        p = count_parameters(m)
        print(f"{name:<32} {format_params(p['total']):>18} {format_params(p['trainable']):>18}")
    print("=" * 78)

    # ----------------------------------------------------------------------
    # 2. Prepare inputs from the same dataset testv1.py uses
    # ----------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    images = sorted(
        glob.glob(base_config['test_dataset'] + '/*.png')
        + glob.glob(base_config['test_dataset'] + '/*.jpg')
    )
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise RuntimeError(f"No images found in {base_config['test_dataset']}")
    print(f"\n[Speed] Using {len(images)} image(s) from {base_config['test_dataset']}")
    print(f"[Speed] warmup={warmup}  runs={runs}")

    stride_h, stride_w = base_config.get('model_stride', [64, 64])

    per_image = []
    pos_tag_prompt = [1]

    for idx, img_path in enumerate(images):
        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]
        pad_h = math.ceil(ori_h / stride_h) * stride_h - ori_h
        pad_w = math.ceil(ori_w / stride_w) * stride_w - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')
        _, _, H, W = img_padded.shape

        # Produce strings/shape once so decode timing does not depend on encoding
        with torch.no_grad():
            out = net.compress(img_padded)
        strings, shape = out["strings"], out["shape"]

        try:
            enc_time = cuda_time(build_encode_fn(net, img_padded), warmup, runs)
            dec_time = cuda_time(build_decode_fn(net, strings, shape, pos_tag_prompt), warmup, runs)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  [{idx:02d}] {os.path.basename(img_path)}  CUDA OOM, skipping.")
                torch.cuda.empty_cache()
                continue
            raise

        per_image.append({
            "name": os.path.basename(img_path),
            "H": H, "W": W,
            "enc": enc_time,
            "dec": dec_time,
        })
        print(f"  [{idx:02d}] {os.path.basename(img_path):<20} {H}x{W}  "
              f"enc={enc_time['avg_ms']:7.2f}±{enc_time['std_ms']:.2f} ms  "
              f"dec={dec_time['avg_ms']:7.2f}±{dec_time['std_ms']:.2f} ms")

    # ----------------------------------------------------------------------
    # 3. Aggregate timing report
    # ----------------------------------------------------------------------
    if per_image:
        enc_avgs = np.array([r["enc"]["avg_ms"] for r in per_image])
        dec_avgs = np.array([r["dec"]["avg_ms"] for r in per_image])
        print("\n" + "=" * 78)
        print("  Speed Report (per-image averages across dataset)")
        print("=" * 78)
        print(f"  Encoding:  mean={enc_avgs.mean():.2f} ms   "
              f"min={enc_avgs.min():.2f}   max={enc_avgs.max():.2f}   "
              f"throughput={1000.0 / enc_avgs.mean():.2f} img/s")
        print(f"  Decoding:  mean={dec_avgs.mean():.2f} ms   "
              f"min={dec_avgs.min():.2f}   max={dec_avgs.max():.2f}   "
              f"throughput={1000.0 / dec_avgs.mean():.2f} img/s")
        print("=" * 78)

    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n[GPU] Peak allocated memory: {mem_mb:.1f} MB")


# CUDA_VISIBLE_DEVICES=0 python src/testv1_complexity.py --method channel --stage 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complexity (params + encode/decode speed) for testv1 checkpoints."
    )
    parser.add_argument("--method", type=str, default="channel")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--base_config_file", type=str, default="./configs/base.yaml")
    parser.add_argument("--test_config_file", type=str, default="./configs/test.yaml")
    parser.add_argument("--warmup", type=int, default=3, help="warmup runs per image")
    parser.add_argument("--runs", type=int, default=10, help="timed runs per image")
    parser.add_argument("--max_images", type=int, default=None,
                        help="limit how many images from test_dataset to time (None = all)")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config_file)
    test_config = load_yaml(args.test_config_file)

    main(args.method, base_config, test_config,
         stage=args.stage, warmup=args.warmup, runs=args.runs,
         max_images=args.max_images)
