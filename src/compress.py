"""Encode and decode single-image Omni-RAD bitstreams using the A0 model."""
import argparse
import json
import math
from pathlib import Path

from bitstream import read_bitstream, write_bitstream


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/test.yaml')
    parser.add_argument('--mode', choices=['roundtrip', 'encode', 'decode'], default='roundtrip')
    parser.add_argument('--img_path', type=Path, help='Input image directory (encode/roundtrip)')
    parser.add_argument('--bin_path', type=Path, required=True)
    parser.add_argument('--rec_path', type=Path, default=Path('results/reconstructions'))
    parser.add_argument('--lmbda', type=float, default=1.0)
    parser.add_argument('--max_images', type=int)
    args = parser.parse_args()
    if args.mode != 'decode' and args.img_path is None:
        parser.error('--img_path is required for encoding')
    if not math.isfinite(args.lmbda) or not 0 < args.lmbda <= 65504:
        parser.error('--lmbda must be a positive finite float16 value')
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max_images must be positive')
    return args


def main(args):
    import torch
    import torch.nn.functional as F
    import yaml
    from PIL import Image
    from torchvision.transforms.functional import to_tensor, to_pil_image
    from StableCodec_variable2_step import StableCodec

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if config.get('use_ema', False):
        raise ValueError('Use an exported non-EMA checkpoint for bitstream coding')
    if not config['model'].get('codec_path'):
        raise ValueError('A trained Omni-RAD checkpoint is required')
    net = StableCodec(sd_path=config['model']['sd_path'], config=config['model']).cuda().eval()
    net.codec.update(force=True)
    args.bin_path.mkdir(parents=True, exist_ok=True)
    args.rec_path.mkdir(parents=True, exist_ok=True)
    if args.mode == 'decode':
        inputs = sorted(args.bin_path.glob('*.ord'))
    else:
        inputs = sorted(p for p in args.img_path.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    inputs = inputs[:args.max_images]
    if not inputs:
        raise ValueError('No input files found')
    if len({p.stem for p in inputs}) != len(inputs):
        raise ValueError('Input image stems must be unique')
    records = []
    with torch.inference_mode():
        for source in inputs:
            target = source if args.mode == 'decode' else args.bin_path / (source.name + '.ord')
            if args.mode != 'decode':
                with Image.open(source) as image:
                    x = to_tensor(image.convert('RGB')).unsqueeze(0).cuda() * 2 - 1
                height, width = x.shape[-2:]
                # 256px alignment makes all four entropy passes shape-compatible.
                ph, pw = (-height) % 256, (-width) % 256
                mode = 'reflect' if ph < height and pw < width else 'replicate'
                x = F.pad(x, (0, pw, 0, ph), mode=mode)
                output = net.compress(x, lmbda=args.lmbda)
                write_bitstream(target, output, height, width)
            payload, shape, (height, width) = read_bitstream(target)
            if args.mode != 'encode':
                reconstruction = net.decompress(payload, shape, [1])[:, :, :height, :width]
                to_pil_image(((reconstruction[0].cpu() + 1) / 2).clamp(0, 1)).save(args.rec_path / (Path(target.stem).stem + '.png'))
            row = {'file': target.name, 'bpp': 8 * target.stat().st_size / (height * width), 'lambda': payload['lmbda_val'][0]}
            records.append(row)
            print(json.dumps(row))
    with open(args.bin_path / 'metrics.json', 'w') as stream:
        json.dump({'rate_type': 'file_bpp_including_header', 'images': records}, stream, indent=2)


if __name__ == '__main__':
    main(parse_args())
