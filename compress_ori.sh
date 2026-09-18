#!/bin/bash
# bash compress_ori.sh

python src/compress_ori.py \
    --sd_path="/data/ssd/liuchaolei/models/sd-turbo" \
    --elic_path="/data/ssd/liuchaolei/models/StableCodec/checkpoints/elic_official.pth" \
    --img_path="/data/ssd/liuchaolei/image_datasets/DFCLIC/valid/" \
    --rec_path="/data/ssd/liuchaolei/results/reconstruct/StableCodec/compress/ft2/DFCLIC_val/rec/" \
    --bin_path="/data/ssd/liuchaolei/results/reconstruct/StableCodec/compress/ft2/DFCLIC_val/bin/" \
    --codec_path="/data/ssd/liuchaolei/models/StableCodec/checkpoints/stablecodec_ft2.pkl" \
    # --color_fix