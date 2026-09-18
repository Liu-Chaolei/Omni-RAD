#!/bin/bash
# bash compress.sh

python src/compress.py \
    --sd_path="/data/ssd/liuchaolei/models/sd-turbo" \
    --elic_path="/home/liuchaolei/Image-Coder/StableCodec/checkpoints/elic_official.pth" \
    --img_path="/data/ssd/liuchaolei/image_datasets/Kodak24/HR/" \
    --rec_path="/data/ssd/liuchaolei/results/reconstruct/StableCodec/ft32/Kodak/rec/" \
    --bin_path="/data/ssd/liuchaolei/results/reconstruct/StableCodec/ft32/Kodak/bin/" \
    --codec_path="/home/liuchaolei/Image-Coder/StableCodec/checkpoints/stablecodec_ft32.pkl" \
    # --color_fix