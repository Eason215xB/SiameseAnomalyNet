#!/usr/bin/env bash
set -euo pipefail
# 在脚本所在目录执行，保证 ./configs 与 ./logs 相对路径一致
cd "$(dirname "$0")"

python val_slide.py \
  --config ./configs/paras_val.json \
  --ckpt ./logs/AbnormalClass/Siamese-Merged-0323-HomBlock/best_siamese_model.pth \
  --split val \
  --no_save_localization
