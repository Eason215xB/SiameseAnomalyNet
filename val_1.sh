#!/usr/bin/env bash
set -euo pipefail
# 在脚本所在目录执行，保证 ./configs 与 ./logs 相对路径一致
cd "$(dirname "$0")"

python val.py \
  --config ./configs/paras-.json \
  --ckpt /home/algo002/work/SiameseAnomalyNet/logs/AbnormalClass/Siamese-Merged-0323-2/best_siamese_model.pth

# 不需要保存定位图时取消下一行注释并传给 python：
#   --no_save_localization
