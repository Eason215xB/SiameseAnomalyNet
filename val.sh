#!/usr/bin/env bash
# 分布式 pair 级验证（val_one.py）。请按环境修改路径；不再使用 --config。
set -euo pipefail

CKPT="/home/algo002/work/SiameseAnomalyNet/logs/AbnormalClass/Siamese-Merged-0323-2/best_siamese_model.pth"
LOG_DIR="$(dirname "$CKPT")"
H5_IMG="/data_b/algo001/dataset/image-PB-abnormal-test.h5"
H5_ANN="/data_b/algo001/dataset/annotation-PB-abnormal-test.h5"

DATASETS=()
for i in $(seq 0 18); do
  DATASETS+=("/data_b/algo001/cvat/202511_abnormal/ztask_${i}/dataset.key.json")
done

torchrun --nproc_per_node=4 val_one.py \
  --master_port=29520 \
  --ckpt "$CKPT" \
  --log-path "$LOG_DIR" \
  --h5-image "$H5_IMG" \
  --h5-annotation "$H5_ANN" \
  --dataset "${DATASETS[@]}" \
  --num-workers 16 \
  --batch-size 16
