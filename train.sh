#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 获取使用的显卡数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "🚀 开始启动多卡分布式训练，使用显卡数量: $NUM_GPUS"

# 使用 torchrun 启动 DDP 训练，并指定 json 配置路径
torchrun --nproc_per_node=$NUM_GPUS \
         --master_port=29500 \
         train.py \
         --config ./configs/paras.json