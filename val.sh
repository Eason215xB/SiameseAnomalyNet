torchrun --nproc_per_node=4 val.py \
    --master_port=29513 \
    --config ./configs/paras.json \
    --ckpt /home/algo002/work/SiameseAnomalyNet/logs/AbnormalClass/Siamese-Merged/best_siamese_model.pth