LOG_DIR="./logs/AbnormalClass/Siamese-Merged-0324-HomBlock-3"
CKPT="${LOG_DIR}/best_siamese_model.pth"
H5_IMG="/data_b/algo001/dataset/image-PB-abnormal-test.h5"
H5_ANN="/data_b/algo001/dataset/annotation-PB-abnormal-test.h5"

DATASETS=()
for i in $(seq 0 18); do
  DATASETS+=("/data_b/algo001/cvat/202511_abnormal/ztask_${i}/dataset.key.json")
done

python val_slide.py \
  --ckpt "$CKPT" \
  --log-path "$LOG_DIR" \
  --h5-image "$H5_IMG" \
  --h5-annotation "$H5_ANN" \
  --dataset "${DATASETS[@]}" \
  --split val \
  --num-workers 16 \
  --batch-size 16 \
  --dropout 0.5 \
  --learning-rate 1e-5 \
  --no_save_localization
