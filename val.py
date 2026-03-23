#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
import cv2
from tqdm import tqdm

from utils.dataloader import get_val_data, SiameseChromosomeDataset
from utils.model import SiameseAnomalyNet


def init_distributed():
    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
    return distributed, local_rank


def _metrics_from_cm(tn, fp, fn, tp):
    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    tot = tn + fp + fn + tp
    acc = _safe_div(tn + tp, tot) if tot > 0 else float("nan")
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if tp + fp == 0 or tp + fn == 0:
        f1 = 0.0
    else:
        p, r = precision, recall
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if math.isnan(f1):
            f1 = 0.0
    return acc, f1, specificity, precision, recall


@torch.no_grad()
def run_val():
    parser = argparse.ArgumentParser(description="Siamese Anomaly Validation & Inference")
    parser.add_argument("--config", type=str, default="./configs/paras.json")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_siamese_model.pth")
    parser.add_argument("--save_heatmaps", action="store_true", help="Save anomaly heatmap visualization")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        paras = json.load(f)

    distributed, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    valid_val_cells = get_val_data(paras, test=True)
    val_dataset = SiameseChromosomeDataset(
        valid_val_cells,
        paras,
        resize=paras["patch_size"][0],
        is_train=False,
        log_build_stats=is_main,
    )

    sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=paras["batch_size"],
        sampler=sampler,
        num_workers=paras["num_workers"],
        pin_memory=paras.get("pin_memory", True),
    )
  
    model = SiameseAnomalyNet(in_channels=3, base_filters=32).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    criterion = nn.BCEWithLogitsLoss()

    all_labels, all_probs, all_preds = [], [], []
    sum_loss, n_loss = 0.0, 0
    save_dir = os.path.join(paras["log_path"], "inference_results")
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        if args.save_heatmaps:
            os.makedirs(os.path.join(save_dir, "visuals"), exist_ok=True)

    pbar = tqdm(val_loader, desc="Evaluating", disable=not is_main)

    for batch in pbar:
        img_A = batch["img_A"].to(device)
        mask_A = batch["mask_A"].to(device)
        img_B = batch["img_B"].to(device)
        mask_B = batch["mask_B"].to(device)
        labels = batch["is_anomaly"].to(device)

        logits, heatmaps = model(img_A, mask_A, img_B, mask_B)
        loss = criterion(logits, labels)
        sum_loss += loss.item()
        n_loss += 1

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(np.float32)

        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(probs)
        all_preds.extend(preds)

        if is_main and args.save_heatmaps:
            for i in range(len(probs)):
                h_map = heatmaps[i][0].cpu().numpy()
                h_map = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
                h_map_resized = cv2.resize(h_map, (224, 224))
                h_map_color = cv2.applyColorMap((h_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                name = batch["image_name_B"][i].replace(":", "_")
                cv2.imwrite(
                    os.path.join(save_dir, "visuals", f"{name}_prob_{probs[i]:.2f}.jpg"),
                    h_map_color,
                )

    if distributed:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (all_labels, all_probs, all_preds, sum_loss, n_loss))
        if is_main:
            all_labels = [x for sub in gathered for x in sub[0]]
            all_probs = [x for sub in gathered for x in sub[1]]
            all_preds = [x for sub in gathered for x in sub[2]]
            sum_loss = sum(sub[3] for sub in gathered)
            n_loss = sum(sub[4] for sub in gathered)

    if is_main:
        val_loss = sum_loss / n_loss if n_loss > 0 else 0.0
        all_labels = np.asarray(all_labels, dtype=np.float64)
        all_preds = np.asarray(all_preds, dtype=np.float64)
        all_probs = np.asarray(all_probs, dtype=np.float64)

        acc = accuracy_score(all_labels, all_preds)
        y_true = all_labels
        if np.unique(y_true).size < 2:
            auc = float("nan")
        else:
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = float("nan")
        try:
            f1 = f1_score(all_labels, all_preds, zero_division=0.0)
        except ValueError:
            f1 = 0.0

        n_total = len(all_labels)
        n_pos = int(all_labels.sum())
        n_neg = n_total - n_pos
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        _, _, spec, prec, rec = _metrics_from_cm(tn, fp, fn, tp)
        spec_s = f"{spec:.4f}" if not math.isnan(spec) else "nan"
        prec_s = f"{prec:.4f}" if not math.isnan(prec) else "nan"
        rec_s = f"{rec:.4f}" if not math.isnan(rec) else "nan"
        auc_s = f"{auc:.4f}" if not math.isnan(auc) else "nan"
        val_cm = f"[[{tn},{fp}],[{fn},{tp}]]"
        val_note = ""
        if n_neg == 0 or n_pos == 0:
            val_note = " | val_note=single_class_labels"

        line = (
            f" VAL | val_loss={val_loss:.4f} | "
            f"n0={n_neg} n1={n_pos} n={n_total} cm={val_cm} | "
            f"acc={acc:.4f} auc={auc_s} f1={f1:.4f} spec={spec_s} prec={prec_s} rec={rec_s} | "
            f"ckpt={args.ckpt} | out={save_dir}{val_note}"
        )
        print(line)

        def _nan_to_none(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)

        results = {
            "val_loss": val_loss,
            "config_lr": paras.get("learning_rate"),
            "accuracy": acc,
            "auc": _nan_to_none(auc),
            "f1_score": f1,
            "specificity": _nan_to_none(spec),
            "precision": _nan_to_none(prec),
            "recall": _nan_to_none(rec),
            "n_normal": n_neg,
            "n_abnormal": n_pos,
            "n_total": n_total,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "checkpoint": args.ckpt,
        }
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_val()
