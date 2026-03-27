import os
import sys
import json
import csv
import shutil
import logging
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

from utils.dataloader import get_train_data, get_val_data, SiameseChromosomeDataset
from utils.model_train import SiameseAnomalyNet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class _TeeStream:
    """同时写入终端与日志文件（用于捕获 tqdm、scheduler 等到 stderr/stdout 的输出）。"""

    def __init__(self, stream, log_fp):
        self._stream = stream
        self._log_fp = log_fp

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._log_fp.write(data)
        self._log_fp.flush()

    def flush(self):
        self._stream.flush()
        self._log_fp.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()


def _setup_train_logger(log_path, local_rank):
    """Rank 0：training.log + 终端；其它 rank：静默。"""
    log = logging.getLogger("train")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    if local_rank != 0:
        log.addHandler(logging.NullHandler())
        return log
    os.makedirs(log_path, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(
        os.path.join(log_path, "training.log"), encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def _tee_stdout_stderr_to_file(log_path, local_rank):
    """
    Rank 0：将 stdout/stderr 复制到 console_full.log（含 tqdm 等）。
    返回 (tee_file_handle, 原 stdout, 原 stderr)；非 rank0 返回 (None, None, None)。
    """
    if local_rank != 0:
        return None, None, None
    os.makedirs(log_path, exist_ok=True)
    fp = open(os.path.join(log_path, "console_full.log"), "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, fp)
    sys.stderr = _TeeStream(old_err, fp)
    return fp, old_out, old_err


def _restore_stdio(tee_fp, old_out, old_err):
    if old_out is not None:
        sys.stdout = old_out
    if old_err is not None:
        sys.stderr = old_err
    if tee_fp is not None:
        tee_fp.close()


def _save_train_code_snapshot(log_path, config_path):
    """复制 train.py、utils/model.py、当前使用的配置文件到 log_path。"""
    shutil.copy2(os.path.join(_REPO_ROOT, "train.py"), os.path.join(log_path, "train.py"))
    shutil.copy2(
        os.path.join(_REPO_ROOT, "utils", "model_train.py"),
        os.path.join(log_path, "model_train.py"),
    )
    shutil.copy2(os.path.abspath(config_path), os.path.join(log_path, "paras.json"))


#----------混淆矩阵指标计算------------
def _metrics_from_cm(tn, fp, fn, tp):
    """acc、特异(specificity)、精确度(precision)、敏感/召回(recall/recall)、排阴(NPV)。"""
    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    tot = tn + fp + fn + tp
    acc = _safe_div(tn + tp, tot) if tot > 0 else float("nan")
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    npv = _safe_div(tn, tn + fn)
    if tp + fp == 0 or tp + fn == 0:
        f1 = 0.0
    else:
        p, r = precision, recall
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if math.isnan(f1):
            f1 = 0.0
    return acc, f1, specificity, precision, recall, npv

#----------训练开始的可视化------------
def visualize_training_samples(dataloader, log_path, logger, num_batches=3, max_samples_per_batch=4):
    IMAGENET_MEAN = np.array([0.0, 0.0, 0.0])
    IMAGENET_STD = np.array([1.0, 1.0, 1.0])
    
    def denormalize(tensor):
        tensor = tensor.cpu().clone()
        for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
    vis_dir = os.path.join(log_path, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    logger.info("正在生成训练样本可视化(前 %d 个 batch)...", num_batches)
    
    batch_count = 0
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_count >= num_batches:
            break
            
        img_A = batch_data["img_A"]  # (B, 3, H, W)
        mask_A = batch_data["mask_A"]  # (B, 1, H, W)
        img_B = batch_data["img_B"]  # (B, 3, H, W)
        mask_B = batch_data["mask_B"]  # (B, 1, H, W)
        pair_labels = batch_data["is_anomaly"]  # (B, 1) - 一对是否有差异
        label_A = batch_data.get("label_A", [None]*len(img_A))  # 单条A的标签
        label_B = batch_data.get("label_B", [None]*len(img_B))  # 单条B的标签
        names_A = batch_data.get("image_name_A", [f"A_{i}" for i in range(len(img_A))])
        names_B = batch_data.get("image_name_B", [f"B_{i}" for i in range(len(img_B))])
        
        n_samples = min(len(img_A), max_samples_per_batch)
        
        # 创建大图:每行显示一个样本的 [img_A, mask_A, img_B, mask_B]
        fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.5*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # 反归一化图像
            img_a_vis = denormalize(img_A[i]).permute(1, 2, 0).numpy()
            img_b_vis = denormalize(img_B[i]).permute(1, 2, 0).numpy()
            mask_a_vis = mask_A[i][0].cpu().numpy()  # (H, W)
            mask_b_vis = mask_B[i][0].cpu().numpy()  # (H, W)
            
            # 获取标签
            lbl_a = label_A[i].item() if label_A[i] is not None else None
            lbl_b = label_B[i].item() if label_B[i] is not None else None
            pair_lbl = pair_labels[i].item()
            
            # 构建标签文本
            a_status = f"A={'ABN' if lbl_a == 1 else 'NOR' if lbl_a == 0 else '?'})"
            b_status = f"B={'ABN' if lbl_b == 1 else 'NOR' if lbl_b == 0 else '?'})"
            pair_status = "DIFFERENT" if pair_lbl == 1.0 else "SAME"
            pair_color = "red" if pair_lbl == 1.0 else "green"
            
            # 显示 img_A
            axes[i, 0].imshow(img_a_vis)
            axes[i, 0].set_title(f"A (Reference)\n{names_A[i]}\n{a_status}", fontsize=7)
            axes[i, 0].axis('off')
            
            # 显示 mask_A(检查是否和img_A对齐)
            axes[i, 1].imshow(mask_a_vis, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title("Mask A", fontsize=7, color='blue')
            axes[i, 1].axis('off')
            
            # 显示 img_B
            axes[i, 2].imshow(img_b_vis)
            axes[i, 2].set_title(
                f"B (Target)\n{names_B[i]}\n{b_status}\nPAIR: {pair_status}", 
                fontsize=7, color=pair_color, fontweight='bold'
            )
            axes[i, 2].axis('off')
            
            # 显示 mask_B
            axes[i, 3].imshow(mask_b_vis, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title("Mask B", fontsize=7, color='blue')
            axes[i, 3].axis('off')
        
        # 添加总标题
        fig.suptitle(
            f"Batch {batch_idx}: vis | ",
            fontsize=10, y=0.995
        )
        
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"batch_{batch_idx:02d}_samples.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("  Batch %d: %d 样本已保存", batch_idx, n_samples)
        batch_count += 1
    
    logger.info("可视化完成,共 %d 张图,保存至: %s", batch_count, vis_dir)


#--------------训练---------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, local_rank):
    model.train()
    sum_loss = 0.0
    n_batch = 0
    tn = fp = fn = tp = 0
    labels_all, probs_all = [], []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, disable=(local_rank != 0))

    for batch_data in pbar:
        img_A = batch_data["img_A"].to(device)
        mask_A = batch_data["mask_A"].to(device)
        img_B = batch_data["img_B"].to(device)
        mask_B = batch_data["mask_B"].to(device)
        labels = batch_data["is_anomaly"].to(device)

        optimizer.zero_grad()
        logits, _ = model(img_A, mask_A, img_B, mask_B)
        smoothed_labels = labels.float() * 0.90 + 0.05 
        loss = criterion(logits, smoothed_labels)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        n_batch += 1
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().view(-1)
            lab = labels.long().view(-1)
            tn += int(((lab == 0) & (preds == 0)).sum().item())
            fp += int(((lab == 0) & (preds == 1)).sum().item())
            fn += int(((lab == 1) & (preds == 0)).sum().item())
            tp += int(((lab == 1) & (preds == 1)).sum().item())
            labels_all.extend(lab.cpu().numpy().flatten().tolist())
            probs_all.extend(probs.cpu().numpy().flatten().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    stats = torch.tensor(
        [sum_loss, n_batch, tn, fp, fn, tp], dtype=torch.float64, device=device
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    g_sum, g_nb, g_tn, g_fp, g_fn, g_tp = (float(x) for x in stats.tolist())
    avg_loss = g_sum / g_nb if g_nb > 0 else 0.0
    t_acc, t_f1, t_spec, t_prec, t_rec, t_npv = _metrics_from_cm(
        int(g_tn), int(g_fp), int(g_fn), int(g_tp)
    )

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (labels_all, probs_all))
    flat_y, flat_p = [], []
    for g in gathered:
        flat_y.extend(g[0])
        flat_p.extend(g[1])
    y_arr = np.asarray(flat_y, dtype=np.float64)
    if np.unique(y_arr).size < 2:
        t_auc = float("nan")
    else:
        try:
            t_auc = roc_auc_score(flat_y, flat_p)
        except ValueError:
            t_auc = float("nan")

    return avg_loss, t_acc, t_f1, t_spec, t_prec, t_rec, t_npv, t_auc


#--------------loss图---------------
def save_loss_curve_png(log_path, epoch_nums, train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epoch_nums, train_losses, "b-", marker="o", markersize=3, label="train loss", linewidth=1.2)
    ax.plot(epoch_nums, val_losses, "r-", marker="s", markersize=3, label="val loss", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / val loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(log_path, "loss_curve.png"), dpi=120)
    plt.close(fig)


#--------------验证---------------
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for batch_data in dataloader:
        img_A = batch_data["img_A"].to(device)
        mask_A = batch_data["mask_A"].to(device)
        img_B = batch_data["img_B"].to(device)
        mask_B = batch_data["mask_B"].to(device)
        labels = batch_data["is_anomaly"].to(device)

        logits, _ = model(img_A, mask_A, img_B, mask_B)
        loss = criterion(logits, labels)
        val_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())

    if len(dataloader) == 0:
        return (
            0.0,
            0.0,
            float("nan"),
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            0,
            0,
            0,
            0,
            0,
        )

    avg_loss = val_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    y_true = np.asarray(all_labels)
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
    n_pos = int(sum(all_labels))
    n_neg = n_total - n_pos
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    npv = _safe_div(tn, tn + fn)

    return (
        avg_loss,
        acc,
        auc,
        f1,
        specificity,
        precision,
        recall,
        npv,
        n_neg,
        n_pos,
        tn,
        fp,
        fn,
        tp,
    )



def main():
    # ================= 1. 解析配置 =================
    parser = argparse.ArgumentParser(description="Siamese Anomaly DDP Training")
    parser.add_argument("--config", type=str, default="./configs/paras.json", help="Path to config json")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        paras = json.load(f)

    # ================= 2. DDP 初始化 =================
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tee_fp, old_out, old_err = _tee_stdout_stderr_to_file(paras["log_path"], local_rank)
    logger = _setup_train_logger(paras["log_path"], local_rank)

    if local_rank == 0:
        log_path = os.path.abspath(paras["log_path"])
        os.makedirs(log_path, exist_ok=True)
        logger.info(
            "DDP 初始化成功,world_size=%d,日志目录=%s",
            dist.get_world_size(),
            log_path,
        )
        with open(
            os.path.join(log_path, "paras_resolved.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(paras, f, indent=2, ensure_ascii=False)
        logger.info("已保存解析后配置至 paras_resolved.json")
        csv_path = os.path.join(log_path, "train_log.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_auc",
                    "train_f1",
                    "train_specificity",
                    "train_recall",
                    "train_precision",
                    "train_npv",
                    "val_loss",
                    "val_acc",
                    "val_auc",
                    "val_f1",
                    "val_specificity",
                    "val_recall",
                    "val_precision",
                    "val_npv",
                ]
            )

    # ================= 3. 数据加载 =================
    if local_rank == 0:
        logger.info("正在进行 H5 数据集校验...")
    valid_train_cells = get_train_data(paras)
    valid_val_cells = get_val_data(paras, test=True)

    train_dataset = SiameseChromosomeDataset(
        valid_train_cells,
        paras,
        resize=paras["patch_size"][0],
        is_train=True,
        log_build_stats=(local_rank == 0),
    )
    
    # 打印训练集统计信息
    if local_rank == 0:
        train_stats = train_dataset.get_stats()
        tot = train_stats["total"]
        p1 = (100.0 * train_stats["label_1_pairs"] / tot) if tot else 0.0
        p0 = (100.0 * train_stats["label_0_pairs"] / tot) if tot else 0.0
        logger.info("%s", "=" * 40)
        logger.info("训练集样本统计")
        logger.info("%s", "=" * 40)
        logger.info("总样本数: %s", train_stats["total"])
        logger.info(
            "  ├─ 有差异对 (label=1): %s (%.1f%%)",
            train_stats["label_1_pairs"],
            p1,
        )
        logger.info("  │   ├─ A正常 vs B异常: %s", train_stats["abnormal_forward"])
        logger.info("  │   └─ A异常 vs B正常: %s", train_stats["abnormal_reverse"])
        logger.info(
            "  └─ 无差异对 (label=0): %s (%.1f%%)",
            train_stats["label_0_pairs"],
            p0,
        )
        logger.info("     └─ A正常 vs B正常: %s", train_stats["normal_pairs"])
        logger.info("同源配对统计:")
        logger.info(
            "  └─ 无同源正常被跳过的异常染色体: %s",
            train_stats["skipped_no_homolog"],
        )
        logger.info("%s", "=" * 60)
    
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=paras["batch_size"], 
        sampler=train_sampler, 
        num_workers=paras["num_workers"], 
        pin_memory=paras["pin_memory"], 
        drop_last=True
    )


    if local_rank == 0:
        vis_loader = DataLoader(
            train_dataset,
            batch_size=paras["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        visualize_training_samples(
            vis_loader,
            paras["log_path"],
            logger,
            num_batches=3,
            max_samples_per_batch=4,
        )
    
    val_loader = None
    val_stats = None
    if local_rank == 0:
        val_dataset = SiameseChromosomeDataset(
            valid_val_cells,
            paras,
            resize=paras["patch_size"][0],
            is_train=False,
            log_build_stats=True,
        )
        
        # 打印验证集统计信息
        val_stats = val_dataset.get_stats()
        vtot = val_stats["total"]
        vp1 = (100.0 * val_stats["label_1_pairs"] / vtot) if vtot else 0.0
        vp0 = (100.0 * val_stats["label_0_pairs"] / vtot) if vtot else 0.0
        logger.info("%s", "=" * 40)
        logger.info("验证集样本统计")
        logger.info("%s", "=" * 40)
        logger.info("总样本数: %s", val_stats["total"])
        logger.info(
            "  ├─ 有差异对 (label=1): %s (%.1f%%)",
            val_stats["label_1_pairs"],
            vp1,
        )
        logger.info("  │   ├─ A正常 vs B异常: %s", val_stats["abnormal_forward"])
        logger.info("  │   └─ A异常 vs B正常: %s", val_stats["abnormal_reverse"])
        logger.info(
            "  └─ 无差异对 (label=0): %s (%.1f%%)",
            val_stats["label_0_pairs"],
            vp0,
        )
        logger.info("     └─ A正常 vs B正常: %s", val_stats["normal_pairs"])
        logger.info("%s", "=" * 40)
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=paras["batch_size"] * 2, 
            shuffle=False, 
            num_workers=paras["num_workers"], 
            pin_memory=paras["pin_memory"]
        )
    # ================= 4. 模型与 DDP 包装 =================
    backbone = paras.get("backbone", "resnet18")        # 可选: resnet18, resnet34, resnet50
    pretrained = paras.get("pretrained", False)         # 是否使用 ImageNet 预训练权重
    pretrained_path = paras.get("pretrained_path", None)  # 本地预训练权重路径
    dropout = paras.get("dropout", 0.3)                 # Dropout 概率
    
    if local_rank == 0:
        logger.info(
            "构建模型: backbone=%s, pretrained=%s, pretrained_path=%s, dropout=%s",
            backbone,
            pretrained,
            pretrained_path,
            dropout,
        )
    
    model = SiameseAnomalyNet(
        in_channels=3, 
        backbone=backbone,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        dropout=dropout
    ).to(device)
    
    if paras.get("model_path", ""):
        if local_rank == 0:
            logger.info("加载预训练权重: %s", paras["model_path"])
        model.load_state_dict(torch.load(paras['model_path'], map_location=device), strict=False)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(device))
    
    # 1. 区分骨干网络和头部的参数
    backbone_params = []
    head_params = []
    for name, param in model.module.named_parameters():
        if 'encoder' in name:
            if param.requires_grad:
                backbone_params.append(param)
        else:
            if param.requires_grad:
                head_params.append(param)

    # 2. 设置差异化学习率 (骨干网络学习率缩小 10 倍)
    base_lr = paras["learning_rate"]
    # optimizer = optim.Adam([
    #     {'params': backbone_params, 'lr': base_lr * 0.1}, 
    #     {'params': head_params, 'lr': base_lr}
    # ], weight_decay=paras.get("weight_decay", 1e-3))
    # 推荐的统一优化器写法：
    optimizer = torch.optim.AdamW(model.parameters(), lr=paras["learning_rate"], weight_decay=paras["weight_decay"])
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=(local_rank == 0)
    )
    
    # ================= 5. 训练 =================
    best_auc = 0.0
    best_val_loss = float("inf")
    ever_saved_by_auc = False
    ever_saved_by_loss = False
    num_epochs = paras["num_epochs"]
    hist_epochs, hist_train_loss, hist_val_loss = [], [], []
    
    # 早停参数
    early_stop_patience = paras.get("early_stop_patience", 15)
    early_stop_counter = 0

    if local_rank == 0:
        _lp = os.path.abspath(paras["log_path"])
        logger.info("训练开始：保存代码与配置快照至 %s", _lp)
        _save_train_code_snapshot(_lp, args.config)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        train_loss, train_acc, train_f1, train_spec, train_prec, train_rec, train_npv, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, local_rank
        )

        if local_rank == 0:
            (
                val_loss,
                val_acc,
                val_auc,
                val_f1,
                val_spec,
                val_prec,
                val_rec,
                val_npv,
                val_n0,
                val_n1,
                val_tn,
                val_fp,
                val_fn,
                val_tp,
            ) = evaluate(model.module, val_loader, criterion, device)

            hist_epochs.append(epoch + 1)
            hist_train_loss.append(train_loss)
            hist_val_loss.append(val_loss)
            save_loss_curve_png(paras["log_path"], hist_epochs, hist_train_loss, hist_val_loss)
            
            # 更新学习率调度器
            scheduler.step(val_loss)
            
            # 早停逻辑 - 基于验证损失
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            t_spec_s = f"{train_spec:.4f}" if not math.isnan(train_spec) else "nan"
            t_prec_s = f"{train_prec:.4f}" if not math.isnan(train_prec) else "nan"
            t_rec_s = f"{train_rec:.4f}" if not math.isnan(train_rec) else "nan"
            t_npv_s = f"{train_npv:.4f}" if not math.isnan(train_npv) else "nan"
            t_auc_s = (
                f"{train_auc:.4f}"
                if isinstance(train_auc, float) and not math.isnan(train_auc)
                else "nan"
            )
            auc_str = (
                f"{val_auc:.4f}"
                if isinstance(val_auc, float) and not math.isnan(val_auc)
                else "nan"
            )
            spec_str = f"{val_spec:.4f}" if not math.isnan(val_spec) else "nan"
            prec_str = f"{val_prec:.4f}" if not math.isnan(val_prec) else "nan"
            rec_str = f"{val_rec:.4f}" if not math.isnan(val_rec) else "nan"
            npv_str = f"{val_npv:.4f}" if not math.isnan(val_npv) else "nan"
            val_cm = f"[[{val_tn},{val_fp}],[{val_fn},{val_tp}]]"
            val_note = ""
            if val_n0 == 0 or val_n1 == 0:
                val_note = " | val_note=single_class_labels"
            
            # 添加早停计数器到日志
            cur_lr = optimizer.param_groups[0]["lr"]
            epoch_line = (
                f"lr={cur_lr:.6g} | "
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train acc={train_acc:.4f} auc={t_auc_s} f1={train_f1:.4f} "
                f"spec={t_spec_s} sens={t_rec_s} prec={t_prec_s} npv={t_npv_s} ||||| "
                f"val_loss={val_loss:.4f} | "
                f"val acc={val_acc:.4f} auc={auc_str} f1={val_f1:.4f} "
                f"spec={spec_str} sens={rec_str} prec={prec_str} npv={npv_str} | "
                f"early_stop={early_stop_counter}/{early_stop_patience}"
                f"{val_note}"
            )

            save_best = False
            best_reason = ""
            if isinstance(val_auc, float) and not math.isnan(val_auc):
                if val_auc > best_auc:
                    best_auc = val_auc
                    save_best = True
                    best_reason = f"AUC={best_auc:.4f}"
            else:
                if val_loss < best_val_loss:
                    save_best = True
                    best_reason = f"val_loss={best_val_loss:.4f} (no val AUC)"

            log_line = epoch_line
            if save_best:
                save_path = os.path.join(paras["log_path"], "best_siamese_model.pth")
                torch.save(model.module.state_dict(), save_path)
                log_line = f"{epoch_line} | BEST_SAVE {best_reason}"
                if isinstance(val_auc, float) and not math.isnan(val_auc):
                    ever_saved_by_auc = True
                else:
                    ever_saved_by_loss = True

            logger.info("%s", log_line)

            with open(os.path.join(paras["log_path"], "train_log.csv"), "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        epoch + 1,
                        train_loss,
                        train_acc,
                        train_auc,
                        train_f1,
                        train_spec,
                        train_rec,
                        train_prec,
                        train_npv,
                        val_loss,
                        val_acc,
                        val_auc,
                        val_f1,
                        val_spec,
                        val_rec,
                        val_prec,
                        val_npv,
                    ]
                )
            
            # 触发早停
            if early_stop_counter >= early_stop_patience:
                logger.info(
                    "Early stopping triggered at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    early_stop_patience,
                )
                break

        dist.barrier()

    if local_rank == 0:
        tail = []
        if ever_saved_by_auc:
            tail.append(f"最佳验证 AUC={best_auc:.4f}")
        if ever_saved_by_loss:
            tail.append(f"按 val_loss 选优时最佳={best_val_loss:.4f}")
        logger.info(
            "DONE | %s",
            (" | ".join(tail) if tail else "no_checkpoint_saved"),
        )

    _restore_stdio(tee_fp, old_out, old_err)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()