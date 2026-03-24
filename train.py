import os
import json
import csv
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
from utils.model import SiameseAnomalyNet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def visualize_training_samples(dataloader, log_path, num_batches=3, max_samples_per_batch=4):
    """
    可视化训练样本,保存前几个batch的输入图片和标签
    显示:单条A的标签、单条B的标签、一对的标签(是否有差异)
    Args:
        dataloader: 训练数据加载器
        log_path: 保存路径
        num_batches: 要可视化的batch数量(默认3)
        max_samples_per_batch: 每个batch最多显示多少样本(默认4)
    """
    # ImageNet 反归一化参数
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def denormalize(tensor):
        """将归一化的tensor还原为可视化格式"""
        tensor = tensor.cpu().clone()
        for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
    vis_dir = os.path.join(log_path, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\n🖼️  正在生成训练样本可视化(前{num_batches}个batch)...")
    print("   布局: [img_A | mask_A | img_B | mask_B] + 详细标签信息")
    
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
        
        print(f"  ✓ Batch {batch_idx}: {n_samples} 样本已保存")
        batch_count += 1
    
    print(f"🎉 可视化完成！共 {batch_count} 张图,保存至: {vis_dir}")

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, local_rank):
    model.train()
    sum_loss = 0.0
    n_batch = 0
    tn = fp = fn = tp = 0
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
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    stats = torch.tensor(
        [sum_loss, n_batch, tn, fp, fn, tp], dtype=torch.float64, device=device
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    g_sum, g_nb, g_tn, g_fp, g_fn, g_tp = (float(x) for x in stats.tolist())
    avg_loss = g_sum / g_nb if g_nb > 0 else 0.0
    t_acc, t_f1, t_spec, t_prec, t_rec = _metrics_from_cm(
        int(g_tn), int(g_fp), int(g_fn), int(g_tp)
    )
    return avg_loss, t_acc, t_f1, t_spec, t_prec, t_rec


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

    return (
        avg_loss,
        acc,
        auc,
        f1,
        specificity,
        precision,
        recall,
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
    
    if local_rank == 0:
        print(f"✅ DDP 初始化成功! 发现 {dist.get_world_size()} 张显卡正在并行训练。")
        os.makedirs(paras["log_path"], exist_ok=True)
        # 保存训练参数
        paras_save_path = os.path.join(paras["log_path"], "paras.json")
        with open(paras_save_path, "w", encoding="utf-8") as f:
            json.dump(paras, f, indent=2, ensure_ascii=False)
        print(f"已保存训练参数至 {paras_save_path}")
        # 初始化训练日志 CSV
        csv_path = os.path.join(paras["log_path"], "train_log.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_f1",
                    "train_specificity",
                    "train_precision",
                    "train_recall",
                    "val_loss",
                    "val_acc",
                    "val_auc",
                    "val_f1",
                    "val_specificity",
                    "val_precision",
                    "val_recall",
                ]
            )

    # ================= 3. 数据加载 =================
    if local_rank == 0:
        print("正在进行 H5 数据集校验...")
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
        print("\n" + "="*60)
        print("📊 训练集样本统计")
        print("="*60)
        print(f"总样本数: {train_stats['total']}")
        print(f"  ├─ 有差异对 (label=1): {train_stats['label_1_pairs']} ({train_stats['label_1_pairs']/train_stats['total']*100:.1f}%)")
        print(f"  │   ├─ A正常 vs B异常: {train_stats['abnormal_forward']}")
        print(f"  │   └─ A异常 vs B正常: {train_stats['abnormal_reverse']}")
        print(f"  └─ 无差异对 (label=0): {train_stats['label_0_pairs']} ({train_stats['label_0_pairs']/train_stats['total']*100:.1f}%)")
        print(f"     └─ A正常 vs B正常: {train_stats['normal_pairs']}")
        print(f"同源配对统计:")
        print(f"  └─ 无同源正常被跳过的异常染色体: {train_stats['skipped_no_homolog']}")
        print("="*60 + "\n")
    
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=paras["batch_size"], 
        sampler=train_sampler, 
        num_workers=paras["num_workers"], 
        pin_memory=paras["pin_memory"], 
        drop_last=True
    )
    
    # 训练前可视化前3个batch的样本(类似YOLO风格)
    if local_rank == 0:
        # 创建一个临时非分布式loader用于可视化(避免DistributedSampler的shuffle干扰)
        vis_loader = DataLoader(
            train_dataset,
            batch_size=paras["batch_size"],
            shuffle=False,  # 固定顺序便于观察
            num_workers=0,  # 单线程避免多进程问题
            pin_memory=False,
        )
        visualize_training_samples(vis_loader, paras["log_path"], num_batches=3, max_samples_per_batch=4)
    
    # 验证集 (仅在主卡 Rank 0 初始化):默认 is_train=False 保留真实分布用于计算客观指标
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
        print("\n" + "="*60)
        print("📊 验证集样本统计 (Siamese对称配对)")
        print("="*60)
        print(f"总样本数: {val_stats['total']}")
        print(f"  ├─ 有差异对 (label=1): {val_stats['label_1_pairs']} ({val_stats['label_1_pairs']/val_stats['total']*100:.1f}%)")
        print(f"  │   ├─ A正常 vs B异常: {val_stats['abnormal_forward']}")
        print(f"  │   └─ A异常 vs B正常: {val_stats['abnormal_reverse']}")
        print(f"  └─ 无差异对 (label=0): {val_stats['label_0_pairs']} ({val_stats['label_0_pairs']/val_stats['total']*100:.1f}%)")
        print(f"     └─ A正常 vs B正常: {val_stats['normal_pairs']}")
        print("="*60 + "\n")
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=paras["batch_size"] * 2, 
            shuffle=False, 
            num_workers=paras["num_workers"], 
            pin_memory=paras["pin_memory"]
        )
    # ================= 4. 模型与 DDP 包装 =================
    # 使用 ResNet 作为 backbone
    backbone = paras.get("backbone", "resnet18")        # 可选: resnet18, resnet34, resnet50
    pretrained = paras.get("pretrained", False)         # 是否使用 ImageNet 预训练权重
    pretrained_path = paras.get("pretrained_path", None)  # 本地预训练权重路径
    dropout = paras.get("dropout", 0.3)                 # Dropout 概率
    
    if local_rank == 0:
        print(f"构建模型: backbone={backbone}, pretrained={pretrained}, "
              f"pretrained_path={pretrained_path}, dropout={dropout}")
    
    model = SiameseAnomalyNet(
        in_channels=3, 
        backbone=backbone,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        dropout=dropout
    ).to(device)
    
    if paras.get("model_path", ""):
        if local_rank == 0:
            print(f"加载预训练权重: {paras['model_path']}")
        model.load_state_dict(torch.load(paras['model_path'], map_location=device), strict=False)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    
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
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': base_lr * 0.1}, 
        {'params': head_params, 'lr': base_lr}
    ], weight_decay=paras.get("weight_decay", 1e-3))
    
    # 添加学习率调度器 - 当验证损失停滞时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=(local_rank == 0)
    )
    
    # ================= 5. 开始训练循环 =================
    best_auc = 0.0
    best_val_loss = float("inf")
    ever_saved_by_auc = False
    ever_saved_by_loss = False
    num_epochs = paras["num_epochs"]
    hist_epochs, hist_train_loss, hist_val_loss = [], [], []
    
    # 早停参数
    early_stop_patience = paras.get("early_stop_patience", 15)
    early_stop_counter = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        train_loss, train_acc, train_f1, train_spec, train_prec, train_rec = train_one_epoch(
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
            auc_str = (
                f"{val_auc:.4f}"
                if isinstance(val_auc, float) and not math.isnan(val_auc)
                else "nan"
            )
            spec_str = f"{val_spec:.4f}" if not math.isnan(val_spec) else "nan"
            prec_str = f"{val_prec:.4f}" if not math.isnan(val_prec) else "nan"
            rec_str = f"{val_rec:.4f}" if not math.isnan(val_rec) else "nan"
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
                f"train acc={train_acc:.4f} f1={train_f1:.4f} spec={t_spec_s} prec={t_prec_s} rec={t_rec_s} ||||| "
                f"val_loss={val_loss:.4f} | "
                f"val acc={val_acc:.4f} auc={auc_str} f1={val_f1:.4f} spec={spec_str} prec={prec_str} rec={rec_str} | "
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

            print(log_line)

            with open(os.path.join(paras["log_path"], "train_log.csv"), "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        epoch + 1,
                        train_loss,
                        train_acc,
                        train_f1,
                        train_spec,
                        train_prec,
                        train_rec,
                        val_loss,
                        val_acc,
                        val_auc,
                        val_f1,
                        val_spec,
                        val_prec,
                        val_rec,
                    ]
                )
            
            # 触发早停
            if early_stop_counter >= early_stop_patience:
                print(f"\n⛔ Early stopping triggered at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break

        dist.barrier()

    if local_rank == 0:
        tail = []
        if ever_saved_by_auc:
            tail.append(f"最佳验证 AUC={best_auc:.4f}")
        if ever_saved_by_loss:
            tail.append(f"按 val_loss 选优时最佳={best_val_loss:.4f}")
        print(
            "DONE | " + (" | ".join(tail) if tail else "no_checkpoint_saved")
        )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()