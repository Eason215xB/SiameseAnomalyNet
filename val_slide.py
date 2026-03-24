#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import json
import argparse
import math
import hashlib
import h5py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc as sklearn_auc,
    f1_score,
    confusion_matrix,
)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

from utils.dataloader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SiameseChromosomeDataset,
    get_train_data,
    get_val_data,
)
from utils.model import SiameseAnomalyNet
from utils.bind_category import pair_sample_bind_category

# slide 预测：该 slide 内 pair 预测为 1 的比例 >= X% 则 slide 预测为 1
SLIDE_PRED_PCT_THRESHOLDS = list(range(50, 81, 5))  # 50, 55, ..., 80


def _slide_id_from_file_name(file_name):
    """
    与数据集 JSON 中 file_name 路径一致；取倒数第 4 段作为 slide 分组键。
    例: .../202405_abnormal/A20200974/slide6/cell0/1.png -> path_parts[-4] -> A20200974
    """
    if not file_name:
        return "unknown"
    parts = str(file_name).replace("\\", "/").split("/")
    parts = [p for p in parts if p]
    if len(parts) < 4:
        return "unknown"
    try:
        return parts[-4]
    except IndexError:
        return "unknown"


def _build_slide_pair_groups(labels, probs, preds, slide_ids):
    """按 slide_id 聚合该 slide 下所有 pair 的 label / prob / pred。"""
    from collections import defaultdict

    g = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})
    for i in range(len(labels)):
        g[slide_ids[i]]["labels"].append(float(labels[i]))
        g[slide_ids[i]]["probs"].append(float(probs[i]))
        g[slide_ids[i]]["preds"].append(float(preds[i]))
    return g


def _slide_level_gt_and_mean_prob(g):
    """slide GT = max(pair labels)；ROC 分数 = 该 slide 内 pair 概率均值（与 AUC 一致）。"""
    sids = sorted(g.keys())
    yt, ymean = [], []
    for sid in sids:
        d = g[sid]
        labs = d["labels"]
        pbs = d["probs"]
        yt.append(float(max(labs)))
        ymean.append(float(np.mean(pbs)))
    return (
        np.asarray(yt, dtype=np.float64),
        np.asarray(ymean, dtype=np.float64),
        sids,
    )


def _youden_threshold_from_roc(y_true, y_score):
    """
    在 ROC 上最大化 Youden J = sensitivity + specificity - 1 = TPR - FPR。
    返回 (threshold, sensitivity, specificity, J)；单类标签时全为 nan。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # thr[i] 对应 ROC 点 (fpr[i+1], tpr[i+1])
    if len(thr) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    J = tpr[1:] - fpr[1:]
    j = int(np.argmax(J))
    best_thr = float(thr[j])
    best_sens = float(tpr[j + 1])
    best_spec = float(1.0 - fpr[j + 1])
    best_J = float(J[j])
    return best_thr, best_sens, best_spec, best_J


def _plot_slide_roc_png(
    out_path,
    fpr,
    tpr,
    auc_val,
    youden_fpr,
    youden_tpr,
    youden_thr,
):
    fig, ax = plt.subplots(figsize=(7, 6))
    auc_s = f"{auc_val:.4f}" if isinstance(auc_val, float) and not math.isnan(auc_val) else "nan"
    ax.plot(fpr, tpr, color="C0", lw=2, label=f"ROC (AUC = {auc_s})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="chance")
    if (
        isinstance(youden_fpr, (float, np.floating))
        and isinstance(youden_tpr, (float, np.floating))
        and isinstance(youden_thr, (float, np.floating))
        and not math.isnan(float(youden_fpr))
        and not math.isnan(float(youden_tpr))
        and not math.isnan(float(youden_thr))
    ):
        ax.scatter(
            [youden_fpr],
            [youden_tpr],
            color="C3",
            s=80,
            zorder=5,
            label=f"Youden (thr={float(youden_thr):.4f})",
        )
    ax.set_xlabel("FPR (1 - specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title("Slide-level ROC (score = mean pair probability)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _slide_level_from_groups(g, pred_pct):
    """
    GT：slide 内任一对 label==1 则 slide GT=1（max）。
    Pred：该 slide 内 pred==1 的 pair 数 / pair 总数 >= pred_pct/100 则 slide pred=1。
    AUC 分数：该 slide 内 pair 概率的均值 mean(prob)。
    """
    thr = float(pred_pct) / 100.0
    sids = sorted(g.keys())
    yt, yp, ymean = [], [], []
    for sid in sids:
        d = g[sid]
        labs = d["labels"]
        prs = d["preds"]
        pbs = d["probs"]
        n = len(labs)
        n_pred_pos = sum(1 for x in prs if float(x) >= 0.5)
        frac = n_pred_pos / n if n > 0 else 0.0
        yt.append(float(max(labs)))
        yp.append(1.0 if frac >= thr else 0.0)
        ymean.append(float(np.mean(pbs)))
    return (
        np.asarray(yt, dtype=np.float64),
        np.asarray(yp, dtype=np.float64),
        np.asarray(ymean, dtype=np.float64),
        sids,
    )


def _per_slide_detail_dicts(g):
    """每个 slide 一行：GT、pair 数、预测为 1 的比例、各 X% 阈值下的 slide pred。"""
    rows = []
    for sid in sorted(g.keys()):
        d = g[sid]
        labs = d["labels"]
        prs = d["preds"]
        pbs = d["probs"]
        n = len(labs)
        n_pred_pos = sum(1 for x in prs if float(x) >= 0.5)
        frac = n_pred_pos / n if n > 0 else 0.0
        gt = int(round(float(max(labs))))
        mean_pb = float(np.mean(pbs))
        pred_at_pct = {
            str(pct): int(frac >= float(pct) / 100.0) for pct in SLIDE_PRED_PCT_THRESHOLDS
        }
        rows.append(
            {
                "slide_id": str(sid),
                "gt": gt,
                "n_pairs": n,
                "n_pred_pos": n_pred_pos,
                "pred_pos_frac": round(frac, 6),
                "mean_pair_prob": mean_pb,
                "pred_slide_at_pct": pred_at_pct,
            }
        )
    return rows


def _metrics_by_bind_category_slide(
    all_labels, all_probs, all_preds, all_cat, all_slide_ids, pred_pct
):
    """按 bind 粗类别分组后，再在各类别内按 slide 聚合（与全局相同 GT / pred% 规则）。"""
    out = {}
    y = np.asarray(all_labels, dtype=np.float64)
    p = np.asarray(all_probs, dtype=np.float64)
    ph = np.asarray(all_preds, dtype=np.float64)
    c = np.asarray(all_cat, dtype=object)
    sids = np.asarray(all_slide_ids, dtype=object)
    for cat in sorted(set(all_cat)):
        mask = c == cat
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        g = _build_slide_pair_groups(
            y[idx].tolist(), p[idx].tolist(), ph[idx].tolist(), sids[idx].tolist()
        )
        yt, yp, ymean, _ = _slide_level_from_groups(g, pred_pct)
        m = _metrics_subset_binary(yt, ymean, yp)
        if m is not None:
            out[str(cat)] = m
    return out


def _safe_filename_component(s):
    """key 中含 / 等字符时展平为单层文件名。"""
    t = str(s).strip()
    for bad in ("/", "\\", "\0", ":", "<", ">", "|", '"', "?", "*"):
        t = t.replace(bad, "_")
    t = t.replace(os.sep, "_")
    return t if t else "unnamed"


def _chw_imagenet_to_bgr_u8(img_chw):
    """反 ImageNet 归一化，得到与 OpenCV 一致的 BGR uint8 (H,W,3)。"""
    t = img_chw.detach().cpu().float().clone()
    mean = torch.tensor(IMAGENET_MEAN, device=t.device, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device, dtype=t.dtype).view(3, 1, 1)
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    rgb = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _struct_tag_abn(x):
    """单条染色体结构标注：0=正常(NOR)，1=异常(ABN)，与 dataloader 中 label_A/B 一致。"""
    return "ABN" if float(x) >= 0.5 else "NOR"


def _mean_gray_bgr_u8(bgr):
    return float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))


def _invert_bgr_u8(bgr):
    return 255 - bgr


def _text_strip_gray(w, h, line1, line2=None):
    """仅文字条，灰底黑字，不压在图像上。"""
    strip = np.full((h, w, 3), 235, dtype=np.uint8)
    cv2.putText(
        strip,
        line1[: min(len(line1), 200)],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    if line2:
        cv2.putText(
            strip,
            line2[: min(len(line2), 200)],
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    return strip


def _text_strip_4cols(panel_w, h, texts):
    """四列说明，文字在条带内、不在图像上。"""
    strip = np.full((h, panel_w, 3), 240, dtype=np.uint8)
    cw = panel_w // 4
    for i, t in enumerate(texts):
        x0 = 6 + i * cw
        for li, part in enumerate(t.split("|")[:2]):
            y = 20 + li * 16
            cv2.putText(
                strip,
                part.strip()[:80],
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
    return strip


def _load_full_cell_bgr(h5_path, cell_key):
    """整细胞 RGB 原图 -> BGR uint8。"""
    with h5py.File(h5_path, "r", swmr=True) as f_img:
        img_bytes = np.array(f_img[cell_key])
    rgb = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _heatmap_target_side(label_a, label_b):
    """热力图/叠加列：优先显示「异常」那条；仅 A 异常且 B 正常时用 A；否则用 B（含双正常、仅 B 异常、双异常）。"""
    a = float(label_a) >= 0.5
    b = float(label_b) >= 0.5
    if a and not b:
        return "A"
    return "B"


def _batch_optional_str(batch, key, i):
    v = batch.get(key)
    if v is None:
        return ""
    x = v[i]
    if torch.is_tensor(x):
        return str(x.item())
    if x is None:
        return ""
    return str(x)


def _abnormal_filename_part(s):
    t = (s or "").strip()
    if not t:
        return "empty"
    return _safe_filename_component(t)[:120]


def _resize_full_to_panel_width(full_bgr, panel_w, max_h):
    fh, fw = full_bgr.shape[:2]
    if fw <= 0:
        return full_bgr
    new_w = int(panel_w)
    new_h = max(1, int(fh * new_w / fw))
    if new_h > max_h:
        new_h = int(max_h)
        new_w = max(1, int(fw * new_h / fh))
    out = cv2.resize(full_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if out.shape[1] < panel_w:
        pad = np.full((out.shape[0], panel_w - out.shape[1], 3), 238, dtype=np.uint8)
        out = np.hstack([out, pad])
    elif out.shape[1] > panel_w:
        out = cv2.resize(out, (panel_w, new_h), interpolation=cv2.INTER_AREA)
    return out


def _build_localization_panel(
    img_A_chw,
    img_B_chw,
    mask_A_1hw,
    mask_B_1hw,
    heatmap_1hw,
    vis_hw,
    label_a,
    label_b,
    pair_gt,
    pair_prob,
    full_bgr_u8,
    dark_mean_thresh,
):
    """
    整图 + [A|B|heatmap|overlay]，文字仅在上下灰条；偏暗 invert；
    热力图与叠加列固定使用「异常」一侧（仅 A 异 B 正用 A；否则用 B）。
    """
    h, w = int(vis_hw), int(vis_hw)
    panel_w = 4 * w
    side = _heatmap_target_side(label_a, label_b)

    bgr_a = _chw_imagenet_to_bgr_u8(img_A_chw)
    bgr_b = _chw_imagenet_to_bgr_u8(img_B_chw)
    bgr_a = cv2.resize(bgr_a, (w, h), interpolation=cv2.INTER_LINEAR)
    bgr_b = cv2.resize(bgr_b, (w, h), interpolation=cv2.INTER_LINEAR)

    full_orig = None
    if full_bgr_u8 is not None:
        full_orig = full_bgr_u8.copy()
        mg = _mean_gray_bgr_u8(full_orig)
    else:
        mg = 0.5 * (_mean_gray_bgr_u8(bgr_a) + _mean_gray_bgr_u8(bgr_b))

    do_invert = mg < float(dark_mean_thresh)

    if do_invert:
        bgr_a = _invert_bgr_u8(bgr_a)
        bgr_b = _invert_bgr_u8(bgr_b)

    m_a = mask_A_1hw[0].detach().cpu().numpy().astype(np.float32)
    m_b = mask_B_1hw[0].detach().cpu().numpy().astype(np.float32)
    m_a = np.clip(cv2.resize(m_a, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    m_b = np.clip(cv2.resize(m_b, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    m_tgt = m_a if side == "A" else m_b

    hm_raw = heatmap_1hw[0].detach().cpu().numpy().astype(np.float32)
    hm_raw = cv2.resize(hm_raw, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = hm_raw * m_tgt
    hmin, hmax = float(hm.min()), float(hm.max())
    if hmax - hmin < 1e-8:
        hm_n = np.zeros_like(hm, dtype=np.float32)
    else:
        hm_n = (hm - hmin) / (hmax - hmin + 1e-8)
    hm_u8 = (hm_n * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    bgr_tgt = bgr_a if side == "A" else bgr_b
    blend = cv2.addWeighted(bgr_tgt, 0.55, heat_bgr, 0.45, 0.0)
    overlay = np.where(m_tgt[..., None] > 0.5, blend, bgr_tgt)

    row4 = np.concatenate([bgr_a, bgr_b, heat_bgr, overlay], axis=1)

    ta = _struct_tag_abn(label_a)
    tb = _struct_tag_abn(label_b)
    pg = float(pair_gt)
    pair_word = "DIFF" if pg >= 0.5 else "SAME"
    inv_note = f"invert=ON meanG={mg:.1f}<{dark_mean_thresh}" if do_invert else f"invert=OFF meanG={mg:.1f}"

    header = _text_strip_gray(
        panel_w,
        48,
        f"GT_pair {pair_word} ({int(pg)})  Pred {float(pair_prob):.3f}  |  A {ta}  B {tb}  |  heat on {side}  |  {inv_note}",
    )

    blocks = [header]

    if full_orig is not None:
        full_disp = _invert_bgr_u8(full_orig) if do_invert else full_orig
        full_rs = _resize_full_to_panel_width(full_disp, panel_w, max_h=3 * h)
        blocks.append(full_rs)

    blocks.append(row4)
    blocks.append(
        _text_strip_4cols(
            panel_w,
            40,
            [
                f"A ref | {ta}",
                f"B tgt | {tb}",
                f"mask {side}",
                f"{side}+heat",
            ],
        )
    )

    return np.vstack(blocks)


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


def _metrics_subset_binary(y_true, y_prob, y_pred):
    """单个子集上的二分类指标（与全局相同定义）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return None
    acc = accuracy_score(y_true, y_pred)
    if np.unique(y_true).size < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
    except ValueError:
        f1 = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    _, _, spec, prec, rec = _metrics_from_cm(tn, fp, fn, tp)
    return {
        "n": n,
        "accuracy": float(acc),
        "auc": auc,
        "f1_score": float(f1),
        "specificity": spec,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


@torch.no_grad()
def run_val():
    parser = argparse.ArgumentParser(description="Siamese Anomaly Validation & Inference")
    parser.add_argument("--config", type=str, default="./configs/paras.json")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_siamese_model.pth")
    parser.add_argument(
        "--no_save_localization",
        action="store_true",
        help="不保存异常定位可视化（默认保存到 log_path/inference_results/localization/）",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "train"),
        default="val",
        help="评估数据划分：val=验证(+test)集，train=训练集（与 train.py 中 get_train_data 一致）",
    )
    args = parser.parse_args()
    save_localization = not args.no_save_localization

    with open(args.config, "r", encoding="utf-8") as f:
        paras = json.load(f)

    distributed, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if args.split == "train":
        eval_cells = get_train_data(paras)
    else:
        eval_cells = get_val_data(paras, test=True)
    if is_main:
        print(f"[eval] split={args.split}  cells={len(eval_cells)}", flush=True)

    cell_key_to_slide = {}
    for cell in eval_cells:
        k = cell.get("key")
        if k is None:
            continue
        fn = cell.get("file_name", "")
        cell_key_to_slide[str(k)] = _slide_id_from_file_name(fn)

    val_dataset = SiameseChromosomeDataset(
        eval_cells,
        paras,
        resize=paras["patch_size"][0],
        is_train=False,
        log_build_stats=is_main,
    )
    if len(val_dataset) == 0:
        st = val_dataset.get_stats()
        if is_main:
            print(
                "[eval] Siamese 数据集样本数为 0，无法验证。常见原因："
                "① 同 cell 同 category_id 下没有 ≥2 条带 segmentation 的正常染色体（无法配正常-正常对）；"
                "② 大量标注缺少 category_id（kid is None）；"
                "③ 异常样本在同源池无正常对照且也无足够正常对。"
                f" stats={st}",
                flush=True,
            )
        raise SystemExit(1)

    sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    val_bs = int(paras.get("val_batch_size", paras["batch_size"] * 2))
    loader_kw = dict(
        batch_size=val_bs,
        num_workers=paras["num_workers"],
        pin_memory=paras.get("pin_memory", True),
        drop_last=False,
    )
    if sampler is not None:
        loader_kw["sampler"] = sampler
    else:
        loader_kw["shuffle"] = False
    val_loader = DataLoader(val_dataset, **loader_kw)

    backbone = paras.get("backbone", "resnet18")
    dropout = paras.get("dropout", 0.3)
    # 与 train 结构一致；推理仅依赖 checkpoint，不再加载 ImageNet 预训练以加快启动、避免路径依赖
    model = SiameseAnomalyNet(
        in_channels=3,
        backbone=backbone,
        pretrained=False,
        pretrained_path=None,
        dropout=dropout,
    ).to(device)
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

    all_labels, all_probs, all_preds, all_cat, all_slide_ids = [], [], [], [], []
    sum_loss, n_loss = 0.0, 0
    # 与权重路径无关：可视化默认写在配置里的 log_path 下（相对路径相对当前工作目录）
    _inf_name = "inference_results_train" if args.split == "train" else "inference_results"
    save_dir = os.path.abspath(os.path.join(paras["log_path"], _inf_name))
    loc_dir = os.path.join(save_dir, "localization")
    n_loc_saved = 0
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        if save_localization:
            os.makedirs(loc_dir, exist_ok=True)
            print(f"[localization] 输出目录（非 ckpt 目录）: {loc_dir}", flush=True)
            if distributed:
                print(
                    "[localization] 提示: 当前为多进程分布式，仅 LOCAL_RANK=0 会写图且只覆盖本分片数据；"
                    "需要全量拼图请单进程运行（unset WORLD_SIZE 或使用单卡）。",
                    flush=True,
                )

    vis_hw = int(paras["patch_size"][0])
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
        preds = (probs > 0.4).astype(np.float32)
        labels_np = labels.cpu().numpy().flatten()
        bs = int(labels_np.shape[0])
        keys_b = batch.get("cell_key")
        for i in range(bs):
            la = batch["label_A"][i]
            lb = batch["label_B"][i]
            la = la.item() if torch.is_tensor(la) else float(la)
            lb = lb.item() if torch.is_tensor(lb) else float(lb)
            sa = _batch_optional_str(batch, "abnormal_content_A", i)
            sb = _batch_optional_str(batch, "abnormal_content_B", i)
            all_cat.append(pair_sample_bind_category(la, lb, sa, sb))
            if keys_b is None:
                ck = None
            else:
                ck = keys_b[i]
                ck = ck.item() if torch.is_tensor(ck) else ck
                ck = str(ck) if ck is not None else None
            sid = cell_key_to_slide.get(ck, "unknown") if ck is not None else "unknown"
            all_slide_ids.append(sid)

        all_labels.extend(labels_np.tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

        if is_main and save_localization:
            img_A_cpu = batch["img_A"].cpu()
            img_B_cpu = batch["img_B"].cpu()
            mask_A_cpu = batch["mask_A"].cpu()
            mask_B_cpu = batch["mask_B"].cpu()
            heatmaps_cpu = heatmaps.cpu()
            h5_img_path = paras["h5_files"]["image"]
            dark_thresh = float(paras.get("vis_dark_mean_threshold", 72.0))
            full_cache = {}
            for i in range(len(probs)):
                na = _safe_filename_component(batch["image_name_A"][i])
                nb = _safe_filename_component(batch["image_name_B"][i])
                la = batch["label_A"][i]
                lb = batch["label_B"][i]
                la = la.item() if torch.is_tensor(la) else float(la)
                lb = lb.item() if torch.is_tensor(lb) else float(lb)
                keys = batch.get("cell_key")
                if keys is None:
                    ck = None
                else:
                    ck = keys[i]
                    ck = ck.item() if torch.is_tensor(ck) else ck
                    ck = str(ck) if ck is not None else None
                if ck not in full_cache:
                    if ck is None:
                        full_cache[ck] = None
                    else:
                        try:
                            full_cache[ck] = _load_full_cell_bgr(h5_img_path, ck)
                        except Exception as e:
                            print(f"[WARN] load whole cell failed key={ck}: {e}", flush=True)
                            full_cache[ck] = None
                sa = _batch_optional_str(batch, "abnormal_content_A", i)
                sb = _batch_optional_str(batch, "abnormal_content_B", i)
                pa = _abnormal_filename_part(sa)
                pb = _abnormal_filename_part(sb)
                side = _heatmap_target_side(la, lb)
                panel = _build_localization_panel(
                    img_A_cpu[i],
                    img_B_cpu[i],
                    mask_A_cpu[i],
                    mask_B_cpu[i],
                    heatmaps_cpu[i],
                    vis_hw,
                    label_a=la,
                    label_b=lb,
                    pair_gt=float(labels[i].item()),
                    pair_prob=float(probs[i]),
                    full_bgr_u8=full_cache[ck],
                    dark_mean_thresh=dark_thresh,
                )
                base = (
                    f"{na}__{nb}__abA_{pa}__abB_{pb}__tgt_{side}_"
                    f"p{probs[i]:.3f}_gt{labels[i].item():.0f}_loc.jpg"
                )
                if len(base.encode("utf-8")) > 220:
                    short = hashlib.sha256(
                        f"{na}|{nb}|{pa}|{pb}|{side}".encode("utf-8")
                    ).hexdigest()[:16]
                    base = (
                        f"{short}__abA_{pa[:40]}__abB_{pb[:40]}__tgt_{side}_"
                        f"p{probs[i]:.3f}_gt{labels[i].item():.0f}_loc.jpg"
                    )
                    if len(base.encode("utf-8")) > 220:
                        base = f"{short}_p{probs[i]:.3f}_gt{labels[i].item():.0f}_loc.jpg"
                fn = os.path.join(loc_dir, base)
                ok = cv2.imwrite(fn, panel)
                if ok:
                    n_loc_saved += 1
                else:
                    print(f"[WARN] cv2.imwrite 失败（检查路径/权限/磁盘）: {fn}", flush=True)

    if distributed:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered,
            (all_labels, all_probs, all_preds, all_cat, all_slide_ids, sum_loss, n_loss),
        )
        if is_main:
            all_labels = [x for sub in gathered for x in sub[0]]
            all_probs = [x for sub in gathered for x in sub[1]]
            all_preds = [x for sub in gathered for x in sub[2]]
            all_cat = [x for sub in gathered for x in sub[3]]
            all_slide_ids = [x for sub in gathered for x in sub[4]]
            sum_loss = sum(sub[5] for sub in gathered)
            n_loss = sum(sub[6] for sub in gathered)

    if is_main:
        val_loss = sum_loss / n_loss if n_loss > 0 else 0.0
        all_labels = np.asarray(all_labels, dtype=np.float64)
        all_preds = np.asarray(all_preds, dtype=np.float64)
        all_probs = np.asarray(all_probs, dtype=np.float64)
        all_slide_ids = list(all_slide_ids)

        n_pairs = int(len(all_labels))
        g_all = _build_slide_pair_groups(
            all_labels.tolist(), all_probs.tolist(), all_preds.tolist(), all_slide_ids
        )

        loc_note = ""
        if save_localization:
            loc_note = f" | localization={loc_dir}"
        print(
            f" VAL | val_loss={val_loss:.4f} | n_pairs={n_pairs} n_slides={len(g_all)} | "
            f"slide GT=max(pair labels); slide pred=(pred_pos_frac>=X%) | "
            f"ckpt={args.ckpt} | out={save_dir}{loc_note}",
            flush=True,
        )

        y_slide_gt, y_slide_score, _slide_ids_roc = _slide_level_gt_and_mean_prob(g_all)
        slide_roc_block = None
        if np.unique(y_slide_gt).size >= 2:
            try:
                fpr_roc, tpr_roc, _thr_roc = roc_curve(y_slide_gt, y_slide_score)
                auc_roc = float(sklearn_auc(fpr_roc, tpr_roc))
            except ValueError:
                fpr_roc, tpr_roc = np.array([0.0, 1.0]), np.array([0.0, 1.0])
                auc_roc = float("nan")
            y_thr, y_sens, y_spec, y_J = _youden_threshold_from_roc(
                y_slide_gt, y_slide_score
            )
            y_fpr = 1.0 - y_spec if not math.isnan(y_spec) else float("nan")
            y_tpr = y_sens
            roc_png = os.path.join(save_dir, "roc_slide_mean_pair_prob.png")
            _plot_slide_roc_png(
                roc_png,
                fpr_roc,
                tpr_roc,
                auc_roc,
                y_fpr,
                y_tpr,
                y_thr,
            )
            slide_roc_block = {
                "score_definition": "mean pair probability per slide",
                "rule_positive_if": "mean_pair_prob >= threshold",
                "auc": auc_roc,
                "youden_threshold_on_mean_prob": y_thr,
                "youden_sensitivity_recall": y_sens,
                "youden_specificity": y_spec,
                "youden_J": y_J,
                "roc_png": os.path.basename(roc_png),
                "note": "Youden J = sensitivity + specificity - 1 = TPR - FPR（在 ROC 上取最大）",
            }
            print(
                "--- slide-level ROC (score = mean pair prob) | Youden 平衡阈值 ---",
                flush=True,
            )
            auc_rs = f"{auc_roc:.4f}" if not math.isnan(auc_roc) else "nan"
            print(
                f"  AUC={auc_rs} | fig -> {roc_png}",
                flush=True,
            )
            print(
                f"  Youden: threshold(mean_prob>={y_thr:.6f}) "
                f"sensitivity={y_sens:.4f} specificity={y_spec:.4f} J={y_J:.4f}",
                flush=True,
            )
        else:
            print(
                "--- slide-level ROC: skipped (only one class in slide GT) ---",
                flush=True,
            )

        print(
            "--- slide-level metrics vs X% (AUC uses mean pair prob per slide) ---",
            flush=True,
        )

        metrics_by_pred_pct = {}
        for pct in SLIDE_PRED_PCT_THRESHOLDS:
            y_slide_t, y_slide_pred, y_slide_mean_prob, slide_ids_ordered = _slide_level_from_groups(
                g_all, pct
            )
            acc = accuracy_score(y_slide_t, y_slide_pred)
            if np.unique(y_slide_t).size < 2:
                auc = float("nan")
            else:
                try:
                    auc = roc_auc_score(y_slide_t, y_slide_mean_prob)
                except ValueError:
                    auc = float("nan")
            try:
                f1 = f1_score(y_slide_t, y_slide_pred, zero_division=0.0)
            except ValueError:
                f1 = 0.0
            n_total = len(y_slide_t)
            n_pos = int(y_slide_t.sum())
            n_neg = n_total - n_pos
            cm = confusion_matrix(y_slide_t, y_slide_pred, labels=[0, 1])
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
            print(
                f"  X={pct}% | n_slides={n_total} n0={n_neg} n1={n_pos} cm={val_cm} | "
                f"acc={acc:.4f} auc={auc_s} f1={f1:.4f} spec={spec_s} prec={prec_s} rec={rec_s}{val_note}",
                flush=True,
            )
            metrics_by_pred_pct[str(pct)] = {
                "n_slides": n_total,
                "n_normal_slides": n_neg,
                "n_abnormal_slides": n_pos,
                "accuracy": float(acc),
                "auc": auc,
                "f1_score": float(f1),
                "specificity": spec,
                "precision": prec,
                "recall": rec,
                "confusion_matrix": [[tn, fp], [fn, tp]],
            }

        per_slide_rows = _per_slide_detail_dicts(g_all)
        print(
            "--- per-slide: GT, n_pairs, pred_pos_frac, pred_slide @ X%=50..80 ---",
            flush=True,
        )
        for row in per_slide_rows:
            sid = row["slide_id"]
            gt = row["gt"]
            frac = row["pred_pos_frac"]
            pa = row["pred_slide_at_pct"]
            parts = [f"p{p}={pa[str(p)]}" for p in SLIDE_PRED_PCT_THRESHOLDS]
            marks = [
                "OK" if gt == pa[str(p)] else "WR"
                for p in SLIDE_PRED_PCT_THRESHOLDS
            ]
            print(
                f"  slide={sid}  GT={gt}  n_pairs={row['n_pairs']}  "
                f"pred_pos_frac={frac:.4f}  mean_prob={row['mean_pair_prob']:.4f}  "
                f"{' '.join(parts)}  mark@X={'/'.join(marks)}",
                flush=True,
            )
        if save_localization:
            print(
                f"[localization] 已保存 {n_loc_saved} 张定位拼图（A|B|heatmap|overlay）-> {loc_dir}",
                flush=True,
            )

        def _nan_to_none(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)

        def _scrub_metrics_dict(m):
            return {
                "n_slides": m["n"],
                "accuracy": m["accuracy"],
                "auc": _nan_to_none(m["auc"]),
                "f1_score": m["f1_score"],
                "specificity": _nan_to_none(m["specificity"]),
                "precision": _nan_to_none(m["precision"]),
                "recall": _nan_to_none(m["recall"]),
                "confusion_matrix": m["confusion_matrix"],
            }

        metrics_by_category_by_pred_pct = {}
        for pct in SLIDE_PRED_PCT_THRESHOLDS:
            by_cat = _metrics_by_bind_category_slide(
                all_labels, all_probs, all_preds, all_cat, all_slide_ids, pct
            )
            metrics_by_category_by_pred_pct[str(pct)] = {
                str(cat): _scrub_metrics_dict(m) for cat, m in by_cat.items()
            }

        print(
            "--- metrics by bind category @ X=60% (full per-X in metrics.json) ---",
            flush=True,
        )
        by_cat_60 = _metrics_by_bind_category_slide(
            all_labels, all_probs, all_preds, all_cat, all_slide_ids, 60
        )
        for cat in sorted(by_cat_60.keys()):
            m = by_cat_60[cat]
            auc_c = m["auc"]
            auc_cs = f"{auc_c:.4f}" if isinstance(auc_c, float) and not math.isnan(auc_c) else "nan"
            spec_c = m["specificity"]
            prec_c = m["precision"]
            rec_c = m["recall"]
            spec_s = f"{spec_c:.4f}" if not math.isnan(spec_c) else "nan"
            prec_s_c = f"{prec_c:.4f}" if not math.isnan(prec_c) else "nan"
            rec_s_c = f"{rec_c:.4f}" if not math.isnan(rec_c) else "nan"
            print(
                f"  [{cat}] n_slides={m['n']} acc={m['accuracy']:.4f} auc={auc_cs} "
                f"f1={m['f1_score']:.4f} spec={spec_s} prec={prec_s_c} rec={rec_s_c} "
                f"cm={m['confusion_matrix']}",
                flush=True,
            )

        per_slide_json = []
        for row in per_slide_rows:
            r = dict(row)
            r["match_at_pct"] = {
                str(p): bool(r["gt"] == r["pred_slide_at_pct"][str(p)])
                for p in SLIDE_PRED_PCT_THRESHOLDS
            }
            per_slide_json.append(r)

        metrics_by_pred_pct_json = {}
        for k, v in metrics_by_pred_pct.items():
            metrics_by_pred_pct_json[k] = {
                "n_slides": v["n_slides"],
                "n_normal_slides": v["n_normal_slides"],
                "n_abnormal_slides": v["n_abnormal_slides"],
                "accuracy": v["accuracy"],
                "auc": _nan_to_none(v["auc"]),
                "f1_score": v["f1_score"],
                "specificity": _nan_to_none(v["specificity"]),
                "precision": _nan_to_none(v["precision"]),
                "recall": _nan_to_none(v["recall"]),
                "confusion_matrix": v["confusion_matrix"],
            }

        slide_roc_json = None
        if slide_roc_block is not None:
            slide_roc_json = {
                "score_definition": slide_roc_block["score_definition"],
                "rule_positive_if": slide_roc_block["rule_positive_if"],
                "auc": _nan_to_none(slide_roc_block["auc"]),
                "youden_threshold_on_mean_prob": _nan_to_none(
                    slide_roc_block["youden_threshold_on_mean_prob"]
                ),
                "youden_sensitivity_recall": _nan_to_none(
                    slide_roc_block["youden_sensitivity_recall"]
                ),
                "youden_specificity": _nan_to_none(slide_roc_block["youden_specificity"]),
                "youden_J": _nan_to_none(slide_roc_block["youden_J"]),
                "roc_png": slide_roc_block["roc_png"],
                "note": slide_roc_block["note"],
            }

        results = {
            "eval_split": args.split,
            "val_loss": val_loss,
            "backbone": backbone,
            "dropout": dropout,
            "localization_dir": loc_dir if save_localization else None,
            "localization_saved_count": int(n_loc_saved) if save_localization else 0,
            "save_localization": bool(save_localization),
            "config_lr": paras.get("learning_rate"),
            "n_total_pairs": n_pairs,
            "slide_id_rule": "file_name path_parts[-4]",
            "slide_gt_rule": "max over pair labels in slide (any pair 1 => slide 1)",
            "slide_pred_rule": "frac of pairs with pred==1 >= X% => slide pred 1; X in 50..80 step 5",
            "slide_auc_score": "mean pair prob per slide",
            "slide_roc": slide_roc_json,
            "metrics_by_pred_pct": metrics_by_pred_pct_json,
            "per_slide": per_slide_json,
            "metrics_by_category_by_pred_pct": metrics_by_category_by_pred_pct,
            "checkpoint": args.ckpt,
        }
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_val()
