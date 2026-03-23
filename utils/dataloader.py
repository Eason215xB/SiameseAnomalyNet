import io
import os
import cv2
import json
import h5py
import pickle
import random
import logging
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as F

logger = logging.getLogger('main')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# IMAGENET_MEAN = [0.0, 0.0, 0.0]
# IMAGENET_STD = [1.0, 1.0, 1.0]

from utils.utils import merge_dataset


def _binary_abnormal_cid(an):
    """
    染色体级二分类：0=结构正常，1=异常。依据标注字段，不使用 category_id
    （category_id 表示核型类别 / 几号染色体，与是否异常无关）。

    与导出逻辑一致：正常为 abnormal 空；异常为 abnormal 非空字符串；
    另：bind_type 字符串中含 "mar" 视为异常。对 JSON null、数值 0、布尔值按正常处理。
    """
    raw = an.get("abnormal")
    
    # 1. 兼容数字标注 (如 1 代表异常，0 代表正常)
    if isinstance(raw, (int, float)):
        if raw != 0:
            return 1
            
    # 2. 严格要求是字符串(文字)且非空
    elif isinstance(raw, str) and raw.strip() != "":
        return 1
        
    # 3. 兜底策略：检查 bind_type 中是否包含 mar
    bind_type_str = str(an.get("bind_type", "") or "").lower()
    if "mar" in bind_type_str:
        return 1

    return 0


def _abnormal_content_str(an):
    """abnormal 字段原文（无异常描述则为空串），供验证可视化文件名。"""
    raw = an.get("abnormal")
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return ""
    if isinstance(raw, (int, float)):
        if raw == 0:
            return ""
        rf = float(raw)
        return str(int(rf)) if rf == int(rf) else str(raw)
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()


def _karyotype_id(an):
    """核型 category_id，转为可比较的 int；缺失或无法解析时返回 None。"""
    v = an.get("category_id", None)
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


# =====================================================================
# 1. 数据集获取与清洗
# =====================================================================

def get_train_data(paras):
    dataset_paths = paras["dataset"]
    if isinstance(dataset_paths, str): 
        dataset_paths = [dataset_paths]

    dataset = merge_dataset(*[json.load(open(d)) for d in dataset_paths])
    [val.update(dict(key=key)) for key, val in dataset["data"].items()]
    
    fold_key = str(paras["fold"]) if str(paras["fold"]) in dataset["fold"] else paras["fold"]
    raw_train_keys = dataset["fold"][fold_key].get("train", [])
    raw_train_ds = [dataset["data"][k] for k in raw_train_keys if k in dataset["data"]]

    train_ds_valid = []
    missing_count = 0
    with h5py.File(paras["h5_files"]["image"], "r", swmr=True) as f_img, \
         h5py.File(paras["h5_files"]["annotation"], "r", swmr=True) as f_ann:
        for x in raw_train_ds:
            if x["key"] in f_img and x["key"] in f_ann:
                train_ds_valid.append(x)
            else:
                missing_count += 1
    
    if missing_count > 0:
        logger.warning(f"训练集扫雷完成：发现并过滤了 {missing_count} 个在 H5 中缺失的样本！")

    logger.info("train cell number (after filtering): %d", len(train_ds_valid))
    return train_ds_valid

def get_val_data(paras, test=True):
    dataset_paths = paras.get("val_dataset", paras["dataset"])
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    dataset = merge_dataset(*[json.load(open(d)) for d in dataset_paths])
    [val.update(dict(key=key)) for key, val in dataset["data"].items()]

    fold_key = str(paras["fold"]) if str(paras["fold"]) in dataset["fold"] else paras["fold"]
    val_keys = dataset["fold"][fold_key].get("val", [])
    if test:
        val_keys += dataset["fold"][fold_key].get("test", [])
        
    raw_val_ds = [dataset["data"][k] for k in val_keys if k in dataset["data"]]

    val_ds_valid = []
    missing_count = 0
    with h5py.File(paras["h5_files"]["image"], "r", swmr=True) as f_img, \
         h5py.File(paras["h5_files"]["annotation"], "r", swmr=True) as f_ann:
        for x in raw_val_ds:
            if x["key"] in f_img and x["key"] in f_ann:
                val_ds_valid.append(x)
            else:
                missing_count += 1
                
    if missing_count > 0:
        logger.warning(f"验证集扫雷完成：过滤了 {missing_count} 个缺失样本！")

    logger.info("val cell number: %d", len(val_ds_valid))
    return val_ds_valid

# =====================================================================
# 2. 单条染色体裁剪助手
# =====================================================================

def crop_single_chromosome(image_np, an, offset=4):
    """
    专门为单条染色体抠图和生成 Mask 优化的函数
    返回: 裁剪后的图像 numpy 数组 和 掩码 numpy 数组
    """
    if "segmentation" not in an or len(an["segmentation"]) == 0:
        raise ValueError("Annotation missing segmentation")

    idx = np.argmax([len(a) for a in an["segmentation"]])
    pnts = np.array(an["segmentation"][idx], dtype=np.int32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(pnts.reshape(-1, 2))
    pnts = [pnts]

    beg = end = offset
    sbox = np.maximum(np.array([y, x]) - beg, 0), np.minimum(
        np.array([y + h, x + w]) + end, image_np.shape[:2]
    )

    slice_box = (slice(sbox[0][0], sbox[1][0]), slice(sbox[0][1], sbox[1][1]))

    cropped_img = image_np[slice_box].copy()

    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, pnts, 255)
    cropped_mask = mask[slice_box].copy()

    return cropped_img, cropped_mask

# =====================================================================
# 3. 双流网络专用 Dataset
# =====================================================================

class SiameseChromosomeDataset(Dataset):
    """
    B 流：异常（标签 _binary_abnormal_cid==1）+ 按配置比例抽样的正常 B（同源池内需另有正常作 A）。
    A 流：恒为同 cell、同 category_id、与 B 不同 anno_idx 的另一条正常染色体。
    """

    def __init__(
        self,
        valid_cells_list,
        paras,
        resize=256,
        imagesize=224,
        is_train=False,
        log_build_stats=None,
    ):
        super().__init__()
        self.paras = paras
        self.h5_img_path = paras["h5_files"]["image"]
        self.h5_ann_path = paras["h5_files"]["annotation"]
        self.crop_offset = int(paras.get("val_crop_offset", 4))
        self.is_train = is_train
        
        # 预处理参数
        self.resize = resize
        self.is_train = is_train
        
        # 几何变换参数（训练时随机，验证时固定）
        self.rotation_degrees = 30 if is_train else 0
        self.translate = (0.15, 0.15) if is_train else (0, 0)
        self.scale = (0.9, 1.1) if is_train else (1.0, 1.0)
        self.flip_p = 0.5 if is_train else 0.0
        
        # 颜色变换（仅图像）
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        ) if is_train else None
        self.gaussian_blur = transforms.GaussianBlur(
            kernel_size=3, sigma=(0.1, 0.5)
        ) if is_train else None
        
        # 归一化
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        # 构建数据集样本
        self.all_items, self._normals_by_key_cat, self._stats = self._build_dataset(
            valid_cells_list, log_build_stats
        )
        
        if self.is_train:
            random.shuffle(self.all_items)
    
    def _sync_transform(self, img, mask, seed=None):
        """
        对图像和mask进行同步几何变换，确保两者对齐
        Args:
            img: numpy array (H, W, C)
            mask: numpy array (H, W)
            seed: 随机种子，确保img和mask的随机操作一致
        Returns:
            tensor_img, tensor_mask
        """
        # 转为PIL
        img_pil = Image.fromarray(img.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode='L')
        
        # Resize
        img_pil = F.resize(img_pil, (self.resize, self.resize))
        mask_pil = F.resize(mask_pil, (self.resize, self.resize))
        
        if self.is_train and seed is not None:
            # 设置随机种子，确保img和mask的随机操作完全一致
            random.seed(seed)
            
            # 同步随机旋转
            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                img_pil = F.rotate(img_pil, angle, fill=0)
                mask_pil = F.rotate(mask_pil, angle, fill=0)
            
            # 同步随机仿射变换
            if self.translate != (0, 0) or self.scale != (1.0, 1.0):
                # 简单的平移+缩放实现
                if random.random() < 0.5:
                    translate_px = (
                        int(random.uniform(-self.translate[0], self.translate[0]) * self.resize),
                        int(random.uniform(-self.translate[1], self.translate[1]) * self.resize)
                    )
                    scale_factor = random.uniform(self.scale[0], self.scale[1])
                    img_pil = F.affine(img_pil, angle=0, translate=translate_px, scale=scale_factor, shear=0, fill=0)
                    mask_pil = F.affine(mask_pil, angle=0, translate=translate_px, scale=scale_factor, shear=0, fill=0)
            
            # 同步翻转
            if random.random() < self.flip_p:
                img_pil = F.hflip(img_pil)
                mask_pil = F.hflip(mask_pil)
            if random.random() < self.flip_p:
                img_pil = F.vflip(img_pil)
                mask_pil = F.vflip(mask_pil)
        
        # 转Tensor
        tensor_img = F.to_tensor(img_pil)
        tensor_mask = F.to_tensor(mask_pil)
        
        # 图像的颜色变换（mask不需要）
        if self.is_train and self.color_jitter is not None:
            tensor_img = self.color_jitter(tensor_img)
        if self.is_train and self.gaussian_blur is not None:
            tensor_img = self.gaussian_blur(tensor_img)
        
        # 归一化（只对图像）
        tensor_img = self.normalize(tensor_img)
        
        return tensor_img, tensor_mask

    def _build_dataset(self, valid_cells_list, log_build_stats):
        """构建数据集样本"""
        # key -> category_id -> 该 cell 内「结构正常」的 anno_idx 列表（同源池）
        normals_by_key_cat = defaultdict(lambda: defaultdict(list))
        abnormal_candidates = []  # (key, anno_idx_B)

        with h5py.File(self.h5_ann_path, "r", swmr=True) as f_ann:
            for cell_data in valid_cells_list:
                key = cell_data["key"]
                ann_bytes = np.array(f_ann[key])
                annotations = pickle.loads(ann_bytes).get("annotations", [])

                for anno_idx, an in enumerate(annotations):
                    if "segmentation" not in an or len(an["segmentation"]) == 0:
                        continue
                    kid = _karyotype_id(an)
                    if kid is None:
                        continue
                    cid = _binary_abnormal_cid(an)
                    if cid == 0:
                        normals_by_key_cat[key][kid].append(anno_idx)
                    else:
                        abnormal_candidates.append((key, anno_idx, kid))

        abnormal_items = []
        abnormal_items_reverse = []  # 新增：反向样本 (A异常, B正常)
        skipped_no_homolog_ab = 0
        for key_B, anno_idx_B, kid in abnormal_candidates:
            pool = normals_by_key_cat[key_B].get(kid, [])
            valid_a = [i for i in pool if i != anno_idx_B]
            if not valid_a:
                skipped_no_homolog_ab += 1
                continue
            # 原样本：A正常, B异常
            abnormal_items.append((key_B, anno_idx_B, kid, True))
            # 反向样本：A异常, B正常（用于训练双向检测）
            # 从同源池选一条正常B'作为新的"B流"，原异常B作为"A流"
            if len(valid_a) > 0:
                anno_idx_normal = random.choice(valid_a)
                abnormal_items_reverse.append((key_B, anno_idx_B, anno_idx_normal, kid, True))

        # 可作「正常 B」的 (key, anno_idx, kid)：同源正常池至少 2 条，才能另选 A
        normal_b_candidates = []
        for key, kid_map in normals_by_key_cat.items():
            for kid, pool in kid_map.items():
                if len(pool) < 2:
                    continue
                for anno_idx_B in pool:
                    normal_b_candidates.append((key, anno_idx_B, kid))

        multiplier = float(self.paras.get("normal_b_multiplier", 1.0))
        seed = int(self.paras.get("seed", 42))
        rng = random.Random(seed)
        n_ab = len(abnormal_items)
        target_nb = int(round(n_ab * multiplier)) if multiplier > 0 else 0
        normal_items = []
        if target_nb > 0 and normal_b_candidates:
            if target_nb <= len(normal_b_candidates):
                picked = rng.sample(normal_b_candidates, target_nb)
            else:
                logger.warning(
                    "正常 B 目标 %d 大于可配对候选 %d，已放回抽样凑满",
                    target_nb,
                    len(normal_b_candidates),
                )
                picked = [rng.choice(normal_b_candidates) for _ in range(target_nb)]
            normal_items = [(k, i, kid, False) for k, i, kid in picked]

        # 合并所有样本：原异常B + 反向异常A + 正常B
        # 使用统一的5元组格式: (key, anno_idx_A, anno_idx_B, kid, is_anomaly)
        # 其中 is_anomaly 表示"这对染色体是否有差异"（有差异=1，无差异=0）
        all_items = []
        
        # 类型1: A正常, B异常 (原设计)
        for key, anno_idx_B, kid, _ in abnormal_items:
            pool = normals_by_key_cat[key][kid]
            valid_a = [i for i in pool if i != anno_idx_B]
            anno_idx_A = rng.choice(valid_a) if len(valid_a) > 0 else None
            if anno_idx_A is not None:
                all_items.append((key, anno_idx_A, anno_idx_B, kid, 1))  # label=1: 有差异
        
        # 类型2: A异常, B正常 (新增反向样本)
        for key, anno_idx_abnormal, anno_idx_normal, kid, _ in abnormal_items_reverse:
            # A是异常，B是正常，label=1（有差异）
            all_items.append((key, anno_idx_abnormal, anno_idx_normal, kid, 1))
        
        # 类型3: A正常, B正常
        for key, anno_idx_B, kid, _ in normal_items:
            pool = normals_by_key_cat[key][kid]
            valid_a = [i for i in pool if i != anno_idx_B]
            anno_idx_A = rng.choice(valid_a) if len(valid_a) > 0 else None
            if anno_idx_A is not None:
                all_items.append((key, anno_idx_A, anno_idx_B, kid, 0))  # label=0: 无差异

        if log_build_stats is None:
            log_build_stats = int(os.environ.get("LOCAL_RANK", "0")) == 0
        if log_build_stats:
            if skipped_no_homolog_ab > 0:
                logger.warning(
                    "异常染色体中，有 %d 条在同 cell 同 category_id 下无可用正常同源，已跳过",
                    skipped_no_homolog_ab,
                )
            logger.info(
                "Siamese 样本: B 异常=%d, B 正常=%d (normal_b_multiplier=%g), 总计=%d",
                n_ab,
                len(normal_items),
                multiplier,
                len(all_items),
            )

        # 保存统计信息供后续查询
        # 重新计算各类样本数
        n_type1 = len([x for x in all_items if x[4] == 1])  # label=1: 有差异（A正常B异常 或 A异常B正常）
        n_type0 = len([x for x in all_items if x[4] == 0])  # label=0: 无差异（A正常B正常）
        
        stats = {
            "total": len(all_items),
            "label_1_pairs": n_type1,  # 有差异的对（原：B异常 或 新增：A异常B正常）
            "label_0_pairs": n_type0,  # 无差异的对（A正常B正常）
            "abnormal_forward": len(abnormal_items),  # A正常B异常
            "abnormal_reverse": len(abnormal_items_reverse),  # A异常B正常（新增）
            "normal_pairs": len(normal_items),  # A正常B正常
            "skipped_no_homolog": skipped_no_homolog_ab,
            "multiplier": multiplier,
        }
        
        return all_items, normals_by_key_cat, stats

    def get_stats(self):
        """返回数据集统计信息"""
        return self._stats

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        # 新的5元组格式: (key, anno_idx_A, anno_idx_B, kid, is_anomaly)
        key, anno_idx_A, anno_idx_B, kid, is_anomaly = self.all_items[idx]
        key_A = key
        key_B = key

        # 开始从 H5 中提取真正的图像字节并抠图
        # 为了防止反复开关文件带来 IO 瓶颈，我们在提取时合并读取
        with h5py.File(self.h5_img_path, "r", swmr=True) as f_img, \
             h5py.File(self.h5_ann_path, "r", swmr=True) as f_ann:
             
            # --- 处理目标流 B ---
            img_bytes_B = np.array(f_img[key_B])
            ann_bytes_B = np.array(f_ann[key_B])
            image_np_B = np.asarray(Image.open(io.BytesIO(img_bytes_B)).convert("RGB"))
            annotations_B = pickle.loads(ann_bytes_B).get("annotations", [])
            an_B = annotations_B[anno_idx_B]
            lbl_B = _binary_abnormal_cid(an_B)
            
            crop_img_B, crop_mask_B = crop_single_chromosome(image_np_B, an_B, self.crop_offset)
            
            # --- 处理基准流 A ---
            # 优化：如果 A 和 B 在同一个细胞，无需重新解码图片
            if key_A == key_B:
                image_np_A = image_np_B
                annotations_A = annotations_B
            else:
                img_bytes_A = np.array(f_img[key_A])
                ann_bytes_A = np.array(f_ann[key_A])
                image_np_A = np.asarray(Image.open(io.BytesIO(img_bytes_A)).convert("RGB"))
                annotations_A = pickle.loads(ann_bytes_A).get("annotations", [])
            
            an_A = annotations_A[anno_idx_A]
            lbl_A = _binary_abnormal_cid(an_A)
            
            # 验证：计算实际的差异标签，应该与all_items中存储的is_anomaly一致
            actual_diff = float(lbl_A != lbl_B)  # A和B标签不同=有差异
            expected_diff = float(is_anomaly)
            if actual_diff != expected_diff and (lbl_A == 1 or lbl_B == 1):
                # 只有一个是异常时应该是有差异的
                logger.warning(
                    f"标签不一致: key={key} A_idx={anno_idx_A}(lbl={lbl_A}) "
                    f"B_idx={anno_idx_B}(lbl={lbl_B}), 期望diff={expected_diff}, 实际diff={actual_diff}"
                )
                
            # A流可以是正常也可以是异常（对称设计）
            crop_img_A, crop_mask_A = crop_single_chromosome(image_np_A, an_A, self.crop_offset)

        # 转换为 Tensor（使用同步变换确保 img 和 mask 对齐）
        # A和B使用不同的随机种子，但同一对(A,B)内的img和mask使用相同种子
        seed_a = random.randint(0, 2**32 - 1)
        seed_b = random.randint(0, 2**32 - 1)
        
        tensor_img_A, tensor_mask_A = self._sync_transform(crop_img_A, crop_mask_A, seed=seed_a)
        tensor_img_B, tensor_mask_B = self._sync_transform(crop_img_B, crop_mask_B, seed=seed_b)

        return {
            "img_A": tensor_img_A,
            "mask_A": tensor_mask_A,
            "img_B": tensor_img_B,
            "mask_B": tensor_mask_B,
            "is_anomaly": torch.tensor([is_anomaly], dtype=torch.float32),
            "image_name_B": f"{key}:{anno_idx_B:02d}",
            "image_name_A": f"{key}:{anno_idx_A:02d}",
            "cell_key": key,
            "label_A": float(lbl_A),
            "label_B": float(lbl_B),
            "abnormal_content_A": _abnormal_content_str(an_A),
            "abnormal_content_B": _abnormal_content_str(an_B),
        }