#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考 count_bind_types.py：遍历细胞 H5 标注，统计指定 category_id（默认 24）的染色体：
  - 该 category 下染色体总条数；
  - 其中结构异常条数；
  - 异常按 abnormal 文本映射的粗粒度类型条数（与 bind_category 一致）。
"""
import os
import json
import argparse
import h5py
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

from utils.dataloader import get_train_data, get_val_data, _binary_abnormal_cid, _karyotype_id
from utils.bind_category import abnormal_str_to_bind_category


def count_category_stats(cells, ann_h5_path, split_name, target_cat_id: int):
    """
    仅统计 category_id == target_cat_id 的染色体：
    - 总条数；
    - 结构正常 / 结构异常条数；
    - 异常子集按 bind 粗类计数。
    """
    print(f"正在统计 {split_name} 中 category_id={target_cat_id} 的染色体...")
    n_total = 0
    n_normal = 0
    n_abnormal = 0
    type_counter = Counter()

    with h5py.File(ann_h5_path, "r", swmr=True) as f_ann:
        for cell in tqdm(cells, desc=split_name):
            key = cell["key"]
            if key not in f_ann:
                continue

            ann_bytes = np.array(f_ann[key])
            annotations = pickle.loads(ann_bytes).get("annotations", [])

            for an in annotations:
                kid = _karyotype_id(an)
                if kid != target_cat_id:
                    continue

                n_total += 1
                if _binary_abnormal_cid(an) == 1:
                    n_abnormal += 1
                    bind_type = abnormal_str_to_bind_category(an.get("abnormal"))
                    type_counter[bind_type] += 1
                else:
                    n_normal += 1

    print(f"\n=========================================")
    print(f"=== {split_name} | category_id = {target_cat_id} ===")
    print(f"=========================================")
    print(f"  染色体总条数:           {n_total}")
    print(f"  其中结构正常:           {n_normal}")
    print(f"  其中结构异常:           {n_abnormal}")
    if n_abnormal > 0:
        print(f"\n  --- 异常类型分布（共 {n_abnormal} 条）---")
        for btype, cnt in type_counter.most_common():
            print(f"    - {btype}: {cnt} 条")
    else:
        print(f"\n  （无结构异常，无类型分布）")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="统计指定 category_id 下染色体条数、异常条数及异常类型分布"
    )
    parser.add_argument("--config", type=str, default="./configs/paras_new.json", help="训练用 paras.json")
    parser.add_argument(
        "--cat-id",
        type=int,
        default=24,
        help="要统计的 category_id（核型编号），默认 24",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        paras = json.load(f)

    ann_h5_path = paras["h5_files"]["annotation"]
    if not os.path.exists(ann_h5_path):
        raise FileNotFoundError(f"找不到 H5 标注文件: {ann_h5_path}")

    print("正在解析配置并加载细胞列表...")
    train_cells = get_train_data(paras)
    val_cells = get_val_data(paras, test=True)

    count_category_stats(train_cells, ann_h5_path, "训练集 (Train)", args.cat_id)
    count_category_stats(val_cells, ann_h5_path, "验证集 (Validation)", args.cat_id)


if __name__ == "__main__":
    main()
