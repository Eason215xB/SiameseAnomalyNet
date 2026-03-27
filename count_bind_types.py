import os
import json
import h5py
import pickle
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm

# 复用你项目中的数据读取和异常判定逻辑
from utils.dataloader import get_train_data, get_val_data, _binary_abnormal_cid
from utils.bind_category import abnormal_str_to_bind_category

NOR_LABEL = "NOR (结构正常)"


def count_bind_types(cells, ann_h5_path, split_name):
    """
    遍历细胞列表，逐条染色体统计：
    - 结构正常：计入 NOR_LABEL；
    - 结构异常：按 abnormal 文本映射为粗粒度 bind 类别（与 bind_category 一致）。
    """
    print(f"正在统计 {split_name} 的染色体（正常 + 异常类别）...")
    counter = Counter()

    with h5py.File(ann_h5_path, "r", swmr=True) as f_ann:
        for cell in tqdm(cells, desc=split_name):
            key = cell["key"]
            if key not in f_ann:
                continue

            ann_bytes = np.array(f_ann[key])
            annotations = pickle.loads(ann_bytes).get("annotations", [])

            for an in annotations:
                if _binary_abnormal_cid(an) == 1:
                    bind_type = abnormal_str_to_bind_category(an.get("abnormal"))
                    counter[bind_type] += 1
                else:
                    counter[NOR_LABEL] += 1

    n_total = sum(counter.values())
    n_nor = counter.get(NOR_LABEL, 0)
    n_abn = n_total - n_nor

    print(f"\n=========================================")
    print(f"=== {split_name} 染色体类别统计（含正常）===")
    print(f"=========================================")
    for btype, count in counter.most_common():
        print(f" - {btype}: {count} 条")
    print(f"\n> 结构正常: {n_nor} 条 | 结构异常: {n_abn} 条 | 合计: {n_total} 条\n")

    return counter

def main():
    parser = argparse.ArgumentParser(description="统计数据集中每条染色体的类别（结构正常 + 异常 bind 粗类）")
    parser.add_argument("--config", type=str, default="./configs/paras.json", help="Path to config json")
    args = parser.parse_args()
    
    # 1. 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        paras = json.load(f)
        
    ann_h5_path = paras["h5_files"]["annotation"]
    if not os.path.exists(ann_h5_path):
        raise FileNotFoundError(f"找不到 H5 标注文件: {ann_h5_path}")
    
    print("正在解析 JSON 获取数据集划分...")
    # 2. 获取训练集和验证集的细胞列表 (自动过滤掉在 H5 中缺失的样本)
    train_cells = get_train_data(paras)
    val_cells = get_val_data(paras, test=True)
    
    # 3. 分别统计并输出结果
    count_bind_types(train_cells, ann_h5_path, "训练集 (Train)")
    count_bind_types(val_cells, ann_h5_path, "验证集 (Validation)")

if __name__ == "__main__":
    main()