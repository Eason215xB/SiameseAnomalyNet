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

def count_abnormal_bind_types(cells, ann_h5_path, split_name):
    """
    遍历指定的细胞列表，统计其中异常染色体的 bind_type
    """
    print(f"正在统计 {split_name} 的异常 bind_type...")
    counter = Counter()
    
    # 打开 H5 标注文件
    with h5py.File(ann_h5_path, "r", swmr=True) as f_ann:
        for cell in tqdm(cells, desc=split_name):
            key = cell["key"]
            if key not in f_ann:
                continue
            
            # 读取并反序列化标注数据
            ann_bytes = np.array(f_ann[key])
            annotations = pickle.loads(ann_bytes).get("annotations", [])
            
            for an in annotations:
                # 使用你项目中的判别逻辑：判断是否为异常染色体
                if _binary_abnormal_cid(an) == 1:
                    # 提取 abnormal 字段作为分类名称
                    raw_str = str(an.get("abnormal", "")).strip().lower()
                    
                    if not raw_str or raw_str in ["none", "null"]:
                        bind_type = "Unspecified (未明确)"
                    elif raw_str.startswith("t(") or "t(" in raw_str:
                        bind_type = "t (易位)"
                    elif raw_str.startswith("der") or "der(" in raw_str:
                        bind_type = "der (衍生)"
                    elif raw_str.startswith("del"):
                        bind_type = "del (缺失)"
                    elif raw_str.startswith("inv"):
                        bind_type = "inv (倒位)"
                    elif raw_str.startswith("ins") or "ins(" in raw_str:
                        bind_type = "ins (插入)"
                    elif raw_str.startswith("add"):
                        bind_type = "add (附加)"
                    elif raw_str.startswith("r(") or raw_str == "r15":
                        bind_type = "r (环状)"
                    elif raw_str.startswith("idic") or raw_str.startswith("dic"):
                        bind_type = "dic/idic (双着丝粒)"
                    elif raw_str.startswith("dup"):
                        bind_type = "dup (重复)"
                    elif raw_str.startswith("i("):
                        bind_type = "i (等臂)"
                    elif "mar" in raw_str:
                        bind_type = "mar (标记)"
                    elif "qh+" in raw_str or "pstk+" in raw_str:
                        bind_type = "Polymorphism (多态性变异)"
                    else:
                        bind_type = f"Other (其他复合/罕见: {raw_str})"
                    
                    counter[bind_type] += 1
                    
    print(f"\n=========================================")
    print(f"=== {split_name} 异常类别 (bind_type) 统计 ===")
    print(f"=========================================")
    # 按数量降序排序输出
    for btype, count in counter.most_common():
        print(f" - {btype}: {count} 条")
    print(f"\n> 总计异常染色体数量: {sum(counter.values())} 条\n")
    
    return counter

def main():
    parser = argparse.ArgumentParser(description="统计数据集中异常染色体的 bind_type")
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
    count_abnormal_bind_types(train_cells, ann_h5_path, "训练集 (Train)")
    count_abnormal_bind_types(val_cells, ann_h5_path, "验证集 (Validation)")

if __name__ == "__main__":
    main()