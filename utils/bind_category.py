"""
与 count_bind_types 一致的 abnormal 文本 -> 粗粒度类别名，用于验证集按类别统计指标。
"""


def abnormal_str_to_bind_category(raw) -> str:
    """
    将单条染色体的 abnormal 字段映射为粗粒度类别（与 count_bind_types 规则一致）。
    raw 可为 None / 数字 / 字符串。
    """
    raw_str = str(raw or "").strip().lower()
    if not raw_str or raw_str in ("none", "null"):
        return "Unspecified (未明确)"
    if raw_str.startswith("t(") or "t(" in raw_str:
        return "t (易位)"
    if raw_str.startswith("der") or "der(" in raw_str:
        return "der (衍生)"
    if raw_str.startswith("del"):
        return "del (缺失)"
    if raw_str.startswith("inv"):
        return "inv (倒位)"
    if raw_str.startswith("ins") or "ins(" in raw_str:
        return "ins (插入)"
    if raw_str.startswith("add"):
        return "add (附加)"
    if raw_str.startswith("r(") or raw_str == "r15":
        return "r (环状)"
    if raw_str.startswith("idic") or raw_str.startswith("dic"):
        return "dic/idic (双着丝粒)"
    if raw_str.startswith("dup"):
        return "dup (重复)"
    if raw_str.startswith("i("):
        return "i (等臂)"
    if "mar" in raw_str:
        return "mar (标记)"
    if "qh+" in raw_str or "pstk+" in raw_str:
        return "Polymorphism (多态性变异)"
    return f"Other (其他复合/罕见: {raw_str})"


def pair_sample_bind_category(label_a, label_b, abnormal_content_a, abnormal_content_b) -> str:
    """
    为一条 Siamese 样本定义「类别」用于分组统计：
    - 仅 A 结构异常：用 A 的 abnormal 映射类别；
    - 仅 B 异常或 A/B 均异常：用 B 的 abnormal 映射（与热力图目标侧一致）；
    - A/B 均结构正常：NOR_pair（无异常描述可归类）。
    """
    a = float(label_a) >= 0.5
    b = float(label_b) >= 0.5
    if a and not b:
        return abnormal_str_to_bind_category(abnormal_content_a)
    if b:
        return abnormal_str_to_bind_category(abnormal_content_b)
    return "NOR_pair"
