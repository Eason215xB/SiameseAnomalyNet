"""验证脚本共用：仅从命令行构造 paras（不读取训练用的 config/paras.json）。"""
import argparse
import os
from typing import Any, Dict


def register_val_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="权重路径，如 best_siamese_model.pth",
    )
    parser.add_argument(
        "--h5-image",
        type=str,
        required=True,
        help="图像 H5（与训练 h5_files.image 一致）",
    )
    parser.add_argument(
        "--h5-annotation",
        type=str,
        required=True,
        help="标注 H5（与训练 h5_files.annotation 一致）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        required=True,
        help="dataset.key.json 路径列表（与训练配置 dataset 一致）",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        nargs="+",
        default=None,
        help="验证用 key 列表；省略则与 --dataset 相同",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="inference_results 与 metrics.json 的根目录；默认与 --ckpt 同目录",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=96, help="patch 正方形边长")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="验证 DataLoader batch；默认为 batch_size×2",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="关闭 DataLoader pin_memory",
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-crop-offset", type=int, default=4)
    parser.add_argument("--invert-p", type=float, default=0.0)
    parser.add_argument(
        "--vis-dark-mean-threshold",
        type=float,
        default=72.0,
        help="定位可视化整图暗度阈值",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="仅写入 metrics.json 的 config_lr，不参与推理",
    )
    parser.add_argument("--normal-b-multiplier", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def paras_dict_from_val_args(args: Any) -> Dict[str, Any]:
    log_path = args.log_path
    if not log_path:
        log_path = os.path.dirname(os.path.abspath(args.ckpt))
    ps = int(args.patch_size)
    paras: Dict[str, Any] = {
        "fold": int(args.fold),
        "seed": int(args.seed),
        "dataset": list(args.dataset),
        "h5_files": {
            "image": args.h5_image,
            "annotation": args.h5_annotation,
        },
        "patch_size": [ps, ps],
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": not bool(args.no_pin_memory),
        "log_path": log_path,
        "backbone": args.backbone,
        "dropout": float(args.dropout),
        "val_crop_offset": int(args.val_crop_offset),
        "invert_p": float(args.invert_p),
        "normal_b_multiplier": float(args.normal_b_multiplier),
        "vis_dark_mean_threshold": float(args.vis_dark_mean_threshold),
    }
    if args.val_dataset is not None:
        paras["val_dataset"] = list(args.val_dataset)
    if args.val_batch_size is not None:
        paras["val_batch_size"] = int(args.val_batch_size)
    if args.learning_rate is not None:
        paras["learning_rate"] = float(args.learning_rate)
    return paras
