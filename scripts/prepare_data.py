#!/usr/bin/env python3
"""
数据准备脚本
用于准备训练数据
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import DataProcessor, create_dataset_info  # noqa: E402
from src.core.logger import setup_logger  # noqa: E402
from src.utils.common import ensure_dir  # noqa: E402

logger = setup_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument("--input", type=str, required=True, help="输入数据文件路径")
    parser.add_argument(
        "--output_dir", type=str, default="./data/processed", help="输出目录"
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt", "openai"],
        help="输入数据格式",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt", "openai"],
        help="输出数据格式",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument(
        "--dataset_name", type=str, default="dataset", help="数据集名称"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 确保输出目录存在
    ensure_dir(args.output_dir)

    # 创建数据处理器
    processor = DataProcessor(
        input_format=args.input_format,
        output_format=args.output_format,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 处理数据
    output_files = processor.process(
        input_file=args.input,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )

    # 为LLaMA-Factory创建dataset_info.json
    if args.output_format == "alpaca":
        create_dataset_info(
            dataset_name=args.dataset_name,
            file_name=f"{args.dataset_name}_train.json",
            output_dir=args.output_dir,
            format_type=args.output_format,
        )

    logger.info("=" * 60)
    logger.info("数据准备完成!")
    logger.info(f"训练数据: {output_files.get('train')}")
    logger.info(f"验证数据: {output_files.get('val')}")
    if "test" in output_files:
        logger.info(f"测试数据: {output_files.get('test')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
