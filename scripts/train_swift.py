#!/usr/bin/env python3
"""
MS-Swift 训练脚本
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.swift_trainer import SwiftTrainer  # noqa: E402
from src.core.logger import setup_logger  # noqa: E402

logger = setup_logger("train_swift")


def main():
    parser = argparse.ArgumentParser(description="使用MS-Swift训练模型")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/swift/qwen_lora.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--dataset", type=str, help="训练数据路径（覆盖配置文件）")
    parser.add_argument("--val_dataset", type=str, help="验证数据路径（覆盖配置文件）")
    parser.add_argument("--output_dir", type=str, help="输出目录（覆盖配置文件）")
    parser.add_argument("--model_name", type=str, help="模型名称（覆盖配置文件）")
    parser.add_argument("--epochs", type=int, help="训练轮数（覆盖配置文件）")
    parser.add_argument("--batch_size", type=int, help="批次大小（覆盖配置文件）")
    parser.add_argument("--learning_rate", type=float, help="学习率（覆盖配置文件）")
    parser.add_argument("--lora_rank", type=int, help="LoRA秩（覆盖配置文件）")
    parser.add_argument(
        "--merge_after_train", action="store_true", help="训练后自动合并模型"
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default="./outputs/merged/swift",
        help="合并模型输出目录",
    )

    args = parser.parse_args()

    # 创建训练器
    trainer = SwiftTrainer(config_path=args.config)

    # 覆盖配置
    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.val_dataset:
        overrides["val_dataset"] = args.val_dataset
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.model_name:
        overrides["model_id_or_path"] = args.model_name
    if args.epochs:
        overrides["num_train_epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.lora_rank:
        overrides["lora_rank"] = args.lora_rank

    trainer.update_config(**overrides)

    # 执行训练
    output_dir = trainer.train()

    # 合并模型（可选）
    if args.merge_after_train:
        logger.info("=" * 60)
        logger.info("开始合并模型")
        logger.info("=" * 60)

        # 获取最新检查点
        latest_checkpoint = trainer.get_latest_checkpoint(output_dir)

        if latest_checkpoint:
            merged_dir = trainer.export(
                checkpoint_dir=latest_checkpoint,
                output_dir=args.merged_dir,
                merge_lora=True,
            )
            logger.info(f"合并完成: {merged_dir}")
        else:
            logger.warning("未找到检查点，跳过合并")

    logger.info("=" * 60)
    logger.info("训练流程完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
