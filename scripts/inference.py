#!/usr/bin/env python3
"""
推理脚本
支持多种推理引擎
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.engine import InferenceEngineFactory  # noqa: E402
from src.core.logger import setup_logger  # noqa: E402

logger = setup_logger("inference")


def main():
    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument(
        "--engine",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="推理引擎类型",
    )
    parser.add_argument("--prompt", type=str, help="单条推理提示")
    parser.add_argument("--interactive", action="store_true", help="交互式对话模式")
    parser.add_argument("--system_prompt", type=str, default="", help="系统提示词")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="最大生成token数"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用4bit量化")
    parser.add_argument("--load_in_8bit", action="store_true", help="使用8bit量化")

    args = parser.parse_args()

    # 创建推理引擎
    engine_kwargs = {
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
    }

    logger.info(f"创建推理引擎: {args.engine}")
    engine = InferenceEngineFactory.create_engine(
        args.engine, args.model_path, **engine_kwargs
    )

    # 加载模型
    logger.info("加载模型...")
    engine.load_model()

    if args.interactive:
        # 交互式模式
        engine.chat(system_prompt=args.system_prompt)
    elif args.prompt:
        # 单条推理
        response = engine.generate(
            args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\n用户: {args.prompt}")
        print(f"助手: {response}\n")
    else:
        # 默认进入交互式模式
        print("未提供--prompt，进入交互式模式...")
        engine.chat(system_prompt=args.system_prompt)


if __name__ == "__main__":
    main()
