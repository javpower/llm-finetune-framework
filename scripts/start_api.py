#!/usr/bin/env python3
"""
API服务启动脚本
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.server import start_server  # noqa: E402
from src.core.logger import setup_logger  # noqa: E402

logger = setup_logger("api")


def main():
    parser = argparse.ArgumentParser(description="启动API服务")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument(
        "--engine",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="推理引擎类型",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用4bit量化")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("启动API服务")
    logger.info("=" * 60)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"推理引擎: {args.engine}")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    # 启动服务
    start_server(
        model_path=args.model_path,
        engine_type=args.engine,
        host=args.host,
        port=args.port,
        load_in_4bit=args.load_in_4bit,
    )


if __name__ == "__main__":
    main()
