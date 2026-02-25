"""
日志管理模块
提供统一的日志配置和管理功能
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    def format(self, record):
        # 保存原始级别名称
        original_levelname = record.levelname

        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )

        # 格式化
        result = super().format(record)

        # 恢复原始级别名称
        record.levelname = original_levelname

        return result


def setup_logger(
    name: str = "llm-finetune",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志目录
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的日志文件数量
        format_string: 自定义格式字符串

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 清除现有处理器
    logger.handlers.clear()

    # 默认格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(format_string)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_to_file and log_dir:
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        # 创建轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 错误日志单独记录
        error_log_file = os.path.join(log_dir, f"{name}_error_{timestamp}.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

    return logger


def get_logger(name: str = "llm-finetune") -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    训练过程日志记录器
    专门用于记录训练过程中的指标和状态
    """

    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化训练日志记录器

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics = {}

        # 创建实验目录
        self.exp_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # 指标日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(self.exp_dir, f"metrics_{timestamp}.csv")

        # 写入CSV头
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            f.write("step,loss,learning_rate,eval_loss,epoch,time\n")

    def log_metric(self, step: int, **kwargs):
        """
        记录指标

        Args:
            step: 训练步数
            **kwargs: 指标键值对
        """
        self.metrics[step] = kwargs

        # 追加到CSV
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            loss = kwargs.get("loss", "")
            lr = kwargs.get("learning_rate", "")
            eval_loss = kwargs.get("eval_loss", "")
            epoch = kwargs.get("epoch", "")
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{step},{loss},{lr},{eval_loss},{epoch},{time_str}\n")

    def log_hyperparameters(self, hparams: dict):
        """
        记录超参数

        Args:
            hparams: 超参数字典
        """
        hparams_file = os.path.join(self.exp_dir, "hyperparameters.json")
        import json

        with open(hparams_file, "w", encoding="utf-8") as f:
            json.dump(hparams, f, ensure_ascii=False, indent=2)

    def log_config(self, config: dict):
        """
        记录配置

        Args:
            config: 配置字典
        """
        config_file = os.path.join(self.exp_dir, "config.json")
        import json

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def get_metrics(self) -> dict:
        """获取所有指标"""
        return self.metrics.copy()
