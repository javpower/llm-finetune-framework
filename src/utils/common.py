"""
通用工具模块
提供各种辅助功能
"""

import os
import json
import hashlib
import random
import string
from typing import Any, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_id(length: int = 12) -> str:
    """
    生成随机ID

    Args:
        length: ID长度

    Returns:
        随机ID字符串
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def generate_timestamp() -> str:
    """
    生成时间戳字符串

    Returns:
        时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_md5(file_path: str) -> str:
    """
    计算文件MD5

    Args:
        file_path: 文件路径

    Returns:
        MD5哈希值
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_dir(dir_path: str):
    """
    确保目录存在

    Args:
        dir_path: 目录路径
    """
    os.makedirs(dir_path, exist_ok=True)


def get_file_size(file_path: str) -> int:
    """
    获取文件大小

    Args:
        file_path: 文件路径

    Returns:
        文件大小（字节）
    """
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的字符串
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    保存JSON文件

    Args:
        data: 数据
        file_path: 文件路径
        indent: 缩进
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: str) -> Any:
    """
    加载JSON文件

    Args:
        file_path: 文件路径

    Returns:
        加载的数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict], file_path: str):
    """
    保存JSONL文件

    Args:
        data: 数据列表
        file_path: 文件路径
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载JSONL文件

    Args:
        file_path: 文件路径

    Returns:
        数据列表
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """
    递归合并字典

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_gpu_info() -> List[Dict]:
    """
    获取GPU信息

    Returns:
        GPU信息列表
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append(
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i),
                }
            )

        return gpu_info
    except ImportError:
        return []


def print_gpu_info():
    """打印GPU信息"""
    gpu_info = get_gpu_info()

    if not gpu_info:
        print("未检测到GPU")
        return

    print("=" * 60)
    print("GPU信息:")
    print("=" * 60)

    for gpu in gpu_info:
        print(f"GPU {gpu['id']}: {gpu['name']}")
        print(f"  总内存: {format_file_size(gpu['memory_total'])}")
        print(f"  已分配: {format_file_size(gpu['memory_allocated'])}")
        print(f"  已预留: {format_file_size(gpu['memory_reserved'])}")

    print("=" * 60)


def count_parameters(model) -> Dict[str, int]:
    """
    统计模型参数

    Args:
        model: 模型

    Returns:
        参数字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_percentage": (
            100 * trainable_params / total_params if total_params > 0 else 0
        ),
    }


def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class ProgressTracker:
    """
    进度跟踪器
    """

    def __init__(self, total: int, desc: str = "Progress"):
        """
        初始化进度跟踪器

        Args:
            total: 总步数
            desc: 描述
        """
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = datetime.now()

    def update(self, n: int = 1):
        """
        更新进度

        Args:
            n: 更新的步数
        """
        self.current += n
        self._print_progress()

    def _print_progress(self):
        """打印进度"""
        percentage = 100 * self.current / self.total if self.total > 0 else 0
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "N/A"

        print(
            f"\r{self.desc}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - ETA: {eta_str}",
            end="",
            flush=True,
        )

        if self.current >= self.total:
            print()  # 换行


class Timer:
    """
    计时器
    """

    def __init__(self, name: str = "Timer"):
        """
        初始化计时器

        Args:
            name: 计时器名称
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """进入上下文"""
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        """退出上下文"""
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} 耗时: {format_time(elapsed)}")

    @property
    def elapsed(self) -> float:
        """获取已耗时"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()
