"""
LLM微调框架
"""

__version__ = "1.0.0"
__author__ = "AI Team"

from .core.config import get_config, ConfigManager
from .core.logger import setup_logger, get_logger

__all__ = [
    "get_config",
    "ConfigManager",
    "setup_logger",
    "get_logger",
]
