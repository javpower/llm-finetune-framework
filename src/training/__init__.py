"""
训练模块
"""

from .swift_trainer import SwiftTrainer, SwiftInference
from .llamafactory_trainer import LLaMAFactoryTrainer, LLaMAFactoryInference

__all__ = [
    "SwiftTrainer",
    "SwiftInference",
    "LLaMAFactoryTrainer",
    "LLaMAFactoryInference",
]
