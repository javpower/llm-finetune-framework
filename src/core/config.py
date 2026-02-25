"""
配置管理模块
提供统一的配置加载、验证和管理功能
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""

    name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_type: str = "qwen2_5-7b-instruct"
    template: str = "qwen"
    context_length: int = 32768


@dataclass
class LoRAConfig:
    """LoRA配置"""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "ALL"


@dataclass
class QLoRAConfig:
    """QLoRA配置"""

    quantization_bit: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"


@dataclass
class TrainingConfig:
    """训练配置"""

    training_type: str = "lora"  # lora, qlora, full
    epochs: int = 5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.5
    max_length: int = 1024
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    gradient_checkpointing: bool = True
    flash_attention: bool = False
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)


@dataclass
class DataConfig:
    """数据配置"""

    format: str = "alpaca"  # alpaca, sharegpt, openai
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    preprocessing_num_workers: int = 16
    overwrite_cache: bool = True


@dataclass
class InferenceConfig:
    """推理配置"""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class APIConfig:
    """API配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


@dataclass
class PathConfig:
    """路径配置"""

    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    merged_dir: str = "./outputs/merged"
    log_dir: str = "./outputs/logs"
    cache_dir: str = "~/.cache/huggingface"


class ConfigManager:
    """
    配置管理器
    提供配置的加载、验证和访问功能
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，默认为None时加载默认配置
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.model: ModelConfig = ModelConfig()
        self.training: TrainingConfig = TrainingConfig()
        self.data: DataConfig = DataConfig()
        self.inference: InferenceConfig = InferenceConfig()
        self.api: APIConfig = APIConfig()
        self.paths: PathConfig = PathConfig()

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._load_default_config()

    def _load_default_config(self):
        """加载默认配置"""
        default_config_path = (
            Path(__file__).parent.parent.parent / "configs" / "base.yaml"
        )
        if default_config_path.exists():
            self.load_config(str(default_config_path))
        else:
            logger.warning(f"默认配置文件不存在: {default_config_path}")

    def load_config(self, config_path: str):
        """
        从YAML文件加载配置

        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            self._parse_config()
            logger.info(f"配置加载成功: {config_path}")
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise

    def _parse_config(self):
        """解析配置到各个配置类"""
        # 解析模型配置
        if "model" in self.config:
            model_cfg = self.config["model"]
            if "default_model" in model_cfg:
                default_model = model_cfg["default_model"]
                if (
                    "supported_models" in model_cfg
                    and default_model in model_cfg["supported_models"]
                ):
                    supported = model_cfg["supported_models"][default_model]
                    # 将配置文件中的 'type' 映射到 ModelConfig 的 'model_type'
                    model_kwargs = supported.copy()
                    if "type" in model_kwargs:
                        model_kwargs["model_type"] = model_kwargs.pop("type")
                    self.model = ModelConfig(**model_kwargs)

        # 解析训练配置
        if "training" in self.config:
            train_cfg = self.config["training"]
            self.training = TrainingConfig(
                training_type=train_cfg.get("type", "lora"),
                epochs=train_cfg.get("epochs", 5),
                batch_size=train_cfg.get("batch_size", 1),
                gradient_accumulation_steps=train_cfg.get(
                    "gradient_accumulation_steps", 8
                ),
                learning_rate=train_cfg.get("learning_rate", 2.0e-4),
                weight_decay=train_cfg.get("weight_decay", 0.01),
                warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
                max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
                max_length=train_cfg.get("max_length", 1024),
                optimizer=train_cfg.get("optimizer", "adamw_torch"),
                lr_scheduler=train_cfg.get("lr_scheduler", "cosine"),
                gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
                flash_attention=train_cfg.get("flash_attention", False),
                save_steps=train_cfg.get("save_steps", 50),
                eval_steps=train_cfg.get("eval_steps", 50),
                logging_steps=train_cfg.get("logging_steps", 10),
                save_total_limit=train_cfg.get("save_total_limit", 3),
                load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
            )

            # 解析LoRA配置
            if "lora" in train_cfg:
                lora_cfg = train_cfg["lora"]
                self.training.lora = LoRAConfig(
                    rank=lora_cfg.get("rank", 16),
                    alpha=lora_cfg.get("alpha", 32),
                    dropout=lora_cfg.get("dropout", 0.05),
                    target_modules=lora_cfg.get("target_modules", "ALL"),
                )

            # 解析QLoRA配置
            if "qlora" in train_cfg:
                qlora_cfg = train_cfg["qlora"]
                self.training.qlora = QLoRAConfig(
                    quantization_bit=qlora_cfg.get("quantization_bit", 4),
                    double_quant=qlora_cfg.get("double_quant", True),
                    quant_type=qlora_cfg.get("quant_type", "nf4"),
                )

        # 解析数据配置
        if "data" in self.config:
            data_cfg = self.config["data"]
            self.data = DataConfig(
                format=data_cfg.get("format", "alpaca"),
                train_ratio=data_cfg.get("train_ratio", 0.9),
                val_ratio=data_cfg.get("val_ratio", 0.1),
                test_ratio=data_cfg.get("test_ratio", 0.0),
                preprocessing_num_workers=data_cfg.get("preprocessing_num_workers", 16),
                overwrite_cache=data_cfg.get("overwrite_cache", True),
            )

        # 解析推理配置
        if "inference" in self.config:
            inf_cfg = self.config["inference"]
            self.inference = InferenceConfig(
                max_new_tokens=inf_cfg.get("max_new_tokens", 512),
                temperature=inf_cfg.get("temperature", 0.7),
                top_p=inf_cfg.get("top_p", 0.9),
                top_k=inf_cfg.get("top_k", 50),
                repetition_penalty=inf_cfg.get("repetition_penalty", 1.1),
                do_sample=inf_cfg.get("do_sample", True),
            )

        # 解析API配置
        if "api" in self.config:
            api_cfg = self.config["api"]
            self.api = APIConfig(
                host=api_cfg.get("host", "0.0.0.0"),
                port=api_cfg.get("port", 8000),
                workers=api_cfg.get("workers", 1),
            )

        # 解析路径配置
        if "paths" in self.config:
            path_cfg = self.config["paths"]
            self.paths = PathConfig(
                data_dir=path_cfg.get("data_dir", "./data"),
                output_dir=path_cfg.get("output_dir", "./outputs"),
                checkpoint_dir=path_cfg.get("checkpoint_dir", "./outputs/checkpoints"),
                merged_dir=path_cfg.get("merged_dir", "./outputs/merged"),
                log_dir=path_cfg.get("log_dir", "./outputs/logs"),
                cache_dir=path_cfg.get("cache_dir", "~/.cache/huggingface"),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的路径
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        设置配置项

        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save_config(self, save_path: str):
        """
        保存配置到文件

        Args:
            save_path: 保存路径
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"配置保存成功: {save_path}")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model.__dict__,
            "training": {
                **self.training.__dict__,
                "lora": self.training.lora.__dict__,
                "qlora": self.training.qlora.__dict__,
            },
            "data": self.data.__dict__,
            "inference": self.inference.__dict__,
            "api": self.api.__dict__,
            "paths": self.paths.__dict__,
        }


# 全局配置实例
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigManager实例
    """
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = ConfigManager(config_path)
    return _config_instance


def reload_config(config_path: Optional[str] = None):
    """
    重新加载配置

    Args:
        config_path: 配置文件路径
    """
    global _config_instance
    _config_instance = ConfigManager(config_path)
