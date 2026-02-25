"""
MS-Swift 训练模块
提供基于MS-Swift的训练功能
"""

import os
import subprocess
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class SwiftTrainer:
    """
    MS-Swift训练器
    封装MS-Swift的训练功能
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化Swift训练器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._set_default_config()

    def _set_default_config(self):
        """设置默认配置"""
        self.config = {
            "model_type": "qwen2_5-7b-instruct",
            "model_id_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "template_type": "qwen",
            "sft_type": "lora",
            "tuner_backend": "peft",
            "dtype": "fp16",
            "output_dir": "./outputs/checkpoints/swift",
            "max_length": 1024,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout_p": 0.05,
            "lora_target_modules": "ALL",
            "num_train_epochs": 5,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2.0e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.5,
            "gradient_checkpointing": True,
            "use_flash_attn": False,
            "quantization_bit": 4,
            "eval_steps": 50,
            "save_steps": 50,
            "save_total_limit": 3,
            "logging_steps": 10,
        }

    def load_config(self, config_path: str):
        """
        加载配置文件

        Args:
            config_path: 配置文件路径
        """
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"加载配置: {config_path}")

    def update_config(self, **kwargs):
        """
        更新配置

        Args:
            **kwargs: 配置项
        """
        self.config.update(kwargs)
        logger.info(f"更新配置: {kwargs}")

    def _build_command(self) -> List[str]:
        """
        构建训练命令

        Returns:
            命令列表
        """
        cmd = ["swift", "sft"]

        for key, value in self.config.items():
            if value is None:
                continue

            # 转换布尔值
            if isinstance(value, bool):
                value = "true" if value else "false"

            cmd.append(f"--{key}")
            cmd.append(str(value))

        return cmd

    def train(
        self,
        dataset_path: str,
        val_dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        执行训练

        Args:
            dataset_path: 训练数据路径
            val_dataset_path: 验证数据路径
            output_dir: 输出目录
            **kwargs: 其他配置参数

        Returns:
            输出目录路径
        """
        # 更新配置
        self.config["dataset"] = dataset_path
        if val_dataset_path:
            self.config["val_dataset"] = val_dataset_path
        if output_dir:
            self.config["output_dir"] = output_dir

        self.config.update(kwargs)

        # 确保输出目录存在
        os.makedirs(self.config["output_dir"], exist_ok=True)

        # 构建命令
        cmd = self._build_command()

        logger.info("=" * 60)
        logger.info("开始训练 (MS-Swift)")
        logger.info("=" * 60)
        logger.info(f"命令: {' '.join(cmd)}")

        # 执行训练
        try:
            subprocess.run(cmd, check=True, capture_output=False, text=True)
            logger.info("训练完成")
            return self.config["output_dir"]
        except subprocess.CalledProcessError as e:
            logger.error(f"训练失败: {e}")
            raise

    def export(
        self, checkpoint_dir: str, output_dir: str, merge_lora: bool = True
    ) -> str:
        """
        导出/合并模型

        Args:
            checkpoint_dir: 检查点目录
            output_dir: 输出目录
            merge_lora: 是否合并LoRA权重

        Returns:
            输出目录路径
        """
        cmd = [
            "swift",
            "export",
            "--ckpt_dir",
            checkpoint_dir,
            "--merge_lora",
            "true" if merge_lora else "false",
            "--output_dir",
            output_dir,
        ]

        logger.info("=" * 60)
        logger.info("开始导出模型")
        logger.info("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        try:
            subprocess.run(cmd, check=True, capture_output=False, text=True)
            logger.info(f"导出完成: {output_dir}")
            return output_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"导出失败: {e}")
            raise

    def get_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        获取最新的检查点

        Args:
            checkpoint_dir: 检查点目录

        Returns:
            最新检查点路径
        """
        if not os.path.exists(checkpoint_dir):
            return None

        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-"):
                checkpoints.append(item)

        if not checkpoints:
            return None

        # 按版本号排序
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest = os.path.join(checkpoint_dir, checkpoints[-1])

        logger.info(f"最新检查点: {latest}")
        return latest


class SwiftInference:
    """
    MS-Swift推理器
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "qwen2_5-7b-instruct",
        load_in_4bit: bool = True,
    ):
        """
        初始化推理器

        Args:
            model_path: 模型路径
            model_type: 模型类型
            load_in_4bit: 是否使用4bit量化
        """
        self.model_path = model_path
        self.model_type = model_type
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.template = None

    def load_model(self):
        """加载模型"""
        try:
            from swift.llm import get_model_tokenizer, get_template

            logger.info(f"加载模型: {self.model_path}")

            self.model, self.tokenizer = get_model_tokenizer(
                model_type=self.model_type,
                model_id_or_path=self.model_path,
                load_in_4bit=self.load_in_4bit,
            )

            self.template = get_template(
                model_type=self.model_type,
                tokenizer=self.tokenizer,
            )

            logger.info("模型加载完成")

        except ImportError:
            logger.error("未安装ms-swift，请运行: pip install ms-swift")
            raise

    def generate(self, query: str, history: Optional[List] = None, **kwargs) -> tuple:
        """
        生成回复

        Args:
            query: 用户输入
            history: 历史对话
            **kwargs: 生成参数

        Returns:
            (回复, 更新后的历史)
        """
        if self.model is None:
            self.load_model()

        try:
            from swift.llm import inference

            response, new_history = inference(
                self.model, self.template, query, history if history else None, **kwargs
            )

            return response, new_history

        except ImportError:
            logger.error("未安装ms-swift")
            raise

    def chat(self):
        """交互式对话"""
        history = []

        print("\n" + "=" * 60)
        print("开始对话（输入 'quit' 或 'exit' 退出）")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("用户: ").strip()
                if user_input.lower() in ["quit", "exit", "退出"]:
                    break

                if not user_input:
                    continue

                response, history = self.generate(user_input, history)
                print(f"助手: {response}\n")

            except KeyboardInterrupt:
                print("\n退出对话")
                break
            except Exception as e:
                logger.error(f"生成失败: {e}")
                print(f"错误: {e}")
