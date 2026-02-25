"""
LLaMA-Factory 训练模块
提供基于LLaMA-Factory的训练功能
"""

import os
import subprocess
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class LLaMAFactoryTrainer:
    """
    LLaMA-Factory训练器
    封装LLaMA-Factory的训练功能
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化LLaMA-Factory训练器

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
            "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "template": "qwen",
            "dataset": "customer_service",
            "cutoff_len": 1024,
            "max_samples": 1000,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "output_dir": "./outputs/checkpoints/llamafactory",
            "logging_steps": 10,
            "save_steps": 50,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "do_train": True,
            "stage": "sft",
            "finetuning_type": "lora",
            "lora_target": "all",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "quantization_bit": 4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2.0e-4,
            "num_train_epochs": 5.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "bf16": False,
            "fp16": True,
            "flash_attn": "disabled",
            "gradient_checkpointing": True,
            "val_size": 0.1,
            "per_device_eval_batch_size": 1,
            "evaluation_strategy": "steps",
            "eval_steps": 50,
            "load_best_model_at_end": True,
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

    def save_config(self, save_path: str):
        """
        保存配置到YAML文件

        Args:
            save_path: 保存路径
        """
        import yaml

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"配置已保存: {save_path}")

    def train(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        执行训练

        Args:
            config_path: 配置文件路径（优先使用）
            output_dir: 输出目录
            **kwargs: 其他配置参数

        Returns:
            输出目录路径
        """
        # 设置环境变量
        os.environ["WANDB_DISABLED"] = "true"

        # 确定配置文件
        if config_path is None:
            if self.config_path:
                config_path = self.config_path
            else:
                # 创建临时配置文件
                config_path = "./temp_train_config.yaml"
                if output_dir:
                    self.config["output_dir"] = output_dir
                self.config.update(kwargs)
                self.save_config(config_path)

        # 确保输出目录存在
        output_dir = self.config.get("output_dir", "./outputs/checkpoints/llamafactory")
        os.makedirs(output_dir, exist_ok=True)

        # 构建命令
        cmd = ["llamafactory-cli", "train", config_path]

        logger.info("=" * 60)
        logger.info("开始训练 (LLaMA-Factory)")
        logger.info("=" * 60)
        logger.info(f"命令: {' '.join(cmd)}")

        # 执行训练
        try:
            subprocess.run(cmd, check=True, capture_output=False, text=True)
            logger.info("训练完成")
            return output_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"训练失败: {e}")
            raise

    def export(
        self,
        adapter_path: str,
        output_dir: str,
        model_name_or_path: Optional[str] = None,
        template: Optional[str] = None,
    ) -> str:
        """
        导出/合并模型

        Args:
            adapter_path: LoRA适配器路径
            output_dir: 输出目录
            model_name_or_path: 基础模型路径
            template: 模板名称

        Returns:
            输出目录路径
        """
        if model_name_or_path is None:
            model_name_or_path = self.config.get("model_name_or_path")
        if template is None:
            template = self.config.get("template")

        cmd = [
            "llamafactory-cli",
            "export",
            "--model_name_or_path",
            model_name_or_path,
            "--adapter_name_or_path",
            adapter_path,
            "--template",
            template,
            "--finetuning_type",
            "lora",
            "--export_dir",
            output_dir,
            "--export_size",
            "2",
            "--export_device",
            "cpu",
            "--export_legacy_format",
            "false",
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

    def start_api(
        self, model_path: str, template: Optional[str] = None, port: int = 8000
    ):
        """
        启动API服务

        Args:
            model_path: 模型路径
            template: 模板名称
            port: 端口号
        """
        if template is None:
            template = self.config.get("template", "qwen")

        cmd = [
            "llamafactory-cli",
            "api",
            "--model_name_or_path",
            model_path,
            "--template",
            template,
            "--finetuning_type",
            "full",
            "--port",
            str(port),
        ]

        logger.info("=" * 60)
        logger.info(f"启动API服务 (端口: {port})")
        logger.info("=" * 60)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"API服务启动失败: {e}")
            raise

    def start_webui(self, port: int = 7860):
        """
        启动Web UI

        Args:
            port: 端口号
        """
        cmd = ["llamafactory-cli", "webui"]

        logger.info("=" * 60)
        logger.info(f"启动Web UI (端口: {port})")
        logger.info("=" * 60)
        logger.info(f"请访问: http://localhost:{port}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Web UI启动失败: {e}")
            raise


class LLaMAFactoryInference:
    """
    LLaMA-Factory推理器
    使用Transformers直接推理
    """

    def __init__(
        self, model_path: str, device_map: str = "auto", torch_dtype: str = "float16"
    ):
        """
        初始化推理器

        Args:
            model_path: 模型路径
            device_map: 设备映射
            torch_dtype: 数据类型
        """
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"加载模型: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            dtype = getattr(torch, self.torch_dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device_map,
                trust_remote_code=True,
            )

            self.model.eval()
            logger.info("模型加载完成")

        except ImportError:
            logger.error("未安装transformers")
            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        生成回复

        Args:
            messages: 消息列表
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: top-p采样
            **kwargs: 其他参数

        Returns:
            生成的回复
        """
        if self.model is None:
            self.load_model()

        try:
            import torch

            # 应用模板
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 编码
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    **kwargs,
                )

            # 解码
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise

    def chat(self, system_prompt: str = ""):
        """
        交互式对话

        Args:
            system_prompt: 系统提示词
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

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

                messages.append({"role": "user", "content": user_input})

                response = self.generate(messages)
                print(f"助手: {response}\n")

                messages.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                print("\n退出对话")
                break
            except Exception as e:
                logger.error(f"生成失败: {e}")
                print(f"错误: {e}")
