"""
推理引擎模块
提供统一的推理接口，支持多种后端
"""

from typing import Optional, List, Generator
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseInferenceEngine(ABC):
    """
    推理引擎基类
    定义统一的推理接口
    """

    def __init__(self, model_path: str, **kwargs):
        """
        初始化推理引擎

        Args:
            model_path: 模型路径
            **kwargs: 其他参数
        """
        self.model_path = model_path
        self.config = kwargs
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def generate(self, prompt: str, history: Optional[List] = None, **kwargs) -> str:
        """
        生成回复

        Args:
            prompt: 输入提示
            history: 历史对话
            **kwargs: 生成参数

        Returns:
            生成的回复
        """
        pass

    @abstractmethod
    def stream_generate(
        self, prompt: str, history: Optional[List] = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成回复

        Args:
            prompt: 输入提示
            history: 历史对话
            **kwargs: 生成参数

        Yields:
            生成的token
        """
        pass

    def chat(self, system_prompt: str = ""):
        """
        交互式对话

        Args:
            system_prompt: 系统提示词
        """
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

                response = self.generate(user_input, history)
                print(f"助手: {response}\n")

                history.append([user_input, response])

            except KeyboardInterrupt:
                print("\n退出对话")
                break
            except Exception as e:
                logger.error(f"生成失败: {e}")
                print(f"错误: {e}")


class TransformersEngine(BaseInferenceEngine):
    """
    Transformers推理引擎
    基于Hugging Face Transformers
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        """
        初始化Transformers推理引擎

        Args:
            model_path: 模型路径
            device_map: 设备映射
            torch_dtype: 数据类型
            load_in_4bit: 是否使用4bit量化
            load_in_8bit: 是否使用8bit量化
        """
        super().__init__(model_path, **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

    def load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"加载模型: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # 设置加载参数
            load_kwargs = {"trust_remote_code": True}

            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.torch_dtype),
                )
            elif self.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            else:
                load_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
                load_kwargs["device_map"] = self.device_map

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs
            )

            self.model.eval()
            logger.info("模型加载完成")

        except ImportError as e:
            logger.error(f"导入失败: {e}")
            raise

    def generate(
        self,
        prompt: str,
        history: Optional[List] = None,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> str:
        """
        生成回复

        Args:
            prompt: 用户输入
            history: 历史对话
            system_prompt: 系统提示词
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: top-p采样
            top_k: top-k采样
            repetition_penalty: 重复惩罚

        Returns:
            生成的回复
        """
        if self.model is None:
            self.load_model()

        try:
            import torch

            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if history:
                for human, assistant in history:
                    messages.append({"role": "user", "content": human})
                    messages.append({"role": "assistant", "content": assistant})

            messages.append({"role": "user", "content": prompt})

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
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
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

    def stream_generate(
        self,
        prompt: str,
        history: Optional[List] = None,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成回复

        Args:
            prompt: 用户输入
            history: 历史对话
            system_prompt: 系统提示词
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: top-p采样

        Yields:
            生成的token
        """
        if self.model is None:
            self.load_model()

        try:
            from transformers import TextIteratorStreamer
            from threading import Thread

            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if history:
                for human, assistant in history:
                    messages.append({"role": "user", "content": human})
                    messages.append({"role": "assistant", "content": assistant})

            messages.append({"role": "user", "content": prompt})

            # 应用模板
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 编码
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # 创建streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # 生成参数
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "streamer": streamer,
                **kwargs,
            }

            # 在后台线程生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # 流式输出
            for text in streamer:
                yield text

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise


class VLLMEngine(BaseInferenceEngine):
    """
    vLLM推理引擎
    基于vLLM的高性能推理
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        quantization: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化vLLM推理引擎

        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU内存利用率
            max_model_len: 最大模型长度
            quantization: 量化方式（awq, gptq等）
        """
        super().__init__(model_path, **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.llm = None

    def load_model(self):
        """加载模型"""
        try:
            from vllm import LLM

            logger.info(f"加载vLLM模型: {self.model_path}")

            load_kwargs = {
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "trust_remote_code": True,
            }

            if self.quantization:
                load_kwargs["quantization"] = self.quantization

            self.llm = LLM(self.model_path, **load_kwargs)

            logger.info("vLLM模型加载完成")

        except ImportError:
            logger.error("未安装vllm，请运行: pip install vllm")
            raise

    def generate(
        self,
        prompt: str,
        history: Optional[List] = None,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs,
    ) -> str:
        """
        生成回复

        Args:
            prompt: 用户输入
            history: 历史对话
            system_prompt: 系统提示词
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: top-p采样
            top_k: top-k采样

        Returns:
            生成的回复
        """
        if self.llm is None:
            self.load_model()

        try:
            from vllm import SamplingParams

            # 构建完整提示
            full_prompt = self._build_prompt(prompt, history, system_prompt)

            # 采样参数
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_new_tokens,
                **kwargs,
            )

            # 生成
            outputs = self.llm.generate(full_prompt, sampling_params)

            # 返回结果
            return outputs[0].outputs[0].text.strip()

        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise

    def stream_generate(
        self,
        prompt: str,
        history: Optional[List] = None,
        system_prompt: str = "",
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成回复

        Args:
            prompt: 用户输入
            history: 历史对话
            system_prompt: 系统提示词

        Yields:
            生成的token
        """
        # vLLM的流式生成需要特殊处理
        # 这里简化处理，直接返回完整结果
        response = self.generate(prompt, history, system_prompt, **kwargs)
        yield response

    def _build_prompt(
        self, prompt: str, history: Optional[List], system_prompt: str
    ) -> str:
        """
        构建完整提示

        Args:
            prompt: 用户输入
            history: 历史对话
            system_prompt: 系统提示词

        Returns:
            完整提示
        """
        # 这里简化处理，实际应该使用模型的chat template
        full_prompt = ""

        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"

        if history:
            for human, assistant in history:
                full_prompt += f"User: {human}\nAssistant: {assistant}\n\n"

        full_prompt += f"User: {prompt}\nAssistant:"

        return full_prompt


class InferenceEngineFactory:
    """
    推理引擎工厂
    根据配置创建对应的推理引擎
    """

    @staticmethod
    def create_engine(
        engine_type: str, model_path: str, **kwargs
    ) -> BaseInferenceEngine:
        """
        创建推理引擎

        Args:
            engine_type: 引擎类型 (transformers, vllm)
            model_path: 模型路径
            **kwargs: 其他参数

        Returns:
            推理引擎实例
        """
        if engine_type == "transformers":
            return TransformersEngine(model_path, **kwargs)
        elif engine_type == "vllm":
            return VLLMEngine(model_path, **kwargs)
        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")
