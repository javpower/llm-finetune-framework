"""
数据处理模块
提供数据加载、转换、验证和预处理功能
"""

import json
import os
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """数据样本"""

    instruction: str = ""
    input: str = ""
    output: str = ""
    history: List[List[str]] = None
    system: str = ""

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "history": self.history,
            "system": self.system,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSample":
        """从字典创建"""
        return cls(
            instruction=data.get("instruction", ""),
            input=data.get("input", ""),
            output=data.get("output", ""),
            history=data.get("history", []),
            system=data.get("system", ""),
        )


class DataConverter:
    """
    数据格式转换器
    支持多种数据格式之间的转换
    """

    @staticmethod
    def alpaca_to_sharegpt(data: List[Dict]) -> List[Dict]:
        """
        将Alpaca格式转换为ShareGPT格式

        Args:
            data: Alpaca格式数据

        Returns:
            ShareGPT格式数据
        """
        sharegpt_data = []
        for item in data:
            messages = []

            # 系统消息
            if item.get("system"):
                messages.append({"from": "system", "value": item["system"]})

            # 历史消息
            for human, assistant in item.get("history", []):
                messages.append({"from": "human", "value": human})
                messages.append({"from": "gpt", "value": assistant})

            # 当前对话
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            if instruction:
                query = f"{instruction}\n{input_text}" if input_text else instruction
            else:
                query = input_text

            messages.append({"from": "human", "value": query})
            messages.append({"from": "gpt", "value": item.get("output", "")})

            sharegpt_data.append({"conversations": messages})

        return sharegpt_data

    @staticmethod
    def sharegpt_to_alpaca(data: List[Dict]) -> List[Dict]:
        """
        将ShareGPT格式转换为Alpaca格式

        Args:
            data: ShareGPT格式数据

        Returns:
            Alpaca格式数据
        """
        alpaca_data = []
        for item in data:
            conversations = item.get("conversations", [])

            # 提取系统消息
            system = ""
            history = []

            i = 0
            while i < len(conversations):
                conv = conversations[i]
                if conv["from"] == "system":
                    system = conv["value"]
                    i += 1
                elif conv["from"] == "human":
                    if (
                        i + 1 < len(conversations)
                        and conversations[i + 1]["from"] == "gpt"
                    ):
                        history.append([conv["value"], conversations[i + 1]["value"]])
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

            # 最后一条作为当前对话
            if history:
                current = history.pop()
                alpaca_data.append(
                    {
                        "instruction": "",
                        "input": current[0],
                        "output": current[1],
                        "history": history,
                        "system": system,
                    }
                )

        return alpaca_data

    @staticmethod
    def openai_to_alpaca(data: List[Dict]) -> List[Dict]:
        """
        将OpenAI格式转换为Alpaca格式

        Args:
            data: OpenAI格式数据

        Returns:
            Alpaca格式数据
        """
        alpaca_data = []
        for item in data:
            messages = item.get("messages", [])

            system = ""
            history = []

            # 提取系统消息
            if messages and messages[0]["role"] == "system":
                system = messages[0]["content"]
                messages = messages[1:]

            # 构建历史
            for i in range(0, len(messages) - 2, 2):
                if (
                    messages[i]["role"] == "user"
                    and messages[i + 1]["role"] == "assistant"
                ):
                    history.append([messages[i]["content"], messages[i + 1]["content"]])

            # 最后一条作为当前对话
            if len(messages) >= 2:
                last_user = messages[-2]
                last_assistant = messages[-1]
                if (
                    last_user["role"] == "user"
                    and last_assistant["role"] == "assistant"
                ):
                    alpaca_data.append(
                        {
                            "instruction": "",
                            "input": last_user["content"],
                            "output": last_assistant["content"],
                            "history": history,
                            "system": system,
                        }
                    )

        return alpaca_data


class DataValidator:
    """
    数据验证器
    验证数据格式和内容的正确性
    """

    @staticmethod
    def validate_alpaca(data: List[Dict]) -> Tuple[bool, List[str]]:
        """
        验证Alpaca格式数据

        Args:
            data: 待验证数据

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        if not isinstance(data, list):
            return False, ["数据必须是列表格式"]

        for i, item in enumerate(data):
            # 检查必要字段
            if "instruction" not in item and "input" not in item:
                errors.append(f"第{i}条数据缺少instruction或input字段")

            if "output" not in item:
                errors.append(f"第{i}条数据缺少output字段")

            # 检查字段类型
            if "history" in item and not isinstance(item["history"], list):
                errors.append(f"第{i}条数据的history必须是列表")

        return len(errors) == 0, errors

    @staticmethod
    def validate_sharegpt(data: List[Dict]) -> Tuple[bool, List[str]]:
        """
        验证ShareGPT格式数据

        Args:
            data: 待验证数据

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        if not isinstance(data, list):
            return False, ["数据必须是列表格式"]

        for i, item in enumerate(data):
            if "conversations" not in item:
                errors.append(f"第{i}条数据缺少conversations字段")
                continue

            conversations = item["conversations"]
            if not isinstance(conversations, list):
                errors.append(f"第{i}条数据的conversations必须是列表")
                continue

            for j, conv in enumerate(conversations):
                if "from" not in conv or "value" not in conv:
                    errors.append(f"第{i}条数据的第{j}条对话缺少from或value字段")

        return len(errors) == 0, errors


class DataProcessor:
    """
    数据处理器
    提供完整的数据处理流程
    """

    def __init__(
        self,
        input_format: str = "alpaca",
        output_format: str = "alpaca",
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int = 42,
    ):
        """
        初始化数据处理器

        Args:
            input_format: 输入数据格式
            output_format: 输出数据格式
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        # 验证比例参数
        for name, value in [
            ("train_ratio", train_ratio),
            ("val_ratio", val_ratio),
            ("test_ratio", test_ratio),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间，当前值: {value}")
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio > 1:
            raise ValueError(
                f"训练、验证和测试比例之和不能超过 1，"
                f"当前总和: {total_ratio}"
            )
        
        self.input_format = input_format
        self.output_format = output_format
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.converter = DataConverter()
        self.validator = DataValidator()

        random.seed(seed)

    def load_data(self, file_path: str) -> List[Dict]:
        """
        加载数据文件

        Args:
            file_path: 数据文件路径

        Returns:
            加载的数据
        """
        logger.info(f"加载数据: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        data = []
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif file_ext == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            logger.info(f"成功加载 {len(data)} 条数据")
            return data

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def save_data(self, data: List[Dict], file_path: str):
        """
        保存数据文件

        Args:
            data: 待保存数据
            file_path: 保存路径
        """
        logger.info(f"保存数据: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == ".json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif file_ext == ".jsonl":
                with open(file_path, "w", encoding="utf-8") as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            logger.info(f"成功保存 {len(data)} 条数据")

        except Exception as e:
            logger.error(f"数据保存失败: {e}")
            raise

    def convert_format(self, data: List[Dict]) -> List[Dict]:
        """
        转换数据格式

        Args:
            data: 原始数据

        Returns:
            转换后的数据
        """
        if self.input_format == self.output_format:
            return data

        logger.info(f"转换数据格式: {self.input_format} -> {self.output_format}")

        # 输入格式 -> Alpaca
        if self.input_format == "sharegpt":
            data = self.converter.sharegpt_to_alpaca(data)
        elif self.input_format == "openai":
            data = self.converter.openai_to_alpaca(data)

        # Alpaca -> 输出格式
        if self.output_format == "sharegpt":
            data = self.converter.alpaca_to_sharegpt(data)

        return data

    def validate(self, data: List[Dict]) -> bool:
        """
        验证数据

        Args:
            data: 待验证数据

        Returns:
            是否有效
        """
        logger.info("验证数据格式...")

        if self.output_format == "alpaca":
            is_valid, errors = self.validator.validate_alpaca(data)
        elif self.output_format == "sharegpt":
            is_valid, errors = self.validator.validate_sharegpt(data)
        else:
            logger.warning(f"未实现的验证格式: {self.output_format}")
            return True

        if not is_valid:
            logger.error("数据验证失败:\n" + "\n".join(errors[:10]))
            if len(errors) > 10:
                logger.error(f"... 还有 {len(errors) - 10} 个错误")
        else:
            logger.info("数据验证通过")

        return is_valid

    def split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        分割数据集

        Args:
            data: 完整数据

        Returns:
            (训练集, 验证集, 测试集)
        """
        logger.info("分割数据集...")

        # 随机打乱
        data_copy = data.copy()
        random.shuffle(data_copy)

        n = len(data_copy)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_data = data_copy[:n_train]
        val_data = data_copy[n_train : n_train + n_val]
        test_data = data_copy[n_train + n_val :] if self.test_ratio > 0 else []

        logger.info(
            f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}"
        )

        return train_data, val_data, test_data

    def process(
        self, input_file: str, output_dir: str, dataset_name: str = "dataset"
    ) -> Dict[str, str]:
        """
        执行完整的数据处理流程

        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            dataset_name: 数据集名称

        Returns:
            输出文件路径字典
        """
        logger.info("=" * 50)
        logger.info("开始数据处理流程")
        logger.info("=" * 50)

        # 1. 加载数据
        data = self.load_data(input_file)

        # 2. 转换格式
        data = self.convert_format(data)

        # 3. 验证数据
        if not self.validate(data):
            raise ValueError("数据验证失败，请检查数据格式")

        # 4. 分割数据
        train_data, val_data, test_data = self.split_data(data)

        # 5. 保存数据
        os.makedirs(output_dir, exist_ok=True)

        output_files = {}

        if self.output_format == "alpaca":
            # Alpaca格式保存为JSON
            train_file = os.path.join(output_dir, f"{dataset_name}_train.json")
            val_file = os.path.join(output_dir, f"{dataset_name}_val.json")

            self.save_data(train_data, train_file)
            self.save_data(val_data, val_file)

            output_files["train"] = train_file
            output_files["val"] = val_file

        elif self.output_format in ["sharegpt", "openai"]:
            # 其他格式保存为JSONL
            train_file = os.path.join(output_dir, "train.jsonl")
            val_file = os.path.join(output_dir, "val.jsonl")

            self.save_data(train_data, train_file)
            self.save_data(val_data, val_file)

            output_files["train"] = train_file
            output_files["val"] = val_file

        # 保存测试集
        if test_data:
            test_file = os.path.join(output_dir, "test.jsonl")
            self.save_data(test_data, test_file)
            output_files["test"] = test_file

        logger.info("=" * 50)
        logger.info("数据处理完成")
        logger.info("=" * 50)

        return output_files


def create_dataset_info(
    dataset_name: str, file_name: str, output_dir: str, format_type: str = "alpaca"
):
    """
    创建LLaMA-Factory的dataset_info.json

    Args:
        dataset_name: 数据集名称
        file_name: 数据文件名
        output_dir: 输出目录
        format_type: 数据格式
    """
    dataset_info = {
        dataset_name: {
            "file_name": file_name,
            "formatting": format_type,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }

    info_file = os.path.join(output_dir, "dataset_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    logger.info(f"创建dataset_info.json: {info_file}")
