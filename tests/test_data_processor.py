"""
数据处理器测试
"""

import pytest
import json
import tempfile
import os
from src.data.processor import DataSample, DataConverter, DataValidator, DataProcessor


class TestDataSample:
    """测试DataSample类"""

    def test_create_sample(self):
        sample = DataSample(instruction="测试指令", input="测试输入", output="测试输出")
        assert sample.instruction == "测试指令"
        assert sample.input == "测试输入"
        assert sample.output == "测试输出"
        assert sample.history == []

    def test_to_dict(self):
        sample = DataSample(instruction="指令", input="输入", output="输出")
        data = sample.to_dict()
        assert data["instruction"] == "指令"
        assert data["input"] == "输入"
        assert data["output"] == "输出"

    def test_from_dict(self):
        data = {
            "instruction": "指令",
            "input": "输入",
            "output": "输出",
            "history": [["历史输入", "历史输出"]],
        }
        sample = DataSample.from_dict(data)
        assert sample.instruction == "指令"
        assert sample.history == [["历史输入", "历史输出"]]


class TestDataConverter:
    """测试DataConverter类"""

    def test_alpaca_to_sharegpt(self):
        alpaca_data = [{"instruction": "指令", "input": "输入", "output": "输出"}]

        sharegpt_data = DataConverter.alpaca_to_sharegpt(alpaca_data)

        assert len(sharegpt_data) == 1
        assert "conversations" in sharegpt_data[0]
        conversations = sharegpt_data[0]["conversations"]
        assert len(conversations) == 2
        assert conversations[0]["from"] == "human"
        assert conversations[1]["from"] == "gpt"

    def test_sharegpt_to_alpaca(self):
        sharegpt_data = [
            {
                "conversations": [
                    {"from": "human", "value": "问题"},
                    {"from": "gpt", "value": "回答"},
                ]
            }
        ]

        alpaca_data = DataConverter.sharegpt_to_alpaca(sharegpt_data)

        assert len(alpaca_data) == 1
        assert alpaca_data[0]["input"] == "问题"
        assert alpaca_data[0]["output"] == "回答"


class TestDataValidator:
    """测试DataValidator类"""

    def test_validate_alpaca_valid(self):
        data = [
            {"instruction": "指令", "input": "输入", "output": "输出"},
            {"instruction": "指令2", "input": "输入2", "output": "输出2"},
        ]

        is_valid, errors = DataValidator.validate_alpaca(data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_alpaca_invalid(self):
        data = [
            {"input": "输入"},  # 缺少output
        ]

        is_valid, errors = DataValidator.validate_alpaca(data)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_sharegpt_valid(self):
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "问题"},
                    {"from": "gpt", "value": "回答"},
                ]
            }
        ]

        is_valid, errors = DataValidator.validate_sharegpt(data)

        assert is_valid is True
        assert len(errors) == 0


class TestDataProcessor:
    """测试DataProcessor类"""

    def test_init(self):
        processor = DataProcessor(
            input_format="alpaca",
            output_format="sharegpt",
            train_ratio=0.8,
            val_ratio=0.2,
        )

        assert processor.input_format == "alpaca"
        assert processor.output_format == "sharegpt"
        assert processor.train_ratio == 0.8
        assert processor.val_ratio == 0.2

    def test_load_save_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试数据
            test_data = [{"instruction": "测试", "input": "输入", "output": "输出"}]

            # 保存
            file_path = os.path.join(tmpdir, "test.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            # 加载
            processor = DataProcessor()
            loaded_data = processor.load_data(file_path)

            assert len(loaded_data) == 1
            assert loaded_data[0]["instruction"] == "测试"

    def test_load_save_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试数据
            test_data = [
                {"instruction": "测试1", "output": "输出1"},
                {"instruction": "测试2", "output": "输出2"},
            ]

            # 保存
            file_path = os.path.join(tmpdir, "test.jsonl")
            with open(file_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # 加载
            processor = DataProcessor()
            loaded_data = processor.load_data(file_path)

            assert len(loaded_data) == 2

    def test_split_data(self):
        processor = DataProcessor(train_ratio=0.8, val_ratio=0.2, seed=42)

        data = [{"id": i} for i in range(100)]
        train, val, test = processor.split_data(data)

        assert len(train) == 80
        assert len(val) == 20
        assert len(test) == 0

    def test_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建输入文件
            input_file = os.path.join(tmpdir, "input.json")
            test_data = [
                {"instruction": f"指令{i}", "input": f"输入{i}", "output": f"输出{i}"}
                for i in range(10)
            ]
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            # 处理数据
            output_dir = os.path.join(tmpdir, "output")
            processor = DataProcessor(
                input_format="alpaca",
                output_format="alpaca",
                train_ratio=0.8,
                val_ratio=0.2,
                seed=42,
            )

            output_files = processor.process(
                input_file=input_file, output_dir=output_dir, dataset_name="test"
            )

            assert "train" in output_files
            assert "val" in output_files
            assert os.path.exists(output_files["train"])
            assert os.path.exists(output_files["val"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
