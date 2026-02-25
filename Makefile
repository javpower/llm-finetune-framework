# ============================================
# LLM微调框架 - Makefile
# ============================================

.PHONY: help install install-swift install-llamafactory data train-swift train-llamafactory merge infer api test clean

# 默认目标
help:
	@echo "LLM微调框架 - 可用命令:"
	@echo ""
	@echo "  make install              - 安装基础依赖"
	@echo "  make install-swift        - 安装MS-Swift"
	@echo "  make install-llamafactory - 安装LLaMA-Factory"
	@echo "  make data                 - 准备训练数据"
	@echo "  make train-swift          - 使用MS-Swift训练"
	@echo "  make train-llamafactory   - 使用LLaMA-Factory训练"
	@echo "  make merge-swift          - 合并MS-Swift模型"
	@echo "  make merge-llamafactory   - 合并LLaMA-Factory模型"
	@echo "  make infer                - 运行推理"
	@echo "  make api                  - 启动API服务"
	@echo "  make test                 - 运行测试"
	@echo "  make clean                - 清理输出文件"
	@echo ""

# 安装依赖
install:
	pip install -r requirements.txt

install-swift:
	pip install "ms-swift[all]" -U

install-llamafactory:
	git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory
	cd /tmp/LLaMA-Factory && pip install -e ".[torch,metrics]"

# 数据准备
data:
	python scripts/prepare_data.py \
		--input data/examples/customer_service.json \
		--output_dir data/processed \
		--input_format alpaca \
		--output_format alpaca \
		--dataset_name customer_service

# 训练
train-swift:
	python scripts/train_swift.py \
		--config configs/swift/qwen_lora.yaml \
		--dataset data/processed/train.jsonl \
		--val_dataset data/processed/val.jsonl \
		--merge_after_train

train-llamafactory:
	python scripts/train_llamafactory.py \
		--config configs/llamafactory/qwen_lora.yaml \
		--dataset customer_service \
		--merge_after_train

# 合并模型
merge-swift:
	@echo "合并MS-Swift模型..."
	python -c "from src.training.swift_trainer import SwiftTrainer; \
		trainer = SwiftTrainer(); \
		ckpt = trainer.get_latest_checkpoint('./outputs/checkpoints/swift'); \
		trainer.export(ckpt, './outputs/merged/swift')"

merge-llamafactory:
	@echo "合并LLaMA-Factory模型..."
	python -c "from src.training.llamafactory_trainer import LLaMAFactoryTrainer; \
		trainer = LLaMAFactoryTrainer(); \
		trainer.export('./outputs/checkpoints/llamafactory', './outputs/merged/llamafactory')"

# 推理
infer:
	python scripts/inference.py \
		--model_path ./outputs/merged/swift \
		--engine transformers \
		--interactive \
		--load_in_4bit

# 启动API
api:
	python scripts/start_api.py \
		--model_path ./outputs/merged/swift \
		--engine transformers \
		--host 0.0.0.0 \
		--port 8000 \
		--load_in_4bit

# 测试
test:
	python -m pytest tests/ -v

# 清理
clean:
	rm -rf outputs/checkpoints/*
	rm -rf outputs/logs/*
	rm -rf data/processed/*
	rm -f temp_*.yaml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 完整流程
pipeline-swift: data train-swift
	@echo "MS-Swift训练流程完成!"

pipeline-llamafactory: data train-llamafactory
	@echo "LLaMA-Factory训练流程完成!"
