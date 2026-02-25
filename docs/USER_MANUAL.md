# LLM微调框架 - 用户手册

## 目录
1. [项目简介](#项目简介)
2. [环境安装](#环境安装)
3. [快速开始](#快速开始)
4. [数据准备](#数据准备)
5. [模型训练](#模型训练)
6. [模型合并](#模型合并)
7. [模型推理](#模型推理)
8. [API部署](#api部署)
9. [配置说明](#配置说明)
10. [常见问题](#常见问题)

---

## 项目简介

本项目是一个**企业级大语言模型微调框架**，整合了业界最流行的两个微调框架：

- **MS-Swift**: 阿里出品，中文友好，训练速度快
- **LLaMA-Factory**: 社区最流行，功能丰富，Web UI支持

### 主要特性

- 统一的接口，支持两种训练框架
- 支持 LoRA、QLoRA、全参数微调
- 支持多种数据格式（Alpaca、ShareGPT、OpenAI）
- OpenAI兼容的API接口
- 高性能推理引擎（Transformers、vLLM）
- Docker一键部署
- 完善的日志和监控

---

## 环境安装

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+), Windows WSL2
- **Python**: 3.10+
- **CUDA**: 11.8+ 或 12.1+
- **GPU**: 显存 >= 16GB (推荐)

### 安装步骤

#### 1. 克隆项目

```bash
git clone <项目仓库>
cd llm-finetune-framework
```

#### 2. 创建虚拟环境

```bash
conda create -n llm-finetune python=3.10
conda activate llm-finetune
```

#### 3. 安装基础依赖

```bash
pip install -r requirements.txt
```

#### 4. 安装训练框架（二选一或都安装）

**安装 MS-Swift:**
```bash
make install-swift
# 或
pip install "ms-swift[all]" -U
```

**安装 LLaMA-Factory:**
```bash
make install-llamafactory
# 或
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

---

## 快速开始

### 一键运行完整流程

#### 使用 MS-Swift

```bash
# 完整流程：数据准备 → 训练 → 合并
make pipeline-swift

# 或分步执行
make data              # 准备数据
make train-swift       # 训练模型
make merge-swift       # 合并模型
make infer             # 推理测试
make api               # 启动API
```

#### 使用 LLaMA-Factory

```bash
# 完整流程
make pipeline-llamafactory

# 或分步执行
make data
make train-llamafactory
make merge-llamafactory
```

---

## 数据准备

### 支持的数据格式

#### 1. Alpaca 格式（推荐）

```json
[
  {
    "instruction": "你是专业客服，回答用户问题",
    "input": "怎么修改收货地址？",
    "output": "您可以在【我的】-【地址管理】中修改。",
    "history": [],
    "system": ""
  }
]
```

#### 2. ShareGPT 格式

```json
[
  {
    "conversations": [
      {"from": "system", "value": "你是专业客服"},
      {"from": "human", "value": "怎么修改收货地址？"},
      {"from": "gpt", "value": "您可以在【我的】-【地址管理】中修改。"}
    ]
  }
]
```

#### 3. OpenAI 格式

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是专业客服"},
      {"role": "user", "content": "怎么修改收货地址？"},
      {"role": "assistant", "content": "您可以在【我的】-【地址管理】中修改。"}
    ]
  }
]
```

### 数据准备命令

```bash
python scripts/prepare_data.py \
    --input data/examples/customer_service.json \
    --output_dir data/processed \
    --input_format alpaca \
    --output_format alpaca \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    --dataset_name customer_service
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入数据文件路径 | 必填 |
| `--output_dir` | 输出目录 | `./data/processed` |
| `--input_format` | 输入格式 | `alpaca` |
| `--output_format` | 输出格式 | `alpaca` |
| `--train_ratio` | 训练集比例 | `0.9` |
| `--val_ratio` | 验证集比例 | `0.1` |
| `--dataset_name` | 数据集名称 | `dataset` |
| `--seed` | 随机种子 | `42` |

---

## 模型训练

### 使用 MS-Swift 训练

#### 命令行方式

```bash
python scripts/train_swift.py \
    --config configs/swift/qwen_lora.yaml \
    --dataset data/processed/train.jsonl \
    --val_dataset data/processed/val.jsonl \
    --output_dir outputs/checkpoints/swift \
    --epochs 5 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --merge_after_train
```

#### 配置文件方式

编辑 `configs/swift/qwen_lora.yaml`:

```yaml
model_type: "qwen2_5-7b-instruct"
model_id_or_path: "Qwen/Qwen2.5-7B-Instruct"
sft_type: "lora"
lora_rank: 16
lora_alpha: 32
num_train_epochs: 5
batch_size: 1
learning_rate: 2.0e-4
quantization_bit: 4
```

然后运行:
```bash
python scripts/train_swift.py --config configs/swift/qwen_lora.yaml
```

### 使用 LLaMA-Factory 训练

#### 命令行方式

```bash
python scripts/train_llamafactory.py \
    --config configs/llamafactory/qwen_lora.yaml \
    --dataset customer_service \
    --output_dir outputs/checkpoints/llamafactory \
    --epochs 5 \
    --merge_after_train
```

#### Web UI 方式

```bash
llamafactory-cli webui
# 然后访问 http://localhost:7860
```

### 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 3-10 |
| `--batch_size` | 批次大小 | 1-4 |
| `--learning_rate` | 学习率 | 1e-4 ~ 5e-4 |
| `--lora_rank` | LoRA秩 | 8, 16, 32 |
| `--lora_alpha` | LoRA alpha | 2 * rank |
| `--quantization_bit` | 量化位数 | 4 (16G显存) |
| `--gradient_accumulation_steps` | 梯度累积步数 | 4-16 |

---

## 模型合并

### MS-Swift 合并

```bash
# 自动找到最新检查点并合并
python -c "
from src.training.swift_trainer import SwiftTrainer
trainer = SwiftTrainer()
ckpt = trainer.get_latest_checkpoint('./outputs/checkpoints/swift')
trainer.export(ckpt, './outputs/merged/swift')
"
```

### LLaMA-Factory 合并

```bash
python -c "
from src.training.llamafactory_trainer import LLaMAFactoryTrainer
trainer = LLaMAFactoryTrainer()
trainer.export(
    './outputs/checkpoints/llamafactory',
    './outputs/merged/llamafactory'
)
"
```

---

## 模型推理

### 交互式对话

```bash
python scripts/inference.py \
    --model_path ./outputs/merged/swift \
    --engine transformers \
    --interactive \
    --load_in_4bit
```

### 单条推理

```bash
python scripts/inference.py \
    --model_path ./outputs/merged/swift \
    --prompt "怎么修改收货地址？" \
    --load_in_4bit
```

### 使用 vLLM 加速

```bash
python scripts/inference.py \
    --model_path ./outputs/merged/swift \
    --engine vllm \
    --interactive
```

---

## API部署

### 启动API服务

```bash
python scripts/start_api.py \
    --model_path ./outputs/merged/swift \
    --engine transformers \
    --host 0.0.0.0 \
    --port 8000 \
    --load_in_4bit
```

### API接口文档

#### 1. 健康检查

```bash
curl http://localhost:8000/health
```

#### 2. 获取模型列表

```bash
curl http://localhost:8000/v1/models
```

#### 3. 聊天完成

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "怎么修改收货地址？"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### 4. 流式输出

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "怎么修改收货地址？"}
    ],
    "stream": true
  }'
```

### Python客户端示例

```python
import requests

def chat(message):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": message}],
            "temperature": 0.7,
            "max_tokens": 512
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# 使用
response = chat("怎么修改收货地址？")
print(response)
```

---

## 配置说明

### 基础配置 (configs/base.yaml)

```yaml
# 模型配置
model:
  default_model: "qwen2_5_7b"
  supported_models:
    qwen2_5_7b:
      name: "Qwen/Qwen2.5-7B-Instruct"
      type: "qwen2_5-7b-instruct"
      template: "qwen"

# 训练配置
training:
  type: "lora"
  epochs: 5
  batch_size: 1
  learning_rate: 2.0e-4
  lora:
    rank: 16
    alpha: 32
    dropout: 0.05

# 数据配置
data:
  format: "alpaca"
  train_ratio: 0.9
  val_ratio: 0.1
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 指定GPU | `0` |
| `WANDB_DISABLED` | 禁用WandB | `true` |
| `HF_CACHE_DIR` | HuggingFace缓存目录 | `~/.cache/huggingface` |

---

## Docker部署

### 构建镜像

```bash
cd docker
docker-compose build
```

### 启动服务

```bash
# 启动训练服务
docker-compose up train

# 启动API服务
docker-compose up api

# 启动Web UI
docker-compose up webui

# 启动所有服务
docker-compose up -d
```

### 查看日志

```bash
docker-compose logs -f train
docker-compose logs -f api
```

---

## 常见问题

### Q1: 显存不足怎么办？

**A:** 尝试以下方法：
1. 使用QLoRA（4bit量化）
2. 减小 `max_length`
3. 减小 `batch_size`
4. 增大 `gradient_accumulation_steps`
5. 启用 `gradient_checkpointing`

### Q2: 训练速度太慢怎么办？

**A:** 
1. 使用Flash Attention（如果支持）
2. 使用DeepSpeed进行分布式训练
3. 使用vLLM进行推理加速

### Q3: 如何选择LoRA的rank？

**A:**
- rank=8: 轻量级微调，显存占用小
- rank=16: 平衡选择（推荐）
- rank=32: 更强的拟合能力，但可能过拟合
- rank=64+: 复杂任务，需要更多数据

### Q4: 训练后模型效果不佳？

**A:**
1. 检查数据质量
2. 增加训练数据量
3. 调整学习率
4. 增加训练轮数
5. 尝试全参数微调

### Q5: 如何切换不同的基础模型？

**A:** 修改配置文件中的 `model_name_or_path`:

```yaml
# Qwen2.5
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"

# LLaMA3
model_name_or_path: "meta-llama/Meta-Llama-3-8B-Instruct"

# ChatGLM
model_name_or_path: "THUDM/chatglm3-6b"
```

---

## 附录

### 项目结构

```
llm-finetune-framework/
├── configs/              # 配置文件
├── src/                  # 源代码
│   ├── core/            # 核心模块
│   ├── data/            # 数据处理
│   ├── training/        # 训练模块
│   ├── inference/       # 推理模块
│   ├── api/             # API服务
│   └── utils/           # 工具函数
├── scripts/             # 执行脚本
├── data/                # 数据目录
├── outputs/             # 输出目录
├── docker/              # Docker文件
├── docs/                # 文档
├── Makefile             # 构建脚本
└── requirements.txt     # 依赖列表
```

### 相关链接

- [MS-Swift文档](https://github.com/modelscope/ms-swift)
- [LLaMA-Factory文档](https://github.com/hiyouga/LLaMA-Factory)
- [Hugging Face](https://huggingface.co/)

---

**祝使用愉快！如有问题，欢迎提交Issue。**
