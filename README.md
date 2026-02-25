# LLM微调框架 (LLM Fine-tune Framework)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>企业级大语言模型微调框架</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> •
  <a href="#功能特性">功能特性</a> •
  <a href="#安装指南">安装指南</a> •
  <a href="#使用文档">使用文档</a> •
  <a href="#API文档">API文档</a>
</p>

---

##  简介

本项目是一个**高可用、模块化、企业级**的大语言模型微调框架，整合了业界最流行的两个微调框架：

- **[MS-Swift](https://github.com/modelscope/ms-swift)** - 阿里出品，中文友好，训练速度快
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** - 社区最流行，功能丰富，Web UI支持

### 为什么选择本项目？

- ✅ **统一接口** - 一套代码支持两种训练框架
- ✅ **高可用架构** - 模块化设计，易于扩展和维护
- ✅ **生产就绪** - 完整的日志、监控、错误处理
- ✅ **OpenAI兼容** - 标准API接口，易于集成
- ✅ **Docker支持** - 一键部署，环境隔离
- ✅ **多引擎推理** - 支持Transformers和vLLM

---

##  快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/javpower/llm-finetune-framework.git
cd llm-finetune-framework

# 创建环境
conda create -n llm-finetune python=3.10
conda activate llm-finetune

# 安装依赖
pip install -r requirements.txt

# 安装训练框架
make install-swift          # 安装MS-Swift
make install-llamafactory   # 安装LLaMA-Factory
```

### 2. 准备数据

```bash
make data
```

### 3. 训练模型

```bash
# 使用MS-Swift
make train-swift

# 或使用LLaMA-Factory
make train-llamafactory
```

### 4. 启动API服务

```bash
make api
```

### 5. 测试API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

---

##  功能特性

### 训练功能

| 特性 | MS-Swift | LLaMA-Factory |
|------|:--------:|:-------------:|
| LoRA微调 | ✅ | ✅ |
| QLoRA (4bit) | ✅ | ✅ |
| 全参数微调 | ✅ | ✅ |
| Flash Attention | ✅ | ✅ |
| DeepSpeed | ✅ | ✅ |
| 多卡训练 | ✅ | ✅ |

### 数据格式

- ✅ Alpaca格式
- ✅ ShareGPT格式
- ✅ OpenAI格式
- ✅ 自动格式转换

### 推理引擎

- ✅ Transformers (Hugging Face)
- ✅ vLLM (高性能)
- ✅ 流式输出
- ✅ 批量推理

### API服务

- ✅ OpenAI兼容接口
- ✅ 流式响应
- ✅ 多模型支持
- ✅ 健康检查

---

##  安装指南

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+), Windows WSL2
- **Python**: 3.10+
- **CUDA**: 11.8+ 或 12.1+
- **GPU**: 显存 >= 16GB (推荐)

### 详细安装步骤

#### 步骤1: 安装CUDA和驱动

确保已安装CUDA 11.8或更高版本：

```bash
nvidia-smi
nvcc --version
```

#### 步骤2: 创建Python环境

```bash
conda create -n llm-finetune python=3.10 -y
conda activate llm-finetune
```

#### 步骤3: 安装PyTorch

```bash
# CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 步骤4: 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 步骤5: 安装训练框架

```bash
# 安装MS-Swift
pip install "ms-swift[all]" -U

# 安装LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory
cd /tmp/LLaMA-Factory
pip install -e ".[torch,metrics]"
```

---

##  使用文档

### 目录结构

```
llm-finetune-framework/
├── configs/              # 配置文件
│   ├── base.yaml        # 基础配置
│   ├── swift/           # MS-Swift配置
│   └── llamafactory/    # LLaMA-Factory配置
├── src/                  # 源代码
│   ├── core/            # 核心模块（配置、日志）
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

### 数据准备

```bash
python scripts/prepare_data.py \
    --input data/examples/customer_service.json \
    --output_dir data/processed \
    --input_format alpaca \
    --output_format alpaca \
    --train_ratio 0.9 \
    --val_ratio 0.1
```

### 模型训练

#### MS-Swift

```bash
python scripts/train_swift.py \
    --config configs/swift/qwen_lora.yaml \
    --dataset data/processed/train.jsonl \
    --val_dataset data/processed/val.jsonl \
    --epochs 5 \
    --merge_after_train
```

#### LLaMA-Factory

```bash
python scripts/train_llamafactory.py \
    --config configs/llamafactory/qwen_lora.yaml \
    --dataset customer_service \
    --epochs 5 \
    --merge_after_train
```

### 模型推理

```bash
python scripts/inference.py \
    --model_path ./outputs/merged/swift \
    --engine transformers \
    --interactive \
    --load_in_4bit
```

---

##  API文档

### 启动服务

```bash
python scripts/start_api.py \
    --model_path ./outputs/merged/swift \
    --host 0.0.0.0 \
    --port 8000
```

### 接口列表

#### 健康检查

```http
GET /health
```

#### 获取模型列表

```http
GET /v1/models
```

#### 聊天完成

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "default",
  "messages": [
    {"role": "user", "content": "你好！"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
```

### Python客户端

```python
import requests

class LLMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def chat(self, message, **kwargs):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": message}],
                **kwargs
            }
        )
        return response.json()["choices"][0]["message"]["content"]

# 使用
client = LLMClient()
response = client.chat("怎么修改收货地址？")
print(response)
```

---

##  Docker部署

### 构建镜像

```bash
docker-compose -f docker/docker-compose.yml build
```

### 启动服务

```bash
# 启动所有服务
docker-compose -f docker/docker-compose.yml up -d

# 查看日志
docker-compose logs -f api
```

### 访问服务

- API服务: http://localhost:8000
- Web UI: http://localhost:7860

---

##  配置说明

### 基础配置

编辑 `configs/base.yaml`:

```yaml
model:
  default_model: "qwen2_5_7b"
  
training:
  type: "lora"
  epochs: 5
  batch_size: 1
  learning_rate: 2.0e-4
  
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
| `HF_CACHE_DIR` | 缓存目录 | `~/.cache/huggingface` |

---

##  性能对比

### 训练速度对比 (Qwen2.5-7B, LoRA, 16G显存)

| 框架 | 量化 | 显存占用 | 训练速度 |
|------|------|----------|----------|
| MS-Swift | 4bit | ~12GB | 快 |
| LLaMA-Factory | 4bit | ~13GB | 中等 |
| 全参数 | 无 | OOM | - |

### 推理速度对比

| 引擎 | 量化 | 吞吐量 (tokens/s) |
|------|------|-------------------|
| Transformers | 4bit | ~50 |
| vLLM | 无 | ~200+ |

---

##  贡献指南

欢迎提交Issue和Pull Request！

### 开发流程

1. Fork项目
2. 创建分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

---

##  许可证

本项目基于 [MIT](LICENSE) 许可证开源。

---

## 🙏 致谢

- [MS-Swift](https://github.com/modelscope/ms-swift) - 阿里ModelScope团队
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - hiyouga
- [Hugging Face](https://huggingface.co/) - Transformers库
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎

---

##  联系方式

如有问题或建议，欢迎：

- 提交 [GitHub Issue](../../issues)
- 发送邮件至: javpower@163.com

---

<p align="center">
  <b>Star ⭐ 本项目如果对你有帮助！</b>
</p>
