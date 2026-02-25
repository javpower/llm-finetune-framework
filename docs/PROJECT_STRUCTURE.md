# 项目结构说明

## 目录树

```
llm-finetune-framework/
├── configs/                          # 配置文件目录
│   ├── base.yaml                    # 基础配置文件
│   ├── swift/                       # MS-Swift配置
│   │   └── qwen_lora.yaml          # Qwen LoRA配置
│   └── llamafactory/                # LLaMA-Factory配置
│       └── qwen_lora.yaml          # Qwen LoRA配置
│
├── src/                             # 源代码目录
│   ├── __init__.py                 # 包初始化
│   ├── core/                        # 核心模块
│   │   ├── __init__.py
│   │   ├── config.py               # 配置管理
│   │   └── logger.py               # 日志管理
│   ├── data/                        # 数据处理模块
│   │   ├── __init__.py
│   │   └── processor.py            # 数据处理器
│   ├── training/                    # 训练模块
│   │   ├── __init__.py
│   │   ├── swift_trainer.py        # MS-Swift训练器
│   │   └── llamafactory_trainer.py # LLaMA-Factory训练器
│   ├── inference/                   # 推理模块
│   │   ├── __init__.py
│   │   └── engine.py               # 推理引擎
│   ├── api/                         # API服务模块
│   │   ├── __init__.py
│   │   └── server.py               # FastAPI服务
│   └── utils/                       # 工具模块
│       ├── __init__.py
│       └── common.py               # 通用工具函数
│
├── scripts/                         # 执行脚本目录
│   ├── __init__.py
│   ├── prepare_data.py             # 数据准备脚本
│   ├── train_swift.py              # Swift训练脚本
│   ├── train_llamafactory.py       # LLaMA-Factory训练脚本
│   ├── inference.py                # 推理脚本
│   └── start_api.py                # API启动脚本
│
├── data/                            # 数据目录
│   ├── examples/                    # 示例数据
│   │   └── customer_service.json   # 客服场景示例
│   ├── raw/                         # 原始数据
│   │   └── .gitkeep
│   └── processed/                   # 处理后数据
│       └── .gitkeep
│
├── outputs/                         # 输出目录
│   ├── checkpoints/                 # 模型检查点
│   │   └── .gitkeep
│   ├── merged/                      # 合并后的模型
│   │   └── .gitkeep
│   └── logs/                        # 日志文件
│       └── .gitkeep
│
├── tests/                           # 测试目录
│   ├── __init__.py
│   └── test_data_processor.py      # 数据处理器测试
│
├── docs/                            # 文档目录
│   ├── USER_MANUAL.md              # 用户手册
│   └── PROJECT_STRUCTURE.md        # 项目结构说明
│
├── docker/                          # Docker配置
│   ├── Dockerfile                  # Docker镜像构建
│   └── docker-compose.yml          # Docker Compose配置
│
├── .gitignore                       # Git忽略文件
├── LICENSE                          # 许可证
├── Makefile                         # 构建脚本
├── README.md                        # 项目说明
├── quickstart.sh                    # 快速启动脚本
└── requirements.txt                 # Python依赖
```

## 模块说明

### 1. 核心模块 (src/core/)

#### config.py
- **功能**: 统一的配置管理
- **主要类**:
  - `ConfigManager`: 配置管理器，支持YAML加载、验证和访问
  - `ModelConfig`: 模型配置
  - `TrainingConfig`: 训练配置
  - `DataConfig`: 数据配置
- **使用示例**:
```python
from src.core.config import get_config

config = get_config("configs/base.yaml")
print(config.training.learning_rate)
```

#### logger.py
- **功能**: 日志管理和训练过程记录
- **主要类**:
  - `setup_logger()`: 设置日志记录器
  - `TrainingLogger`: 训练过程日志记录器
  - `Timer`: 计时器上下文管理器
- **使用示例**:
```python
from src.core.logger import setup_logger

logger = setup_logger("my_module")
logger.info("This is an info message")
```

### 2. 数据处理模块 (src/data/)

#### processor.py
- **功能**: 数据加载、转换、验证和分割
- **主要类**:
  - `DataSample`: 数据样本类
  - `DataConverter`: 数据格式转换器
  - `DataValidator`: 数据验证器
  - `DataProcessor`: 数据处理器
- **支持格式**:
  - Alpaca
  - ShareGPT
  - OpenAI
- **使用示例**:
```python
from src.data.processor import DataProcessor

processor = DataProcessor(
    input_format="alpaca",
    output_format="sharegpt"
)
output_files = processor.process(
    input_file="data.json",
    output_dir="output/"
)
```

### 3. 训练模块 (src/training/)

#### swift_trainer.py
- **功能**: MS-Swift训练器封装
- **主要类**:
  - `SwiftTrainer`: Swift训练器
  - `SwiftInference`: Swift推理器
- **使用示例**:
```python
from src.training.swift_trainer import SwiftTrainer

trainer = SwiftTrainer("configs/swift/qwen_lora.yaml")
trainer.train(
    dataset_path="data/train.jsonl",
    output_dir="outputs/"
)
```

#### llamafactory_trainer.py
- **功能**: LLaMA-Factory训练器封装
- **主要类**:
  - `LLaMAFactoryTrainer`: LLaMA-Factory训练器
  - `LLaMAFactoryInference`: LLaMA-Factory推理器
- **使用示例**:
```python
from src.training.llamafactory_trainer import LLaMAFactoryTrainer

trainer = LLaMAFactoryTrainer("configs/llamafactory/qwen_lora.yaml")
trainer.train()
```

### 4. 推理模块 (src/inference/)

#### engine.py
- **功能**: 统一的推理引擎接口
- **主要类**:
  - `BaseInferenceEngine`: 推理引擎基类
  - `TransformersEngine`: Transformers推理引擎
  - `VLLMEngine`: vLLM推理引擎
  - `InferenceEngineFactory`: 推理引擎工厂
- **使用示例**:
```python
from src.inference.engine import InferenceEngineFactory

engine = InferenceEngineFactory.create_engine(
    "transformers",
    "path/to/model"
)
engine.load_model()
response = engine.generate("Hello!")
```

### 5. API服务模块 (src/api/)

#### server.py
- **功能**: OpenAI兼容的API服务
- **主要组件**:
  - `FastAPI`应用
  - `InferenceManager`: 推理管理器
  - 标准端点: `/v1/chat/completions`, `/v1/models`, `/health`
- **使用示例**:
```python
from src.api.server import start_server

start_server(
    model_path="path/to/model",
    host="0.0.0.0",
    port=8000
)
```

### 6. 工具模块 (src/utils/)

#### common.py
- **功能**: 通用工具函数
- **主要功能**:
  - ID生成
  - 时间格式化
  - 文件操作
  - GPU信息获取
  - 进度跟踪

## 脚本说明

### prepare_data.py
数据准备脚本，用于数据格式转换和分割。

### train_swift.py
MS-Swift训练脚本，支持命令行参数覆盖配置。

### train_llamafactory.py
LLaMA-Factory训练脚本，支持命令行参数覆盖配置。

### inference.py
推理脚本，支持交互式和单条推理模式。

### start_api.py
API服务启动脚本。

## 配置文件说明

### base.yaml
基础配置文件，包含所有默认配置。

### swift/qwen_lora.yaml
MS-Swift的LoRA训练配置。

### llamafactory/qwen_lora.yaml
LLaMA-Factory的LoRA训练配置。

## 数据流

```
原始数据 (data/raw/)
    ↓
数据准备 (prepare_data.py)
    ↓
处理后数据 (data/processed/)
    ↓
模型训练 (train_*.py)
    ↓
模型检查点 (outputs/checkpoints/)
    ↓
模型合并 (merge)
    ↓
合并模型 (outputs/merged/)
    ↓
推理/API服务 (inference.py / start_api.py)
```

## 扩展指南

### 添加新的训练框架

1. 在 `src/training/` 创建新的训练器类
2. 继承基类并实现必要方法
3. 在 `scripts/` 创建对应的训练脚本
4. 在 `configs/` 添加配置文件

### 添加新的推理引擎

1. 在 `src/inference/engine.py` 创建新的引擎类
2. 继承 `BaseInferenceEngine`
3. 在 `InferenceEngineFactory` 中注册

### 添加新的数据格式

1. 在 `src/data/processor.py` 的 `DataConverter` 中添加转换方法
2. 在 `DataValidator` 中添加验证方法
3. 在 `DataProcessor` 中支持新格式

## 最佳实践

1. **配置管理**: 使用YAML配置文件，通过环境变量或命令行参数覆盖
2. **日志记录**: 所有模块使用统一的日志记录器
3. **错误处理**: 使用try-except块捕获并记录异常
4. **代码风格**: 遵循PEP 8规范，使用类型提示
5. **测试**: 为新功能编写单元测试
