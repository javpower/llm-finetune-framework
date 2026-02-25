# ============================================
# LLM微调框架 - Docker镜像
# ============================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 设置Python3为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 升级pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 设置工作目录
WORKDIR /workspace

# 复制项目文件
COPY requirements.txt .
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装MS-Swift（可选）
RUN pip install --no-cache-dir "ms-swift[all]"

# 安装LLaMA-Factory（可选）
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory && \
    cd /tmp/LLaMA-Factory && \
    pip install --no-cache-dir -e ".[torch,metrics]"

# 创建必要的目录
RUN mkdir -p data/raw data/processed outputs/checkpoints outputs/merged outputs/logs

# 暴露端口
EXPOSE 8000 7860

# 设置入口点
ENTRYPOINT ["/bin/bash"]
