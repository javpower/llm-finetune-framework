#!/bin/bash
# ============================================
# LLM微调框架 - 快速启动脚本
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "  LLM微调框架 - 快速启动"
    echo "========================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# 检查环境
check_environment() {
    print_info "检查环境..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "未找到Python，请先安装Python 3.10+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python版本: $PYTHON_VERSION"
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_success "CUDA版本: $CUDA_VERSION"
    else
        print_error "未找到CUDA，GPU训练可能无法使用"
    fi
    
    # 检查conda环境
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_error "未检测到conda环境，建议先创建环境: conda create -n llm-finetune python=3.10"
        exit 1
    fi
    
    print_success "当前环境: $CONDA_DEFAULT_ENV"
}

# 安装依赖
install_dependencies() {
    print_info "安装依赖..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "未找到requirements.txt"
        exit 1
    fi
    
    pip install -q -r requirements.txt
    print_success "依赖安装完成"
}

# 安装训练框架
install_frameworks() {
    print_info "安装训练框架..."
    
    echo "请选择要安装的训练框架:"
    echo "1) MS-Swift (推荐，中文友好)"
    echo "2) LLaMA-Factory (功能丰富)"
    echo "3) 两者都安装"
    echo "4) 跳过"
    
    read -p "请输入选项 [1-4]: " choice
    
    case $choice in
        1)
            print_info "安装MS-Swift..."
            pip install -q "ms-swift[all]"
            print_success "MS-Swift安装完成"
            ;;
        2)
            print_info "安装LLaMA-Factory..."
            if [ ! -d "/tmp/LLaMA-Factory" ]; then
                git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory
            fi
            cd /tmp/LLaMA-Factory
            pip install -q -e ".[torch,metrics]"
            cd -
            print_success "LLaMA-Factory安装完成"
            ;;
        3)
            print_info "安装MS-Swift..."
            pip install -q "ms-swift[all]"
            print_success "MS-Swift安装完成"
            
            print_info "安装LLaMA-Factory..."
            if [ ! -d "/tmp/LLaMA-Factory" ]; then
                git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory
            fi
            cd /tmp/LLaMA-Factory
            pip install -q -e ".[torch,metrics]"
            cd -
            print_success "LLaMA-Factory安装完成"
            ;;
        4)
            print_info "跳过框架安装"
            ;;
        *)
            print_error "无效选项"
            exit 1
            ;;
    esac
}

# 准备数据
prepare_data() {
    print_info "准备训练数据..."
    
    if [ ! -f "data/examples/customer_service.json" ]; then
        print_error "未找到示例数据"
        exit 1
    fi
    
    python scripts/prepare_data.py \
        --input data/examples/customer_service.json \
        --output_dir data/processed \
        --dataset_name customer_service
    
    print_success "数据准备完成"
}

# 训练模型
train_model() {
    print_info "训练模型..."
    
    echo "请选择训练框架:"
    echo "1) MS-Swift"
    echo "2) LLaMA-Factory"
    
    read -p "请输入选项 [1-2]: " choice
    
    case $choice in
        1)
            if ! command -v swift &> /dev/null; then
                print_error "MS-Swift未安装，请先安装"
                return
            fi
            
            print_info "使用MS-Swift训练..."
            python scripts/train_swift.py \
                --config configs/swift/qwen_lora.yaml \
                --dataset data/processed/train.jsonl \
                --val_dataset data/processed/val.jsonl \
                --merge_after_train
            ;;
        2)
            if ! command -v llamafactory-cli &> /dev/null; then
                print_error "LLaMA-Factory未安装，请先安装"
                return
            fi
            
            print_info "使用LLaMA-Factory训练..."
            python scripts/train_llamafactory.py \
                --config configs/llamafactory/qwen_lora.yaml \
                --dataset customer_service \
                --merge_after_train
            ;;
        *)
            print_error "无效选项"
            return
            ;;
    esac
    
    print_success "训练完成"
}

# 启动API
start_api() {
    print_info "启动API服务..."
    
    # 查找合并后的模型
    MODEL_PATH=""
    if [ -d "outputs/merged/swift" ]; then
        MODEL_PATH="outputs/merged/swift"
    elif [ -d "outputs/merged/llamafactory" ]; then
        MODEL_PATH="outputs/merged/llamafactory"
    elif [ -d "outputs/checkpoints/swift" ]; then
        # 查找最新的checkpoint
        MODEL_PATH=$(ls -d outputs/checkpoints/swift/checkpoint-* 2>/dev/null | sort -V | tail -1)
    fi
    
    if [ -z "$MODEL_PATH" ]; then
        print_error "未找到训练好的模型，请先训练"
        return
    fi
    
    print_info "使用模型: $MODEL_PATH"
    
    echo "启动API服务，按Ctrl+C停止..."
    python scripts/start_api.py \
        --model_path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --load_in_4bit
}

# 主菜单
show_menu() {
    echo ""
    echo "请选择操作:"
    echo "1) 完整流程（安装 → 数据 → 训练 → API）"
    echo "2) 仅安装环境"
    echo "3) 仅准备数据"
    echo "4) 仅训练模型"
    echo "5) 仅启动API"
    echo "6) 退出"
    echo ""
}

# 主函数
main() {
    print_header
    
    # 检查是否在项目根目录
    if [ ! -f "requirements.txt" ]; then
        print_error "请在项目根目录运行此脚本"
        exit 1
    fi
    
    check_environment
    
    while true; do
        show_menu
        read -p "请输入选项 [1-6]: " choice
        
        case $choice in
            1)
                install_dependencies
                install_frameworks
                prepare_data
                train_model
                start_api
                break
                ;;
            2)
                install_dependencies
                install_frameworks
                ;;
            3)
                prepare_data
                ;;
            4)
                train_model
                ;;
            5)
                start_api
                break
                ;;
            6)
                print_info "退出"
                exit 0
                ;;
            *)
                print_error "无效选项"
                ;;
        esac
    done
}

# 运行主函数
main
