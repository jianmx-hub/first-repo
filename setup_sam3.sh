#!/usr/bin/env bash
# =============================================================================
# SAM3 环境自动化搭建脚本
# 使用方法：bash setup_sam3.sh
# =============================================================================

set -e  # 遇到错误立即退出

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------- 默认配置 ----------
ENV_NAME="sam3"
PYTHON_VERSION="3.10"
CUDA_VERSION="cu121"   # 可改为 cu118
INSTALL_DIR="$HOME/projects/sam2"
MODEL_SIZE="large"     # 可选: large, base_plus, small, tiny

# ---------- 帮助信息 ----------
usage() {
    cat <<EOF
用法: bash setup_sam3.sh [选项]

选项:
  -e ENV_NAME       conda 虚拟环境名称 (默认: sam3)
  -c CUDA_VERSION   CUDA 版本，如 cu121 或 cu118 (默认: cu121)
  -d INSTALL_DIR    项目安装目录 (默认: ~/projects/sam2)
  -m MODEL_SIZE     模型大小: large/base_plus/small/tiny (默认: large)
  -h                显示帮助信息

示例:
  bash setup_sam3.sh -c cu118 -m small
EOF
    exit 0
}

while getopts "e:c:d:m:h" opt; do
    case $opt in
        e) ENV_NAME="$OPTARG" ;;
        c) CUDA_VERSION="$OPTARG" ;;
        d) INSTALL_DIR="$OPTARG" ;;
        m) MODEL_SIZE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# ---------- 参数校验 ----------
SUPPORTED_CUDA="cu121 cu118 cu117 cpu"
if ! echo "$SUPPORTED_CUDA" | grep -qw "$CUDA_VERSION"; then
    log_error "不支持的 CUDA 版本: ${CUDA_VERSION}"
    log_error "支持的版本: ${SUPPORTED_CUDA}"
    exit 1
fi
SUPPORTED_MODELS="large base_plus small tiny"
if ! echo "$SUPPORTED_MODELS" | grep -qw "$MODEL_SIZE"; then
    log_error "不支持的模型大小: ${MODEL_SIZE}"
    log_error "支持的大小: ${SUPPORTED_MODELS}"
    exit 1
fi

# ---------- 步骤 1：检查 GPU / CUDA ----------
echo ""
echo "=============================================="
echo "  SAM3 环境搭建脚本 - 开始执行"
echo "=============================================="
echo ""

log_info "步骤 1/7：检查 GPU 和 CUDA 环境..."
if command -v nvidia-smi &>/dev/null; then
    log_ok "检测到 NVIDIA GPU："
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
else
    log_warn "未检测到 nvidia-smi，将使用 CPU 模式运行（速度较慢）"
fi

# ---------- 步骤 2：检查 / 安装 Miniconda ----------
log_info "步骤 2/7：检查 Conda 环境..."
if ! command -v conda &>/dev/null; then
    log_warn "未找到 conda，开始安装 Miniconda..."
    MINICONDA_SCRIPT="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O "$MINICONDA_SCRIPT"
    bash "$MINICONDA_SCRIPT" -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    "$HOME/miniconda3/bin/conda" init bash
    source ~/.bashrc
    log_ok "Miniconda 安装完成"
else
    log_ok "conda 已安装：$(conda --version)"
    eval "$(conda shell.bash hook)"
fi

# ---------- 步骤 3：创建虚拟环境 ----------
log_info "步骤 3/7：创建 conda 虚拟环境 [${ENV_NAME}]..."
if conda env list | grep -q "^${ENV_NAME} "; then
    log_warn "虚拟环境 '${ENV_NAME}' 已存在，跳过创建"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    log_ok "虚拟环境创建成功"
fi
conda activate "$ENV_NAME"
log_ok "已激活虚拟环境：$ENV_NAME"

# ---------- 步骤 4：安装 PyTorch ----------
log_info "步骤 4/7：安装 PyTorch（CUDA: ${CUDA_VERSION}）..."
if python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    log_ok "PyTorch 已安装且 CUDA 可用，跳过"
else
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" -q
    python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
    log_ok "PyTorch 安装完成"
fi

# ---------- 步骤 5：克隆 SAM3 仓库 ----------
log_info "步骤 5/7：克隆 SAM3 代码仓库..."
mkdir -p "$(dirname "$INSTALL_DIR")"
if [ -d "$INSTALL_DIR/.git" ]; then
    log_warn "仓库已存在于 ${INSTALL_DIR}，执行 git pull 更新..."
    git -C "$INSTALL_DIR" pull
else
    git clone https://github.com/facebookresearch/sam2.git "$INSTALL_DIR"
    log_ok "代码克隆完成：${INSTALL_DIR}"
fi
cd "$INSTALL_DIR"

# ---------- 步骤 6：安装 SAM3 依赖 ----------
log_info "步骤 6/7：安装 SAM3 依赖..."
pip install -e ".[dev]" -q
pip install opencv-python matplotlib Pillow jupyter notebook -q
log_ok "所有依赖安装完成"

# ---------- 步骤 7：下载模型权重 ----------
log_info "步骤 7/7：下载预训练模型权重（${MODEL_SIZE}）..."
mkdir -p "$INSTALL_DIR/checkpoints"

declare -A MODEL_URLS=(
    ["large"]="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    ["base_plus"]="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    ["small"]="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    ["tiny"]="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
)

MODEL_URL="${MODEL_URLS[$MODEL_SIZE]}"
MODEL_FILE="$INSTALL_DIR/checkpoints/$(basename "$MODEL_URL")"

if [ -f "$MODEL_FILE" ]; then
    log_warn "模型权重文件已存在，跳过下载：$(basename "$MODEL_FILE")"
else
    log_info "开始下载：$(basename "$MODEL_URL")"
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_FILE"
    log_ok "模型权重下载完成"
fi

# ---------- 完成 ----------
echo ""
echo "=============================================="
echo -e "${GREEN}  SAM3 环境搭建完成！${NC}"
echo "=============================================="
echo ""
echo "  安装目录  : $INSTALL_DIR"
echo "  虚拟环境  : $ENV_NAME"
echo "  模型权重  : $(basename "$MODEL_FILE")"
echo ""
echo "  快速开始："
echo "    conda activate $ENV_NAME"
echo "    cd $INSTALL_DIR"
echo "    python demo_sam3.py"
echo ""
