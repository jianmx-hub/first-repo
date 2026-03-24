# SAM3 服务器环境搭建指南

> 本文档详细介绍如何在 Linux 服务器上从零开始搭建 SAM3（Segment Anything Model 3）框架，并成功运行 Demo 示例。

---

## 目录

1. [环境要求](#1-环境要求)
2. [安装系统依赖](#2-安装系统依赖)
3. [安装 Miniconda（Python 环境管理）](#3-安装-minicondapython-环境管理)
4. [创建并激活虚拟环境](#4-创建并激活虚拟环境)
5. [安装 PyTorch（含 CUDA）](#5-安装-pytorch含-cuda)
6. [克隆 SAM3 代码仓库](#6-克隆-sam3-代码仓库)
7. [安装 SAM3 依赖](#7-安装-sam3-依赖)
8. [下载预训练模型权重](#8-下载预训练模型权重)
9. [运行基础调用 Demo](#9-运行基础调用-demo)
10. [常见问题排查](#10-常见问题排查)

---

## 1. 环境要求

在开始之前，请确认你的服务器满足以下条件：

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Ubuntu 20.04 | Ubuntu 22.04 LTS |
| GPU | NVIDIA GPU (8GB 显存) | NVIDIA A100 / RTX 3090 |
| CUDA 版本 | 11.8 | 12.1 |
| RAM | 16 GB | 32 GB+ |
| 磁盘空间 | 50 GB | 100 GB+ |
| Python 版本 | 3.9 | 3.10 / 3.11 |

**验证 GPU 和 CUDA 是否可用：**

```bash
# 查看 GPU 信息
nvidia-smi

# 查看 CUDA 版本
nvcc --version
```

如果 `nvidia-smi` 命令不存在，需要先安装 NVIDIA 驱动（见下文）。

---

## 2. 安装系统依赖

```bash
# 更新系统包列表
sudo apt-get update && sudo apt-get upgrade -y

# 安装基础工具
sudo apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# 安装 NVIDIA 驱动（如果尚未安装）
# 先检查是否已安装
nvidia-smi
# 若未安装，执行以下命令（以 Ubuntu 22.04 为例）
sudo apt-get install -y nvidia-driver-535
sudo reboot  # 安装驱动后需要重启
```

---

## 3. 安装 Miniconda（Python 环境管理）

使用 Miniconda 可以方便地管理 Python 版本和依赖包，避免版本冲突。

```bash
# 下载 Miniconda 安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 执行安装（-b 表示静默安装，-p 指定安装路径）
bash miniconda.sh -b -p $HOME/miniconda3

# 初始化 conda
$HOME/miniconda3/bin/conda init bash

# 使配置生效（重新加载 shell 配置）
source ~/.bashrc

# 验证安装成功
conda --version
```

---

## 4. 创建并激活虚拟环境

```bash
# 创建名为 sam3 的虚拟环境，Python 版本 3.10
conda create -n sam3 python=3.10 -y

# 激活虚拟环境
conda activate sam3

# 验证 Python 版本
python --version  # 应显示 Python 3.10.x

# 升级 pip
pip install --upgrade pip
```

> **注意**：之后所有命令都应在 `sam3` 虚拟环境激活的状态下执行。  
> 每次打开新的终端会话，都需要重新执行 `conda activate sam3`。

---

## 5. 安装 PyTorch（含 CUDA）

根据你服务器的 CUDA 版本选择对应的 PyTorch 安装命令。

```bash
# 先确认 CUDA 版本
nvcc --version
# 或者
nvidia-smi  # 右上角显示 CUDA 版本

# 安装 PyTorch（以 CUDA 12.1 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 如果你的 CUDA 版本是 11.8，使用以下命令
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 PyTorch 是否正确识别 GPU
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count())"
```

输出应类似于：
```
PyTorch 版本: 2.3.0+cu121
CUDA 可用: True
GPU 数量: 1
```

---

## 6. 克隆 SAM3 代码仓库

```bash
# 进入你的工作目录
cd ~
mkdir -p projects && cd projects

# 克隆 SAM3 官方仓库（Meta 发布）
git clone https://github.com/facebookresearch/sam2.git

# 进入项目目录
cd sam2

# 查看项目结构
ls -la
```

> **说明**：SAM3 是 Meta 发布的 Segment Anything Model 系列最新版本，  
> 代码仓库托管在 `facebookresearch/sam2`（包含 SAM2 和 SAM3 的代码）。

---

## 7. 安装 SAM3 依赖

```bash
# 确保当前在项目根目录下，且虚拟环境已激活
pwd  # 应该显示 .../projects/sam2
conda activate sam3

# 安装 SAM3 及其所有依赖（以可编辑模式安装）
pip install -e ".[dev]"

# 安装额外依赖
pip install opencv-python matplotlib Pillow jupyter notebook

# 验证安装
python -c "import sam2; print('SAM3 安装成功')"
```

---

## 8. 下载预训练模型权重

```bash
# 创建模型权重存放目录
mkdir -p checkpoints
cd checkpoints

# 下载 SAM3 大型模型权重（推荐，效果最好）
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# 下载 SAM3 基础模型权重（更轻量）
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# 下载小型模型（GPU 显存不足时使用）
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

# 返回项目根目录
cd ..

# 验证权重文件已下载
ls -lh checkpoints/
```

---

## 9. 运行基础调用 Demo

### 9.1 使用本项目提供的 Demo 脚本

本项目提供了 `demo_sam3.py` 脚本，可以快速验证环境是否搭建成功：

```bash
# 在项目根目录下运行
python demo_sam3.py
```

### 9.2 图像分割示例（完整代码）

创建测试脚本 `test_sam3.py`：

```python
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入 SAM3 模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===== 1. 加载模型 =====
print("正在加载模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型配置文件路径（相对于 sam2 项目根目录）
model_cfg = "sam2_hiera_l.yaml"
# 模型权重路径
checkpoint = "checkpoints/sam2_hiera_large.pt"

# 构建并加载模型
sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print("模型加载完成！")

# ===== 2. 准备输入图像 =====
# 方法一：使用测试图片（需要先下载）
# wget -O test_image.jpg https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg
image_path = "test_image.jpg"
image = np.array(Image.open(image_path).convert("RGB"))
print(f"图像尺寸: {image.shape}")

# ===== 3. 设置图像 =====
predictor.set_image(image)

# ===== 4. 通过点提示进行分割 =====
# 指定前景点（图像中想要分割的目标位置）
input_point = np.array([[500, 375]])  # [x, y] 坐标
input_label = np.array([1])           # 1 表示前景点

# 执行预测
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # 输出多个候选掩码
)
print(f"生成掩码数量: {masks.shape[0]}, 各掩码置信度: {scores}")

# ===== 5. 可视化结果 =====
fig, axes = plt.subplots(1, masks.shape[0] + 1, figsize=(20, 5))

# 显示原图
axes[0].imshow(image)
axes[0].plot(input_point[:, 0], input_point[:, 1], 'r*', markersize=15)
axes[0].set_title("原始图像")
axes[0].axis('off')

# 显示每个掩码
for i, (mask, score) in enumerate(zip(masks, scores)):
    axes[i + 1].imshow(image)
    axes[i + 1].imshow(mask, alpha=0.5, cmap='viridis')
    axes[i + 1].set_title(f"掩码 {i+1}\n置信度: {score:.3f}")
    axes[i + 1].axis('off')

plt.tight_layout()
plt.savefig("segmentation_result.png", dpi=150, bbox_inches='tight')
print("分割结果已保存到 segmentation_result.png")
plt.show()
```

运行脚本：

```bash
# 先下载测试图片
wget -O test_image.jpg "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"

# 运行分割脚本
python test_sam3.py
```

### 9.3 使用官方 Jupyter Notebook Demo

```bash
# 启动 Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# 打开官方示例（浏览器访问 http://服务器IP:8888）
# 进入 notebooks/ 目录，打开 image_predictor_example.ipynb
```

---

## 10. 常见问题排查

### Q1: `nvidia-smi` 命令找不到

```bash
# 安装 NVIDIA 驱动
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Q2: CUDA out of memory（显存不足）

```python
# 在代码中添加以下设置，使用较小的模型
checkpoint = "checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

# 或者使用半精度推理
sam2_model = build_sam2(model_cfg, checkpoint, device=device)
sam2_model = sam2_model.half()  # 使用 FP16 减少显存占用
```

### Q3: `pip install` 速度很慢

```bash
# 使用国内镜像源加速
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者使用阿里云镜像
pip install -e ".[dev]" -i https://mirrors.aliyun.com/pypi/simple/
```

### Q4: `conda activate` 无效

```bash
# 重新初始化 conda
conda init bash
source ~/.bashrc
# 重新激活
conda activate sam3
```

### Q5: 下载模型权重速度慢

```bash
# 使用 aria2 多线程下载（更快）
sudo apt-get install -y aria2
aria2c -x 16 -s 16 https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### Q6: `ModuleNotFoundError: No module named 'sam2'`

```bash
# 确保在 sam2 项目目录下，且以可编辑模式重新安装
cd ~/projects/sam2
conda activate sam3
pip install -e .
```

---

## 快速开始总结

```bash
# 一键快速配置（所有步骤汇总）
conda create -n sam3 python=3.10 -y && conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e ".[dev]"
mkdir -p checkpoints && wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
python demo_sam3.py
```

---

*如有任何问题，欢迎在 [Issues](https://github.com/jianmx-hub/first-repo/issues) 中提问。*
