#!/usr/bin/env python3
"""
SAM3 基础调用 Demo
演示如何使用 SAM3（Segment Anything Model 3）对图像进行分割。

运行前准备：
    1. 完成环境搭建（参考 README.md 或运行 setup_sam3.sh）
    2. 下载模型权重到 checkpoints/ 目录
    3. 激活虚拟环境：conda activate sam3

运行方式：
    python demo_sam3.py
    python demo_sam3.py --image path/to/your/image.jpg --model small
"""

import argparse
import os
import sys

import numpy as np

# ===== 参数解析 =====
def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 图像分割 Demo")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="输入图像路径（不指定则自动生成测试图像）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large",
        choices=["large", "base_plus", "small", "tiny"],
        help="模型大小 (默认: large)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="模型权重目录 (默认: checkpoints/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_segmentation.png",
        help="结果图像保存路径 (默认: output_segmentation.png)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备 (默认: auto)",
    )
    return parser.parse_args()


# ===== 检查依赖 =====
def check_dependencies():
    missing = []
    for pkg in ["torch", "numpy", "PIL", "matplotlib"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
        except ImportError:
            missing.append(pkg if pkg != "PIL" else "Pillow")
    if missing:
        print(f"[错误] 缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        sys.exit(1)


# ===== 创建测试图像 =====
def create_test_image(save_path="test_input.jpg"):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 480), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)

    # 绘制几个简单的几何图形作为分割目标
    draw.ellipse([100, 100, 300, 300], fill=(255, 100, 50), outline=(200, 60, 20), width=3)
    draw.rectangle([350, 150, 550, 380], fill=(50, 180, 100), outline=(20, 130, 60), width=3)
    draw.polygon(
        [(160, 380), (240, 200), (320, 380)],
        fill=(100, 100, 255),
        outline=(60, 60, 200),
    )
    draw.text((20, 440), "SAM3 Demo - 测试图像", fill=(50, 50, 50))

    img.save(save_path)
    print(f"[信息] 已生成测试图像: {save_path}")
    return save_path


# ===== 可视化分割结果 =====
def show_mask(mask, ax, random_color=False):
    import matplotlib.pyplot as plt

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(
        pos[:, 0], pos[:, 1],
        color="green", marker="*", s=marker_size, edgecolors="white", linewidths=1.25,
        label="前景点",
    )
    ax.scatter(
        neg[:, 0], neg[:, 1],
        color="red", marker="*", s=marker_size, edgecolors="white", linewidths=1.25,
        label="背景点",
    )


# ===== 主程序 =====
def main():
    args = parse_args()
    check_dependencies()

    import torch
    import matplotlib.pyplot as plt
    from PIL import Image

    # ----- 选择运行设备 -----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[信息] 使用设备: {device}")
    if device.type == "cuda":
        print(f"[信息] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[信息] 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ----- 定位模型权重 -----
    model_filename_map = {
        "large":     "sam2_hiera_large.pt",
        "base_plus": "sam2_hiera_base_plus.pt",
        "small":     "sam2_hiera_small.pt",
        "tiny":      "sam2_hiera_tiny.pt",
    }
    model_cfg_map = {
        "large":     "sam2_hiera_l.yaml",
        "base_plus": "sam2_hiera_b+.yaml",
        "small":     "sam2_hiera_s.yaml",
        "tiny":      "sam2_hiera_t.yaml",
    }

    checkpoint = os.path.join(args.checkpoints_dir, model_filename_map[args.model])
    model_cfg  = model_cfg_map[args.model]

    if not os.path.exists(checkpoint):
        print(f"[错误] 模型权重文件不存在: {checkpoint}")
        print("请先下载权重文件，参考 README.md 第 8 步，或运行:")
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/{model_filename_map[args.model]}"
        print(f"  wget {url} -P {args.checkpoints_dir}/")
        sys.exit(1)

    # ----- 加载 SAM3 模型 -----
    print(f"[信息] 正在加载 SAM3 模型（{args.model}）...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("[错误] 未找到 sam2 模块，请在 sam2 项目目录下以可编辑模式安装:")
        print("  pip install -e .")
        sys.exit(1)

    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("[信息] 模型加载完成！")

    # ----- 准备输入图像 -----
    if args.image and os.path.exists(args.image):
        image_path = args.image
    else:
        if args.image:
            print(f"[警告] 指定图像不存在: {args.image}，使用自动生成的测试图像")
        image_path = create_test_image()

    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    print(f"[信息] 图像尺寸: {w} x {h}")

    # ----- 执行分割（点提示模式） -----
    print("[信息] 开始执行图像分割...")
    predictor.set_image(image)

    # 使用图像中心附近的点作为前景提示
    cx, cy = w // 2, h // 2
    input_point = np.array([[cx, cy]])
    input_label = np.array([1])  # 1 = 前景，0 = 背景

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

    # 按置信度排序
    sorted_idx = np.argsort(scores)[::-1]
    masks  = masks[sorted_idx]
    scores = scores[sorted_idx]

    print(f"[信息] 共生成 {len(masks)} 个候选掩码")
    for i, s in enumerate(scores):
        print(f"        掩码 {i+1}: 置信度 = {s:.4f}")

    # ----- 可视化 & 保存 -----
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(5 * (len(masks) + 1), 5))
    fig.suptitle("SAM3 图像分割结果", fontsize=14, fontweight="bold")

    # 原图 + 点位标注
    axes[0].imshow(image)
    show_points(input_point, input_label, axes[0])
    axes[0].set_title("原始图像\n(★ = 提示点)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].axis("off")

    # 各候选掩码
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i + 1].imshow(image)
        show_mask(mask, axes[i + 1])
        show_points(input_point, input_label, axes[i + 1])
        axes[i + 1].set_title(f"掩码 {i+1}\n置信度: {score:.3f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[完成] 分割结果已保存到: {args.output}")

    # 控制台输出结果摘要
    best_mask = masks[0]
    coverage = best_mask.sum() / (h * w) * 100
    print(f"[完成] 最优掩码覆盖率: {coverage:.1f}%")
    print("=" * 50)
    print("  SAM3 Demo 运行成功！")
    print("=" * 50)


if __name__ == "__main__":
    main()
