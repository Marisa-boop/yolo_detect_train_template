import cv2
import numpy as np
from pathlib import Path
import os


def generate_class_mapping(template_dir):
    """通过模板图片生成全局类别映射表"""
    template_mask = next(Path(template_dir).glob("*.png"))
    mask = cv2.imread(str(template_mask), cv2.IMREAD_GRAYSCALE)
    unique_vals = np.unique(mask)
    unique_vals = unique_vals[unique_vals != 0]  # 排除背景0
    return {orig_id: new_id for new_id, orig_id in enumerate(unique_vals, start=1)}


def convert_segment_masks_to_yolo_det(masks_dir, output_dir, class_mapping):
    """
    将分割掩码转换为YOLO格式的检测标签（边界框）
    使用全局类别映射确保类别ID一致性
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for mask_file in Path(masks_dir).glob("*.png"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]

        # 获取所有非零唯一值（每个值代表一个对象实例）
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # 排除背景

        # 创建对应的标签文件
        label_file = Path(output_dir) / f"{mask_file.stem}.txt"

        with open(label_file, "w") as f:
            for orig_id in instance_ids:
                # 应用全局类别映射
                if orig_id not in class_mapping:
                    continue  # 跳过未映射的类别

                class_id = class_mapping[orig_id] - 1  # 转换为0-based的类别ID

                # 创建当前类别的二进制掩码
                binary_mask = (mask == orig_id).astype(np.uint8)

                # 计算边界框
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue

                # 合并所有轮廓（处理同一类别的多个实例）
                all_contours = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_contours)

                # 转换为YOLO格式（归一化中心坐标+宽高）
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                bbox_width = w / width
                bbox_height = h / height

                # 写入标签文件
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {
                        bbox_width:.6f} {bbox_height:.6f}\n"
                )

        print(f"生成检测标签: {mask_file} → {label_file}")


if __name__ == "__main__":
    # ====== 配置路径 ======
    base_dir = "./merged_data"
    input_dir = "./merged_data/masks"  # 原始分割掩码目录
    labels_dir = "./merged_data/labels_det"  # 输出检测标签目录
    template_dir = "./template_image"  # 包含所有类别的模板图片目录

    # ====== 目录创建 ======
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    # ====== 核心流程 ======
    # 1. 生成全局类别映射表
    class_mapping = generate_class_mapping(template_dir)
    print(f"生成全局映射表: {class_mapping}")

    # 2. 直接转换为YOLO检测格式
    convert_segment_masks_to_yolo_det(
        masks_dir=input_dir, output_dir=labels_dir, class_mapping=class_mapping
    )

    print("所有检测标签生成完毕！")
