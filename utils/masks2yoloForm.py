import cv2
import numpy as np
from pathlib import Path
import os
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg


def generate_class_mapping(template_dir):
    """通过模板图片生成全局类别映射表"""
    template_mask = next(Path(template_dir).glob("*.png"))
    mask = cv2.imread(str(template_mask), cv2.IMREAD_GRAYSCALE)
    unique_vals = np.unique(mask)
    unique_vals = unique_vals[unique_vals != 0]  # 排除背景0
    return {orig_id: new_id for new_id, orig_id in enumerate(unique_vals, start=1)}


def normalized_masks_global(input_dir, output_dir, class_mapping):
    """
    使用全局映射表处理掩码
    :param class_mapping: 全局映射字典 {原始ID: 新ID}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for mask_file in Path(input_dir).glob("*.png"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        new_mask = np.zeros_like(mask)  # 创建背景为0的新掩码

        # 应用全局映射
        for orig_id, new_id in class_mapping.items():
            new_mask[mask == orig_id] = new_id

        # 保存结果
        output_path = Path(output_dir) / mask_file.name
        cv2.imwrite(str(output_path), new_mask)
        print(f"映射完成: {mask_file} → {output_path}")


if __name__ == "__main__":
    # 配置路径
    base_dir = "./merged_data"
    input_dir = "./merged_data/masks"
    normalized_dir = "./merged_data/masks_normalized"
    labels_dir = "./merged_data/labels"
    template_dir = "./template_image"  # 包含所有类别的模板图片目录

    # 步骤1：通过模板生成全局映射表
    class_mapping = generate_class_mapping(template_dir)
    print(f"生成全局映射表: {class_mapping}")

    # 步骤2：应用全局映射处理所有掩码
    normalized_masks_global(input_dir, normalized_dir, class_mapping)

    # 步骤3：转换YOLO格式（需指定总类别数）
    convert_segment_masks_to_yolo_seg(
        masks_dir=normalized_dir,
        output_dir=labels_dir,
        classes=len(class_mapping),  # 动态获取类别数
    )
    print("所有处理完成！")
