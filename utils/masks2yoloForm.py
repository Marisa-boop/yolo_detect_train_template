import cv2
import numpy as np
from pathlib import Path


def normalized_masks(input_dir, output_dir):
    # 遍历所有掩码文件
    for mask_file in Path(input_dir).glob("*.png"):
        # 读取掩码图像
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        # 获取非零唯一像素值（排除背景0）
        unique_vals = np.unique(mask)
        unique_vals = unique_vals[unique_vals != 0]

        # 创建重映射字典
        remap_dict = {
            old_val: new_val for new_val, old_val in enumerate(unique_vals, start=1)
        }

        # 应用像素值重映射
        for old_val, new_val in remap_dict.items():
            mask[mask == old_val] = new_val

        # 保存处理后的掩码
        output_path = Path(output_dir) / mask_file.name
        cv2.imwrite(str(output_path), mask)
        print(f"处理完成: {mask_file} → {output_path}")

    print("所有掩码文件处理完毕！")


def convert_segment_masks_to_yolo_det(masks_dir, output_dir, classes):
    """
    将分割掩码转换为YOLO格式的检测标签（边界框）
    每个对象一行：<class_id> <x_center> <y_center> <width> <height>
    所有值归一化到[0,1]范围
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 处理每个掩码文件
    for mask_file in Path(masks_dir).glob("*.png"):
        # 读取掩码图像
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]

        # 获取所有非零唯一值（每个值代表一个对象实例）
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # 排除背景

        # 创建对应的标签文件
        label_file = Path(output_dir) / f"{mask_file.stem}.txt"

        with open(label_file, "w") as f:
            # 处理每个对象实例
            for instance_id in instance_ids:
                # 创建当前实例的二进制掩码
                binary_mask = (mask == instance_id).astype(np.uint8)

                # 计算边界框
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    # 获取最大轮廓的边界框
                    cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)

                    # 转换为YOLO格式（归一化中心坐标+宽高）
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    bbox_width = w / width
                    bbox_height = h / height

                    # 写入标签文件
                    # 注意：这里的类别ID假设实例ID就是类别ID（通常需要根据实际情况调整）
                    class_id = instance_id - 1  # 因为YOLO类别ID从0开始
                    f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

        print(f"生成检测标签: {mask_file} → {label_file}")

    print("所有检测标签生成完毕！")


if __name__ == "__main__":
    # 创建输出目录
    base_dir = "./merged_data"
    input_dir = "./merged_data/masks"
    output_dir = "./merged_data/masks_normalized"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels_dir = "./merged_data/labels"
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    # 1. 归一化掩码（保持原功能）
    normalized_masks(input_dir, output_dir)

    # 2. 转换为检测格式（边界框）而不是分割格式
    # 注意：classes参数在这里不再需要，但保留以保持接口一致
    convert_segment_masks_to_yolo_det(
        masks_dir=output_dir, output_dir=labels_dir, classes=6  # 使用归一化后的掩码
    )
