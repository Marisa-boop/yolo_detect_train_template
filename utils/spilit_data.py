import os
import shutil
import random
from tqdm import tqdm


def split_dataset(merged_data_dir, output_dir, ratios=(0.7, 0.2, 0.1)):
    """
    将合并后的数据集分割为训练集、验证集和测试集

    参数:
    merged_data_dir: 合并后的数据集目录路径
    output_dir: 输出目录路径
    ratios: 分割比例 (训练集, 验证集, 测试集)
    """
    # 验证比例总和为1
    if round(sum(ratios), 9) != 1.0:
        raise ValueError("比例总和必须为1")

    # 创建目标目录结构
    datasets = {
        "train": os.path.join(output_dir, "data", "train"),
        "valid": os.path.join(output_dir, "data", "valid"),
        "test": os.path.join(output_dir, "data", "test"),
    }

    # 为每个数据集创建images和labels目录
    for dataset in datasets.values():
        os.makedirs(os.path.join(dataset, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset, "labels"), exist_ok=True)

    # 读取图像文件列表（从images目录）
    images_dir = os.path.join(merged_data_dir, "images")
    image_files = [
        f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算分割点
    total_count = len(image_files)
    train_end = int(total_count * ratios[0])
    valid_end = train_end + int(total_count * ratios[1])

    # 分割文件列表
    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]

    # 标签目录路径
    labels_dir = os.path.join(merged_data_dir, "labels")

    # 检查标签目录是否存在
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    # 复制训练集
    print(f"\n复制训练集 ({ratios[0]*100:.0f}%, {len(train_files)} 个文件)...")
    for img_file in tqdm(train_files, desc="训练集"):
        # 复制图像文件
        src_img = os.path.join(images_dir, img_file)
        dest_img = os.path.join(datasets["train"], "images", img_file)
        shutil.copy2(src_img, dest_img)

        # 查找并复制对应的标签文件
        img_base, img_ext = os.path.splitext(img_file)
        label_file = f"{img_base}.txt"
        src_label = os.path.join(labels_dir, label_file)

        if os.path.exists(src_label):
            dest_label = os.path.join(datasets["train"], "labels", label_file)
            shutil.copy2(src_label, dest_label)
        else:
            print(f"警告: 找不到标签文件 {label_file} 对应图像 {img_file}")

    # 复制验证集
    print(f"\n复制验证集 ({ratios[1]*100:.0f}%, {len(valid_files)} 个文件)...")
    for img_file in tqdm(valid_files, desc="验证集"):
        src_img = os.path.join(images_dir, img_file)
        dest_img = os.path.join(datasets["valid"], "images", img_file)
        shutil.copy2(src_img, dest_img)

        img_base, _ = os.path.splitext(img_file)
        label_file = f"{img_base}.txt"
        src_label = os.path.join(labels_dir, label_file)

        if os.path.exists(src_label):
            dest_label = os.path.join(datasets["valid"], "labels", label_file)
            shutil.copy2(src_label, dest_label)
        else:
            print(f"警告: 找不到标签文件 {label_file} 对应图像 {img_file}")

    # 复制测试集
    print(f"\n复制测试集 ({ratios[2]*100:.0f}%, {len(test_files)} 个文件)...")
    for img_file in tqdm(test_files, desc="测试集"):
        src_img = os.path.join(images_dir, img_file)
        dest_img = os.path.join(datasets["test"], "images", img_file)
        shutil.copy2(src_img, dest_img)

        img_base, _ = os.path.splitext(img_file)
        label_file = f"{img_base}.txt"
        src_label = os.path.join(labels_dir, label_file)

        if os.path.exists(src_label):
            dest_label = os.path.join(datasets["test"], "labels", label_file)
            shutil.copy2(src_label, dest_label)
        else:
            print(f"警告: 找不到标签文件 {label_file} 对应图像 {img_file}")

    # 创建yaml配置文件
    create_yaml_config(output_dir)

    return {
        "训练集": len(train_files),
        "验证集": len(valid_files),
        "测试集": len(test_files),
        "总计": total_count,
    }


def create_yaml_config(output_dir):
    """创建YOLO训练所需的YAML配置文件"""
    # 确保路径正确
    data_path = os.path.abspath(os.path.join(output_dir, "data"))

    yaml_content = f"""# YOLO数据集配置文件
path: {data_path}  # 数据集根目录
train: train/images  # 训练集图像相对路径
val: valid/images    # 验证集图像相对路径
test: test/images    # 测试集图像相对路径

# 类别名称
names:
  0: "0"
  1: "1"
  2: "2"
  3: "3"
  4: "4"
  5: "5"
"""

    config_path = os.path.join(output_dir, "data/data.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        f.write(yaml_content)

    print(f"\nYOLO配置文件已创建: {config_path}")


if __name__ == "__main__":
    # 配置路径
    merged_data_dir = "merged_data"  # 合并后的数据集目录
    output_dir = "datasets"  # 输出目录

    # 分割比例 (训练集, 验证集, 测试集)
    ratios = (0.7, 0.2, 0.1)  # 70% 训练集, 20% 验证集, 10% 测试集

    # 打印配置信息
    print("数据集分割配置:")
    print(f"合并数据集目录: {merged_data_dir}")
    print(f"输出目录: {output_dir}")
    train_per = ratios[0] * 100
    valid_per = ratios[1] * 100
    test_per = ratios[2] * 100
    print(f"分割比例: 训练集 {train_per}%, 验证集 {valid_per}%, 测试集 {test_per}%")
    print("=" * 50)

    # 执行分割
    result = split_dataset(merged_data_dir, output_dir, ratios)

    # 打印结果
    print("\n分割结果:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # 打印目录结构
    print("\n创建的目录结构:")
    print(f"{output_dir}/")
    print("├── data")
    print("│   ├── test")
    print("│   │   ├── images")
    print("│   │   └── labels")
    print("│   ├── train")
    print("│   │   ├── images")
    print("│   │   └── labels")
    print("│   └── valid")
    print("│       ├── images")
    print("│       └── labels")
    print("└── data.yaml")
