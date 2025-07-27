import os
import shutil
from tqdm import tqdm


def merge_datasets(source_dirs, target_dir):
    """
    合并多个源目录到统一目录，保持原始图像和蒙版图分离

    参数:
    source_dirs: 源目录列表
    target_dir: 目标目录路径
    """
    # 创建目标目录结构
    target_images = os.path.join(target_dir, "images")
    target_masks = os.path.join(target_dir, "masks")

    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_masks, exist_ok=True)

    # 计数器
    file_count = 0
    missing_mask_count = 0
    skipped_dirs = []

    # 遍历每个源目录
    for source_dir in source_dirs:
        # 获取图像和蒙版图目录路径
        images_path = os.path.join(source_dir, "images")
        masks_path = os.path.join(source_dir, "masks")

        # 检查源目录是否存在
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            skipped_dirs.append(os.path.basename(source_dir))
            print(f"跳过: {os.path.basename(source_dir)} 缺少必要的子目录")
            continue

        # 获取图像文件列表
        images_files = [
            f for f in os.listdir(images_path) if f.endswith((".jpg", ".jpeg", ".png"))
        ]

        if not images_files:
            print(f"警告: {os.path.basename(source_dir)} 的图像目录为空")
            continue

        # 处理进度条
        desc = f"处理目录 {os.path.basename(source_dir)}"
        progress_bar = tqdm(images_files, desc=desc)

        # 处理每个文件
        for img_file in progress_bar:
            # 获取不带扩展名的文件名
            img_base, img_ext = os.path.splitext(img_file)

            # 构建新文件名：源目录名 + 原文件名
            new_prefix = f"{os.path.basename(source_dir)}_"
            new_img_name = new_prefix + img_file

            # 源文件路径
            src_img = os.path.join(images_path, img_file)

            # 目标路径
            dest_img = os.path.join(target_images, new_img_name)

            # 复制图像文件
            if os.path.exists(src_img):
                shutil.copy2(src_img, dest_img)

            # 查找对应的蒙版文件
            mask_file = img_base + ".png"  # 蒙版文件为同名.png
            src_mask = os.path.join(masks_path, mask_file)

            if os.path.exists(src_mask):
                new_mask_name = new_prefix + mask_file
                dest_mask = os.path.join(target_masks, new_mask_name)
                shutil.copy2(src_mask, dest_mask)
                file_count += 1  # 计入有效配对
            else:
                # 检查是否存在其他扩展名的蒙版文件
                possible_extensions = [".png", ".jpg", ".jpeg"]
                found = False

                for ext in possible_extensions:
                    if ext == img_ext:
                        continue  # 跳过与原图相同的扩展名

                    mask_file_alt = img_base + ext
                    src_mask_alt = os.path.join(masks_path, mask_file_alt)

                    if os.path.exists(src_mask_alt):
                        new_mask_name = new_prefix + mask_file_alt
                        dest_mask = os.path.join(target_masks, new_mask_name)
                        shutil.copy2(src_mask_alt, dest_mask)
                        file_count += 1
                        found = True
                        break

                if not found:
                    missing_mask_count += 1
                    if missing_mask_count < 5:  # 只显示前几个警告
                        print(
                            f"警告: 找不到蒙版文件 {mask_file} (或替代) 对应图像 {img_file}"
                        )

        print(f"已处理 {len(images_files)} 个图像文件")

    print(f"\n合并完成! 共复制 {file_count} 个有效文件对")
    print(f"图像目录: {target_images} (共 {len(os.listdir(target_images))} 个文件)")
    print(f"蒙版目录: {target_masks} (共 {len(os.listdir(target_masks))} 个文件)")

    if skipped_dirs:
        print(f"\n跳过的目录: {', '.join(skipped_dirs)}")
    if missing_mask_count > 0:
        print(f"警告: 共有 {missing_mask_count} 个图像找不到对应的蒙版文件")


if __name__ == "__main__":
    # 定义基础目录
    base_dir = "datasets_origin"  # 数据集基础路径
    target_dir_name = "merged_data"  # 目标目录名称

    # 设置目标目录路径
    target_directory = os.path.join(target_dir_name)

    # 获取base_dir中的所有子目录（排除目标目录）
    source_directories = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item != target_dir_name:
            source_directories.append(full_path)

    # 打印配置信息
    print("配置信息:")
    print(f"基础目录: {base_dir}")
    print(f"目标目录: {target_directory}")
    print(f"找到 {len(source_directories)} 个潜在源目录:")
    for i, path in enumerate(source_directories, 1):
        print(f"{i}. {os.path.basename(path)}")
    print("=" * 50)

    # 执行合并
    merge_datasets(source_directories, target_directory)
