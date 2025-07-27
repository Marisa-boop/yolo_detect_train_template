from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLOv11 model
    model = YOLO("./yolov11/runs/train_detect/weights/last.pt")

    # Train the model on a custom dataset
    model.train(
        data="datasets/data/data.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=640,  # 输入图像大小
        batch=32,  # 批量大小
        workers=2,  # 数据加载线程数
        project="yolov11/runs",  # 指定输出目录为 yolov11 下的 runs 文件夹
        name="train_detect",  # 修改实验名称
        amp=True,  # 混合精度训练
        # resume=True
    )
