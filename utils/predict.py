import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class VideoSegmenter:
    def __init__(self, model_path, video_path, conf_threshold=0.5):
        self.model_path = Path(model_path)

        # 加载YOLOv11分割模型
        self.model = YOLO(model_path)
        self.model.fuse()

        # 打开视频流
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        # 获取视频属性
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.conf_threshold = conf_threshold

        # 创建透明覆盖层
        self.overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.colors = self._generate_colors(len(self.model.names))

    def _generate_colors(self, n_classes):
        return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n_classes)]

    def _draw_segmentation(self, frame, results):
        self.overlay[:] = 0
        for result in results:
            if result.masks is None:
                continue
            for i, (mask, box) in enumerate(zip(result.masks.xy, result.boxes)):
                if box.conf[0] < self.conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                color = self.colors[cls_id]
                mask_points = np.array([mask], dtype=np.int32)
                cv2.fillPoly(self.overlay, mask_points, color)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{self.model.names[cls_id]} {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.addWeighted(self.overlay, 0.4, frame, 0.6, 0, frame)
        return frame

    def process_video(self, output_path=None, show=True):
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 改为mp4v编码器
            out = cv2.VideoWriter(output_path, fourcc,
                                  self.fps, (self.width, self.height))
            # 验证写入器是否成功打开
            if not out.isOpened():
                raise RuntimeError(f"无法创建输出视频: {output_path}")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 执行分割预测（优化推理尺寸）
            results = self.model.predict(
                frame,
                task='segment',
                imgsz=min(640, max(self.width, self.height)),  # 动态调整
                conf=self.conf_threshold,
                device='cuda' if self.model.device.type != 'cpu' else 'cpu'
            )

            processed_frame = self._draw_segmentation(frame.copy(), results)

            fps_text = f"FPS: {self.cap.get(cv2.CAP_PROP_FPS):.1f} | Model: {
                self.model_path.stem}"
            cv2.putText(processed_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if show:
                cv2.imshow("YOLOv11 Video Segmentation", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if output_path:
                out.write(processed_frame)

        self.cap.release()
        if output_path:
            out.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    MODEL_PATH = "./yolov11/runs/train_seg/weights/best.pt"
    VIDEO_PATH = "./video/homework.webm"
    OUTPUT_PATH = "./video/export/output_segmented.mp4"

    segmenter = VideoSegmenter(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        conf_threshold=0.5
    )
    segmenter.process_video(output_path=OUTPUT_PATH, show=True)
