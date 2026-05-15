"""Fine-tunes YOLOv8n on the tube-lid detection dataset."""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="yolo_dataset/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device="mps",
        patience=30,
        project="runs/detect",
        name="tube_detector",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()