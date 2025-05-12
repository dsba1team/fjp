from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.train(
        data="input path of data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,  # 배치 크기 줄이기 (예: 8로 변경)
        name="yolov8m_custom"
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
