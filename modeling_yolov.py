from ultralytics import YOLO

def main():
    model = YOLO("yolov8s-seg.pt") #자신이 사용할 모델 넣기
    model.train(
        data=r"C:\Users\User\Desktop\total\dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=50,  # early stopping: 50 epoch 동안 개선 없으면 중단
        save_period=5,  # 5 epochs마다 모델 저장
        name="yolov8s-seg_custom_again"
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()


