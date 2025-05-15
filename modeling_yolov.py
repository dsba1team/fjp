from ultralytics import YOLO

def main():


    model = YOLO("yolov8n-seg.pt") #자신이 사용할 모델 넣기
    model.train(
        data=r"C:/Users/User/Desktop/cross/dataset/data.yaml",
        epochs=10,
        imgsz=256,
        batch=64,
        patience=10,  # early stopping: 10 epoch 동안 개선 없으면 중단
        save_period=5,  # 5 epochs마다 모델 저장
        workers=8,
        name="yolov8n-seg_crosswalk"
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
