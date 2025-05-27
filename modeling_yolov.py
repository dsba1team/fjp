from ultralytics import YOLO

def main():

    model = YOLO(r"C:\Users\User\PycharmProjects\PythonProject1\runs\segment\yolo11n-seg_finfin\weights\last.pt") #자신이 사용할 모델 넣기
    model.train(
        task="segment",  # 또는 'detect'
        data=r"C:\Users\User\Desktop\dataset_finfin\data.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        patience=20,
        save_period=20,
        device=0,  # GPU 번호
        workers=8,  # 워커 수: multiprocessing 에러가 계속 나면 0으로 낮춰보세요
        resume=True,
        name="yolo11n-seg_finfin"
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
