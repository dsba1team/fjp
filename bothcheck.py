from ultralytics import YOLO
import os
import cv2
import numpy as np

# ───── 경로 설정 ─────
image_dir = r'C:\Users\User\Desktop\datasetfin\images'
output_dir = r'C:\Users\User\Desktop\datasetfin\vis_multi_class_all'
os.makedirs(output_dir, exist_ok=True)

# ───── 모델 경로 등록 ─────
model_paths = [
    r'C:\Users\User\Downloads\김민석 kick\민석님\runs\segment\train\weights\best.pt',               # 킥보드
    r'C:\Users\User\Downloads\인도보행 영상\서피스마스킹\runs\segment\train\weights\best.pt',       # 점자블록
    r'C:\Users\User\Desktop\datasetfin\best.pt',                                                    # 횡단보도
    r'C:\Users\User\Desktop\datasetfin\traffic_best.pt',                                            # 신호등
    r'C:\Users\User\Desktop\datasetfin\all_best.pt'                                                 # 기타 장애물
]

# ───── 모델 로드 및 클래스 오프셋 계산 ─────
models = {}
offset = 0
for i, path in enumerate(model_paths):
    model = YOLO(path)
    models[i] = {"model": model, "base_class": offset}
    print(f"🔧 모델 {i} 로드됨: {os.path.basename(path)} (클래스 수: {len(model.names)}, 시작 ID: {offset})")
    offset += len(model.names)

# ───── 색상 설정 함수 ─────
def get_color(cls_id):
    np.random.seed(cls_id)
    return tuple(int(x) for x in np.random.randint(0, 255, size=3))

# ───── 이미지 순회 및 시각화 ─────
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        continue

    h, w = img.shape[:2]
    img_drawn = img.copy()
    found_classes = set()

    for model_id, item in models.items():
        model = item["model"]
        base_class = item["base_class"]

        # 이미지 크기 고정 → 마스크 불일치 방지
        results = model.predict(image_path, save=False, conf=0.4, verbose=False, imgsz=(h, w))

        for r in results:
            if r.masks is not None:
                for seg, cls in zip(r.masks.data, r.boxes.cls):
                    cls = int(cls.item())
                    global_cls = base_class + cls
                    found_classes.add(global_cls)

                    mask = seg.cpu().numpy()
                    mask_bool = mask > 0.5

                    if mask_bool.shape != img_drawn.shape[:2]:
                        print(f"⚠️ 마스크 크기 불일치 → 건너뜀: {mask_bool.shape} vs {img_drawn.shape[:2]}")
                        continue

                    color = get_color(global_cls)
                    color_np = np.array(color, dtype=np.uint8)
                    img_drawn[mask_bool] = img_drawn[mask_bool] * 0.5 + color_np * 0.5

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, img_drawn)
    print(f"✅ 저장됨: {out_path} (포함된 클래스: {sorted(found_classes) if found_classes else '없음'})")
