from ultralytics import YOLO
import os
import cv2
import numpy as np

# ───── 설정 ─────
image_dir = r'C:\Users\User\Desktop\datasetfin\images'
label_dir = r'C:\Users\User\Desktop\datasetfin\labels'
output_dir = r'C:\Users\User\Desktop\datasetfin\vis_2class_only'
os.makedirs(output_dir, exist_ok=True)

# 모델 로드 (class 0: 킥보드, class 1: 점자블록)
models = {
    0: YOLO(r'C:\Users\User\Downloads\김민석 kick\민석님\runs\segment\train\weights\best.pt'),
    1: YOLO(r'C:\Users\User\Downloads\인도보행 영상\서피스마스킹\runs\segment\train10\weights\best.pt')
}

# 클래스별 시각화 색상 (BGR)
colors = {
    0: (0, 255, 0),     # 킥보드: 초록
    1: (0, 0, 255)      # 점자블록: 빨강
}

# ───── 1. 클래스 0, 1 모두 있는 라벨 파일 찾기 ─────
matched_files = []
for filename in os.listdir(label_dir):
    if not filename.endswith('.txt'):
        continue

    with open(os.path.join(label_dir, filename), 'r') as f:
        lines = f.readlines()

    class_ids = set(line.strip().split()[0] for line in lines if line.strip())
    if {'0', '1'}.issubset(class_ids):
        matched_files.append(filename)

print(f"\n✅ 클래스 0, 1 모두 포함된 파일 수: {len(matched_files)}\n")

# ───── 2. 해당 이미지 예측 후 시각화 ─────
for label_file in matched_files:
    base_name = os.path.splitext(label_file)[0]

    # 이미지 확장자 찾기 (.jpg, .png 우선순위)
    image_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        try_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(try_path):
            image_path = try_path
            break

    if not image_path:
        print(f"❌ 이미지 없음: {base_name}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 읽기 실패: {image_path}")
        continue

    h, w = img.shape[:2]
    img_drawn = img.copy()

    for cls_id, model in models.items():
        results = model.predict(image_path, save=False, conf=0.4, verbose=False)
        for r in results:
            if r.masks is not None:
                for mask in r.masks.data:
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (w, h))
                    color = colors.get(cls_id, (255, 255, 0))
                    img_drawn[mask > 0.5] = img_drawn[mask > 0.5] * 0.5 + np.array(color) * 0.5

    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, img_drawn)
    print(f"✅ 저장됨: {out_path}")
