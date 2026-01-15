import os
import cv2
import numpy as np

# ───── 경로 설정 ─────
image_dir = r'C:\Users\User\Desktop\datasetfin\images'
label_dir = r'C:\Users\User\Desktop\datasetfin\labels'
output_dir = r'C:\Users\User\Desktop\datasetfin\vis_from_labels_only'
os.makedirs(output_dir, exist_ok=True)

# ───── 고정된 색상 테이블 생성 ─────
np.random.seed(42)
color_table = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(100)]

def get_color(cls_id):
    return color_table[cls_id % len(color_table)]

# ───── 라벨 기반 시각화 ─────
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    name, _ = os.path.splitext(filename)
    label_path = os.path.join(label_dir, f"{name}.txt")
    if not os.path.exists(label_path):
        print(f"🚫 라벨 없음 → 제외됨: {filename}")
        continue

    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        continue

    h, w = img.shape[:2]
    img_drawn = img.copy()

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue

        try:
            cls_id = int(parts[0])
            points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= w
            points[:, 1] *= h
            points = np.round(points).astype(np.int32)

            # 🔽 작은 객체 필터링 제거
            cv2.fillPoly(img_drawn, [points], color=get_color(cls_id))
        except Exception as e:
            print(f"⚠️ 오류 발생: {line} → {e}")
            continue

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, img_drawn)
    print(f"✅ 시각화 저장됨: {out_path}")
