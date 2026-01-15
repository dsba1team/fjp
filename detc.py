from ultralytics import YOLO
import os
import cv2
import numpy as np
from collections import defaultdict

image_dir = r'C:\Users\User\Desktop\datasetfin\images'
label_dir = r'C:\Users\User\Desktop\datasetfin\labels'
os.makedirs(label_dir, exist_ok=True)

model_paths = [
    r'C:\Users\User\Downloads\김민석 kick\민석님\runs\segment\train\weights\best.pt',
    r'C:\Users\User\Downloads\인도보행 영상\서피스마스킹\runs\segment\train\weights\best.pt',
    r'C:\Users\User\Desktop\datasetfin\best.pt',
    r'C:\Users\User\Desktop\datasetfin\traffic_best.pt',
    r'C:\Users\User\Desktop\datasetfin\all_best.pt'
]

models = {}
offset = 0
for i, path in enumerate(model_paths):
    model = YOLO(path)
    models[i] = (model, offset)
    print(f"🔧 모델 {i} 로드됨: {path} (클래스 수: {len(model.names)}, 시작 ID: {offset})")
    offset += len(model.names)

min_area_px = 50
object_count = defaultdict(int)
label_file_count = 0

def mask_to_yolo_seg(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) >= 6:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue
            norm_pts = []
            for pt in cnt:
                x = pt[0][0] / img_w
                y = pt[0][1] / img_h
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    return None
                norm_pts.extend([x, y])
            lines.append([str(round(p, 6)) for p in norm_pts])
    return lines

# 마스크 겹침 제거를 위한 병합
def apply_non_overlap(masks, h, w):
    final_mask = np.zeros((h, w), dtype=np.int32)
    final_labels = []

    for mask_np, conf, cls_id in sorted(masks, key=lambda x: -x[1]):  # conf 기준 내림차순
        resized_mask = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        new_region = (resized_mask > 0) & (final_mask == 0)

        if np.count_nonzero(new_region) < min_area_px:
            continue

        final_mask[new_region] = cls_id + 1  # +1 to avoid zero
        final_labels.append((cls_id, new_region.astype(np.uint8)))

    return final_labels

for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 이미지 로드 실패: {img_path}")
        continue

    h, w = img.shape[:2]
    print(f"\n📷 처리 중: {img_file}")
    candidate_masks = []

    for idx, (model, offset_cls) in models.items():
        results = model.predict(img_path, save=False, conf=0.5, verbose=False)
        for r in results:
            if r.masks is not None and r.boxes is not None:
                for i, mask in enumerate(r.masks.data):
                    conf = float(r.boxes.conf[i].item())
                    true_cls = int(r.boxes.cls[i].item())
                    real_cls = offset_cls + true_cls
                    candidate_masks.append((mask.cpu().numpy(), conf, real_cls))

    final_labels = apply_non_overlap(candidate_masks, h, w)
    label_lines = []

    for cls_id, mask in final_labels:
        segs = mask_to_yolo_seg(mask, w, h)
        if segs is None:
            print(f"⚠️ 정규화 오류 → 라벨 제외됨 (클래스 {cls_id})")
            continue
        for s in segs:
            label_lines.append(f"{cls_id} {' '.join(s)}")
            object_count[cls_id] += 1

    if not label_lines:
        print(f"⚠️ 객체 없음 → 라벨 파일 생성 생략됨: {img_file}")
        continue

    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))
    label_file_count += 1
    print(f"✅ 라벨 저장 완료: {label_path} (총 객체 수: {len(label_lines)})")

print("\n📊 전체 처리 요약:")
print(f"📂 라벨 파일 생성 수: {label_file_count}")
print("📌 클래스별 객체 수:")
for cls_id in sorted(object_count):
    print(f"  - 클래스 {cls_id}: {object_count[cls_id]}개")
