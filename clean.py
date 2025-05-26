import os

# ————————————————————————————————————————————————————————
# 환경에 맞게 경로만 수정하세요
IMAGES_DIR = r'C:\Users\User\Downloads\인도보행 영상\폴리곤세크멘테이션\dataset\images\train'
LABELS_DIR = r'C:\Users\User\Downloads\인도보행 영상\폴리곤세크멘테이션\dataset\labels\train'
# ————————————————————————————————————————————————————————

# 지원할 이미지 확장자
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

# 1) 이미지 중에 대응하는 .txt 가 없으면 삭제
for fn in os.listdir(IMAGES_DIR):
    base, ext = os.path.splitext(fn)
    if ext.lower() in IMG_EXTS:
        txt_path = os.path.join(LABELS_DIR, base + '.txt')
        img_path = os.path.join(IMAGES_DIR, fn)
        if not os.path.exists(txt_path):
            print(f"[삭제] 레이블 없음 → {img_path}")
            os.remove(img_path)

# 2) 레이블(.txt) 중에 대응하는 이미지가 없으면 삭제
for fn in os.listdir(LABELS_DIR):
    base, ext = os.path.splitext(fn)
    if ext.lower() == '.txt':
        img_found = False
        for ie in IMG_EXTS:
            if os.path.exists(os.path.join(IMAGES_DIR, base + ie)):
                img_found = True
                break
        if not img_found:
            txt_path = os.path.join(LABELS_DIR, fn)
            print(f"[삭제] 이미지 없음 → {txt_path}")
            os.remove(txt_path)

print("✅ 고아(Orphan) 이미지·레이블 정리 완료")
