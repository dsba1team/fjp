import os
import glob
import xml.etree.ElementTree as ET

# 1) 상위 폴더 (하위에 Polygon_7_new, Polygon_8_new, … 가 있다고 가정)
PARENT_DIR = r'C:\Users\User\Downloads\인도보행 영상\폴리곤세크멘테이션'

# 2) dataset 구조
DATASET_DIR      = os.path.join(PARENT_DIR, 'dataset')
LABELS_DIR       = os.path.join(DATASET_DIR, 'labels')
TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, 'train')
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)

# 3) classes.txt 경로
CLASSES_TXT = os.path.join(LABELS_DIR, 'classes.txt')

# 4) 모든 Polygon_*_new 하위의 XML 파일 모으기
xml_paths = glob.glob(
    os.path.join(PARENT_DIR, 'Polygon_*_new', '**', '*.xml'),
    recursive=True
)
if not xml_paths:
    raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {PARENT_DIR}/Polygon_*_new/**")

# 5) 클래스 목록 추출 (첫 XML 기준)
tree = ET.parse(xml_paths[0])
root = tree.getroot()
label_elems = root.findall('meta/task/labels/label')
classes = [lbl.find('name').text for lbl in label_elems]
class_map = {name: idx for idx, name in enumerate(classes)}

# 6) classes.txt 쓰기
with open(CLASSES_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(classes))
print(f"[Info] classes.txt 생성됨: {CLASSES_TXT} ({len(classes)}개 클래스)")

# 7) 모든 XML 순회하면서 YOLOv8 segmentation 포맷으로 변환
for xml_file in xml_paths:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for img in root.findall('image'):
        img_name = img.get('name')                 # e.g. ZED3_KSC_063326_L_P014901.png
        base, _  = os.path.splitext(img_name)
        w = float(img.get('width'))
        h = float(img.get('height'))

        out_txt = os.path.join(TRAIN_LABELS_DIR, base + '.txt')
        lines = []
        for poly in img.findall('polygon'):
            cls_id = class_map[poly.get('label')]
            pts = poly.get('points').split(';')
            coords = []
            for p in pts:
                x, y = map(float, p.split(','))
                coords += [f"{x/w:.6f}", f"{y/h:.6f}"]
            lines.append(f"{cls_id} " + " ".join(coords))

        # 비어 있어도 빈 파일 하나를 찍어 주는 게 좋습니다.
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

print("✅ 레이블 변환 완료")
print(f"   → {TRAIN_LABELS_DIR} 에 총 {len(os.listdir(TRAIN_LABELS_DIR))}개의 .txt 파일이 생성되었습니다.")
print(f"   → 클래스 목록: {CLASSES_TXT}")