import os

# 라벨 파일들이 있는 폴더
label_dir = r'C:\Users\User\Desktop\datasetfin\labels'

# 검사 시작
error_files = []

for filename in os.listdir(label_dir):
    if not filename.endswith('.txt'):
        continue

    path = os.path.join(label_dir, filename)
    with open(path, 'r') as f:
        lines = f.readlines()

    has_error = False
    for idx, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) < 3:
            print(f"❌ {filename} - {idx+1}행: 좌표 부족")
            has_error = True
            continue

        class_id = tokens[0]
        coords = tokens[1:]

        # 짝수 좌표 개수 확인
        if len(coords) % 2 != 0:
            print(f"❌ {filename} - {idx+1}행: 좌표 개수가 홀수개 (짝수여야 함)")
            has_error = True

        # 정규화 범위 확인
        for i, v in enumerate(coords):
            try:
                val = float(v)
                if val < 0.0 or val > 1.0:
                    print(f"❌ {filename} - {idx+1}행: 좌표 {val} 범위 초과")
                    has_error = True
            except ValueError:
                print(f"❌ {filename} - {idx+1}행: 숫자 아님 → '{v}'")
                has_error = True

    if has_error:
        error_files.append(filename)

print("\n검사 완료 ✅")
if error_files:
    print(f"\n❗ 오류가 있는 라벨 파일 총 {len(error_files)}개:")
    for f in error_files:
        print(" -", f)
else:
    print("👍 모든 라벨 파일이 정상입니다.")
