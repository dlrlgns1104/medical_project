import cv2
import numpy as np
from PIL import Image
import os
from collections import defaultdict


# 1. 한글 경로 이미지 읽기 함수 정의
def read_image_with_korean_path(image_path):
    try:
        pil_image = Image.open(image_path)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"이미지를 로드할 수 없습니다: {image_path}, 오류: {e}")
        return None


# 2. 폴더 내 모든 이미지 경로 가져오기
def get_image_paths(folder_path):
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # 지원 이미지 포맷
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(supported_formats)
    ]


# 3. 중복 이미지 탐지 함수
def find_duplicate_image_groups(folder_path):
    image_paths = get_image_paths(folder_path)
    print(f"폴더에서 {len(image_paths)}개의 이미지를 찾았습니다.")

    hash_to_paths = defaultdict(list)
    duplicates = []

    # 모든 이미지 로드 및 해시 생성
    for path in image_paths:
        image = read_image_with_korean_path(path)
        if image is not None:
            # 이미지를 해시값으로 변환 (픽셀값 기반)
            image_hash = hash(image.tobytes())
            hash_to_paths[image_hash].append(path)

    # 해시값을 기준으로 그룹화
    for hash_value, paths in hash_to_paths.items():
        if len(paths) > 1:  # 중복된 이미지가 2개 이상인 경우만
            duplicates.append(paths)

    return duplicates


# 4. 중복 이미지 제거 함수
def remove_duplicate_images(duplicate_groups):
    removed_images = []
    for group in duplicate_groups:
        # 그룹에서 첫 번째 이미지를 제외한 나머지 삭제
        for path in group[1:]:
            try:
                os.remove(path)
                removed_images.append(path)
                print(f"이미지 삭제: {path}")
            except Exception as e:
                print(f"이미지를 삭제할 수 없습니다: {path}, 오류: {e}")
    print(f"총 {len(removed_images)}개의 중복된 이미지를 제거했습니다.")


# 5. 결과 출력 및 실행
folder_path = r"D:/2024_07_오창교교수님_의료프젝/image_0401/crop incomp/"  # 분석할 폴더 경로
# folder_path = r"C:/Users/leegihun/OneDrive - 한림대학교/바탕 화면/2024_07_오창교교수님_의료프젝/54_cropped"
duplicate_groups = find_duplicate_image_groups(folder_path)

if duplicate_groups:
    print(f"총 {len(duplicate_groups)}개의 중복된 이미지 그룹이 발견되었습니다.")
    for idx, group in enumerate(duplicate_groups, start=1):
        print(f"\n그룹 {idx}:")
        for path in group:
            print(f"  {path}")

    # 중복 이미지 제거 실행
    remove_duplicate_images(duplicate_groups)
else:
    print("중복된 이미지가 없습니다.")
