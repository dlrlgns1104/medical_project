import os
import shutil

def copy_images_with_unique_names(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)

        if not os.path.isfile(src_path):
            continue

        # 동일한 이름이 있다면 숫자 붙이기
        base, ext = os.path.splitext(filename)
        new_filename = filename
        counter = 1

        while os.path.exists(os.path.join(dst_folder, new_filename)):
            new_filename = f"{base}_{counter}{ext}"
            counter += 1

        dst_path = os.path.join(dst_folder, new_filename)
        shutil.copy2(src_path, dst_path)
        print(f"복사됨: {src_path} → {dst_path}")


src = r"D:/2024_07_오창교교수님_의료프젝/image_0331/3차" #복사할 이미지가 있는 폴더
dst = r"D:/2024_07_오창교교수님_의료프젝/image_0401/crop incomp" #이미지들이 합쳐질 폴더
copy_images_with_unique_names(src, dst)
