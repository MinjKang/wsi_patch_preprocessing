import os
from PIL import Image
import glob

def save_center_patch(image_path, save_folder, patch_size=(512, 512)):
    img = Image.open(image_path)
    img_width, img_height = img.size

    # 패치의 중앙 좌표 계산
    center_x = img_width // 2
    center_y = img_height // 2

    # 크롭할 박스 영역 계산
    left = max(center_x - patch_size[0] // 2, 0)
    upper = max(center_y - patch_size[1] // 2, 0)
    right = min(center_x + patch_size[0] // 2, img_width)
    lower = min(center_y + patch_size[1] // 2, img_height)

    box = (left, upper, right, lower)
    patch = img.crop(box)

    # 파일명 생성 (원본 파일명 사용)
    base_name = os.path.basename(image_path).split('.')[0]  # 원본 파일명 (확장자 제외)
    new_filename = f"{os.path.basename(image_path)}"
    patch.save(os.path.join(save_folder, new_filename))

def process_all_images_in_folder(image_folder, save_root_folder):
    subfolders = sorted(os.listdir(image_folder))

    for subfolder in subfolders:
        subfolder_path = os.path.join(image_folder, subfolder)
        
        if os.path.isdir(subfolder_path):  # 하위 폴더인지 확인
            save_folder = os.path.join(save_root_folder, subfolder)

            # 이미 처리된 폴더가 있는지 확인
            if os.path.exists(save_folder):
                print(f"'{subfolder}' 폴더는 이미 처리되었습니다. 건너뜁니다.")
                continue

            print(f"처리 중인 폴더: {subfolder}")
            image_list = glob.glob(os.path.join(subfolder_path, '*.png'))

            # 하위 폴더 생성
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for image_path in image_list:
                save_center_patch(image_path, save_folder)

# 사용 예시
image_folder = '/workspace/AS_pred/0.Data/Patches/TCGA-CRC/'  # 상위 폴더 경로
save_root_folder = '/workspace/mjkang/TCGA_CRC_512(center)'  # 저장 경로

process_all_images_in_folder(image_folder, save_root_folder)
