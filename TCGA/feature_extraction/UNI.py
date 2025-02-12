import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import glob
import pandas as pd
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

device_id = 6
torch.cuda.set_device(device_id)
torch.cuda.set_per_process_memory_fraction(0.6, device=device_id)
torch.set_num_threads(16)
device = torch.device(f'cuda:{device_id}')

login()

class ImageDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.files_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_path = self.files_list[idx]
        try:
            img = read_image(img_path).float() / 255.0  # 이미지를 읽고 정규화
            if img is None or img.numel() == 0:  # 이미지가 비어있는 경우
                raise ValueError("Empty image")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return None  # 빈 이미지인 경우 None 반환

        sample = {'input': img, 'input_path': img_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

def print_folder_structure(image_folder):
    subfolders = sorted(os.listdir(image_folder))
    total_images = 0
    folder_image_counts = []

    print(f'Number of WSIs: {len(subfolders)}')
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(image_folder, subfolder)
        if os.path.isdir(subfolder_path):
            image_list = glob.glob(os.path.join(subfolder_path, '*.png'))
            num_images = len(image_list)
            # print(f'{subfolder}: {num_images} images')
            total_images += num_images
            folder_image_counts.append(num_images)
    
    if folder_image_counts:
        average_images = total_images / len(folder_image_counts)
    else:
        average_images = 0

    print(f'Total number of images: {total_images}')
    print(f'Average number of images per folder: {average_images:.2f}')

def save_feature_vectors(image_folder, model, save_folder):
    # 저장 폴더가 존재하지 않으면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    subfolders = sorted(os.listdir(image_folder))[310:]
    
    for index, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(image_folder, subfolder)
        csv_save_path = os.path.join(save_folder, f'{subfolder}.csv')
        
        if os.path.exists(csv_save_path):
            print(f'Skipping folder {index + 1}/{len(subfolders)}: {subfolder_path} (already processed)')
            continue  # 이미 처리된 폴더는 건너뜀

        print(f'Processing folder {index + 1}/{len(subfolders)}: {subfolder_path}')
        
        if os.path.isdir(subfolder_path):  # 하위 폴더인지 확인
            image_list = glob.glob(os.path.join(subfolder_path, '*.png'))
            
            if not image_list:
                continue  # 하위 폴더에 이미지가 없으면 넘어감
            
            # 데이터셋 및 데이터로더 생성
            dataset = ImageDataset(image_list)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            feature_vectors = []
            image_names = []

            model.eval()  # 모델을 평가 모드로 설정
            with torch.no_grad():
                for batch in dataloader:
                    if batch is None or 'input' not in batch:  # 빈 이미지 체크
                        continue  # 빈 이미지가 있는 경우 건너뜀
                    
                    img = batch['input'].to(device)  # GPU 사용 시
                    feats = model(img)  # 특징 벡터 추출
                    feature_vectors.append(feats.cpu().numpy().flatten())  # CPU로 이동 후 리스트에 추가
                    image_names.append(os.path.basename(batch['input_path'][0]))  # 이미지 파일명 저장

            # 결과를 DataFrame으로 변환
            if feature_vectors:  # feature_vectors가 비어있지 않은 경우에만 저장
                df = pd.DataFrame(feature_vectors)
                df.index = image_names  # 인덱스를 이미지 파일명으로 설정
                
                # 열 이름 추가
                df.columns = [str(i) for i in range(df.shape[1])]  # 0부터 시작하는 숫자 열 이름 설정

                # CSV 파일로 저장
                df.to_csv(csv_save_path, index=True, header=True)  # header를 True로 설정하여 열 이름 포함


# feature extractor
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
model = model.to(device)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

image_folder = '/workspace/mjkang/TCGA_CRC_512(non-white)'  # 상위 폴더 경로
save_folder = '/workspace/mjkang/AttentionDeepMIL/uni_features/512(crop_4)'  # CSV 파일 저장 경로

# 폴더 구조 출력
print_folder_structure(image_folder)

# 특징 벡터 저장
save_feature_vectors(image_folder, model, save_folder)
