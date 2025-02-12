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
import numpy as np

device_id = 4
torch.cuda.set_device(device_id)
torch.cuda.set_per_process_memory_fraction(0.8, device=device_id)
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
        img = read_image(img_path).float() / 255.0
        return {'input': img, 'input_path': img_path}
                

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
            total_images += num_images
            folder_image_counts.append(num_images)
    
    if folder_image_counts:
        average_images = total_images / len(folder_image_counts)
    else:
        average_images = 0

    print(f'Total number of images: {total_images}')
    print(f'Average number of images per folder: {average_images:.2f}')

def save_feature_vectors(image_folder, model, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    subfolders = sorted(os.listdir(image_folder))[170:]

    for index, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(image_folder, subfolder)
        print(subfolder_path)
        csv_save_path = os.path.join(save_folder, f'{subfolder}.csv')

        if os.path.exists(csv_save_path):
            print(f'Skipping folder {index + 1}/{len(subfolders)}: {subfolder_path} (already processed)')
            continue

        print(f'Processing folder {index + 1}/{len(subfolders)}: {subfolder_path}')

        if os.path.isdir(subfolder_path):
            image_list = glob.glob(os.path.join(subfolder_path, '*.png'))
            print(len(image_list))

            dataset = ImageDataset(image_list)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

            feature_vectors = []
            image_names = []

            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    img = batch['input'].to(device)
                    try:
                        feats = model(img)
                        feature_vectors.extend(feats.cpu().numpy())
                        image_names.extend([os.path.basename(path) for path in batch['input_path']])
                    except Exception as e:
                        print(f"Error during model inference: {e}")

            if feature_vectors:
                df = pd.DataFrame(feature_vectors)
                df.index = image_names
                df.columns = [str(i) for i in range(df.shape[1])]
                print(f"Saving CSV with {len(df)} rows")
                df.to_csv(csv_save_path, index=True, header=True)

# feature extractor
model = timm.create_model("resnet34", pretrained=True, num_classes=0)
model = model.to(device)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

image_folder = '/workspace/minjungkang_990901/mjkang/1.Raw/patches/CRC/512/center_crop(non_white)'
save_folder = '/workspace/minjungkang_990901/mjkang/2.Features/CRC/ResNet34/512'

# 폴더 구조 출력
print_folder_structure(image_folder)

# 특징 벡터 저장
save_feature_vectors(image_folder, model, save_folder)
