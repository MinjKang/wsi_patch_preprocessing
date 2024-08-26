## wsi_patch_preprocessing


TCGA
- patch_center_crop.py: 1024*1024 patch 이미지 중앙 crop 후 512 사이즈 패치로 저장
- WSI_patch_overlay.py: patch 이미지를 WSI 위에 overlay(TCGA annotation이 된 patch 확인 가능)
- UNI_feature_extractor.py: patch 이미지에서 UNI feature vector 추출

Tiff WSI
- WSI_to_patch.ipynb: overlap 없이 256*256 patch slicing
- image_preprocessing.ipynb: Image resize, .tiff to .png
