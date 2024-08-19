## Image_preprocessing-pathology_image


TCGA
- patch_center_crop.py: 1024*1024 patch 이미지 중앙 crop 후 512 사이즈 패치로 저장

Tiff WSI
- WSI_to_patch.ipynb: overlap 없이 256*256 patch slicing
- image_preprocessing.ipynb: Image resize, .tiff to .png
