import openslide
from PIL import Image, ImageDraw
import os
import re

# WSI 이미지 경로
wsi_path = 'C:/Users/minjk/WSI/TCGA-EI-6883-01Z-00-DX1.cbb30a39-9428-45f0-9178-40616322b073.svs'
# 패치 이미지 디렉토리
patches_dir = 'C:/Users/minjk/WSI/TCGA-EI-6883-01Z-00-DX1.cbb30a39-9428-45f0-9178-40616322b073/'

# WSI 이미지 열기
slide = openslide.OpenSlide(wsi_path)
overlay_image = Image.new('RGBA', slide.dimensions)

# 패치 이미지 처리
for filename in os.listdir(patches_dir):
    if filename.endswith('.png'):
        # 파일명에서 위치 정보 추출
        match = re.search(r'x=(\d+),y=(\d+)', filename)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            patch_path = os.path.join(patches_dir, filename)
            print(match, x, y)

            # 패치 이미지 열기
            patch = Image.open(patch_path).convert('RGBA')  # RGBA로 변환

            # 검정 테두리 추가
            border_size = 5  # 테두리 두께
            bordered_patch = Image.new('RGBA', (patch.width + border_size * 2, patch.height + border_size * 2))
            draw = ImageDraw.Draw(bordered_patch)

            # 테두리 그리기
            draw.rectangle([0, 0, bordered_patch.width, bordered_patch.height], outline='black', width=border_size)
            bordered_patch.paste(patch, (border_size, border_size), patch)  # 패치를 테두리 안에 붙임

            # 오버랩 이미지에 패치 추가
            overlay_image.paste(bordered_patch, (x, y), bordered_patch)

# 원본 WSI 이미지와 오버레이 이미지 결합
wsi_image = slide.get_thumbnail(slide.dimensions)  # 전체 WSI 이미지를 가져옴
wsi_image = wsi_image.convert('RGBA')  # RGBA로 변환

# 투명도 조정
wsi_image = wsi_image.point(lambda p: p * 0.8)  # 투명도 80%로 설정

# 오버레이 이미지와 결합
combined_image = Image.alpha_composite(wsi_image, overlay_image)

# 최종 이미지 크기 조정 (1/10 크기)
final_image = combined_image.resize((combined_image.width // 10, combined_image.height // 10))

# 최종 오버랩 이미지 저장
final_image.save('TCGA-EI-6883-01Z-00-DX1.cbb30a39-9428-45f0-9178-40616322b073_overlay.png')
