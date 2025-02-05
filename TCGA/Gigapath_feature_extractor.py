import os
import time
import pandas as pd
from gigapath.pipeline import tile_one_slide, load_tile_slide_encoder, run_inference_with_tile_encoder, run_inference_with_slide_encoder

# 기본 경로 설정
slide_root_path = '/workspace/minjungkang_990901/mjkang/1.Raw/WSIs/TCGA_STAD/TCGA_STAD'
filtered_svs_dir = '/workspace/mjkang/filtered_svs'
filtered_csv_dir = '/workspace/mjkang/filtered_csv_files'
output_dir = 'outputs/others5/'
os.makedirs(output_dir, exist_ok=True)

# 모델 로드
tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

# filtered_svs에 존재하는 파일 이름 추출 (확장자 제거)
filtered_svs_files = set([os.path.splitext(file)[0] for file in os.listdir(filtered_svs_dir) if file.endswith('.svs')])

# filtered_csv_files에 존재하는 파일 이름 추출 (확장자 제거)
existing_csv_files = set([os.path.splitext(file)[0] for file in os.listdir(filtered_csv_dir) if file.endswith('.csv')])

# 모든 .svs 파일에 대해 inference 실행
for root, dirs, files in os.walk(slide_root_path):
    for file in files:
        if file.endswith('.svs'):
            # 확장자를 제거한 파일명 비교
            svs_file_name = os.path.splitext(file)[0]

            # 조건 1: filtered_svs와 일치하는 파일명
            # 조건 2: filtered_csv_files에 이미 존재하지 않는 경우만 처리
            if svs_file_name in filtered_svs_files and svs_file_name not in existing_csv_files:
                slide_path = os.path.join(root, file)
                print(f"Processing slide: {file}")

                # 시작 시간 기록
                start_time = time.time()

                # 타일 추출
                tile_one_slide(slide_path, save_dir=output_dir, level=1)

                # 타일 이미지 로드
                slide_dir = os.path.join(output_dir, 'output', os.path.basename(slide_path))
                image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]

                # 타일 인코더로 inference 실행
                tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)

                # 슬라이드 인코더로 inference 실행
                slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)

                # 임베딩을 DataFrame으로 저장
                embedding_records = []
                for layer_name, embed in slide_embeds.items():
                    embedding_records.append({
                        'layer_name': layer_name,
                        'embedding': embed.flatten().tolist()  # 각 임베딩을 1차원 리스트로 변환
                    })

                # 각 슬라이드에 대한 개별 CSV 파일 생성
                csv_output_path = os.path.join(output_dir, f"{svs_file_name}.csv")
                embedding_df = pd.DataFrame(embedding_records)
                embedding_df.to_csv(csv_output_path, index=False)
                print(f"Embeddings saved to {csv_output_path}")

                # 종료 시간 기록 및 처리 시간 출력
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Processing time for {file}: {elapsed_time:.2f} seconds")
            else:
                reason = []
                if svs_file_name not in filtered_svs_files:
                    reason.append("not in filtered_svs")
                if svs_file_name in existing_csv_files:
                    reason.append("already processed (exists in filtered_csv_files)")
                print(f"Skipping slide: {file} ({', '.join(reason)})")

print("Processing completed.")
