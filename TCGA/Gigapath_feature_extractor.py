import os
import time
import torch
import pandas as pd
from gigapath.pipeline import tile_one_slide, load_tile_slide_encoder, run_inference_with_tile_encoder, run_inference_with_slide_encoder

os.environ["HF_TOKEN"] = "hf_"

# Please set your Hugging Face API token
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

device_id = 6
torch.cuda.set_device(device_id)
torch.cuda.set_per_process_memory_fraction(0.6, device=device_id)
torch.set_num_threads(16)
device = torch.device(f'cuda:{device_id}')

# 기본 경로 설정
slide_root_path = '/workspace/minjungkang_990901/mjkang/1.Raw/WSIs/TCGA_STAD'
output_dir = 'outputs/stad2/'
os.makedirs(output_dir, exist_ok=True)

# 모델 로드
tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

# 모든 .svs 파일에 대해 inference 실행
for root, dirs, files in os.walk(slide_root_path):
    for file in files:
        if file.endswith('.svs'):
            slide_path = os.path.join(root, file)
            svs_file_name = os.path.splitext(file)[0]
            
            # 출력 CSV 경로
            csv_output_path = os.path.join(output_dir, f"{svs_file_name}.csv")

            # 이미 처리된 경우 건너뛰기
            if os.path.exists(csv_output_path):
                print(f"Skipping slide: {file} (already processed)")
                continue

            print(f"Processing slide: {file}")

            # 시작 시간 기록
            start_time = time.time()

            try:
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

                # CSV 파일 저장
                embedding_df = pd.DataFrame(embedding_records)
                embedding_df.to_csv(csv_output_path, index=False)
                print(f"Embeddings saved to {csv_output_path}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

            # 종료 시간 기록 및 처리 시간 출력
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processing time for {file}: {elapsed_time:.2f} seconds")

print("Processing completed.")
