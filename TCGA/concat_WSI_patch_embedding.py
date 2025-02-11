import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np

# 경로 설정
path_uni = "/workspace/minjungkang_990901/mjkang/2.Features/CRC/concat/uni1024_uni512"
path_wsi = "/workspace/minjungkang_990901/mjkang/2.Features/WSI/CRC"
output_dir = "/workspace/minjungkang_990901/mjkang/2.Features/STAD/concat/WSI_uni1024_uni512"
os.makedirs(output_dir, exist_ok=True)

def process_wsi_vector(wsi_vector_column):
    """
    WSI 벡터 컬럼의 리스트 값을 개별 값으로 분리
    """
    # 리스트 형태의 문자열 값을 파싱하여 배열로 변환
    return np.array([float(x) for x in wsi_vector_column.strip('[]').split(',')])

def concat_csv(file_name):
    file_uni_path = os.path.join(path_uni, file_name)
    file_wsi_path = os.path.join(path_wsi, file_name)

    try:
        # uni1024_512 CSV 파일 읽기
        if not os.path.exists(file_uni_path):
            return f"File not found in uni1024_512 folder: {file_name}"
        df_uni = pd.read_csv(file_uni_path, dtype={'FileName': str})  # FileName을 문자열로 처리

        # WSI CSV 파일 읽기
        if not os.path.exists(file_wsi_path):
            return f"File not found in HCMI_WSI folder: {file_name}"
        df_wsi = pd.read_csv(file_wsi_path, header=None)  # 열 이름 없이 읽기
        df_wsi = df_wsi.iloc[:, 1:]  # 첫 번째 열 제외

        # WSI 데이터 열 이름 생성
        start_col = len(df_uni.columns) - 1  # uni 데이터의 마지막 열 번호
        wsi_columns = [str(start_col + i) for i in range(df_wsi.shape[1])]
        df_wsi.columns = wsi_columns

        # WSI 데이터의 `2049` 열 벡터 값을 분리하여 개별 셀로 변환
        wsi_vector = process_wsi_vector(df_wsi.iloc[-1, -1])  # `2049` 열의 벡터 분리
        wsi_vector_df = pd.DataFrame([wsi_vector] * len(df_uni))  # uni 길이에 맞게 반복

        # uni 데이터와 벡터 데이터 병합
        merged_df = pd.concat([df_uni.reset_index(drop=True), wsi_vector_df.reset_index(drop=True)], axis=1)

        # 열 이름 재정렬: FileName, 0부터 마지막 열 번호까지
        all_columns = ['FileName'] + [str(i) for i in range(merged_df.shape[1] - 1)]
        merged_df.columns = all_columns

        # 결과 저장
        output_path = os.path.join(output_dir, file_name)
        merged_df.to_csv(output_path, index=False)
        return f"Processed and saved: {output_path}"

    except Exception as e:
        return f"Error processing {file_name}: {e}"

def main():
    file_names = [f for f in os.listdir(path_uni) if f.endswith('.csv')]

    num_workers = min(cpu_count(), len(file_names))
    with Pool(num_workers) as pool:
        results = pool.map(concat_csv, file_names)

    # 결과 출력
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
