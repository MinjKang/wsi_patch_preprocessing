import os
import glob
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import concurrent.futures

csv_dir = "/workspace/minjungkang_990901/mjkang/2.Features/CRC/"
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

def process_csv(csv_file):
    """
    각 CSV 파일을 읽고, AS 정보를 이진 값(AS>=10: 1, AS<10: 0)으로 변환
    불필요한 컬럼을 제거하고, feature vector를 추출하여 반환
    """
  
    df = pd.read_csv(csv_file)
    as_values = df['AS'].values
    binary_as = (as_values >= 10).astype(int)
    df = df.drop(columns=['AS', 'MSI'])
    feat = df.iloc[:, 1:].values
    wsi_label = [os.path.basename(csv_file)] * feat.shape[0]
    return feat, binary_as, wsi_label

# 멀티 스레딩을 이용하여 CSV 파일들을 병렬로 읽기
features_list = []
as_labels = []
wsi_labels = []

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_csv, csv_files))

for feat, binary_as, wsi_label in results:
    features_list.append(feat)
    as_labels.extend(binary_as)
    wsi_labels.extend(wsi_label)

features = np.concatenate(features_list, axis=0)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=as_labels, cmap='coolwarm', s=5, alpha=0.7)
plt.title("UMAP Projection of Patch Features")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter, label='AS Threshold (0: AS<10, 1: AS>=10)')

# umap 시각화 결과 png로 저장
plt.savefig("umap_projection.png", dpi=300, bbox_inches='tight')
plt.close()
