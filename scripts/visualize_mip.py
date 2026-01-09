import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm

# ================= USER SETTINGS =================
# 1. 파일 경로 설정
CSV_PATH = "../data/vessel_analysis_result.csv"
IMG_ROOTS = [
    "../data/Plate-25250_A11-H11",  
    "../data/Plate-25251_A11-H11" 
]
OUT_PATH = "./results/mip_comparison_result.png"

# 2. 시각화 설정
SAMPLES_PER_ROW = 6   # 각 조건(행)당 보여줄 이미지 개수
FIG_SIZE_W = 20       # 전체 그림 가로 크기
FIG_SIZE_H = 8        # 전체 그림 세로 크기
# ===============================================

def find_image_paths(img_roots):
    """폴더 내의 모든 TIFF 파일 경로를 찾아서 ID와 매핑"""
    path_map = {}
    print("Scanning image directories...")
    for r in img_roots:
        files = glob.glob(os.path.join(r, "**", "*.tif*"), recursive=True)
        for f in files:
            # 파일명에서 ID 추출 (예: blah-504002.vessel.tiff -> 504002)
            fname = os.path.basename(f)
            try:
                # 파일명 규칙에 따라 ID 파싱
                clean_name = fname.replace(".tiff", "").replace(".tif", "").replace(".vessel", "")
                img_id = clean_name.split("-")[-1]
                path_map[img_id] = f
            except:
                continue
    print(f"Found {len(path_map)} images.")
    return path_map

def generate_mip(img_path):
    """3D 이미지를 읽어서 MIP(2D) 변환 및 정규화"""
    try:
        # 이미지 로드
        vol = tifffile.imread(img_path)
        
        # 3D인 경우 (Depth, Height, Width) -> MIP (Height, Width)
        if vol.ndim == 3:
            mip = np.max(vol, axis=0)
        else:
            mip = vol # 이미 2D면 그대로 사용
            
        # 시각화를 위한 정규화 (하위 1% ~ 상위 1% 기준으로 Contrast 조절)
        p1, p99 = np.percentile(mip, (1, 99))
        mip = np.clip(mip, p1, p99)
        mip = (mip - p1) / (p99 - p1 + 1e-8)
        return mip
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

def main():
    # 1. 데이터 로드
    df = pd.read_csv(CSV_PATH)
    
    # 2. 이미지 경로 매칭
    path_map = find_image_paths(IMG_ROOTS)
    
    # Image ID를 문자열로 변환하여 매핑
    df['Image ID'] = df['Image ID'].astype(str)
    df['file_path'] = df['Image ID'].map(path_map)
    
    # 경로가 있는 데이터만 필터링
    df_valid = df.dropna(subset=['file_path'])
    print(f"Matched {len(df_valid)} images from CSV.")

    # 3. 그룹별로 나누기
    groups = df_valid.groupby('group_name')
    n_groups = len(groups)
    
    if n_groups == 0:
        print("Error: 표시할 그룹이 없습니다.")
        return

    # 4. 플롯 설정 (행: 그룹 수, 열: 샘플 수)
    fig, axes = plt.subplots(
        nrows=n_groups, 
        ncols=SAMPLES_PER_ROW, 
        figsize=(FIG_SIZE_W, 4 * n_groups), # 세로 크기는 그룹 수에 비례
        squeeze=False
    )
    
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    print("\nGenerating MIP visualization...")

    # 5. 그리기 루프
    for i, (group_name, group_df) in enumerate(groups):
        # 해당 그룹에서 랜덤하게(또는 순서대로) 샘플 뽑기
        samples = group_df.head(SAMPLES_PER_ROW) # 앞에서부터 N개
        
        # 행 제목 (그룹명)
        axes[i, 0].set_ylabel(group_name, fontsize=14, fontweight='bold', rotation=0, labelpad=80, ha='right')
        
        for j in range(SAMPLES_PER_ROW):
            ax = axes[i, j]
            
            if j < len(samples):
                row = samples.iloc[j]
                img_path = row['file_path']
                
                # MIP 생성
                mip_img = generate_mip(img_path)
                
                if mip_img is not None:
                    ax.imshow(mip_img, cmap='gray')
                    # 제목에 ID와 주요 특징 표시 (Volume 등)
                    vol_val = row.get('Total Vessel Volume (μm^3)', 0)
                    tort_val = row.get('Mean Tortuosity', 0)
                    title = f"ID: {row['Image ID']}\nVol: {vol_val:.1e}\nTort: {tort_val:.2f}"
                    ax.set_title(title, fontsize=9)
                else:
                    ax.text(0.5, 0.5, "Load Error", ha='center', va='center')
            else:
                # 샘플 부족하면 빈칸 처리
                ax.axis('off')
                
            # 축 눈금 제거
            ax.set_xticks([])
            ax.set_yticks([])

    # 6. 저장
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\n[Done] Result saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
    