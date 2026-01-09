import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from utils import set_seed, safe_mkdir
from dataset import ImageTableDataset
import analysis
from models import ViTVAE 
from engine import train_vit_vae, extract_vit_latents

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ================= USER SETTINGS =================
CSV_PATH = "../data/vessel_analysis_result.csv"
IMG_ROOTS = [
    "../data/Plate-25250_A11-H11",  
    "../data/Plate-25251_A11-H11" 
]
OUT_DIR = "./results"
CHECKPOINT_PATH = "./checkpoints/vit_vae_epoch_470.pth" 

IMAGE_EXT = ".vessel.tiff"
RESIZE_H = 384
RESIZE_W = 640
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
LATENT_DIM = 512
# ===============================================

def smart_load_weights(model, checkpoint_path, device):
    """
    크기가 안 맞는 가중치는 보간(Interpolate)하거나 건너뛰고 로딩하는 함수
    """
    print(f"\n[Smart Load] Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k in model_state:
            # 1. 크기가 같으면 그냥 로딩
            if v.shape == model_state[k].shape:
                new_state_dict[k] = v
            
            # 2. Pos Embedding 크기가 다르면 (이미지 크기가 달라서 발생) -> 보간(Resize) 수행
            elif k == "pos_embedding":
                print(f"  Resizing {k}: {v.shape} -> {model_state[k].shape}")
                # (1, N+1, D) -> (1, D, N+1) for interpolate
                v_cls = v[:, :1, :] # CLS token
                v_grid = v[:, 1:, :] # Grid tokens
                
                # Checkpoint Grid Size (960 patches -> 24x40)
                n_src = v_grid.shape[1]
                h_src, w_src = 24, 40 
                
                # Target Grid Size (240 patches -> 12x20)
                n_tgt = model_state[k].shape[1] - 1
                h_tgt, w_tgt = 12, 20
                
                # Reshape & Interpolate
                v_grid = v_grid.transpose(1, 2).reshape(1, 256, h_src, w_src)
                v_grid_new = F.interpolate(v_grid, size=(h_tgt, w_tgt), mode='bicubic', align_corners=False)
                v_grid_new = v_grid_new.flatten(2).transpose(1, 2) # (1, 240, 256)
                
                # Combine
                v_new = torch.cat((v_cls, v_grid_new), dim=1)
                new_state_dict[k] = v_new
                
            # 3. Decoder Input (Linear) 크기가 다르면 -> 건너뜀 (학습)
            elif "decoder_input" in k:
                print(f"  Skipping {k} due to shape mismatch (will be initialized randomly)")
                new_state_dict[k] = model_state[k]
                
            else:
                print(f"  Skipping {k} due to unknown mismatch: {v.shape} vs {model_state[k].shape}")
                new_state_dict[k] = model_state[k]
        else:
            pass # 모델에 없는 키는 무시

    model.load_state_dict(new_state_dict, strict=False)
    print("[Smart Load] Done! Encoder loaded successfully.\n")

def main():
    safe_mkdir(OUT_DIR)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 데이터 준비
    print("Loading CSV & Dataset...")
    df_full = pd.read_csv(CSV_PATH)
    
    DROP_COLS = [
        "Plate", "Well", "Image ID", "group_name", "Chip Barcode", "Well Number",
        "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", 
        "Unnamed: 21", "Unnamed: 22", "Unnamed: 23"
    ]

    dataset = ImageTableDataset(
        df=df_full,
        image_root=IMG_ROOTS,
        image_ext=IMAGE_EXT,
        resize_hw=(RESIZE_H, RESIZE_W)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 모델 생성 (Stride 32에 맞게 patch_size=32)
    model = ViTVAE(
        img_size=(RESIZE_H, RESIZE_W),
        patch_size=32,          # ★ models.py 구조에 맞춰 32로 설정
        in_channels=1,
        embed_dim=256,
        depth=6,
        heads=8,
        latent_dim=LATENT_DIM
    ).to(device)

    # 3. 스마트 가중치 로드
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        smart_load_weights(model, CHECKPOINT_PATH, device)
    else:
        print("[Info] No checkpoint found. Starting from scratch...")

    # Phase 2: Latent Extraction
    print("\n>>> Phase 2: Extracting Latents (Z)...")
    Z_all = extract_vit_latents(model, loader, device)
    
    # Phase 3: Ridge Regression (Z -> M)
    print("\n>>> Phase 3: Analyzing Latent Space (Ridge Regression)...")
    
    feature_cols = [c for c in df_full.columns if c not in DROP_COLS]
    M_filtered = dataset.df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    
    print(f"Inputs: Z={Z_all.shape}, M={M_filtered.shape}")

    translator, rank_df, _, _ = analysis.fit_translator_ridge(
        Z=Z_all, 
        M=M_filtered,
        feature_names=feature_cols
    )
    
    save_path = os.path.join(OUT_DIR, "trackA_ranking.csv")
    rank_df.to_csv(save_path, index=False)
    print(f"\n[Done] Ranking Saved: {save_path}")
    print(rank_df.head(10))

if __name__ == "__main__":
    main()
