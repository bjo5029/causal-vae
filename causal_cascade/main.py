import torch
from torch.utils.data import DataLoader
from dataset import CausalDataset
from models import CausalBioVAE
from train import train_one_epoch
from analyze import run_sensitivity_analysis
from utils import set_seed
import os

# ============ 설정 ============
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CSV_PATH = "/home/jeongeun.baek/workspace/causal-vae/data/vessel_analysis_result.csv"
IMG_ROOTS = [
    "/home/jeongeun.baek/workspace/causal-vae/data/Plate-25250_A11-H11",  
    "??" 
]

# 이미지 크기 설정 (비율 유지 권장)
# 원본 비율(1:1.66) 고려하여 (384, 640) 추천. OOM나면 (256, 426)
TARGET_SIZE = (512, 960) 

BATCH_SIZE = 8   
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==============================

def main():
    set_seed(42)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    print(f"Loading Data from {IMG_ROOT}...")
    dataset = CausalDataset(CSV_PATH, IMG_ROOT, img_size=TARGET_SIZE, is_train=True)
    
    if len(dataset) == 0:
        print("Error: 데이터셋 찾을 수 없음")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    print(f"Total Images: {len(dataset)}")
    print(f"Conditions: {dataset.num_groups}, Features: {len(dataset.m_cols)}")

    model = CausalBioVAE(
        img_channels=1,
        m_dim=len(dataset.m_cols),
        t_dim=dataset.num_groups
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, DEVICE)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "./checkpoints/causal_model.pth")
    print("Training Done.")

    run_sensitivity_analysis(
        model, dataset.num_groups, dataset.m_cols, DEVICE, 
        "/results/ranking.csv"
    )

if __name__ == "__main__":
    main()
