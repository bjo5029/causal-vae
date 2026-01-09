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

CSV_PATH = "../data/vessel_analysis_result.csv"
IMG_ROOTS = [
    "../data/Plate-25250_A11-H11",  
    "../data/Plate-25251_A11-H11" 
]

# 이미지 크기 설정
TARGET_SIZE = (512, 960) 

BATCH_SIZE = 8   
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==============================

def main():
    set_seed(42)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    print(f"Loading Data from {IMG_ROOTS}...")
    dataset = CausalDataset(CSV_PATH, IMG_ROOTS, img_size=TARGET_SIZE, is_train=True)
    
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

    best_loss = float('inf') 

    print("Starting Training...")
    for epoch in range(EPOCHS):
        # 1. 학습 진행
        loss = train_one_epoch(model, loader, optimizer, DEVICE)
        
        # 2. 로그 출력
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

        # 3. Best Model 저장
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "./checkpoints/causal_model_best.pth")

    # 4. 마지막 모델도 저장
    torch.save(model.state_dict(), "./checkpoints/causal_model_final.pth")
    print(f"Training Done. Best Loss: {best_loss:.4f}")

    # Best Model 사용
    print("Loading Best Model for Analysis...")
    model.load_state_dict(torch.load("./checkpoints/causal_model_best.pth"))
    
    run_sensitivity_analysis(
        model, dataset.num_groups, dataset.m_cols, DEVICE, 
        "./results/ranking.csv"
    )

if __name__ == "__main__":
    main()
