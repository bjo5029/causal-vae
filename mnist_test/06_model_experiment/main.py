import torch
import numpy as np
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import MorphMNIST12
from train import train_model, train_external_classifier
from visualize import (
    export_intervention_csv_10x10,
    visualize_intervention_grid_with_original,
    visualize_z_clustering,
    verify_visualization,
    validate_and_analyze_outliers
)

if __name__ == "__main__":
    # 0. 시드 설정
    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])

    # 1. 모델 학습
    trained_model = train_model()

    # 2. 데이터 샘플링 
    FIXED_SEED = 999 
    torch.manual_seed(FIXED_SEED) 
    np.random.seed(FIXED_SEED)

    print(f"\n[Collecting Data] Finding representative samples (Fixed Seed: {FIXED_SEED})...")
    test_dataset = MorphMNIST12(train=False, limit_count=None) 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    digit_samples = {}
    for x, m, t in test_loader:
        label = torch.argmax(t).item()
        if label not in digit_samples:
            digit_samples[label] = (x, m, t)
        if len(digit_samples) == 10: break
    print("[Info] Random selection complete.")

    # 3. 불확실성 정량화 분석 
    from visualize import analyze_feature_uncertainty
    analyze_feature_uncertainty(trained_model)
    
    print("\n[Done] Analysis complete.")
