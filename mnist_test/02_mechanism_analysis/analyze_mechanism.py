
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from config import CONFIG, FEATURE_NAMES
from dataset import MorphMNIST12
from models import CausalMorphVAE12, LatentDiscriminator
from train import train_model

def analyze_mechanism():
    device = CONFIG["DEVICE"]
    print(f"[Mechanism Analysis] Device: {device}")
    
    # 1. Load or Train Model
    print("[Mechanism Analysis] Loading Model...")
    vae = train_model()
    vae.eval()
    
    # 2. Prepare Test Data
    test_dataset = MorphMNIST12(train=False, limit_count=10000)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    all_m_true = []
    all_m_pred = []
    
    print("[Mechanism Analysis] Predicting M from T...")
    with torch.no_grad():
        for x, m, t in test_loader:
            x, m, t = x.to(device), m.to(device), t.to(device)
            
            # Mechanism: T -> M_hat
            m_hat = vae.morph_predictor(t)
            
            all_m_true.append(m.cpu().numpy())
            all_m_pred.append(m_hat.cpu().numpy())
            
    all_m_true = np.concatenate(all_m_true, axis=0)
    all_m_pred = np.concatenate(all_m_pred, axis=0)
    
    # 3. Calculate Metrics
    print("\n[Mechanism Analysis] Metrics per Feature:")
    print(f"{'Feature Name':<20} | {'MSE':<10} | {'R2 Score':<10}")
    print("-" * 50)
    
    r2_scores = []
    mse_scores = []
    
    for i, name in enumerate(FEATURE_NAMES):
        mse = mean_squared_error(all_m_true[:, i], all_m_pred[:, i])
        r2 = r2_score(all_m_true[:, i], all_m_pred[:, i])
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        
        print(f"{name:<20} | {mse:.4f}     | {r2:.4f}")
        
    avg_r2 = np.mean(r2_scores)
    print("-" * 50)
    print(f"Average R2 Score: {avg_r2:.4f}")
    
    if avg_r2 > 0.5:
        print("\n>> PASS: The mechanism (T->M) is valid and predictive.")
    else:
        print("\n>> FAIL: The mechanism (T->M) is poor. Feature importance analysis may be unreliable.")

if __name__ == "__main__":
    analyze_mechanism()
