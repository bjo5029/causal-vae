
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, '01_baseline_causal_vae'))
sys.path.append(os.path.join(parent_dir, '02_mechanism_analysis'))
sys.path.append(os.path.join(parent_dir, '03_measurement_approach'))

from config import CONFIG, FEATURE_NAMES
from train import train_model as train_v1
from cvae_train import train_cvae as train_v2
from dataset import extract_refined_features

def print_pairwise_data(digit_a, digit_b):
    device = CONFIG["DEVICE"]
    print(f"\n--- Data for {digit_a} vs {digit_b} ---")
    
    # Phase 1
    vae_v1 = train_v1()
    vae_v1.eval()
    t_a = torch.zeros(1, 10).to(device); t_a[:, digit_a] = 1.0
    t_b = torch.zeros(1, 10).to(device); t_b[:, digit_b] = 1.0
    with torch.no_grad():
        m_a = vae_v1.morph_predictor(t_a).cpu().numpy().flatten()
        m_b = vae_v1.morph_predictor(t_b).cpu().numpy().flatten()
    diff_p1 = np.abs(m_a - m_b)
    diff_p1_norm = diff_p1 / (diff_p1.max() + 1e-9) * 100
    
    # Phase 2
    model_v2 = train_v2()
    model_v2.eval()
    z_fixed = torch.randn(50, CONFIG["Z_DIM"]).to(device)
    t_a_p2 = torch.zeros(50, 10).to(device); t_a_p2[:, digit_a] = 1.0
    t_b_p2 = torch.zeros(50, 10).to(device); t_b_p2[:, digit_b] = 1.0
    
    with torch.no_grad():
        x_a = model_v2.decode(z_fixed, t_a_p2).cpu()
        x_b = model_v2.decode(z_fixed, t_b_p2).cpu()
        
    diffs = []
    for i in range(50):
        fa = extract_refined_features(x_a[i])
        fb = extract_refined_features(x_b[i])
        diffs.append(torch.abs(fa - fb).numpy())
    
    diff_p2 = np.mean(diffs, axis=0)
    diff_p2_norm = diff_p2 / (diff_p2.max() + 1e-9) * 100
    
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "P1_Raw": diff_p1,
        "P1_Norm": diff_p1_norm,
        "P2_Raw": diff_p2,
        "P2_Norm": diff_p2_norm
    })
    print(df.sort_values("P2_Norm", ascending=False).to_string())

if __name__ == "__main__":
    print_pairwise_data(1, 7)
    print_pairwise_data(3, 8)
