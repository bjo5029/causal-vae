
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, '01_baseline_causal_vae'))
sys.path.append(os.path.join(parent_dir, '02_mechanism_analysis'))
sys.path.append(os.path.join(parent_dir, '03_measurement_approach'))

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CONFIG, FEATURE_NAMES
from train import train_model as train_v1

# Phase 2 imports
sys.path.insert(0, os.path.join(parent_dir, '03_measurement_approach'))
from cvae_train import train_cvae as train_v2
from dataset import extract_refined_features

def analyze_pairwise_comparison(digit_a, digit_b):
    device = CONFIG["DEVICE"]
    print(f"[Pairwise Analysis] Comparing Digit {digit_a} vs Digit {digit_b}...")
    
    # --- Phase 1: Prediction (Mechanism) ---
    print("  -> Running Phase 1 (Prediction)...")
    vae_v1 = train_v1()
    vae_v1.eval()
    
    t_a_p1 = torch.zeros(1, 10).to(device)
    t_a_p1[:, digit_a] = 1.0
    t_b_p1 = torch.zeros(1, 10).to(device)
    t_b_p1[:, digit_b] = 1.0
    
    with torch.no_grad():
        m_a_pred = vae_v1.morph_predictor(t_a_p1).cpu().numpy().flatten()
        m_b_pred = vae_v1.morph_predictor(t_b_p1).cpu().numpy().flatten()
        
    diff_p1 = np.abs(m_a_pred - m_b_pred)
    
    # --- Phase 2: Measurement (Realization) ---
    print("  -> Running Phase 2 (Measurement)...")
    model_v2 = train_v2()
    model_v2.eval()
    
    num_samples = 50
    z_fixed = torch.randn(num_samples, CONFIG["Z_DIM"]).to(device)
    diffs_p2 = []
    
    with torch.no_grad():
        t_a_p2 = torch.zeros(num_samples, 10).to(device)
        t_a_p2[:, digit_a] = 1.0
        t_b_p2 = torch.zeros(num_samples, 10).to(device)
        t_b_p2[:, digit_b] = 1.0
        
        x_a = model_v2.decode(z_fixed, t_a_p2)
        x_b = model_v2.decode(z_fixed, t_b_p2)
        
        for i in range(num_samples):
            feat_a = extract_refined_features(x_a[i].cpu())
            feat_b = extract_refined_features(x_b[i].cpu())
            diffs_p2.append(torch.abs(feat_a - feat_b).numpy())
            
    diff_p2 = np.mean(diffs_p2, axis=0)
    
    # --- Combine & Normalize ---
    # Normalize each to 0-100 relative to its own max to see relative importance order
    diff_p1_norm = diff_p1 / (diff_p1.max() + 1e-9) * 100
    diff_p2_norm = diff_p2 / (diff_p2.max() + 1e-9) * 100
    
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Phase 1 (Prediction)": diff_p1_norm,
        "Phase 2 (Measurement)": diff_p2_norm
    })
    
    # Sort by Phase 2 for consistent plotting
    df = df.sort_values(by="Phase 2 (Measurement)", ascending=False)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    df_melted = df.melt(id_vars="Feature", var_name="Method", value_name="Relative Importance")
    
    sns.barplot(x="Relative Importance", y="Feature", hue="Method", data=df_melted, palette=["#6a0dad", "#d9534f"])
    
    plt.title(f"Feature Importance Comparison: {digit_a} vs {digit_b}", fontsize=16)
    plt.xlabel("Relative Importance Score (Normalized)", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"pairwise_comparison_{digit_a}_vs_{digit_b}.png"
    plt.savefig(filename)
    print(f"Saved comparison plot to {filename}")

if __name__ == "__main__":
    # 3 vs 8 (Topological)
    analyze_pairwise_comparison(3, 8)
    
    # 1 vs 7 (Geometrical)
    analyze_pairwise_comparison(1, 7)
