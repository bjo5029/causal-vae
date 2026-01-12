
import sys
import os

# Add paths to import modules from reorganized folders
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, '01_baseline_causal_vae'))
sys.path.append(os.path.join(parent_dir, '02_mechanism_analysis'))
sys.path.append(os.path.join(parent_dir, '03_measurement_approach'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import from respective modules
from config import CONFIG, FEATURE_NAMES
from train import train_model as train_v1

# Phase 2 imports
sys.path.insert(0, os.path.join(parent_dir, '03_measurement_approach'))
from cvae_train import train_cvae as train_v2
from dataset import extract_refined_features

def get_phase1_importance(device):
    print("[Comparison] Running Phase 1 Analysis (Prediction)...")
    # Quick re-implementation of analyze_importance logic to avoid print spam/figures
    vae = train_v1()
    vae.eval()
    
    t_input = torch.eye(10).to(device)
    with torch.no_grad():
        m_pred = vae.morph_predictor(t_input).cpu().numpy()
        
    sensitivity = np.std(m_pred, axis=0)
    return sensitivity

def get_phase2_importance(device):
    print("[Comparison] Running Phase 2 Analysis (Measurement)...")
    model = train_v2()
    model.eval()
    
    num_samples = 50 # Reduce for speed
    z_fixed = torch.randn(num_samples, CONFIG["Z_DIM"]).to(device)
    all_measured_m = np.zeros((num_samples, 10, len(FEATURE_NAMES)))
    
    with torch.no_grad():
        for t_idx in range(10):
            t_batch = torch.zeros(num_samples, 10).to(device)
            t_batch[:, t_idx] = 1.0
            x_hat = model.decode(z_fixed, t_batch)
            
            for i in range(num_samples):
                img_tensor = x_hat[i].cpu()
                features = extract_refined_features(img_tensor)
                all_measured_m[i, t_idx, :] = features.numpy()
                
    std_per_sample = np.std(all_measured_m, axis=1)
    avg_sensitivity = np.mean(std_per_sample, axis=0)
    return avg_sensitivity

def main():
    device = CONFIG["DEVICE"]
    
    # Get Importance Scores
    score_p1 = get_phase1_importance(device)
    score_p2 = get_phase2_importance(device)
    
    # Normalize scores for better visual comparison (0-1 scaling)
    # Because units might differ (prediction vs measurement)
    score_p1_norm = (score_p1 - np.min(score_p1)) / (np.max(score_p1) - np.min(score_p1))
    score_p2_norm = (score_p2 - np.min(score_p2)) / (np.max(score_p2) - np.min(score_p2))
    
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Phase 1 (Prediction)": score_p1_norm,
        "Phase 2 (Measurement)": score_p2_norm
    })
    
    # Sort by Phase 2 ranking
    df = df.sort_values(by="Phase 2 (Measurement)", ascending=False)
    
    print("\n[Comparison Result]")
    print(df)
    
    # Plot Side by Side
    plt.figure(figsize=(14, 8))
    
    # Reshape for seaborn
    df_melted = df.melt(id_vars="Feature", var_name="Method", value_name="Normalized Sensitivity")
    
    sns.barplot(x="Normalized Sensitivity", y="Feature", hue="Method", data=df_melted, palette="magma")
    
    plt.title("Constraint Causality: Comparison of Feature Importance (Phase 1 vs Phase 2)", fontsize=16)
    plt.xlabel("Normalized Sensitivity (0-1)", fontsize=12)
    plt.ylabel("Morphological Feature", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = "comparison_feature_importance.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Done] Comparison plot saved to {save_path}")

if __name__ == "__main__":
    main()
