import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

"""
K-Fold Ensemble Pairwise Analysis
5개 fold 모델의 앙상블로 all_pairwise_report.csv 생성
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG

def generate_pairwise_report():
    print("="*60)
    print("K-Fold Ensemble Pairwise Analysis")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    base_save_dir = CONFIG["SAVE_DIR"]
    base_result_dir = CONFIG["RESULT_DIR"]
    
    # Load full dataset
    print("\n[1/3] Loading dataset...")
    full_dataset = VesselDataset(mode='all')
    
    # Extract treatment labels
    t_labels = []
    for i in range(len(full_dataset.data_source)):
        t_labels.append(full_dataset.data_source[i]['t'])
    t_labels = np.array(t_labels)
    
    # Get unique treatment groups
    unique_groups = np.unique(t_labels)
    n_groups = len(unique_groups)
    print(f"Found {n_groups} treatment groups")
    
    # Load all 5 models
    print("\n[2/3] Loading 5 K-Fold models...")
    models = []
    for fold in range(5):
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")
        model_path = os.path.join(fold_save_dir, "model_latest.pt")
        
        if not os.path.exists(model_path):
            print(f"[Warning] Model not found: {model_path}")
            continue
        
        pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
        model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"  Loaded fold {fold}")
    
    if len(models) == 0:
        print("\n[Error] No models found!")
        return
    
    print(f"\n[3/3] Generating pairwise predictions for {n_groups}x{n_groups} combinations...")
    
    # Feature names
    feature_names = [
        "Node count", "Extremity Count", "Junction Count", "Edge count",
        "Segment Count", "Branch Count", "Isolated Edge Count",
        "Subnetwork Count", "Total Vessel Length", "Mean Tortuosity",
        "Total Vessel Volume", "Average Vessel Radius"
    ]
    
    # Prepare results storage
    results = []
    
    # For each treatment pair
    for t_from in range(n_groups):
        for t_to in range(n_groups):
            # Create one-hot vectors
            t_from_vec = torch.zeros(1, CONFIG["T_DIM"]).to(device)
            t_to_vec = torch.zeros(1, CONFIG["T_DIM"]).to(device)
            t_from_vec[0, t_from] = 1.0
            t_to_vec[0, t_to] = 1.0
            
            # Ensemble prediction: average across all models
            all_m_from = []
            all_m_to = []
            
            with torch.no_grad():
                for model in models:
                    # Predict M for each treatment
                    h = model.morph_predictor_shared(t_from_vec)
                    m_from_mu = model.morph_predictor_mu(h)
                    
                    h = model.morph_predictor_shared(t_to_vec)
                    m_to_mu = model.morph_predictor_mu(h)
                    
                    all_m_from.append(m_from_mu.cpu().numpy())
                    all_m_to.append(m_to_mu.cpu().numpy())
            
            # Average across models
            m_from = np.mean(all_m_from, axis=0).squeeze()  # (12,)
            m_to = np.mean(all_m_to, axis=0).squeeze()
            
            # Compute difference
            m_diff = m_to - m_from
            
            # Store results
            row = {
                'Treatment_From': full_dataset.group_names[t_from],
                'Treatment_To': full_dataset.group_names[t_to],
            }
            
            for i, feat_name in enumerate(feature_names):
                row[feat_name] = m_diff[i]
            
            results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save
    save_path = os.path.join(base_result_dir, "all_pairwise_report.csv")
    df.to_csv(save_path, index=False)
    
    print(f"\n✓ Saved: {save_path}")
    print(f"  Shape: {df.shape} ({n_groups}x{n_groups} = {n_groups*n_groups} rows, {len(feature_names)} features)")
    
    # Show sample
    print("Sample (First row):")
    print(df.iloc[[0]][['Treatment_From', 'Treatment_To'] + feature_names[:3]].to_string(index=False))
    
    print("\n" + "="*60)
    print("Pairwise Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    generate_pairwise_report()
