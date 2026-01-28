import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG
import re

def extract_uncertainty_by_treatment():
    print("="*60)
    print("Extracting Uncertainty by Treatment (Per-Condition Confidence)")
    print("="*60)
    
    # Force CPU to avoid CUDA mismatches
    device = torch.device('cpu') 
    
    # Correct path including 'outputs'
    base_save_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_saved_models_kfold_morph10000"
    base_result_dir = CONFIG["RESULT_DIR"] # Keep this or update if needed, but it looked okay? No, let's use the same base for results
    base_result_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000"
    
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir, exist_ok=True)
    
    # 1. Load Dataset to get Treatment Info
    print("\n[1/3] Loading dataset...")
    full_dataset = VesselDataset(mode='all')
    
    # Get treatment mapping
    group_names = sorted(full_dataset.df['group_name'].dropna().unique())
    treatment_map = {i: name for i, name in enumerate(group_names)}
    n_treatments = len(group_names)
    print(f"Found {n_treatments} treatments.")

    # 2. Load Ensemble Models (5 Folds)
    print("\n[2/3] Loading 5 K-Fold models...")
    models = []
    for fold in range(5):
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")
        model_path = os.path.join(fold_save_dir, "model_latest.pt")
        
        if not os.path.exists(model_path):
            print(f"Skipping fold {fold} (Model not found)")
            continue
            
        # Initialize & Load
        pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
        model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"  Loaded fold {fold}")

    feature_names = [
        "Node count", "Extremity Count", "Junction Count", "Edge count",
        "Segment Count", "Branch Count", "Isolated Edge Count",
        "Subnetwork Count", "Total Vessel Length", "Mean Tortuosity",
        "Total Vessel Volume", "Average Vessel Radius"
    ]
    
    # 3. Predict Uncertainty for Each Treatment
    print("\n[3/3] Calculating uncertainty for each treatment...")
    
    results = []
    
    with torch.no_grad():
        for t_idx in range(n_treatments):
            treatment_name = treatment_map[t_idx]
            
            # Create input tensor for this treatment
            t_tensor = torch.zeros(1, CONFIG["T_DIM"]).to(device)
            t_tensor[0, t_idx] = 1.0
            
            # Ensemble Prediction
            fold_uncertainties = [] # Stores (1, 12) from each fold
            
            for model in models:
                # Get probabilistic output (mu, logvar) from Morph Predictor
                # P(M|T)
                h = model.morph_predictor_shared(t_tensor)
                # m_mu = model.morph_predictor_mu(h)
                m_logvar = model.morph_predictor_logvar(h)
                m_logvar = torch.clamp(m_logvar, min=-10, max=10)
                
                # Uncertainty = Standard Deviation (sigma) = exp(0.5 * logvar)
                m_std = torch.exp(0.5 * m_logvar).cpu().numpy().flatten() # (12,)
                fold_uncertainties.append(m_std)
            
            # Average uncertainty across folds associated with this treatment
            # (Aleatoric Uncertainty averaged over Epistemic Ensemble)
            avg_uncertainty = np.mean(fold_uncertainties, axis=0) # (12,)
            
            # Store Result
            row = {'Treatment': treatment_name}
            for i, feat in enumerate(feature_names):
                row[feat] = avg_uncertainty[i]
            results.append(row)
            
    # 4. Save to CSV
    df = pd.DataFrame(results)
    
    # Reorder columns: Treatment first, then features
    cols = ['Treatment'] + feature_names
    df = df[cols]
    
    save_path = os.path.join(base_result_dir, "uncertainty_by_treatment.csv")
    df.to_csv(save_path, index=False)
    
    print(f"\n✓ Saved: {save_path}")
    print(df.head())

if __name__ == "__main__":
    extract_uncertainty_by_treatment()
