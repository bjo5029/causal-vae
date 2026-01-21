import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

from config import CONFIG
from models import CausalVesselVAE
from dataset import VesselDataset

# Feature names from dataset.py
FEATURE_NAMES = [
    "Node count", "Extremity Count", "Junction Count", "Edge count", 
    "Segment Count", "Branch Count", "Isolated Edge Count", 
    "Subnetwork Count(edge count >= 3)", "Total Vessel Length", 
    "Mean Tortuosity", "Total Vessel Volume", "Average Vessel Radius"
]

def analyze_feature_uncertainty(model, save_path):
    """
    [Output 1] Uncertainty Analysis (Sigma Heatmap)
    Calculates P(M|T) uncertainty (sigma) for each Group (T) and Feature (M).
    """
    print("[Analysis] Calculating Uncertainty (Sigma)...")
    model.eval()
    
    with torch.no_grad():
        # Complete T matrix (19 groups)
        t_all = torch.eye(CONFIG["T_DIM"]).to(CONFIG["DEVICE"])
        
        # Predict P(M|T) parameters
        h = model.morph_predictor_shared(t_all)
        # mu = model.morph_predictor_mu(h)
        logvar = model.morph_predictor_logvar(h)
        sigma = torch.exp(0.5 * logvar) # (19, 12)
        
        sigma_np = sigma.cpu().numpy()
        
    # Create DataFrame
    # Rows: Groups (0..18), Cols: Features
    df = pd.DataFrame(sigma_np, columns=FEATURE_NAMES)
    df.index.name = "Group_ID"
    
    # Save CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_path = save_path.replace(".png", ".csv")
    df.to_csv(csv_path)
    print(f" -> Saved uncertainty CSV to {csv_path}")
    
    # Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Uncertainty (Sigma)'})
    plt.title("Model Uncertainty by Group and Feature")
    plt.ylabel("Group ID (Condition)")
    plt.xlabel("Morphological Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" -> Saved uncertainty heatmap to {save_path}")

def analyze_feature_importance(model, save_path):
    """
    [Output 2] Feature Importance (Mediation Analysis)
    Ranking which M features contribute most to visual changes between conditions.
    Approximation: Gradient-based or Perturbation-based sensitivity.
    Here we use a perturbation approach similar to the plan:
    Simulate transitions between random pairs of groups and measure M contribution.
    """
    print("[Analysis] Calculating Feature Importance...")
    model.eval()
    
    # We need Z samples to generate images
    # Collect some Zs from random inputs
    # To save time, just sample from Normal(0,1) since it's VAE
    # Or better, use encoded Zs from validation set
    
    # Simpler approach for global ranking:
    # 1. Generate many (M, Z) pairs.
    # 2. Perturb each M feature and measure |Delta X|.
    
    z_dim = CONFIG["Z_DIM"]
    device = CONFIG["DEVICE"]
    
    n_samples = 100
    importance_scores = {name: [] for name in FEATURE_NAMES}
    
    with torch.no_grad():
        # Sample Z from N(0,I)
        z_samples = torch.randn(n_samples, z_dim).to(device)
        
        # Sample random M vectors (from data distribution)
        # Use dataset to get real Ms
        # Or just random normal since we normalized M
        m_samples = torch.randn(n_samples, CONFIG["M_DIM"]).to(device)
        
        # Baseline Image
        dec_input = torch.cat([m_samples, z_samples], dim=1)
        # Reshape to (512, 6, 10) matching model structure
        h = model.dec_fc(dec_input).view(-1, 512, 6, 10)
        x_base = model.dec_conv(h)
        
        # Perturb each feature by +1 sigma (which is 1.0 in normalized space)
        for i, fname in enumerate(FEATURE_NAMES):
            m_perturbed = m_samples.clone()
            m_perturbed[:, i] += 1.0 # Perturb feature i
            
            dec_input_p = torch.cat([m_perturbed, z_samples], dim=1)
            h_p = model.dec_fc(dec_input_p).view(-1, 512, 6, 10)
            x_p = model.dec_conv(h_p)
            
            # Measure L2 difference
            diff = (x_p - x_base).view(x_p.size(0), -1).norm(dim=1) # (N,)
            mean_diff = diff.mean().item()
            importance_scores[fname] = mean_diff
            
    # Sort and Plot
    sorted_feats = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    names, scores = zip(*sorted_feats)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(scores), y=list(names), palette="viridis")
    plt.title("Feature Importance (Visual Sensitivity to Perturbation)")
    plt.xlabel("Mean Visual Change (L2 Norm)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" -> Saved feature importance plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    # Load Model
    model = CausalVesselVAE().to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(args.model_path, map_location=CONFIG["DEVICE"]))
    print(f"Loaded model from {args.model_path}")
    
    # Run Analyses
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    
    analyze_feature_uncertainty(model, os.path.join(CONFIG["RESULT_DIR"], "uncertainty_map.png"))
    analyze_feature_importance(model, os.path.join(CONFIG["RESULT_DIR"], "feature_importance.png"))

if __name__ == "__main__":
    main()
