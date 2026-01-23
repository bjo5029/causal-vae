import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

from config import CONFIG
from config import CONFIG
from models import CausalVesselVAE, CausalViTVAE
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
    """
    print("[Analysis] Calculating Feature Importance...")
    model.eval()
    
    z_dim = CONFIG["Z_DIM"]
    device = CONFIG["DEVICE"]
    
    n_samples = 100
    importance_scores = {name: [] for name in FEATURE_NAMES}
    
    with torch.no_grad():
        # Sample Z from N(0,I)
        z_samples = torch.randn(n_samples, z_dim).to(device)
        
        # Sample random M vectors
        m_samples = torch.randn(n_samples, CONFIG["M_DIM"]).to(device)
        
        # Baseline Image
        dec_input = torch.cat([m_samples, z_samples], dim=1)
        
        # Check architecture type
        if hasattr(model, 'dec_adapter'): # ViT Model
            z_vit = model.dec_adapter(dec_input)
            x_base = model.backbone.decode(z_vit)
        else: # CNN Model
            h = model.dec_fc(dec_input).view(-1, 512, 6, 10)
            x_base = model.dec_conv(h)
        
        # Perturb each feature by +1 sigma (1.0)
        for i, fname in enumerate(FEATURE_NAMES):
            m_perturbed = m_samples.clone()
            m_perturbed[:, i] += 1.0 # Perturb feature i
            
            dec_input_p = torch.cat([m_perturbed, z_samples], dim=1)
            
            if hasattr(model, 'dec_adapter'): # ViT Model
                z_vit_p = model.dec_adapter(dec_input_p)
                x_p = model.backbone.decode(z_vit_p)
            else: # CNN Model
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

def analyze_pairwise_difference(model, group_a, group_b, save_path):
    """
    [Output 3] Pairwise Group Difference
    Identifies features that distinguish Group A from Group B.
    Metric: |Mu_A - Mu_B| / sqrt(Sigma_A^2 + Sigma_B^2) (Bhattacharyya distance approx)
    """
    print(f"[Analysis] Calculating Pairwise Difference: Group {group_a} vs {group_b}...")
    model.eval()
    
    z_dim = CONFIG["Z_DIM"]
    t_dim = CONFIG["T_DIM"]
    device = CONFIG["DEVICE"]
    
    with torch.no_grad():
        # Prepare T vectors
        t_a = torch.zeros(1, t_dim).to(device)
        t_a[0, group_a] = 1.0
        
        t_b = torch.zeros(1, t_dim).to(device)
        t_b[0, group_b] = 1.0
        
        # Predict parameters for A
        h_a = model.morph_predictor_shared(t_a)
        mu_a = model.morph_predictor_mu(h_a).cpu().numpy().flatten()
        logvar_a = model.morph_predictor_logvar(h_a)
        sigma_a = torch.exp(0.5 * logvar_a).cpu().numpy().flatten()
        
        # Predict parameters for B
        h_b = model.morph_predictor_shared(t_b)
        mu_b = model.morph_predictor_mu(h_b).cpu().numpy().flatten()
        logvar_b = model.morph_predictor_logvar(h_b)
        sigma_b = torch.exp(0.5 * logvar_b).cpu().numpy().flatten()
        
        # Calculate Discriminative Score (Normalized Difference)
        # Using a simplistic "Z-score" of the difference distribution
        # Diff ~ N(mu_a - mu_b, sigma_a^2 + sigma_b^2)
        diff_mu = np.abs(mu_a - mu_b)
        combined_sigma = np.sqrt(sigma_a**2 + sigma_b**2)
        scores = diff_mu / (combined_sigma + 1e-9)
        
    # Plot
    # Sort features by score
    data = []
    for i, name in enumerate(FEATURE_NAMES):
        data.append({"Feature": name, "Score": scores[i], "Diff": mu_a[i] - mu_b[i]})
        
    df = pd.DataFrame(data).sort_values(by="Score", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Score", y="Feature", palette="magma")
    plt.title(f"Discriminative Features: Group {group_a} vs {group_b}")
    plt.xlabel("Separation Score (Standardized Difference)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" -> Saved pairwise comparison to {save_path}")
    
    # Save CSV details
    csv_path = save_path.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Saved pairwise CSV to {csv_path}")

def generate_full_report(model, baseline_idx=0, save_path="results_test/full_diff_report.csv"):
    """
    Compares ALL groups against Baseline (Group 0).
    Saves a summary CSV with Top 3 features for each group.
    """
    print(f"[Analysis] Generating Full Report vs Baseline (Group {baseline_idx})...")
    model.eval()
    
    t_dim = CONFIG["T_DIM"]
    device = CONFIG["DEVICE"]
    
    results = []
    
    with torch.no_grad():
        # Clean way: Get all mus/sigmas first
        t_all = torch.eye(t_dim).to(device)
        h_all = model.morph_predictor_shared(t_all)
        mu_all = model.morph_predictor_mu(h_all).cpu().numpy() # (19, 12)
        logvar_all = model.morph_predictor_logvar(h_all)
        sigma_all = torch.exp(0.5 * logvar_all).cpu().numpy() # (19, 12)
        
        # Baseline stats
        mu_base = mu_all[baseline_idx]
        sigma_base = sigma_all[baseline_idx]
        
        for i in range(t_dim):
            if i == baseline_idx:
                continue
                
            # Compare Group i vs Baseline
            mu_target = mu_all[i]
            sigma_target = sigma_all[i]
            
            diff_mu = np.abs(mu_target - mu_base)
            combined_sigma = np.sqrt(sigma_target**2 + sigma_base**2)
            scores = diff_mu / (combined_sigma + 1e-9)
            
            # Rank features
            # Get indices sorted by score descending
            sorted_indices = np.argsort(scores)[::-1]
            
            row = {"Group_ID": i}
            
            # Top 3 Features
            for rank in range(3):
                feat_idx = sorted_indices[rank]
                feat_name = FEATURE_NAMES[feat_idx]
                score = scores[feat_idx]
                real_diff = mu_target[feat_idx] - mu_base[feat_idx] # (+ means Target > Base)
                
                row[f"Top{rank+1}_Feature"] = feat_name
                row[f"Top{rank+1}_Score"] = round(score, 3)
                row[f"Top{rank+1}_Diff"] = round(real_diff, 3)
                
            results.append(row)
            
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f" -> Saved full report to {save_path}")

def generate_all_pairwise_report(model, save_path="results_test/all_pairwise_report.csv"):
    """
    Generates pairwise comparisons for ALL combinations of groups (19x19).
    Saves a summary CSV with Top 3 features for each pair.
    """
    print(f"[Analysis] Generating All Pairwise Reports (19x19)...")
    model.eval()
    
    t_dim = CONFIG["T_DIM"]
    device = CONFIG["DEVICE"]
    
    results = []
    
    with torch.no_grad():
        # Get all predictions first (Optimization)
        t_all = torch.eye(t_dim).to(device)
        h_all = model.morph_predictor_shared(t_all)
        mu_all = model.morph_predictor_mu(h_all).cpu().numpy() # (19, 12)
        logvar_all = model.morph_predictor_logvar(h_all)
        sigma_all = torch.exp(0.5 * logvar_all).cpu().numpy() # (19, 12)
        
        for idx_a in range(t_dim):
            for idx_b in range(t_dim):
                if idx_a == idx_b:
                    continue
                    
                # Compare Group A (Target) vs Group B (Base)
                mu_a = mu_all[idx_a]
                sigma_a = sigma_all[idx_a]
                
                mu_b = mu_all[idx_b]
                sigma_b = sigma_all[idx_b]
                
                diff_mu = np.abs(mu_a - mu_b)
                combined_sigma = np.sqrt(sigma_a**2 + sigma_b**2)
                scores = diff_mu / (combined_sigma + 1e-9)
                
                # Rank features
                sorted_indices = np.argsort(scores)[::-1]
                
                row = {
                    "Group_A": idx_a,
                    "Group_B": idx_b
                }
                
                for rank in range(3):
                    feat_idx = sorted_indices[rank]
                    feat_name = FEATURE_NAMES[feat_idx]
                    score = scores[feat_idx]
                    real_diff = mu_a[feat_idx] - mu_b[feat_idx] # (A - B)
                    
                    row[f"Top{rank+1}_Feature"] = feat_name
                    row[f"Top{rank+1}_Score"] = round(score, 3)
                    row[f"Top{rank+1}_Diff"] = round(real_diff, 3)
                    
                results.append(row)
                
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f" -> Saved all pairwise report to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    # Load Model (Try ViT first, usually works for both if we check keys, but explicit class is better)
    # Using CausalViTVAE as that's what we are training now
    try:
        model = CausalViTVAE(pretrained_path=None).to(CONFIG["DEVICE"])
        # If loading fails due to architecture mismatch, we might need CausalVesselVAE
        # But user is running ViT now.
        model.load_state_dict(torch.load(args.model_path, map_location=CONFIG["DEVICE"]), strict=False)
        print(f"[Main] Loaded CausalViTVAE from {args.model_path}")
    except Exception as e:
        print(f"[Main] Failed to load as ViTVAE: {e}. Trying CausalVesselVAE...")
        model = CausalVesselVAE().to(CONFIG["DEVICE"])
        model.load_state_dict(torch.load(args.model_path, map_location=CONFIG["DEVICE"]))
        print(f"[Main] Loaded CausalVesselVAE from {args.model_path}")
    
    # Run Analyses
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    
    analyze_feature_uncertainty(model, os.path.join(CONFIG["RESULT_DIR"], "uncertainty_map.png"))
    analyze_feature_importance(model, os.path.join(CONFIG["RESULT_DIR"], "feature_importance.png"))
    
    # Run Pairwise for 0 vs 1
    # analyze_pairwise_difference(model, 0, 1, os.path.join(CONFIG["RESULT_DIR"], "pairwise_0_vs_1.png"))
    
    # Full Report vs Baseline (0)
    generate_full_report(model, baseline_idx=0, save_path=os.path.join(CONFIG["RESULT_DIR"], "full_diff_report.csv"))

    # All Pairwise Report (19x19)
    generate_all_pairwise_report(model, save_path=os.path.join(CONFIG["RESULT_DIR"], "all_pairwise_report.csv"))

    # Reconstruction Quality Verification (Validation Set)
    analyze_reconstruction_quality(model, save_dir=CONFIG["RESULT_DIR"])

def analyze_reconstruction_quality(model, save_dir, n_samples=8):
    """
    Verifies reconstruction quality on held-out VALIDATION data.
    Plots Original vs Reconstruction for random samples.
    """
    print(f"[Analysis] verifying Reconstruction Quality on Validation Set...")
    model.eval()
    device = CONFIG["DEVICE"]
    
    # 1. Load Test Dataset (completely unseen data)
    # We rely on the Dataset class's internal seed (42) to match the training split
    from dataset import VesselDataset
    from torch.utils.data import DataLoader
    
    test_dataset = VesselDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=n_samples, shuffle=True)
    
    # Get one batch
    real_imgs, m_vecs, t_vecs = next(iter(test_loader))
    real_imgs = real_imgs.to(device)
    m_vecs = m_vecs.to(device)
    t_vecs = t_vecs.to(device)
    
    with torch.no_grad():
        # Forward Pass
        # We want to see 'recon_x' using the proper M and T
        recon_imgs, _, _, _, _, _ = model(real_imgs, m_vecs, t_vecs)
        
    # Stats Logging
    print(f"[Debug] Recon Image Stats: Min={recon_imgs.min().item():.4f}, Max={recon_imgs.max().item():.4f}, Mean={recon_imgs.mean().item():.4f}")
    
    # Plotting
    real_imgs = real_imgs.cpu()
    recon_imgs = recon_imgs.cpu()
    
    # Do NOT threshold for debugging, visualize the raw output range
    # Matplotlib will clip to 0-1 if we don't specify, but let's normalize for visualization if needed
    # Or just show as is if it's supposed to be [0, 1]
    
    plt.figure(figsize=(20, 5))
    for i in range(n_samples):
        # Original
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(real_imgs[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Original")
        
        # Recon
        plt.subplot(2, n_samples, i + 1 + n_samples)
        # Force vmin=0, vmax=1 to see if values are just very low
        plt.imshow(recon_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        if i == 0: plt.title(f"Reconstruction\nMax: {recon_imgs[i].max():.2f}")
        
    save_path = os.path.join(save_dir, "reconstruction_test_quality_debug.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" -> Saved DEBUG reconstruction plot to {save_path}")

if __name__ == "__main__":
    main()
