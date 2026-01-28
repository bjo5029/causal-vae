import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_detailed_reliability():
    print("="*60)
    print("Plotting Detailed Reliability (R2 vs Uncertainty per Treatment-Feature)")
    print("="*60)
    
    device = torch.device('cpu')
    base_save_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_saved_models_kfold_morph10000"
    result_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000"
    
    # 1. Load Data
    print("[1/4] Loading Dataset...")
    full_dataset = VesselDataset(mode='all')
    
    # Group indices by treatment
    # We need to know which image belongs to which treatment to calc R2 per treatment
    # full_dataset.data_source is a list of dicts: {'x': path, 'm': ..., 't': label}
    
    from collections import defaultdict
    treatment_indices = defaultdict(list)
    
    # Map label index to name
    group_names = sorted(full_dataset.df['group_name'].dropna().unique())
    label_to_name = {i: name for i, name in enumerate(group_names)}
    
    all_x = []
    all_m = []
    all_t = []
    
    # Pre-load data to tensor for batch processing (small dataset so OK)
    # Actually let's just use the loader
    from torch.utils.data import DataLoader
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    # 2. Load Models
    print("[2/4] Loading 5 Models...")
    models = []
    for fold in range(5):
        path = os.path.join(base_save_dir, f"fold_{fold}", "model_latest.pt")
        if os.path.exists(path):
            m = CausalViTVAE(pretrained_path="/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth").to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
            
    if not models:
        print("Error: No models found.")
        return

    # 3. Inference
    print("[3/4] Running Inference...")
    
    # Storage structure: treatment_name -> feature_idx -> {'true': [], 'pred': [], 'unc': []}
    data_store = defaultdict(lambda: defaultdict(lambda: {'true': [], 'pred': [], 'unc': []}))
    
    feature_names = [
        "Node count", "Extremity Count", "Junction Count", "Edge count",
        "Segment Count", "Branch Count", "Isolated Edge Count",
        "Subnetwork Count", "Total Vessel Length", "Mean Tortuosity",
        "Total Vessel Volume", "Average Vessel Radius"
    ]
    
    with torch.no_grad():
        for x, m_true, t in loader:
            x, t = x.to(device), t.to(device)
            m_true_np = m_true.numpy() # (B, 12)
            t_np = t.argmax(dim=1).cpu().numpy() # (B,) labels
            
            # Ensemble Prediction
            preds_mu = []
            preds_sigma = []
            
            for model in models:
                # Predict M from T
                h = model.morph_predictor_shared(t) # (B, H)
                mu = model.morph_predictor_mu(h) # (B, 12)
                logvar = model.morph_predictor_logvar(h) # (B, 12)
                std = torch.exp(0.5 * logvar)
                
                preds_mu.append(mu.cpu().numpy())
                preds_sigma.append(std.cpu().numpy())
            
            # Average across folds
            avg_mu = np.mean(preds_mu, axis=0) # (B, 12)
            avg_sigma = np.mean(preds_sigma, axis=0) # (B, 12)
            
            # Store
            for i in range(len(t_np)):
                t_label = t_np[i]
                t_name = label_to_name[t_label]
                
                for f_idx in range(12):
                    data_store[t_name][feature_names[f_idx]]['true'].append(m_true_np[i, f_idx])
                    data_store[t_name][feature_names[f_idx]]['pred'].append(avg_mu[i, f_idx])
                    data_store[t_name][feature_names[f_idx]]['unc'].append(avg_sigma[i, f_idx])

    # 4. Calculate R2 and Uncertainty per Group
    print("[4/4] Calculating Metrics & Plotting...")
    
    plot_data = []
    
    for t_name, features in data_store.items():
        for f_name, lists in features.items():
            true_vals = np.array(lists['true'])
            pred_vals = np.array(lists['pred'])
            unc_vals = np.array(lists['unc'])
            
            # R2 Score (handling single sample or constant input case)
            if len(true_vals) < 2:
                r2 = np.nan
            else:
                # Note: Standard R2 can be negative if prediction is worse than mean
                r2 = r2_score(true_vals, pred_vals)
                
            mean_unc = np.mean(unc_vals)
            
            # Categorize
            if "Ramucirumab" in t_name:
                category = "Ramucirumab (Chaotic)"
            elif "PBS" in t_name or "Isotype" in t_name:
                category = "Control (Stable)"
            else:
                category = "Others"
            
            plot_data.append({
                'Treatment': t_name,
                'Feature': f_name,
                'Uncertainty': mean_unc,
                'R2_Score': r2,
                'Category': category
            })
            
    df_plot = pd.DataFrame(plot_data)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Main Scatter
    sns.scatterplot(
        data=df_plot, 
        x='Uncertainty', 
        y='R2_Score', 
        hue='Category',
        style='Category',
        palette={'Ramucirumab (Chaotic)': 'red', 'Control (Stable)': 'green', 'Others': 'blue'},
        s=80,
        alpha=0.7
    )
    
    # Threshold Lines
    plt.axvline(x=0.6, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.6)')
    plt.axhline(y=0.0, color='gray', linestyle=':', linewidth=1)
    
    # Annotate Extremes
    # Annotate high uncertainty points
    worst_points = df_plot.sort_values('Uncertainty', ascending=False).head(5)
    for _, row in worst_points.iterrows():
        plt.text(row['Uncertainty'], row['R2_Score'], f"{row['Feature']}\n({row['Treatment'][:10]}..)", fontsize=7, color='red')
        
    # Annotate best R2 points
    best_points = df_plot.sort_values('R2_Score', ascending=False).head(5)
    for _, row in best_points.iterrows():
        plt.text(row['Uncertainty'], row['R2_Score'], f"{row['Feature']}", fontsize=7, color='green')

    plt.title('Detailed Reliability Check: Per-Treatment Per-Feature R2 vs Uncertainty')
    plt.xlabel('Uncertainty (Std Dev) - Lower is Better')
    plt.ylabel('R2 Score (Accuracy) - Higher is Better')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # Save
    save_path = os.path.join(result_dir, "detailed_dropoff_plot.png")
    plt.savefig(save_path, dpi=200)
    print(f"✓ Saved plot to: {save_path}")
    
    # Print stats
    print("\n[Threshold Validation]")
    low_unc = df_plot[df_plot['Uncertainty'] <= 0.6]['R2_Score'].mean()
    high_unc = df_plot[df_plot['Uncertainty'] > 0.8]['R2_Score'].mean()
    print(f"Mean R2 for Uncertainty <= 0.6: {low_unc:.3f} (Reliable)")
    print(f"Mean R2 for Uncertainty > 0.8:  {high_unc:.3f} (Unreliable)")
    
    # Save CSV for user to inspect
    csv_save_path = os.path.join(result_dir, "detailed_reliability_stats.csv")
    df_plot.to_csv(csv_save_path, index=False)
    print(f"✓ Saved stats to: {csv_save_path}")

if __name__ == "__main__":
    plot_detailed_reliability()
