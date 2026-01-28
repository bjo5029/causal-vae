
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG
import re

def create_color_mapping(treatments, treatment_names):
    """
    Copy of color mapping logic for consistency
    """
    drug_info = {}
    for t_idx in treatments:
        name = treatment_names[t_idx]
        match = re.match(r'(.+?)\s+([\d.]+)\s*(\w+)', name)
        if match:
            drug = match.group(1).strip()
            conc = float(match.group(2))
            unit = match.group(3)
        else:
            drug = name
            conc = 0.0
            unit = ""
        drug_info[t_idx] = {'drug': drug, 'concentration': conc, 'unit': unit}
    
    from collections import defaultdict
    drug_groups = defaultdict(list)
    for t_idx in treatments:
        drug_groups[drug_info[t_idx]['drug']].append(t_idx)
    
    unique_drugs = sorted(drug_groups.keys())
    base_colors_map = {
        'IsotypeControl': 'black', # Changed from gray to black for better visibility
        'PBS-Buffer-1X': 'black',
        'Ramucirumab': 'brown',
        'TIE2': 'blue',
        'VEGFTrap': 'green',
        'aTIE2VEGFTrap-Bispecific': 'red'
    }
    
    colors = {}
    for drug, t_indices in drug_groups.items():
        base_color = base_colors_map.get(drug, 'purple')
        t_indices_sorted = sorted(t_indices, key=lambda x: drug_info[x]['concentration'])
        n_conc = len(t_indices_sorted)
        
        if n_conc == 1:
            colors[t_indices_sorted[0]] = base_color
        else:
            for i, t_idx in enumerate(t_indices_sorted):
                alpha = 0.4 + (0.6 * i / (n_conc - 1)) # Slightly higher base alpha for visibility
                colors[t_idx] = (base_color, alpha)
    return colors, drug_info

def load_global_stats(stats_path):
    df = pd.read_csv(stats_path)
    # Create dict: feature_name -> (mean, std)
    stats = {}
    for _, row in df.iterrows():
        stats[row['Feature']] = (row['Mean'], row['Std'])
    return stats

def main():
    print("="*60)
    print("Visualizing Real vs Predicted Distributions (Overlap)")
    print("="*60)
    
    base_result_dir = CONFIG["RESULT_DIR"]
    device = CONFIG["DEVICE"]
    
    # 1. Load Dataset (Real Data)
    print("\n[1/4] Loading Dataset...")
    dataset = VesselDataset(mode='all')
    
    # 2. Load Model
    print("\n[2/4] Loading Model...")
    fold_save_dir = os.path.join(CONFIG["SAVE_DIR"], "fold_0")
    model_path = os.path.join(fold_save_dir, "model_latest.pt")
    pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
    model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Load Global Stats
    stats_path = os.path.join(base_result_dir, "feature_stats.csv")
    global_stats = load_global_stats(stats_path)
    print(f"Loaded global stats for {len(global_stats)} features.")
    
    feature_names = [
        "Node count", "Extremity Count", "Junction Count", "Edge count",
        "Segment Count", "Branch Count", "Isolated Edge Count",
        "Subnetwork Count(edge count >= 3)", "Total Vessel Length (μm)",
        "Mean Tortuosity", "Total Vessel Volume (μm^3)", "Average Vessel Radius (μm)"
    ] # Matching dataset.py
    
    # 4. Process Data per Feature
    print("\n[3/4] Processing and Plotting...")
    output_dir = os.path.join(base_result_dir, "dist_overlap_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all treatments
    all_treatments = sorted(list(set([d['t'] for d in dataset.data_source])))
    treatment_names = dataset.group_names
    color_map, drug_info = create_color_mapping(all_treatments, treatment_names)
    
    # Custom Sort Order (Grouping by Drug)
    # aTIE2 -> Iso -> PBS -> Ramu -> TIE2 -> VEGF
    # Actually, let's group logically: Controls first, then drugs
    # Order: PBS, Isotype, Ramu, TIE2, VEGF, Bispecific
    
    drug_order_pref = ['PBS-Buffer-1X', 'IsotypeControl', 'Ramucirumab', 'TIE2', 'VEGFTrap', 'aTIE2VEGFTrap-Bispecific']
    
    sorted_treatments = []
    # Group by drug first
    from collections import defaultdict
    t_by_drug = defaultdict(list)
    for t in all_treatments:
        t_by_drug[drug_info[t]['drug']].append(t)
        
    for drug in drug_order_pref:
        if drug in t_by_drug:
            # Sort by concentration within drug
            ts = sorted(t_by_drug[drug], key=lambda x: drug_info[x]['concentration'])
            sorted_treatments.extend(ts)
            
    # Iterate over features
    for f_idx, feat_name in enumerate(feature_names):
        print(f"  - Plotting: {feat_name}")
        
        # Determine Global Mean/Std for this feature
        g_mean, g_std = global_stats.get(feat_name, (0, 1))
        
        # Collect Data for Plotting
        plot_data_real = [] # List of (Treatment Name, Value) tuples for Boxplot
        plot_data_pred = [] # List of (Treatment Name, Mean_Real, Std_Real, Color)
        
        # To align boxplot and scatter, we need numeric X indices
        # We will map treatment -> x_index
        t_to_x = {t: i for i, t in enumerate(sorted_treatments)}
        
        # 4.1 Collect Real Data
        real_values_per_t = defaultdict(list)
        for i in range(len(dataset)):
            sample = dataset.data_source[i]
            t = sample['t']
            val_norm = sample['m'][f_idx] # Raw Real Value (Dataset stores raw in 'm', norm in 'm_norm')
            # Wait, dataset stores 'm' as raw in temp_data?
            # In dataset.py: "m_values = row[self.feature_cols].values.astype(float)" -> This is raw.
            # Then "item['m'] = m_values". So yes, 'm' is raw.
            real_values_per_t[t].append(val_norm)
            
        # 4.2 Collect Predicted Data
        # We need to run inference for each treatment to get uncertainty
        # OR we can just run inference once for any sample of that treatment?
        # The model is Causal VAE: T -> M_pred. It doesn't depend on input X for M_pred (Morph Predictor part).
        # It only depends on T.
        # So we can just run a dummy batch of T.
        
        with torch.no_grad():
            t_tensor = torch.eye(CONFIG["T_DIM"])[sorted_treatments].to(device) # (N_treatments, T_DIM)
            # Morphology Predictor is part of the model. 
            # forward method: ..., m_mu, m_logvar = model(x, m, t)
            # But we want just the predictor: m_mu, m_logvar = model.morph_predictor(t)
            # Let's check model definition. Usually valid.
            
            # Morphology Predictor is split into shared, mu, and logvar in the model
            try:
                # Manual forward pass for morphology predictor
                h = model.morph_predictor_shared(t_tensor)
                m_mu = model.morph_predictor_mu(h)
                m_logvar = model.morph_predictor_logvar(h)
                
                m_std = torch.exp(0.5 * m_logvar)
                
                m_mu = m_mu.cpu().numpy()[:, f_idx]   # (N_treatments,) Z-score
                m_std = m_std.cpu().numpy()[:, f_idx] # (N_treatments,) Z-score
                
                # Convert to Real Scale
                pred_means_real = m_mu * g_std + g_mean
                # Std scales linearly
                pred_stds_real = m_std * g_std 
                
            except AttributeError as e:
                print(f"Error accessing model components: {e}")
                continue

        # Prepare Real Data for Boxplot (List of arrays ordered by sorted_treatments)
        real_data_list = [real_values_per_t[t] for t in sorted_treatments]
        
        # Prepare Plot
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # 1. Plot Real Data (Boxplot) using Seaborn
        # We need a DataFrame for seaborn
        df_real_plot = []
        for t, values in zip(sorted_treatments, real_data_list):
            name = treatment_names[t]
            for v in values:
                df_real_plot.append({'Treatment': name, 'Value': v, 'Type': 'Real'})
                
        df_real = pd.DataFrame(df_real_plot)
        
        # Boxplot background (Real Distribution)
        # Disable fliers in boxplot because stripplot will show them all
        sns.boxplot(data=df_real, x='Treatment', y='Value', ax=ax, 
                    color='lightgray', showfliers=False, boxprops=dict(alpha=0.3))
        
        # Overlay Strip plot for ALL real data points
        sns.stripplot(data=df_real, x='Treatment', y='Value', ax=ax, 
                      color='gray', alpha=0.3, jitter=True, size=12, zorder=0)
        
        # 2. Plot Predicted Data (Overlay)
        # We iterate and plot error bars
        x_indices = np.arange(len(sorted_treatments))
        
        for i, t in enumerate(sorted_treatments):
            mu = pred_means_real[i]
            sigma = pred_stds_real[i]
            
            color_info = color_map[t]
            if isinstance(color_info, tuple):
                color, alpha = color_info
                # Make alpha stronger for the mean dot
                alpha = min(1.0, alpha + 0.2) 
            else:
                color, alpha = color_info, 1.0
                
            # Plot Mean Dot
            ax.scatter(i, mu, color=color, s=150, zorder=10, edgecolors='white', linewidth=1.5, label='Model Prediction' if i==0 else "")
            
            # Plot Uncertainty Interval (Error Bar) - 1 Sigma
            ax.errorbar(i, mu, yerr=sigma, color=color, fmt='none', capsize=5, elinewidth=2.5, zorder=9, alpha=0.9)
            
            # Plot Uncertainty Interval (Error Bar) - 2 Sigma (Fainter)
            # ax.errorbar(i, mu, yerr=2*sigma, color=color, fmt='none', capsize=0, elinewidth=1, zorder=8, alpha=0.3)

        # Formatting
        ax.set_title(f"Real Data Distribution vs. Model Prediction: {feat_name}", fontsize=16, fontweight='bold')
        ax.set_ylabel(f"Real Scale Value ({feat_name})", fontsize=12)
        ax.set_xticklabels([treatment_names[t] for t in sorted_treatments], rotation=45, ha='right')
        
        # Add a custom legend for "Real" and "Predicted"
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=4, alpha=0.5, label='Real Data Distribution (Boxplot)'),
            Line2D([0], [0], marker='o', color='w', label='Model Prediction (Mean)', markerfacecolor='red', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], color='red', lw=2, label='Model Uncertainty (1 Sigma)')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{feat_name.replace(' ', '_').replace('/', '_')}_overlap.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f"\n[4/4] Saved all plots to: {output_dir}")

if __name__ == "__main__":
    main()
