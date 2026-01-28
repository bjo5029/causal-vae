import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG

def extract_predictions_by_treatment(model, dataset, device):
    """
    각 treatment 조건별로 morphological feature 예측값 추출
    """
    model.eval()
    
    feature_names = [
        "Node count", "Extremity Count", "Junction Count", "Edge count",
        "Segment Count", "Branch Count", "Isolated Edge Count",
        "Subnetwork Count", "Total Vessel Length", "Mean Tortuosity",
        "Total Vessel Volume", "Average Vessel Radius"
    ]
    
    # Get all unique treatments and their names
    all_treatments = []
    for i in range(len(dataset.data_source)):
        all_treatments.append(dataset.data_source[i]['t'])
    unique_treatment_indices = sorted(set(all_treatments))
    
    # Map treatment indices to names
    treatment_names = {idx: dataset.group_names[idx] for idx in unique_treatment_indices}
    
    print(f"Found {len(unique_treatment_indices)} unique treatments:")
    for idx in unique_treatment_indices:
        print(f"  {idx}: {treatment_names[idx]}")
    
    # Collect predictions for each treatment
    treatment_predictions = {idx: [] for idx in unique_treatment_indices}
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, m, t = dataset[i]
            # t is one-hot encoded, get the index
            t_val = t.argmax().item()
            
            # Add batch dimension
            x = x.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)
            t = t.unsqueeze(0).to(device)
            
            # Forward pass
            _, _, _, _, m_mu, m_logvar = model(x, m, t)
            
            # Store prediction
            treatment_predictions[t_val].append(m_mu.cpu().numpy()[0])
    
    # Compute mean prediction for each treatment
    mean_predictions = {}
    std_predictions = {}
    
    for t_idx in unique_treatment_indices:
        preds = np.array(treatment_predictions[t_idx])  # (N_samples, 12)
        mean_predictions[t_idx] = preds.mean(axis=0)
        std_predictions[t_idx] = preds.std(axis=0)
    
    return feature_names, unique_treatment_indices, mean_predictions, std_predictions, treatment_names

def create_color_mapping(treatments, treatment_names):
    """
    약물별로 같은 색상, 농도별로 진하기를 다르게 설정
    """
    import re
    
    # Parse treatment names to extract drug and concentration
    drug_info = {}
    for t_idx in treatments:
        name = treatment_names[t_idx]
        
        # Extract drug name and concentration
        # Pattern: "DrugName concentration unit"
        match = re.match(r'(.+?)\s+([\d.]+)\s*(\w+)', name)
        if match:
            drug = match.group(1).strip()
            conc = float(match.group(2))
            unit = match.group(3)
        else:
            # For controls without concentration
            drug = name
            conc = 0.0
            unit = ""
        
        drug_info[t_idx] = {'drug': drug, 'concentration': conc, 'unit': unit}
    
    # Group by drug
    from collections import defaultdict
    drug_groups = defaultdict(list)
    for t_idx in treatments:
        drug_groups[drug_info[t_idx]['drug']].append(t_idx)
    
    # Assign base colors to each drug
    unique_drugs = sorted(drug_groups.keys())
    base_colors_map = {
        'IsotypeControl': 'gray',
        'PBS-Buffer-1X': 'black',
        'Ramucirumab': 'brown',
        'TIE2': 'blue',
        'VEGFTrap': 'green',
        'aTIE2VEGFTrap-Bispecific': 'red'
    }
    
    # Create color for each treatment
    colors = {}
    for drug, t_indices in drug_groups.items():
        base_color = base_colors_map.get(drug, 'purple')
        
        # Sort by concentration
        t_indices_sorted = sorted(t_indices, key=lambda x: drug_info[x]['concentration'])
        n_conc = len(t_indices_sorted)
        
        if n_conc == 1:
            # Single concentration: use full intensity
            colors[t_indices_sorted[0]] = base_color
        else:
            # Multiple concentrations: vary intensity
            for i, t_idx in enumerate(t_indices_sorted):
                # Alpha from 0.3 (lightest) to 1.0 (darkest)
                alpha = 0.3 + (0.7 * i / (n_conc - 1))
                colors[t_idx] = (base_color, alpha)
    
    return colors, drug_info

def plot_predictions_by_treatment(feature_names, treatments, mean_predictions, std_predictions, treatment_names, output_dir):
    """
    각 조건의 예측값을 plot
    가로축: 형태학적 피처
    세로축: 예측값
    각 점: 각 조건
    약물별로 같은 색상, 농도별로 진하기 다르게
    """
    n_features = len(feature_names)
    n_treatments = len(treatments)
    
    # Create color mapping
    color_map, drug_info = create_color_mapping(treatments, treatment_names)
    
    # Group treatments by drug for legend organization
    from collections import defaultdict
    drug_groups = defaultdict(list)
    for t_idx in treatments:
        drug_groups[drug_info[t_idx]['drug']].append(t_idx)
    
    # Prepare data for plotting with broken axis
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(22, 24), 
                                                   gridspec_kw={'height_ratios': [2, 10, 2], 'hspace': 0.08})
    
    # Find actual data range
    all_values = []
    for t in treatments:
        all_values.extend(mean_predictions[t])
    y_min, y_max = min(all_values), max(all_values)
    
    # Set y-axis ranges
    y_ranges = [
        (1.0, y_max + 0.3),      # Top panel: > 1.0
        (-1.0, 1.0),              # Middle panel: -1.0 to 1.0 (main focus)
        (y_min - 0.3, -1.0)      # Bottom panel: < -1.0
    ]
    
    axes = [ax_top, ax_mid, ax_bot]
    
    # Plot on each axis
    for ax_idx, (ax, (y_low, y_high)) in enumerate(zip(axes, y_ranges)):
        # Plot each treatment, grouped by drug
        for drug in sorted(drug_groups.keys()):
            t_indices = sorted(drug_groups[drug], key=lambda x: drug_info[x]['concentration'])
            
            for t in t_indices:
                means = mean_predictions[t]
                stds = std_predictions[t]
                
                # Get color with alpha
                color_info = color_map[t]
                if isinstance(color_info, tuple):
                    color, alpha = color_info
                else:
                    color, alpha = color_info, 1.0
                
                x_pos = np.arange(n_features)
                
                # Only plot if data is in this y-range
                mask = (means >= y_low) & (means <= y_high)
                if mask.any():
                    # Only add label in middle panel
                    label = treatment_names[t] if ax == ax_mid else ""
                    ax.scatter(x_pos[mask], means[mask], label=label,
                              color=color, s=150, alpha=alpha, edgecolors='black', linewidth=0.8, zorder=3)
                    
                    # Add error bars
                    ax.errorbar(x_pos[mask], means[mask], yerr=stds[mask], fmt='none',
                               ecolor=color, alpha=alpha*0.4, capsize=4, linewidth=1.5, zorder=2)
        
        # Set y limits
        ax.set_ylim(y_low, y_high)
        
        # Set x labels
        ax.set_xticks(range(n_features))
        if ax == ax_bot:
            ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14)
            ax.set_xlabel('Morphological Features', fontsize=18, fontweight='bold', labelpad=10)
        else:
            ax.set_xticklabels([])
        
        # Increase tick label sizes
        ax.tick_params(axis='y', labelsize=14)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=1)
        
        # Add horizontal line at 0 on middle axis
        if ax == ax_mid:
            ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.4, zorder=1)
    
    # Hide spines between axes
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['bottom'].set_visible(False)
    ax_mid.tick_params(labeltop=False, top=False, labelbottom=False, bottom=False)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.tick_params(labeltop=False, top=False)
    
    # Add diagonal break lines
    d = 0.5  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_mid.plot([0, 1], [1, 1], transform=ax_mid.transAxes, **kwargs)
    ax_mid.plot([0, 1], [0, 0], transform=ax_mid.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)
    
    # Set y-label
    fig.text(0.04, 0.5, 'Predicted Value (Original Scale)', 
            va='center', rotation='vertical', fontsize=18, fontweight='bold')
    
    # Add legend on middle axis with larger font
    ax_mid.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, ncol=1,
                 framealpha=0.95, edgecolor='black', fancybox=True,
                 borderpad=1.2, labelspacing=1.0, handletextpad=0.8)
    
    # Add title
    fig.suptitle('Morphological Feature Predictions by Treatment Condition\n(Same drug = Same color, Higher concentration = Darker)',
                fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0.05, 0, 0.98, 0.99])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predictions_by_treatment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()
    
    # Create a second version: one subplot per feature
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()
    
    for feat_idx, feat_name in enumerate(feature_names):
        ax = axes[feat_idx]
        
        # Custom treatment order: aTIE2 → Iso → PBS → Ramu → TIE2 → VEGF
        # Group treatments by drug type
        treatment_order = []
        
        # aTIE2VEGFTrap-Bispecific (14-18)
        # Sort by concentration for consistent ordering within drug group
        atgf_treatments = sorted([t for t in treatments if 14 <= t <= 18], key=lambda x: drug_info[x]['concentration'])
        treatment_order.extend(atgf_treatments)
        
        # IsotypeControl (0-1)
        iso_treatments = sorted([t for t in treatments if 0 <= t <= 1], key=lambda x: drug_info[x]['concentration'])
        treatment_order.extend(iso_treatments)
        
        # PBS-Buffer (2)
        pbs_treatments = sorted([t for t in treatments if t == 2], key=lambda x: drug_info[x]['concentration'])
        treatment_order.extend(pbs_treatments)
        
        # Ramucirumab (3)
        ramu_treatments = sorted([t for t in treatments if t == 3], key=lambda x: drug_info[x]['concentration'])
        treatment_order.extend(ramu_treatments)
        
        # TIE2 (4-8)
        tie2_treatments = sorted([t for t in treatments if 4 <= t <= 8], key=lambda x: drug_info[x]['concentration'])
        treatment_order.extend(tie2_treatments)
        
        # VEGFTrap (9-13)
        vegf_treatments = sorted([t for t in treatments if 9 <= t <= 13], key=lambda x: drug_info[x]['concentration'])
        treatment_order.extend(vegf_treatments)
        
        # Collect values for this feature across all treatments
        values = [mean_predictions[t][feat_idx] for t in treatment_order]
        errors = [std_predictions[t][feat_idx] for t in treatment_order]
        
        # Get colors for each treatment
        point_colors = []
        point_alphas = []
        for t in treatment_order: # Changed from 'treatments' to 'treatment_order'
            color_info = color_map[t]
            if isinstance(color_info, tuple):
                color, alpha = color_info
            else:
                color, alpha = color_info, 1.0
            point_colors.append(color)
            point_alphas.append(alpha)
        
        x_pos = np.arange(n_treatments)
        
        # Plot points with individual colors and alphas
        for i, (x, val, err, col, alph) in enumerate(zip(x_pos, values, errors, point_colors, point_alphas)):
            ax.scatter(x, val, color=col, s=100, alpha=alph, 
                      edgecolors='black', linewidth=0.8, zorder=3)
            ax.errorbar(x, val, yerr=err, fmt='none', 
                       ecolor=col, alpha=alph*0.4, capsize=3, linewidth=1.2, zorder=2)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([treatment_names[t] for t in treatment_order], rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('Predicted Value', fontsize=11, fontweight='bold')
        ax.set_title(feat_name, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Morphological Feature Predictions by Treatment (Individual Features)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path_grid = os.path.join(output_dir, 'predictions_by_treatment_grid.png')
    plt.savefig(output_path_grid, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path_grid}")
    plt.close()
    
    # Create heatmap version
    fig, ax = plt.subplots(figsize=(14, max(8, n_treatments * 0.4)))
    
    # Prepare data matrix (treatments x features)
    heatmap_data = np.array([mean_predictions[t] for t in treatments])
    
    sns.heatmap(heatmap_data, 
                xticklabels=feature_names,
                yticklabels=[treatment_names[t] for t in treatments],
                cmap='viridis', 
                annot=True, fmt='.2f', 
                cbar_kws={'label': 'Predicted Value'},
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Morphological Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Treatment Conditions', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap: Feature Predictions by Treatment', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    
    output_path_heatmap = os.path.join(output_dir, 'predictions_heatmap.png')
    plt.savefig(output_path_heatmap, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path_heatmap}")
    plt.close()
    
    # Save data to CSV
    df = pd.DataFrame(heatmap_data, columns=feature_names)
    df.insert(0, 'Treatment', [treatment_names[t] for t in treatments])
    
    csv_path = os.path.join(output_dir, 'predictions_by_treatment.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

def plot_grouped_treatments(feature_names, treatments, mean_predictions, std_predictions, treatment_names, output_dir):
    """
    조건을 세 그룹으로 나눠서 한 이미지에 표시
    - Group 1: 4-8 (TIE2)
    - Group 2: 9-13 (VEGFTrap)
    - Group 3: 14-18 (aTIE2VEGFTrap-Bispecific)
    약물별로 같은 색상, 농도별로 진하기 다르게
    """
    import matplotlib.colors as mcolors
    
    # Create color mapping
    color_map, drug_info = create_color_mapping(treatments, treatment_names)
    
    # Define groups
    groups = [
        (range(4, 9), "TIE2 (Various Concentrations)", 'blue'),
        (range(9, 14), "VEGFTrap (Various Concentrations)", 'green'),
        (range(14, 19), "aTIE2VEGFTrap-Bispecific (Various Concentrations)", 'red')
    ]
    
    n_features = len(feature_names)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    
    for ax_idx, (treatment_range, title, base_color) in enumerate(groups):
        ax = axes[ax_idx]
        
        # Filter treatments in this range
        group_treatments = [t for t in treatments if t in treatment_range]
        n_treatments = len(group_treatments)
        
        if n_treatments == 0:
            ax.text(0.5, 0.5, f'No data for {title}', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue
        
        # Sort by concentration
        group_treatments_sorted = sorted(group_treatments, 
                                        key=lambda x: drug_info[x]['concentration'])
        
        # Plot each treatment
        for i, t in enumerate(group_treatments_sorted):
            means = mean_predictions[t]
            stds = std_predictions[t]
            
            # Get color with alpha
            color_info = color_map[t]
            if isinstance(color_info, tuple):
                color, alpha = color_info
            else:
                color, alpha = color_info, 1.0
            
            x_pos = np.arange(n_features)
            
            # Create label with concentration
            conc = drug_info[t]['concentration']
            unit = drug_info[t]['unit']
            label = f"{conc} {unit}"
            
            ax.scatter(x_pos, means, label=label, 
                      color=color, s=120, alpha=alpha, 
                      edgecolors='black', linewidth=0.8, zorder=3)
            
            # Add error bars
            ax.errorbar(x_pos, means, yerr=stds, fmt='none', 
                       ecolor=color, alpha=alpha*0.5, capsize=4, zorder=2)
        
        # Set labels
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Predicted Value (Normalized)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # Add legend - sorted by concentration
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=9, 
                 ncol=2, framealpha=0.9, title='Concentration')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
        ax.grid(axis='x', alpha=0.2, linestyle=':', zorder=1)
        
        # Set y-axis limits for better comparison
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    
    plt.suptitle('Morphological Feature Predictions by Treatment Groups\n(Same drug = Same color, Higher concentration = Darker)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predictions_grouped_4-8_9-13_14-18.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

def main():
    print("="*60)
    print("Extract and Plot Predictions by Treatment")
    print("="*60)
    
    base_save_dir = CONFIG["SAVE_DIR"]
    base_result_dir = CONFIG["RESULT_DIR"]
    device = CONFIG["DEVICE"]
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    full_dataset = VesselDataset(mode='all')
    print(f"Total samples: {len(full_dataset)}")
    
    # Load best model from k-fold (use fold 0 as representative)
    print("\n[2/3] Loading model...")
    fold_save_dir = os.path.join(base_save_dir, "fold_0")
    model_path = os.path.join(fold_save_dir, "model_latest.pt")
    
    if not os.path.exists(model_path):
        print(f"[Error] Model not found: {model_path}")
        print("Please train the model first using train_kfold.py")
        return
    
    pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
    model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    # Extract predictions
    print("\n[3/3] Extracting predictions by treatment...")
    feature_names, treatments, mean_predictions, std_predictions, treatment_names = extract_predictions_by_treatment(
        model, full_dataset, device
    )
    
    # Plot
    print("\n[4/5] Creating visualizations...")
    plot_predictions_by_treatment(feature_names, treatments, mean_predictions, std_predictions, treatment_names, base_result_dir)
    
    # Plot grouped treatments (4-8, 9-13, 14-18)
    print("\n[5/5] Creating grouped visualizations...")
    plot_grouped_treatments(feature_names, treatments, mean_predictions, std_predictions, treatment_names, base_result_dir)
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print(f"\nResults saved to: {base_result_dir}")

if __name__ == "__main__":
    main()
