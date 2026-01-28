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

def analyze_feature_importance_single_fold(model, val_loader, device):
    """
    각 fold의 validation set에서 feature importance 측정
    Returns: dict with feature names and importance scores
    """
    model.eval()
    
    feature_names = [
        "Node count", "Extremity Count", "Junction Count", "Edge count",
        "Segment Count", "Branch Count", "Isolated Edge Count",
        "Subnetwork Count", "Total Vessel Length", "Mean Tortuosity",
        "Total Vessel Volume", "Average Vessel Radius"
    ]
    
    # Collect predictions
    all_m_pred = []
    all_m_true = []
    all_m_std = []  # Uncertainty
    
    with torch.no_grad():
        for x, m, t in val_loader:
            x = x.to(device)
            m = m.to(device)
            t = t.to(device)
            
            # Forward pass
            _, _, _, _, m_mu, m_logvar = model(x, m, t)
            m_std = torch.exp(0.5 * m_logvar)
            
            all_m_pred.append(m_mu.cpu().numpy())
            all_m_true.append(m.cpu().numpy())
            all_m_std.append(m_std.cpu().numpy())
    
    # Concatenate
    m_pred = np.concatenate(all_m_pred, axis=0)  # (N, 12)
    m_true = np.concatenate(all_m_true, axis=0)
    m_std = np.concatenate(all_m_std, axis=0)
    
    # Feature importance: R² score per feature
    from sklearn.metrics import r2_score
    importance_scores = []
    
    for i in range(m_pred.shape[1]):
        r2 = r2_score(m_true[:, i], m_pred[:, i])
        importance_scores.append(r2)
    
    return {
        'feature_names': feature_names,
        'importance': np.array(importance_scores),
        'uncertainty': m_std.mean(axis=0)  # Average uncertainty per feature
    }

def main():
    print("="*60)
    print("K-Fold Feature Importance Aggregation")
    print("="*60)
    
    base_save_dir = CONFIG["SAVE_DIR"]
    base_result_dir = CONFIG["RESULT_DIR"]
    
    n_folds = 5
    device = CONFIG["DEVICE"]
    
    # Load full dataset
    print("\n[1/4] Loading dataset...")
    full_dataset = VesselDataset(mode='all')
    
    # Extract treatment labels for stratification
    t_labels = []
    for i in range(len(full_dataset.data_source)):
        t_labels.append(full_dataset.data_source[i]['t'])
    t_labels = np.array(t_labels)
    
    # K-Fold split (same as training)
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Collect results from all folds
    all_importance = []
    all_uncertainty = []
    feature_names = None
    
    print("\n[2/4] Analyzing each fold...")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(t_labels)), t_labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Load model
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")
        model_path = os.path.join(fold_save_dir, "model_latest.pt")
        
        if not os.path.exists(model_path):
            print(f"[Warning] Model not found: {model_path}")
            print(f"Skipping fold {fold}")
            continue
        
        # Initialize model
        pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
        model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
        
        # Create validation loader
        val_subset = Subset(full_dataset, val_idx.tolist())
        val_loader = DataLoader(val_subset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)
        
        # Analyze
        results = analyze_feature_importance_single_fold(model, val_loader, device)
        
        if feature_names is None:
            feature_names = results['feature_names']
        
        all_importance.append(results['importance'])
        all_uncertainty.append(results['uncertainty'])
        
        print(f"Top 3 features: {np.argsort(results['importance'])[-3:][::-1]}")
    
    if len(all_importance) == 0:
        print("\n[Error] No models found. Please train models first using train_kfold.py")
        return
    
    # Convert to arrays
    all_importance = np.array(all_importance)  # (n_folds, n_features)
    all_uncertainty = np.array(all_uncertainty)
    
    print(f"\n[3/4] Aggregating results from {len(all_importance)} folds...")
    
    # Compute statistics
    mean_importance = all_importance.mean(axis=0)
    std_importance = all_importance.std(axis=0)
    cv_importance = std_importance / (np.abs(mean_importance) + 1e-6)  # Coefficient of Variation
    
    mean_uncertainty = all_uncertainty.mean(axis=0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': mean_importance,
        'Std_Importance': std_importance,
        'CV': cv_importance,
        'Mean_Uncertainty': mean_uncertainty
    })
    
    # Add individual fold results
    for i in range(len(all_importance)):
        df[f'Fold_{i}'] = all_importance[i]
    
    # Sort by mean importance
    df = df.sort_values('Mean_Importance', ascending=False)
    
    # Save CSV
    csv_path = os.path.join(base_result_dir, "feature_importance_kfold.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n-> Saved: {csv_path}")
    
    print("\n" + "="*60)
    print("Feature Importance Summary (Top 5)")
    print("="*60)
    print(df[['Feature', 'Mean_Importance', 'Std_Importance', 'CV']].head(5).to_string(index=False))
    
    # [4/4] Visualization
    print("\n[4/4] Creating visualizations...")
    
    # Plot 1: Bar plot with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(feature_names))
    
    # Sort by mean importance
    sorted_idx = np.argsort(mean_importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_means = mean_importance[sorted_idx]
    sorted_stds = std_importance[sorted_idx]
    
    bars = ax.bar(x_pos, sorted_means, yerr=sorted_stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Feature Importance (R² Score)')
    ax.set_title(f'Feature Importance across {len(all_importance)} Folds (Mean ± Std)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(base_result_dir, "feature_importance_kfold.png")
    plt.savefig(plot_path, dpi=150)
    print(f"-> Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Heatmap of importance across folds
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data for heatmap (features x folds)
    heatmap_data = all_importance.T  # (n_features, n_folds)
    
    sns.heatmap(heatmap_data[sorted_idx], 
                xticklabels=[f'Fold {i}' for i in range(len(all_importance))],
                yticklabels=sorted_names,
                annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'R² Score'})
    
    ax.set_title('Feature Importance Heatmap Across Folds')
    plt.tight_layout()
    heatmap_path = os.path.join(base_result_dir, "fold_comparison_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    print(f"-> Saved: {heatmap_path}")
    plt.close()
    
    # Plot 3: Coefficient of Variation (stability indicator)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_cv = cv_importance[sorted_idx]
    colors = ['green' if cv < 0.2 else 'orange' if cv < 0.5 else 'red' for cv in sorted_cv]
    
    bars = ax.bar(x_pos, sorted_cv, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Coefficient of Variation (CV = Std/Mean)')
    ax.set_title('Feature Importance Stability (Lower CV = More Stable)')
    ax.axhline(y=0.2, color='green', linestyle='--', linewidth=1, label='Stable (CV < 0.2)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Moderate (CV < 0.5)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    cv_path = os.path.join(base_result_dir, "feature_stability_cv.png")
    plt.savefig(cv_path, dpi=150)
    print(f"-> Saved: {cv_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to: {base_result_dir}")
    print("\n안정적인 피처 (CV < 0.2):")
    stable_features = df[df['CV'] < 0.2]['Feature'].tolist()
    if stable_features:
        for feat in stable_features:
            print(f"  - {feat}")
    else:
        print("  (없음)")
    
    print("\n불안정한 피처 (CV > 0.5):")
    unstable_features = df[df['CV'] > 0.5]['Feature'].tolist()
    if unstable_features:
        for feat in unstable_features:
            print(f"  - {feat}")
    else:
        print("  (없음)")

if __name__ == "__main__":
    main()
