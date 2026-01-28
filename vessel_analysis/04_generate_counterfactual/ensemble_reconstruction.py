import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG

def generate_ensemble_reconstruction(n_samples=8):
    """
    5개 K-Fold 모델의 앙상블 reconstruction 생성
    """
    print("="*60)
    print("Ensemble Reconstruction from 5 K-Fold Models")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    base_save_dir = CONFIG["SAVE_DIR"]
    base_result_dir = CONFIG["RESULT_DIR"]
    
    # Load full dataset
    print("\n[1/4] Loading dataset...")
    full_dataset = VesselDataset(mode='all')
    
    # Extract treatment labels for stratification
    t_labels = []
    for i in range(len(full_dataset.data_source)):
        t_labels.append(full_dataset.data_source[i]['t'])
    t_labels = np.array(t_labels)
    
    # K-Fold split (same as training)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use fold 0's validation set for visualization
    train_idx, val_idx = list(kfold.split(np.zeros(len(t_labels)), t_labels))[0]
    
    val_subset = Subset(full_dataset, val_idx.tolist())
    val_loader = DataLoader(val_subset, batch_size=n_samples, shuffle=True, num_workers=4)
    
    # Get one batch
    real_imgs, m_vecs, t_vecs = next(iter(val_loader))
    real_imgs = real_imgs.to(device)
    m_vecs = m_vecs.to(device)
    t_vecs = t_vecs.to(device)
    
    print(f"\n[2/4] Loading 5 models and generating predictions...")
    
    # Load all 5 models and get predictions
    all_recons = []
    
    for fold in range(5):
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")
        model_path = os.path.join(fold_save_dir, "model_latest.pt")
        
        if not os.path.exists(model_path):
            print(f"[Warning] Model not found: {model_path}")
            continue
        
        # Load model
        pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
        model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print(f"  Fold {fold}: Loaded {model_path}")
        
        # Generate reconstruction
        with torch.no_grad():
            recon_imgs, _, _, _, _, _ = model(real_imgs, m_vecs, t_vecs)
            all_recons.append(recon_imgs.cpu())
    
    if len(all_recons) == 0:
        print("\n[Error] No models found!")
        return
    
    print(f"\n[3/4] Computing ensemble (average of {len(all_recons)} models)...")
    
    # Ensemble: Average of all reconstructions
    ensemble_recon = torch.stack(all_recons).mean(dim=0)
    
    # Also compute std to show uncertainty
    ensemble_std = torch.stack(all_recons).std(dim=0)
    
    print(f"  Ensemble stats: Min={ensemble_recon.min():.4f}, Max={ensemble_recon.max():.4f}, Mean={ensemble_recon.mean():.4f}")
    print(f"  Uncertainty (std): Min={ensemble_std.min():.4f}, Max={ensemble_std.max():.4f}, Mean={ensemble_std.mean():.4f}")
    
    print(f"\n[4/4] Creating visualization...")
    
    # Move to CPU for plotting
    real_imgs = real_imgs.cpu()
    
    # Plot: Original vs Ensemble Reconstruction
    fig, axes = plt.subplots(3, n_samples, figsize=(20, 8))
    
    for i in range(n_samples):
        # Row 1: Original
        axes[0, i].imshow(real_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12, fontweight='bold')
        
        # Row 2: Ensemble Reconstruction
        axes[1, i].imshow(ensemble_recon[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Ensemble Recon\n(5 models avg)", fontsize=12, fontweight='bold')
        
        # Row 3: Uncertainty (std)
        im = axes[2, i].imshow(ensemble_std[i].squeeze(), cmap='hot', vmin=0, vmax=ensemble_std.max())
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title("Uncertainty (std)", fontsize=12, fontweight='bold')
    
    # Add colorbar for uncertainty
    fig.colorbar(im, ax=axes[2, :], orientation='horizontal', fraction=0.05, pad=0.05)
    
    plt.suptitle(f"K-Fold Ensemble Reconstruction ({len(all_recons)} models)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(base_result_dir, "ensemble_reconstruction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    # Also save individual fold reconstructions for comparison
    fig, axes = plt.subplots(len(all_recons) + 1, n_samples, figsize=(20, 2*(len(all_recons)+1)))
    
    # Row 0: Original
    for i in range(n_samples):
        axes[0, i].imshow(real_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10, fontweight='bold')
    
    # Rows 1-5: Each fold's reconstruction
    for fold_idx, recon in enumerate(all_recons):
        for i in range(n_samples):
            axes[fold_idx + 1, i].imshow(recon[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[fold_idx + 1, i].axis('off')
            if i == 0:
                axes[fold_idx + 1, i].set_ylabel(f"Fold {fold_idx}", fontsize=10)
    
    plt.suptitle("Individual Fold Reconstructions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = os.path.join(base_result_dir, "fold_reconstruction_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {comparison_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("Ensemble Reconstruction Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. {save_path}")
    print(f"  2. {comparison_path}")

if __name__ == "__main__":
    generate_ensemble_reconstruction(n_samples=8)
