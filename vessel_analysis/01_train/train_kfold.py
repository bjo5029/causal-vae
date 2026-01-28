import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

from dataset import VesselDataset
from models import CausalVesselVAE
from config import CONFIG

def loss_function(recon_x, x, m_hat, m, mu, logvar, m_mu, m_logvar):
    """Same loss function as train.py"""
    # 1. Reconstruction Loss (Weighted MSE)
    mse = F.mse_loss(recon_x, x, reduction='none')

    # Dynamic Weighting for Class Imbalance
    with torch.no_grad():
        n_pos = x.sum()
        n_total = x.numel()
        pos_fraction = n_pos / (n_total + 1e-6)
        calculated_weight = (1.0 - pos_fraction) / (pos_fraction + 1e-6)
        pos_weight = torch.clamp(calculated_weight, min=1.0, max=50.0)
    
    weight = 1.0 + (pos_weight - 1.0) * x
    recon_loss = torch.sum(mse * weight)
    
    # 1b. Sparsity Loss - Suppress background noise
    background_mask = (x < 0.1).float()
    sparsity_loss = torch.sum(torch.abs(recon_x) * background_mask) 
    
    # 2. KLD Loss (Z)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. Morphology Prediction Loss (Gaussian NLL)
    m_error = (m - m_mu) ** 2
    m_var = torch.exp(m_logvar)
    morph_loss = 0.5 * torch.sum(m_logvar + m_error / m_var)
    
    return recon_loss, kld_loss, morph_loss, sparsity_loss

def train_one_epoch(epoch, vae, train_loader, opt_vae):
    vae.train()
    
    total_loss = 0
    
    total_recon = 0
    total_kld = 0
    total_morph = 0
    
    for batch_idx, (x, m, t) in enumerate(train_loader):
        x = x.to(CONFIG["DEVICE"])
        m = m.to(CONFIG["DEVICE"])
        t = t.to(CONFIG["DEVICE"])
        t_indices = torch.argmax(t, dim=1)
        
        # --- Train VAE ---
        opt_vae.zero_grad()
        recon_x, m_hat, mu, logvar, m_mu, m_logvar = vae(x, m, t)
        
        recon, kld, morph, sparsity = loss_function(recon_x, x, m_hat, m, mu, logvar, m_mu, m_logvar)
        
        loss = recon + CONFIG["BETA"] * kld + CONFIG["LAMBDA_MORPH"] * morph + 0.3 * sparsity
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
        opt_vae.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()
        total_morph += morph.item()
        
    dataset_size = len(train_loader.dataset)
    return total_loss / dataset_size

def validate(vae, val_loader):
    """Same validation loop as train.py"""
    vae.eval()
    val_loss = 0
    total_recon = 0
    total_kld = 0
    total_morph = 0
    
    with torch.no_grad():
        for x, m, t in val_loader:
            x = x.to(CONFIG["DEVICE"])
            m = m.to(CONFIG["DEVICE"])
            t = t.to(CONFIG["DEVICE"])
            
            recon_x, m_hat, mu, logvar, m_mu, m_logvar = vae(x, m, t)
            recon, kld, morph, sparsity = loss_function(recon_x, x, m_hat, m, mu, logvar, m_mu, m_logvar)
            
            loss = recon + CONFIG["BETA"] * kld + CONFIG["LAMBDA_MORPH"] * morph + 0.3 * sparsity
            val_loss += loss.item()
            
            total_recon += recon.item()
            total_kld += kld.item()
            total_morph += morph.item()
            
    avg_recon = total_recon / len(val_loader.dataset)
    avg_kld = total_kld / len(val_loader.dataset)
    avg_morph = total_morph / len(val_loader.dataset)
    
    print(f"   [Val] Recon: {avg_recon:.1f} | KLD: {avg_kld:.1f} | Morph: {avg_morph:.1f}")
    
    return val_loss / len(val_loader.dataset)

def verify_stratification():
    """Verify that K-Fold splits maintain treatment group distribution"""
    print("\n=== Verifying Stratification ===")
    
    # Load full dataset
    full_dataset = VesselDataset(mode='all')
    
    # Extract treatment labels
    t_labels = []
    for i in range(len(full_dataset.data_source)):
        t_labels.append(full_dataset.data_source[i]['t'])
    t_labels = np.array(t_labels)
    
    # K-Fold split
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(t_labels)), t_labels)):
        train_groups = np.unique(t_labels[train_idx])
        val_groups = np.unique(t_labels[val_idx])
        
        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_idx)} samples, {len(train_groups)} groups")
        print(f"  Val:   {len(val_idx)} samples, {len(val_groups)} groups")
        print(f"  Groups in both: {len(set(train_groups) & set(val_groups))}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150, help='Epochs per fold')
    parser.add_argument('--verify', action='store_true', help='Only verify stratification')
    args = parser.parse_args()
    
    if args.verify:
        verify_stratification()
        return
    
    # Create save directory
    base_save_dir = CONFIG["SAVE_DIR"]
    base_result_dir = CONFIG["RESULT_DIR"]
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(base_result_dir, exist_ok=True)
    
    # Load full dataset (without split)
    print("[K-Fold] Loading full dataset...")
    full_dataset = VesselDataset(mode='all')
    
    # Extract treatment labels for stratification
    t_labels = []
    for i in range(len(full_dataset.data_source)):
        t_labels.append(full_dataset.data_source[i]['t'])
    t_labels = np.array(t_labels)
    
    print(f"[K-Fold] Total samples: {len(t_labels)}")
    print(f"[K-Fold] Unique treatment groups: {len(np.unique(t_labels))}")
    
    # K-Fold split
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(t_labels)), t_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'='*60}")
        
        # Create fold-specific directories
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")
        fold_result_dir = os.path.join(base_result_dir, f"fold_{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)
        os.makedirs(fold_result_dir, exist_ok=True)
        
        # Create train/val subsets
        train_subset = Subset(full_dataset, train_idx.tolist())
        val_subset = Subset(full_dataset, val_idx.tolist())
        
        train_loader = DataLoader(train_subset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)
        
        print(f"Train: {len(train_subset)} samples")
        print(f"Val:   {len(val_subset)} samples")
        
        # Initialize model
        from models import CausalViTVAE
        pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
        vae = CausalViTVAE(pretrained_path=pretrained_path).to(CONFIG["DEVICE"])
        
        opt_vae = optim.Adam(vae.parameters(), lr=CONFIG["LEARNING_RATE"])
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(epoch, vae, train_loader, opt_vae)
            val_loss = validate(vae, val_loader)
            
            print(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(fold_save_dir, "model_best.pt")
                torch.save(vae.state_dict(), save_path)
                print(f" -> Best model saved: {save_path}")
            
            # Save latest
            latest_path = os.path.join(fold_save_dir, "model_latest.pt")
            torch.save(vae.state_dict(), latest_path)
            
            # Save checkpoint every 50 epochs
            if epoch % 50 == 0 and epoch > 0:
                checkpoint_path = os.path.join(fold_save_dir, f"model_epoch{epoch}.pt")
                torch.save(vae.state_dict(), checkpoint_path)
            
            # Optional: Save sample reconstruction every 50 epochs
            if epoch % 50 == 0:
                with torch.no_grad():
                    x, m, t = next(iter(val_loader))
                    x = x.to(CONFIG["DEVICE"])
                    m = m.to(CONFIG["DEVICE"])
                    t = t.to(CONFIG["DEVICE"])
                    recon_x, _, _, _, _, _ = vae(x, m, t)
                    
                    # Plot first one
                    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                    ax[0].imshow(x[0].cpu().squeeze(), cmap='gray')
                    ax[0].set_title("Original")
                    ax[1].imshow(recon_x[0].cpu().squeeze(), cmap='gray')
                    ax[1].set_title("Reconstruction")
                    plt.savefig(os.path.join(fold_result_dir, f"epoch_{epoch}_sample.png"))
                    plt.close()
        
        print(f"\nFold {fold + 1} completed. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
