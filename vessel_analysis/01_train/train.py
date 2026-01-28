import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import VesselDataset
from models import CausalVesselVAE
from config import CONFIG
import numpy as np

def loss_function(recon_x, x, m_hat, m, mu, logvar, m_mu, m_logvar):
    # 1. Reconstruction Loss (Weighted BCE)
    # Penalize missing vessels (x=1) much more than background (x=0)
    # x is [0, 1] (Binarized)
    
    # Calculate pixel-wise loss (no reduction yet)
    # bce = F.binary_cross_entropy(recon_x, x, reduction='none')
    
    # Switch to MSE for Pretrained ViT VAE (which uses Linear Output)
    mse = F.mse_loss(recon_x, x, reduction='none')

    # Dynamic Weighting for Class Imbalance
    with torch.no_grad():
        n_pos = x.sum()
        n_total = x.numel()
        pos_fraction = n_pos / (n_total + 1e-6)
        # Balance influence: pos_weight * pos_frac = 1 * (1 - pos_frac)
        calculated_weight = (1.0 - pos_fraction) / (pos_fraction + 1e-6)
        pos_weight = torch.clamp(calculated_weight, min=1.0, max=50.0)
    
    weight = 1.0 + (pos_weight - 1.0) * x
    
    # Weighted Mean/Sum
    recon_loss = torch.sum(mse * weight)
    
    # 1b. Sparsity Loss - Suppress background noise
    # Penalize non-zero values where original is 0 (background)
    background_mask = (x < 0.1).float()  # Background pixels
    sparsity_loss = torch.sum(torch.abs(recon_x) * background_mask) 
    
    # 2. KLD Loss (Z)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. Morphology Prediction Loss (Gaussian NLL)
    # L_morph = -log P(M|T) = 0.5 * sum(logvar + (m - mu)^2 / exp(logvar))
    # m_hat here is m_mu (mean)
    # We use m_mu and m_logvar from predictor
    # m is Real M
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
        
        loss = recon + CONFIG["BETA"] * kld + morph + 0.3 * sparsity
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0) # Clip Gradients
        opt_vae.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()
        total_morph += morph.item()
        
        if batch_idx % 5 == 0:
             print(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item() / x.size(0):.4f}")
        
    dataset_size = len(train_loader.dataset)
    print(f"   [Train Breakdown] Recon: {total_recon/dataset_size:.1f} | KLD: {total_kld/dataset_size:.1f}")
    return total_loss / dataset_size

def validate(vae, val_loader):
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
            
            # Note: Usually, VGG or adversarial losses are not included in validation metrics
            # unless necessary for model selection.
            # However, to keep consistency with the training loss setup,
            # only Recon / KLD / Morph losses are tracked here
            
            loss = recon + CONFIG["BETA"] * kld + morph + 0.3 * sparsity
            val_loss += loss.item()
            
            total_recon += recon.item()
            total_kld += kld.item()
            total_morph += morph.item()
            
    avg_recon = total_recon / len(val_loader.dataset)
    avg_kld = total_kld / len(val_loader.dataset)
    avg_morph = total_morph / len(val_loader.dataset)
    
    print(f"   [Val Breakdown] Recon: {avg_recon:.1f} | KLD: {avg_kld:.1f} | Morph: {avg_morph:.1f}")
    
    return val_loss / len(val_loader.dataset)

def main():
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    
    # Dataset
    train_dataset = VesselDataset(mode='train')
    val_dataset = VesselDataset(mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)
    
    # Model
    # vae = CausalVesselVAE().to(CONFIG["DEVICE"])
    from models import CausalViTVAE
    pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
    vae = CausalViTVAE(pretrained_path=pretrained_path).to(CONFIG["DEVICE"])
    
    opt_vae = optim.Adam(vae.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    best_val_loss = float('inf')
    
    print(f"Start Training: Epochs={CONFIG['EPOCHS']}, Batch={CONFIG['BATCH_SIZE']}, Image={CONFIG['IMG_HEIGHT']}x{CONFIG['IMG_WIDTH']}")
    
    for epoch in range(CONFIG["EPOCHS"]):
        train_loss = train_one_epoch(epoch, vae, train_loader, opt_vae)
        val_loss = validate(vae, val_loader)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = f"model_ep{CONFIG['EPOCHS']}_bs{CONFIG['BATCH_SIZE']}_lr{CONFIG['LEARNING_RATE']}_beta{CONFIG['BETA']}_best.pt"
            save_path = os.path.join(CONFIG["SAVE_DIR"], model_name)
            torch.save(vae.state_dict(), save_path)
            print(f" -> Best model saved to {save_path}")
        
        # Save Latest Model (every epoch)
        latest_model_name = f"model_ep{CONFIG['EPOCHS']}_bs{CONFIG['BATCH_SIZE']}_lr{CONFIG['LEARNING_RATE']}_beta{CONFIG['BETA']}_latest.pt"
        latest_save_path = os.path.join(CONFIG["SAVE_DIR"], latest_model_name)
        torch.save(vae.state_dict(), latest_save_path)
        
        # Save Checkpoint every 50 epochs
        if epoch % 50 == 0 and epoch > 0:
            checkpoint_name = f"model_ep{CONFIG['EPOCHS']}_bs{CONFIG['BATCH_SIZE']}_lr{CONFIG['LEARNING_RATE']}_beta{CONFIG['BETA']}_epoch{epoch}.pt"
            checkpoint_path = os.path.join(CONFIG["SAVE_DIR"], checkpoint_name)
            torch.save(vae.state_dict(), checkpoint_path)
            print(f" -> Checkpoint saved to {checkpoint_path}")
            
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
                plt.savefig(os.path.join(CONFIG["RESULT_DIR"], f"epoch_{epoch}_sample.png"))
                plt.close()

if __name__ == "__main__":
    main()
