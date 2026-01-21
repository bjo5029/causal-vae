import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import VesselDataset
from models import CausalVesselVAE, LatentDiscriminator
from config import CONFIG
import numpy as np

def loss_function(recon_x, x, m_hat, m, mu, logvar, m_mu, m_logvar):
    # 1. Reconstruction Loss (Weighted BCE)
    # Penalize missing vessels (x=1) much more than background (x=0)
    # x is [0, 1] (Binarized)
    
    # Calculate pixel-wise loss (no reduction yet)
    bce = F.binary_cross_entropy(recon_x, x, reduction='none')
    
    # Weight: 5.0 for Vessel(1), 1.0 for Background(0)
    # Reduced from 20.0 to avoid "all white" output
    pos_weight = 2.0
    weight = 1.0 + (pos_weight - 1.0) * x
    
    # Weighted Mean/Sum
    recon_loss = torch.sum(bce * weight) 
    
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
    
    return recon_loss, kld_loss, morph_loss
    
    # Debug NaN
    if torch.isnan(recon_loss) or torch.isnan(kld_loss) or torch.isnan(morph_loss):
        print(f"[DEBUG] Loss NaN Detected!")
        print(f"Recon: {recon_loss.item()}, KLD: {kld_loss.item()}, Morph: {morph_loss.item()}")
        print(f"Mu max: {mu.max().item()}, Logvar max: {logvar.max().item()}")
        print(f"M_Mu max: {m_mu.max().item()}, M_Logvar max: {m_logvar.max().item()}")
        
    return recon_loss, kld_loss, morph_loss

def train_one_epoch(epoch, vae, discriminator, train_loader, opt_vae, opt_d):
    vae.train()
    discriminator.train()
    
    total_loss = 0
    total_recon = 0
    total_kld = 0
    total_morph = 0
    total_adv = 0
    
    for batch_idx, (x, m, t) in enumerate(train_loader):
        x = x.to(CONFIG["DEVICE"])
        m = m.to(CONFIG["DEVICE"])
        t = t.to(CONFIG["DEVICE"])
        t_indices = torch.argmax(t, dim=1)
        
        # --- Train Discriminator ---
        opt_d.zero_grad()
        with torch.no_grad():
            _, _, mu, logvar, _, _ = vae(x, m, t)
            z = vae.reparameterize(mu, logvar).detach()
        d_logits = discriminator(z)
        loss_d = F.cross_entropy(d_logits, t_indices)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0) # Clip Gradients
        opt_d.step()
        
        # --- Train VAE ---
        opt_vae.zero_grad()
        recon_x, m_hat, mu, logvar, m_mu, m_logvar = vae(x, m, t)
        
        recon, kld, morph = loss_function(recon_x, x, m_hat, m, mu, logvar, m_mu, m_logvar)
        
        # Adversarial Loss (Fool discriminator)
        z_sample = vae.reparameterize(mu, logvar)
        d_logits_fake = discriminator(z_sample)
        # Target: Uniform distribution (confusion)
        target_uniform = torch.full_like(d_logits_fake, 1.0 / CONFIG["T_DIM"])
        loss_adv = F.kl_div(F.log_softmax(d_logits_fake, dim=1), target_uniform, reduction='batchmean') * CONFIG["LAMBDA_ADV"] * 100
        
        loss = recon + CONFIG["BETA"] * kld + morph + loss_adv
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0) # Clip Gradients
        opt_vae.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()
        total_morph += morph.item()
        total_adv += loss_adv.item()
        
        if batch_idx % 5 == 0:
             print(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item() / x.size(0):.4f}")
        
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def validate(vae, val_loader):
    vae.eval()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    total_morph = 0
    with torch.no_grad():
        for x, m, t in val_loader:
            x = x.to(CONFIG["DEVICE"])
            m = m.to(CONFIG["DEVICE"])
            t = t.to(CONFIG["DEVICE"])
            
            recon_x, m_hat, mu, logvar, m_mu, m_logvar = vae(x, m, t)
            recon, kld, morph = loss_function(recon_x, x, m_hat, m, mu, logvar, m_mu, m_logvar)
            
            loss = recon + CONFIG["BETA"] * kld + morph
            
            if torch.isnan(loss):
                 print(f"[VAL DEBUG] Batch Loss is NaN")
                 break
            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            total_morph += morph.item()
            
    avg_loss = total_loss / len(val_loader.dataset)
    avg_recon = total_recon / len(val_loader.dataset)
    avg_kld = total_kld / len(val_loader.dataset)
    avg_morph = total_morph / len(val_loader.dataset)
    
    print(f"   [Val Breakdown] Recon: {avg_recon:.1f} | KLD: {avg_kld:.1f} | Morph: {avg_morph:.1f}")
    
    return avg_loss

def main():
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    
    # Dataset
    train_dataset = VesselDataset(train=True)
    val_dataset = VesselDataset(train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)
    
    # Model
    vae = CausalVesselVAE().to(CONFIG["DEVICE"])
    discriminator = LatentDiscriminator().to(CONFIG["DEVICE"])
    
    opt_vae = optim.Adam(vae.parameters(), lr=CONFIG["LEARNING_RATE"])
    opt_d = optim.Adam(discriminator.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    best_val_loss = float('inf')
    
    print(f"Start Training: Epochs={CONFIG['EPOCHS']}, Batch={CONFIG['BATCH_SIZE']}, Image={CONFIG['IMG_HEIGHT']}x{CONFIG['IMG_WIDTH']}")
    
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        train_loss = train_one_epoch(epoch, vae, discriminator, train_loader, opt_vae, opt_d)
        val_loss = validate(vae, val_loader)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = f"model_ep{CONFIG['EPOCHS']}_bs{CONFIG['BATCH_SIZE']}_lr{CONFIG['LEARNING_RATE']}_beta{CONFIG['BETA']}_lam{CONFIG['LAMBDA_ADV']}_best.pt"
            save_path = os.path.join(CONFIG["SAVE_DIR"], model_name)
            torch.save(vae.state_dict(), save_path)
            print(f" -> Best model saved to {save_path}")
            
        # Optional: Save sample reconstruction every 10 epochs
        if epoch % 10 == 0:
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
