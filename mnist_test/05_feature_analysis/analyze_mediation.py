import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../01_baseline_causal_vae')))

from dataset import MorphMNIST12
from models import CausalMorphVAE12, LatentDiscriminator
from config import CONFIG
import warnings

warnings.filterwarnings("ignore")

def train_temp_model():
    """Trains a temporary model for analysis"""
    print("[1] Training Causal VAE (Quick Train)...")

    EPOCHS = 30 
    
    train_dataset = MorphMNIST12(train=True, limit_count=60000)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    vae = CausalMorphVAE12().to(CONFIG["DEVICE"])
    discriminator = LatentDiscriminator().to(CONFIG["DEVICE"])
     
    opt_vae = optim.Adam(vae.parameters(), lr=CONFIG["LR"])
    opt_d = optim.Adam(discriminator.parameters(), lr=CONFIG["LR"])
    
    vae.train()
    discriminator.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, m, t in train_loader:
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            t_indices = torch.argmax(t, dim=1)
            
            # Train D
            opt_d.zero_grad()
            with torch.no_grad():
                _, _, mu, logvar = vae(x, m, t)
                z = vae.reparameterize(mu, logvar).detach()
            d_logits = discriminator(z)
            loss_d = F.cross_entropy(d_logits, t_indices)
            loss_d.backward()
            opt_d.step()
            
            # Train VAE
            opt_vae.zero_grad()
            recon_x, m_hat, mu, logvar = vae(x, m, t)
            loss_recon = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
            kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kld = kld_element * CONFIG["BETA"]
            loss_morph = F.mse_loss(m_hat, m, reduction='sum') * 100
            
            z_sample = vae.reparameterize(mu, logvar)
            d_logits_fake = discriminator(z_sample)
            target_uniform = torch.full_like(d_logits_fake, 1.0 / CONFIG["T_DIM"])
            loss_adv = F.kl_div(F.log_softmax(d_logits_fake, dim=1), target_uniform, reduction='batchmean') * CONFIG["LAMBDA_ADV"] * 100
            
            loss = loss_recon + loss_kld + loss_morph + loss_adv
            loss.backward()
            opt_vae.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_dataset):.1f}")
        
    return vae

def analyze_mediation(model):
    print("\n[2] Performing Mediation Analysis (1 -> 8)...")
    model.eval()
    
    # Define features mapping
    feature_names = [
        "Area", "Perimeter", "Thickness", "MajorAxis", "Eccentricity", 
        "Orientation", "Solidity", "Extent", "AspectRatio", "Euler", 
        "H_Symmetry", "V_Symmetry"
    ]
    
    # Define Target Digits
    DIGIT_A_IDX = 1 # Source
    DIGIT_B_IDX = 8 # Target
    
    # 1. Prepare M vectors for Digit A and B (Ideal / Average)
    t_a = torch.zeros(1, 10).to(CONFIG["DEVICE"]); t_a[0, DIGIT_A_IDX] = 1.0 
    t_b = torch.zeros(1, 10).to(CONFIG["DEVICE"]); t_b[0, DIGIT_B_IDX] = 1.0 
    
    with torch.no_grad():
        m_a_hat = model.morph_predictor(t_a) # (1, 16)
        m_b_hat = model.morph_predictor(t_b) # (1, 16)
        
    # 2. Collect Z distribution for Digit A and B
    print(f"\n[Analysis] Collecting Z samples for Digit {DIGIT_A_IDX} and {DIGIT_B_IDX}...")
    z_a_list = []
    z_b_list = []
    dataset = MorphMNIST12(train=True, limit_count=2000) 
    
    with torch.no_grad():
        for img, m, t in dataset:
            digit = torch.argmax(t).item()
            if digit == DIGIT_A_IDX:
                img = img.unsqueeze(0).to(CONFIG["DEVICE"])
                _, _, mu, _ = model(img, m.unsqueeze(0).to(CONFIG["DEVICE"]), t.unsqueeze(0).to(CONFIG["DEVICE"]))
                z_a_list.append(mu)
            elif digit == DIGIT_B_IDX:
                img = img.unsqueeze(0).to(CONFIG["DEVICE"])
                _, _, mu, _ = model(img, m.unsqueeze(0).to(CONFIG["DEVICE"]), t.unsqueeze(0).to(CONFIG["DEVICE"]))
                z_b_list.append(mu)
                
    if not z_a_list or not z_b_list:
        print("Error: Not enough samples for Z collection.")
        return

    # 3. Perform Random Sampling Analysis
    N_SAMPLES = 50
    contributions = {name: [] for name in feature_names}
    global_m_contribs = []
    global_z_contribs = []
    
    print(f"\n[Analysis] Running {N_SAMPLES} Monte Carlo simulations...")
    
    for i in range(N_SAMPLES):
        # Sample random style pair (Bootstrap)
        import random
        z_a = random.choice(z_a_list)
        z_b = random.choice(z_b_list)
        
        with torch.no_grad():
            # Base Image: A (M_A, Z_A)
            dec_in_base = torch.cat([m_a_hat, z_a], dim=1)
            x_base = model.dec_conv(model.dec_fc(dec_in_base).view(-1, 64, 7, 7))
            
            # Target Image: B (M_B, Z_B)
            dec_in_target = torch.cat([m_b_hat, z_b], dim=1)
            x_target = model.dec_conv(model.dec_fc(dec_in_target).view(-1, 64, 7, 7))
            
            # Total Difference
            total_diff = torch.norm(x_target - x_base).item() + 1e-9
            
            # --- Global Explainability ---
            
            # 1. Combined M Effect: Change M (A->B) while holding Z fixed (Z_A)
            # Counterfactual: (M_B, Z_A)
            dec_in_m_swap = torch.cat([m_b_hat, z_a], dim=1)
            x_m_swap = model.dec_conv(model.dec_fc(dec_in_m_swap).view(-1, 64, 7, 7))
            diff_m = torch.norm(x_m_swap - x_base).item()
            global_m_contribs.append((diff_m / total_diff) * 100.0)
            
            # 2. Unmeasured Z Effect: Change Z (A->B) while holding M fixed (M_A)
            # Counterfactual: (M_A, Z_B)
            dec_in_z_swap = torch.cat([m_a_hat, z_b], dim=1)
            x_z_swap = model.dec_conv(model.dec_fc(dec_in_z_swap).view(-1, 64, 7, 7))
            diff_z = torch.norm(x_z_swap - x_base).item()
            global_z_contribs.append((diff_z / total_diff) * 100.0)
            
            # --- Individual Feature Analysis ---
            # Measure contribution of each feature k (keeping Z=Z_A fixed)
            for k, fname in enumerate(feature_names):
                m_swapped = m_a_hat.clone()
                m_swapped[0, k] = m_b_hat[0, k] 
                
                dec_in_feat_swap = torch.cat([m_swapped, z_a], dim=1)
                x_feat_swap = model.dec_conv(model.dec_fc(dec_in_feat_swap).view(-1, 64, 7, 7))
                
                diff_k = torch.norm(x_feat_swap - x_base).item()
                pct_k = (diff_k / total_diff) * 100.0
                contributions[fname].append(pct_k)
                
    # 4. Aggregation
    summary_global = []
    summary_global.append(("Measured Features (M)", np.mean(global_m_contribs), np.std(global_m_contribs)))
    summary_global.append(("Unmeasured (Z)", np.mean(global_z_contribs), np.std(global_z_contribs)))
    
    summary_features = []
    for fname in feature_names:
        vals = contributions[fname]
        summary_features.append((fname, np.mean(vals), np.std(vals)))
        
    # Sort features
    summary_features.sort(key=lambda x: x[1], reverse=True)

    print("\n[3] Generating Report...")
    print("\n=== [Table 1] Global Explainability (Overall M vs Z) ===")
    print(f"{'Category':<20} | {'Effect Contrib (%)':<20} | {'Confidence (Std)':<15}")
    print("-" * 65)
    for name, mean, std in summary_global:
        print(f"{name:<20} | {mean:>18.2f}% | {std:>14.2f}")
    print("-" * 65)

    print("\n=== [Table 2] Feature Importance Ranking (Detailed) ===")
    print(f"{'Feature':<20} | {'Effect Contrib (%)':<20} | {'Confidence (Std)':<15}")
    print("-" * 65)
    for name, mean, std in summary_features:
        print(f"{name:<20} | {mean:>18.2f}% | {std:>14.2f}")
    print("-" * 65)
    
    print("\nInterpretation: Table 1 shows how well M explains the total natural change (including style shift).")
    print("                Table 2 shows which morphological feature drives the structural change.")


if __name__ == "__main__":
    model = train_temp_model()
    analyze_mediation(model)
