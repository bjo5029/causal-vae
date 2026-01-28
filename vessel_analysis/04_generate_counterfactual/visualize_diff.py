import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG

def visualize_difference_map():
    print("="*60)
    print("Visualizing Difference Map (M Intervention)")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    save_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/diff_check"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Model
    model_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_saved_models_kfold_morph10000/fold_0/model_latest.pt"
    pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
    model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load Sample
    dataset = VesselDataset(mode='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    real_x, real_m, real_t = next(iter(loader))
    real_x, real_m, real_t = real_x.to(device), real_m.to(device), real_t.to(device)
    
    # 3. Generate Counterfactual
    with torch.no_grad():
        _, _, mu, logvar, _, _ = model(real_x, real_m, real_t)
        z = model.reparameterize(mu, logvar)
        
        # Original
        dec_input_orig = torch.cat([real_m, z], dim=1)
        recon_orig = model.backbone.decode(model.dec_adapter(dec_input_orig))
        
        # Modified (Extreme)
        m_prime = real_m.clone()
        m_prime[0, :] = m_prime[0, :] + 5.0 # +5 sigma shift
        
        dec_input_mod = torch.cat([m_prime, z], dim=1)
        recon_mod = model.backbone.decode(model.dec_adapter(dec_input_mod))
        
        # 4. Compute Difference
        diff = (recon_orig - recon_mod).abs()
        
        # 5. Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # A. Original
        axes[0].imshow(recon_orig.cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Recon (Original M)")
        axes[0].axis('off')
        
        # B. Modified
        axes[1].imshow(recon_mod.cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Recon (M + 5.0)")
        axes[1].axis('off')
        
        # C. Difference
        im = axes[2].imshow(diff.cpu().squeeze(), cmap='hot')
        axes[2].set_title(f"Difference Map\n(Mean Diff: {diff.mean().item():.4f})")
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2])
        
        save_path = os.path.join(save_dir, "diff_map.png")
        plt.savefig(save_path)
        print(f"✓ Saved difference map to: {save_path}")
        
        # Check Value Range
        print(f"Original Range: {recon_orig.min().item():.4f} ~ {recon_orig.max().item():.4f}")
        print(f"Modified Range: {recon_mod.min().item():.4f} ~ {recon_mod.max().item():.4f}")

if __name__ == "__main__":
    visualize_difference_map()
