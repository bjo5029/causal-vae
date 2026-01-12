
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../v0')))

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from train import train_model
from dataset import MorphMNIST12
from torch.utils.data import DataLoader
from config import CONFIG
import numpy as np

def visualize_z_permute():
    device = CONFIG["DEVICE"]
    print(f"Device: {device}")

    # 1. Load Phase 1 Model
    print("Loading Phase 1 Model...")
    vae = train_model()
    vae.eval()

    # 2. Get a batch of data
    full_dataset = MorphMNIST12(train=False, limit_count=32) 
    loader = DataLoader(full_dataset, batch_size=32, shuffle=True) # Get 32 images
    
    x, m, t = next(iter(loader))
    x, m, t = x.to(device), m.to(device), t.to(device)

    with torch.no_grad():
        # Get Z and M_hat from original images
        recon_x, m_hat, mu, logvar = vae(x, m, t)
        
        # Z comes from reparameterize(mu, logvar) which happened inside vae()
        # To get Z explicitly:
        z = vae.reparameterize(mu, logvar)
        
        # Experiment: Permute Z
        # We want to see if the digit identity stays with M or moves with Z.
        # Shift Z by 1 (Z_0 gets Z_1, Z_1 gets Z_2, ...)
        z_permuted = torch.roll(z, shifts=1, dims=0)
        
        # Decode using Original M_hat but Permuted Z
        # If Decoder(M_hat, Z_perm) looks like Original T -> M is dominant (Good)
        # If Decoder(M_hat, Z_perm) looks like Shifted T -> Z is dominant (Leakage)
        dec_input = torch.cat([m_hat, z_permuted], dim=1)
        
        # Manually run decoder parts (since vae.decode isn't separated in this version)
        h = vae.dec_fc(dec_input)
        h = h.view(-1, 64, 7, 7)
        x_swapped = vae.dec_conv(h)
        
    # 3. Visualization
    # Row 1: Original X
    # Row 2: Reconstructed X (M + Z)
    # Row 3: Swapped Recon (M + Z_shifted)
    
    # We create a grid where we match:
    # Col i: 
    #   Top: X[i] (Digit A)
    #   Mid: Recon[i] (Digit A)
    #   Bot: Swap[i] (Uses M from A, Z from B)
    # Ideally, Bot should still look like Digit A.
    
    img_grid_orig = make_grid(x.cpu(), nrow=16, padding=2)
    img_grid_recon = make_grid(recon_x.view(-1, 1, 28, 28).cpu(), nrow=16, padding=2)
    img_grid_swap = make_grid(x_swapped.view(-1, 1, 28, 28).cpu(), nrow=16, padding=2)
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(3, 1, 1)
    plt.imshow(img_grid_orig.permute(1, 2, 0), cmap='gray')
    plt.title("1. Original T (Source of M)")
    plt.axis('off')
    
    plt.subplot(3, 1, 2)
    plt.imshow(img_grid_recon.permute(1, 2, 0), cmap='gray')
    plt.title("2. Normal Recon (M + Z)")
    plt.axis('off')
    
    plt.subplot(3, 1, 3)
    plt.imshow(img_grid_swap.permute(1, 2, 0), cmap='gray')
    plt.title("3. Swapped Z Recon (M_original + Z_neighbor) -> Does it show Original T or Neighbor T?")
    plt.axis('off')
    
    save_path = "phase1_z_permute_test.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    visualize_z_permute()
