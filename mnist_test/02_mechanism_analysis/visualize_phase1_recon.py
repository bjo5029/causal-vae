
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../v0')))

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from train import train_model # Re-use the training logic from v1
from dataset import MorphMNIST12
from torch.utils.data import DataLoader
from config import CONFIG

def visualize_phase1_recon():
    device = CONFIG["DEVICE"]
    print(f"Device: {device}")

    # 1. Load Phase 1 Model (Mechanism VAE)
    print("Loading Phase 1 Model...")
    # This train_model() inside v1_mechanism/train.py trains CausalMorphVAE12
    # We will just train it for 1 epoch or so if no checkpoint exists, 
    # but ideally we want to see the result of a trained model.
    # Since we are in the 'archive', let's assume train_model returns a trained model.
    vae = train_model()
    vae.eval()

    # 2. Load Data
    full_dataset = MorphMNIST12(train=False, limit_count=16) # Just 16 images
    loader = DataLoader(full_dataset, batch_size=16, shuffle=True)
    
    x, m, t = next(iter(loader))
    x, m, t = x.to(device), m.to(device), t.to(device)

    with torch.no_grad():
        # Phase 1 Forward: (x, m, t) -> recon_x
        # Internally: t -> m_hat, then (m_hat, z) -> recon_x
        recon_x, m_hat, mu, logvar = vae(x, m, t)
        
        # Reshape
        recon_x = recon_x.view(-1, 1, 28, 28)
        
    # 3. Visualize Comparison
    # Row 1: Original
    # Row 2: Reconstruction
    # Row 3: Residual (Absolute Difference)
    
    # Calculate Residual
    residual = torch.abs(x - recon_x)
    
    img_grid_orig = make_grid(x.cpu(), nrow=16, padding=2)
    img_grid_recon = make_grid(recon_x.cpu(), nrow=16, padding=2)
    img_grid_res = make_grid(residual.cpu(), nrow=16, padding=2)
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(3, 1, 1)
    plt.imshow(img_grid_orig.permute(1, 2, 0), cmap='gray')
    plt.title("1. Original X (Ground Truth)")
    plt.axis('off')
    
    plt.subplot(3, 1, 2)
    plt.imshow(img_grid_recon.permute(1, 2, 0), cmap='gray')
    plt.title("2. Reconstructed X_hat (Looks good but blurry)")
    plt.axis('off')
    
    plt.subplot(3, 1, 3)
    plt.imshow(img_grid_res.permute(1, 2, 0), cmap='magma') # Use colormap to highlight difference
    plt.title("3. Residual |X - X_hat| (Contains Edges & Details!)")
    plt.axis('off')
    
    save_path = "phase1_recon_and_residual.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    visualize_phase1_recon()
