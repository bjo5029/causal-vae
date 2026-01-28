
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MorphMNIST12
from models import CausalMorphVAE12
from config import CONFIG

def check_mnist_counterfactual():
    print("="*60)
    print("MNIST Baseline: Checking Counterfactual Generation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model (Assume saved model exists)
    model_path = "/home/jeongeun.baek/workspace/causal-vae/mnist_test/01_baseline_causal_vae/results/model_final.pt" 
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Try to find any model
        model_dir = "/home/jeongeun.baek/workspace/causal-vae/mnist_test/01_baseline_causal_vae/results"
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            pt_files = [f for f in files if f.endswith('.pt')]
            if pt_files:
                model_path = os.path.join(model_dir, pt_files[0])
                print(f"Found alternative model: {model_path}")
            else:
                print("No .pt files found in results dir.")
                return
        else:
            return

    print(f"[1/3] Loading Model from {model_path}...")
    model = CausalMorphVAE12().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load Sample
    print("[2/3] Loading a Sample...")
    # Need to check dataset args. Usually root is needed.
    # Assuming config has it or defaulting to './data'
    dataset = MorphMNIST12(train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    real_x, real_m, real_t = next(iter(loader))
    real_x, real_m, real_t = real_x.to(device), real_m.to(device), real_t.to(device)
    
    # 3. Generate Counterfactuals
    print("[3/3] Performing Intervention...")
    
    # MNIST Features: Thickness, Intensity, Slant (Just guessing names based on typical CausalMNIST)
    # usually m_dim=3. Let's intervene on index 0 and 1.
    # User Request: Euler(9), H-Sym(10), V-Sym(11)
    target_indices = [9, 10, 11]
    
    # MNIST Features based on dataset.py
    feature_names = [
        "Area", "Perimeter", "Thickness", "Major Axis", "Eccentricity",
        "Orientation", "Solidity", "Extent", "Aspect Ratio", "Euler",
        "H-Symmetry", "V-Symmetry"
    ]
    
    with torch.no_grad():
        _, _, mu, logvar = model(real_x, real_m, real_t)
        z = model.reparameterize(mu, logvar)
        
        # Original Reconstruct
        dec_input_orig = torch.cat([real_m, z], dim=1)
        h_orig = model.dec_fc(dec_input_orig)
        h_orig = h_orig.view(-1, 64, 7, 7)
        recon_orig = model.dec_conv(h_orig)
        
        fig, axes = plt.subplots(len(target_indices) + 1, 5, figsize=(15, 10))
        
        # Original Image
        axes[0, 2].imshow(real_x.cpu().squeeze(), cmap='gray')
        axes[0, 2].set_title(f"Original\n(M: {real_m[0].cpu().numpy().round(2)})")
        for ax in axes[0]: ax.axis('off')
        
        for row_idx, responsible_idx in enumerate(target_indices):
            feat_name = feature_names[responsible_idx]
            original_val = real_m[0, responsible_idx].item()
            sweep_vals = np.linspace(original_val - 2.0, original_val + 2.0, 5)
            
            for col_idx, val in enumerate(sweep_vals):
                m_prime = real_m.clone()
                m_prime[0, responsible_idx] = val
                
                dec_input_mod = torch.cat([m_prime, z], dim=1)
                # Manual Decode for CausalMorphVAE12
                # dec_input_mod = torch.cat([m_prime, z], dim=1)
                # h = self.dec_fc(dec_input)
                # h = h.view(-1, 64, 7, 7)
                # recon_x = self.dec_conv(h)
                
                h = model.dec_fc(dec_input_mod)
                h = h.view(-1, 64, 7, 7)
                fake_x = model.dec_conv(h)
                
                ax = axes[row_idx + 1, col_idx]
                ax.imshow(fake_x.cpu().squeeze(), cmap='gray')
                ax.axis('off')
                
                title = f"{feat_name}: {val:.2f}"
                if col_idx == 2: title += "\n(Original)"
                ax.set_title(title, fontsize=9)
    
    save_path = "mnist_counterfactual_check.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved check to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    check_mnist_counterfactual()
