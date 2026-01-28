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

def generate_counterfactual_demo():
    print("="*60)
    print("Generating Counterfactual Images (Intervention on M)")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    save_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/counterfactual_demo"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Model (latest from fold 0)
    model_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_saved_models_kfold_morph10000/fold_0/model_latest.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"[1/3] Loading Model from {model_path}...")
    pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
    model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load One Sample (Control group preferably)
    print("[2/3] Loading a Control Sample...")
    dataset = VesselDataset(mode='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Find a sample that is likely Control (based on weak assumption or just pick random)
    # We will just pick the first one and manipulate it.
    real_x, real_m, real_t = next(iter(loader))
    real_x = real_x.to(device)
    real_m = real_m.to(device)
    real_t = real_t.to(device)
    
    # 3. Abduct Z (Get Style)
    print("[3/3] Performing Counterfactual Generation...")
    with torch.no_grad():
        # Encode to get Z
        # We need to access encoder manually or use forward to get mu/logvar
        # The forward returns: recon_x, m_hat, mu, logvar, m_mu, m_logvar
        _, _, mu, logvar, _, _ = model(real_x, real_m, real_t)
        z = model.reparameterize(mu, logvar) # This is the "Style" of this specific image
        
        # Define Features to Intervene
        feature_names = [
            "Node count", "Extremity Count", "Junction Count", "Edge count",
            "Segment Count", "Branch Count", "Isolated Edge Count",
            "Subnetwork Count", "Total Vessel Length", "Mean Tortuosity",
            "Total Vessel Volume", "Average Vessel Radius"
        ]
        
        # Let's intervene on "Branch Count" (Index 5) and "Total Vessel Length" (Index 8)
        target_features = [5, 8] 
        
        # Create plot
        fig, axes = plt.subplots(len(target_features) + 1, 5, figsize=(20, 12))
        
        # Row 0: Original
        axes[0, 2].imshow(real_x.cpu().squeeze(), cmap='gray')
        axes[0, 2].set_title(f"Original Image\n(Real M: {real_m[0][5]:.2f})")
        for ax in axes[0]: ax.axis('off')
        
        # Generate Manipulations
        for row_idx, responsible_idx in enumerate(target_features):
            feat_name = feature_names[responsible_idx]
            original_val = real_m[0, responsible_idx].item()
            
            # Sweep from -3 sigma to +3 sigma (assuming standardized M)
            # Or just add delta. Let's do a sweep range centered on original.
            sweep_vals = np.linspace(original_val - 5.0, original_val + 5.0, 5)
            
            for col_idx, val in enumerate(sweep_vals):
                # Create Counterfactual M
                m_prime = real_m.clone()
                m_prime[0, responsible_idx] = val
                
                # Decode: P(X | Z, M')
                # We need to call decoder manually.
                # In CausalViTVAE:
                # dec_input_our = torch.cat([m, z], dim=1)
                # z_vit = self.dec_adapter(dec_input_our)
                # recon_x = self.backbone.decode(z_vit)
                
                fake_input = torch.cat([m_prime, z], dim=1)
                z_vit = model.dec_adapter(fake_input)
                fake_x = model.backbone.decode(z_vit)
                
                ax = axes[row_idx + 1, col_idx]
                ax.imshow(fake_x.cpu().squeeze(), cmap='gray')
                ax.axis('off')
                
                diff = val - original_val
                title = f"{feat_name}\n{val:.2f} ({diff:+.2f})"
                if col_idx == 2: title += " (Original)"
                ax.set_title(title, fontsize=10)
                
    save_path = os.path.join(save_dir, "counterfactual_demo.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved demo to: {save_path}")
    print("If you see the vessel morphology changing (e.g. more branches) while the background texture stays consistency,")
    print("it means the model successfully learned 'Vision as Inverse Graphics'.")

if __name__ == "__main__":
    generate_counterfactual_demo()
