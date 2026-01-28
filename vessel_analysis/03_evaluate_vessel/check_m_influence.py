import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import VesselDataset
from models import CausalViTVAE
from config import CONFIG

def check_m_influence():
    print("="*60)
    print("DIAGNOSTIC: Checking Influence of M on Decoder")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    
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
    
    # 3. Get Z
    with torch.no_grad():
        _, _, mu, logvar, _, _ = model(real_x, real_m, real_t)
        z = model.reparameterize(mu, logvar)
        
        # 4. Generate with Original M
        # Manual Decode
        dec_input_orig = torch.cat([real_m, z], dim=1)
        z_vit_orig = model.dec_adapter(dec_input_orig)
        recon_orig = model.backbone.decode(z_vit_orig)
        
        # 5. Generate with Modified M (Extreme Change)
        m_prime = real_m.clone()
        m_prime[0, :] = m_prime[0, :] + 10.0 # Huge shift
        
        dec_input_mod = torch.cat([m_prime, z], dim=1)
        z_vit_mod = model.dec_adapter(dec_input_mod)
        recon_mod = model.backbone.decode(z_vit_mod)
        
        # 6. Compare
        diff = (recon_orig - recon_mod).abs().mean().item()
        print(f"Original M: {real_m[0, :3].cpu().numpy()}...")
        print(f"Modified M: {m_prime[0, :3].cpu().numpy()}...")
        
        print("\n[Results]")
        print(f"Mean Pixel Difference: {diff:.6f}")
        
        if diff < 1e-4:
            print(">> CRITICAL: The Decoder is IGNORING 'M'. (Posterior Collapse or Disconnect)")
            print("   The images are mathematically identical despite massive M change.")
        else:
            print(">> The Decoder is using 'M', but the visual change might be subtle.")
            
        # Check Weights
        print("\n[Weight Check]")
        print("Decoder Adapter First Layer Weights (connected to M):")
        # dec_adapter[0] is Linear(z_dim + m_dim -> 256)
        # z_dim=64, m_dim=12. So input 0-11 is M? No, usually cat([m, z]) or [z, m]?
        # In forward: dec_input_our = torch.cat([m, z], dim=1)
        # So first 12 weights are M, next 64 are Z.
        
        weight = model.dec_adapter[0].weight.data # (256, 76)
        m_weights = weight[:, :12].abs().mean().item()
        z_weights = weight[:, 12:].abs().mean().item()
        
        print(f"Average Absolute Weight for M inputs: {m_weights:.6f}")
        print(f"Average Absolute Weight for Z inputs: {z_weights:.6f}")
        
        if m_weights < z_weights * 0.1:
            print(">> Warning: Weights for M are much smaller than Z. The model relies mostly on Z.")

if __name__ == "__main__":
    check_m_influence()
