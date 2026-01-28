
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add 00_core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

from config import CONFIG
from models import CausalViTVAE, CausalVesselVAE
from dataset import VesselDataset


def load_ensemble_models():
    """Loads model_latest.pt from all available folds."""
    # base_dir = CONFIG["SAVE_DIR"] # e.g. .../7_saved_models_kfold_morph10000
    base_dir = '/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/8_saved_models_beta0.5'
    models = []
    
    device = CONFIG["DEVICE"]
    
    # Iterate folds
    for fold in range(5):
        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        # Explicitly requested "latest"
        path_latest = os.path.join(fold_dir, "model_latest.pt")
        
        if not os.path.exists(path_latest):
            print(f"Fold {fold}: Model not found at {path_latest}")
            continue
            
        print(f"[Ensemble] Loading Fold {fold} from {path_latest}...")
        try:
            # Try ViT first
            pretrained_path = "/home/jeongeun.baek/workspace/causal-vae/saved_models/vit_vae_epoch_470.pth"
            model = CausalViTVAE(pretrained_path=pretrained_path).to(device)
            model.load_state_dict(torch.load(path_latest, map_location=device), strict=False)
        except Exception as e:
            print(f"Fold {fold}: ViT Load failed ({e}), trying CNN...")
            model = CausalVesselVAE().to(device)
            model.load_state_dict(torch.load(path_latest, map_location=device))
        
        model.eval()
        models.append(model)
        
    return models

def check_mechanism_z_perm():
    print("="*60)
    print("Experiment 2: Z-Permutation Test (Ensemble Average)")
    print("Using 'model_latest.pt' from all folds.")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    
    # 1. Load Models (Ensemble)
    models = load_ensemble_models()
    if not models:
        print("Error: No trained models found.")
        return
    print(f"Loaded {len(models)} models for ensemble.")

    # 2. Load Validation Samples
    print("[2/4] Loading Samples...")
    dataset = VesselDataset(mode='val')
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    
    real_x, real_m, real_t = next(iter(loader))
    real_x = real_x.to(device)
    real_m = real_m.to(device)
    real_t = real_t.to(device)
    
    n_samples = real_x.size(0)
    
    # 3. Prepare Grid Generation Function
    def generate_grid_for_scale(scale):
        print(f"\n[Scale={scale}] Generating Ensemble Permutations...")
        fig, axes = plt.subplots(n_samples + 1, n_samples + 1, figsize=(18, 18))
        
        # Clear axes
        for ax in axes.flatten():
            ax.axis('off')
            
        # Headers
        axes[0, 0].text(0.5, 0.5, f"Scale Z * {scale}\nRow: M | Col: Z", ha='center', va='center', fontsize=12, fontweight='bold')
        
        for j in range(n_samples):
            ax = axes[0, j+1]
            ax.imshow(real_x[j].cpu().squeeze(), cmap='gray')
            ax.set_title(f"Z Source {j}", fontsize=9)
            
        for i in range(n_samples):
            ax = axes[i+1, 0]
            ax.imshow(real_x[i].cpu().squeeze(), cmap='gray')
            ax.set_title(f"M Source {i}", fontsize=9)
            
        with torch.no_grad():
            for i in range(n_samples): # Row (M from i)
                m_source = real_m[i].unsqueeze(0) # (1, M_DIM)
                
                for j in range(n_samples): # Col (Z from j)
                    # Ensemble Inference
                    ensemble_preds = []
                    
                    for model in models:
                        x_j = real_x[j].unsqueeze(0)
                        m_j = real_m[j].unsqueeze(0)
                        t_j = real_t[j].unsqueeze(0)
                        
                        _, _, mu, logvar, _, _ = model(x_j, m_j, t_j)
                        z_source = mu # Mean
                        
                        # Apply Scaling
                        z_source = z_source * scale
                        
                        dec_input = torch.cat([m_source, z_source], dim=1)
                        
                        if hasattr(model, 'dec_adapter'): # ViT
                            z_vit = model.dec_adapter(dec_input)
                            x_rec = model.backbone.decode(z_vit)
                        else: # CNN
                            h = model.dec_fc(dec_input).view(-1, 512, 6, 10)
                            x_rec = model.dec_conv(h)
                        
                        ensemble_preds.append(x_rec)
                    
                    # Average
                    x_rec_avg = torch.stack(ensemble_preds).mean(dim=0)
                    
                    # Plot
                    ax = axes[i+1, j+1]
                    ax.imshow(x_rec_avg.cpu().squeeze(), cmap='gray')
                    
                    if i == j:
                        ax.set_title("Recon", color='blue', fontsize=8)
                        for spine in ax.spines.values():
                            spine.set_edgecolor('blue')
                            spine.set_linewidth(2)

        plt.tight_layout()
        save_dir = os.path.join(CONFIG["RESULT_DIR"], "mechanism_check")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"z_perm_scale_{scale}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✓ Saved ({scale}): {save_path}")

    # 4. Run for multiple scales
    scales = [1.0, 0.5, 0.3, 0.1, 0.0]
    for s in scales:
        generate_grid_for_scale(s)


if __name__ == "__main__":
    check_mechanism_z_perm()
