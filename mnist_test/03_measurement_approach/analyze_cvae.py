
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchvision.utils import make_grid

from config import CONFIG, FEATURE_NAMES
from cvae_train import train_cvae
from dataset import extract_refined_features 

def analyze_cvae():
    device = CONFIG["DEVICE"]
    print(f"[CVAE Analysis] Device: {device}")
    
    # 1. Train/Load CVAE Model (T -> X)
    print("[CVAE Analysis] Training Conditional VAE...")
    model = train_cvae()
    model.eval()
    
    # 2. Generate Counterfactuals (T -> X)
    print("\n[CVAE Analysis] Generating counterfactual images...")
    
    num_samples = 100
    z_fixed = torch.randn(num_samples, CONFIG["Z_DIM"]).to(device)
    
    # Storage for measured features
    all_measured_m = np.zeros((num_samples, 10, len(FEATURE_NAMES)))
    
    vis_images = [] 
    
    with torch.no_grad():
        for t_idx in range(10):
            # Create T batch for digit 't_idx'
            # (num_samples, 10)
            t_batch = torch.zeros(num_samples, 10).to(device)
            t_batch[:, t_idx] = 1.0
            
            # Decode: (Z, T) -> X_hat
            # Our CVAE decode takes (z, t)
            x_hat = model.decode(z_fixed, t_batch) 
            # Output of decode might be logits or sigmoid? 
            # In cvae_models.py, I used sigmoid at the end. So it is 0-1.
            
            # Store first sample for visualization
            vis_images.append(x_hat[0].cpu())
            
            # 3. Measure Features on Generated Images (X -> M)
            for i in range(num_samples):
                img_tensor = x_hat[i].cpu() # (1, 28, 28)
                features = extract_refined_features(img_tensor) # Returns Tensor(16,)
                all_measured_m[i, t_idx, :] = features.numpy()
                
    # 4. Visualization: Counterfactual Grid
    global_grid = make_grid(vis_images, nrow=10, padding=2)
    plt.figure(figsize=(15, 2))
    plt.imshow(global_grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title("CVAE Counterfactuals: Same Z, Varying T (0 -> 9)")
    plt.axis('off')
    plt.savefig("cvae_counterfactual_grid.png", dpi=150)
    print("[Done] Counterfactual grid saved to 'cvae_counterfactual_grid.png'")

    # 5. Sensitivity Analysis based on MEASURED M
    std_per_sample = np.std(all_measured_m, axis=1) # (100, 16)
    avg_sensitivity = np.mean(std_per_sample, axis=0) # (16,)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Measured Sensitivity": avg_sensitivity
    })
    
    df = df.sort_values(by="Measured Sensitivity", ascending=False)
    
    print("\n[CVAE Feature Importance Ranking (Measurement-Centric)]")
    print(df)
    
    # Bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Measured Sensitivity", y="Feature", data=df, palette="viridis")
    plt.title("Biomarker Discovery using CVAE (T->X->M)", fontsize=16)
    plt.xlabel("Standard Deviation of Re-measured Features (Higher = More Sensitive)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    plt.savefig("cvae_biomarker_importance.png", dpi=300)
    print("\n[Done] CVAE plot saved to 'cvae_biomarker_importance.png'")
    
    top_3 = df.head(3)["Feature"].tolist()
    print(f"\n>> CVAE Top 3 Features: {', '.join(top_3)}")

if __name__ == "__main__":
    analyze_cvae()
