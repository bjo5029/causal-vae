
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchvision.utils import make_grid

from config import CONFIG, FEATURE_NAMES
from train import train_model
from dataset import extract_refined_features # Import feature extractor

def analyze_counterfactual():
    device = CONFIG["DEVICE"]
    print(f"[Counterfactual Analysis] Device: {device}")
    
    # 1. Load Model
    print("[Counterfactual Analysis] Loading Model...")
    vae = train_model()
    vae.eval()
    
    # 2. Generate Counterfactuals (T -> X)
    print("\n[Counterfactual Analysis] Generating counterfactual images...")
    
    # Fix Z (random noise) to control for all other factors except T
    # We will generate one 'base' Z and apply different T's (0-9) to it.
    # To be robust, let's do this for multiple random Z samples (e.g., 100 samples)
    num_samples = 100
    z_fixed = torch.randn(num_samples, CONFIG["Z_DIM"]).to(device)
    
    # Storage for measured features
    # Shape: (num_samples, 10_digits, M_DIM)
    all_measured_m = np.zeros((num_samples, 10, len(FEATURE_NAMES)))
    
    # To visualize, we'll pick the first sample
    vis_images = [] 
    
    with torch.no_grad():
        for t_idx in range(10):
            # Create T batch for digit 't_idx'
            t_batch = torch.zeros(num_samples, 10).to(device)
            t_batch[:, t_idx] = 1.0
            
            # Mechanism Step: T -> M_hat (Predicted Morphology)
            # Even in T->X->M view, the model still uses T->M path for generation logic if defined so (CausalVAE).
            # But wait, our model is:
            # recon_x, m_hat, mu, logvar = model(x, m, t)
            # During generation (inference), we usually do:
            # m_hat = morph_predictor(t)
            # z = sample_normal
            # x_hat = decoder(z, m_hat)
            
            m_hat = vae.morph_predictor(t_batch)
            
            # Decode: (Z, M_hat) -> X_hat
            # Note: The decoder takes concatenated z and m_hat
            z_m_combined = torch.cat([z_fixed, m_hat], dim=1)
            
            # We need to access decoder parts.
            # model.decode(z, m_hat) doesn't exist explicitly in forward, let's allow accessing internal modules.
            # In models.py:
            # h = self.dec_fc(z_combined)
            # h = h.view(-1, 64, 7, 7)
            # recon_x = self.dec_conv(h)
            
            h = vae.dec_fc(z_m_combined)
            h = h.view(-1, 64, 7, 7)
            x_hat = vae.dec_conv(h)
            x_hat = torch.sigmoid(x_hat) # Ensure 0-1 range
            
            # Store first sample for visualization
            vis_images.append(x_hat[0].cpu())
            
            # 3. Measure Features on Generated Images (X -> M)
            # We need to measure M for EACH generated image
            for i in range(num_samples):
                img_tensor = x_hat[i].cpu() # (1, 28, 28)
                # Extract features using the standardized function
                features = extract_refined_features(img_tensor) # Returns Tensor(16,)
                all_measured_m[i, t_idx, :] = features.numpy()
                
    # 4. Visualization: Counterfactual Grid
    # Combine 10 images of the first sample (0 to 9)
    # Shape of vis_images: List of 10 tensors (1, 28, 28)
    grid = make_grid(vis_images, nrow=10, padding=2)
    plt.figure(figsize=(15, 2))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title("Counterfactual Generation: Same Z, Varying T (0 -> 9)")
    plt.axis('off')
    plt.savefig("counterfactual_grid.png", dpi=150)
    print("[Done] Counterfactual grid saved to 'counterfactual_grid.png'")

    # 5. Sensitivity Analysis based on MEASURED M
    # Now we analyze the variance of 'all_measured_m' across the T axis (axis=1)
    # We avail average variance over the num_samples
    
    # Variance per sample per feature: (num_samples, M_DIM)
    # Actually, let's use Std Dev to be consistent with previous analysis
    std_per_sample = np.std(all_measured_m, axis=1) # (100, 16)
    
    # Average Std Dev across all samples
    avg_sensitivity = np.mean(std_per_sample, axis=0) # (16,)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Measured Sensitivity": avg_sensitivity
    })
    
    df = df.sort_values(by="Measured Sensitivity", ascending=False)
    
    print("\n[Phase 2: Measurement-Centric Feature Importance]")
    print(df)
    
    # Bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Measured Sensitivity", y="Feature", data=df, palette="magma")
    plt.title("Phase 2: Biomarker Discovery (Sensitivity of Measured M on T->X)", fontsize=16)
    plt.xlabel("Standard Deviation of Re-measured Features (Higher = More Sensitive)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    plt.savefig("phase2_biomarker_importance.png", dpi=300)
    print("\n[Done] Phase 2 plot saved to 'phase2_biomarker_importance.png'")
    
    top_3 = df.head(3)["Feature"].tolist()
    print(f"\n>> Phase 2 Top 3 Features: {', '.join(top_3)}")

if __name__ == "__main__":
    analyze_counterfactual()
