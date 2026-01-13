
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

def analyze_pairwise_1vs7():
    """
    Analyze discriminative features for a specific pair (1 vs 7).
    """
    device = CONFIG["DEVICE"]
    print("\n[Pairwise Analysis] Comparison: Digit 1 vs Digit 7")
    
    # 1. Load Model
    model = train_cvae()
    model.eval()
    
    # 2. Generate samples for 1 and 7 with SAME styles (Z)
    num_samples = 200
    z_fixed = torch.randn(num_samples, CONFIG["Z_DIM"]).to(device)
    
    t_1 = torch.zeros(num_samples, 10).to(device); t_1[:, 1] = 1.0
    t_7 = torch.zeros(num_samples, 10).to(device); t_7[:, 7] = 1.0
    
    print(" -> Generating images...")
    with torch.no_grad():
        x_1 = model.decode(z_fixed, t_1).cpu()
        x_7 = model.decode(z_fixed, t_7).cpu()
        
    # 3. Measure Features
    print(" -> Measuring features...")
    feats_1 = []
    feats_7 = []
    
    for i in range(num_samples):
        f1 = extract_refined_features(x_1[i]).numpy()
        f7 = extract_refined_features(x_7[i]).numpy()
        feats_1.append(f1)
        feats_7.append(f7)
        
    feats_1 = np.array(feats_1) # (200, 16)
    feats_7 = np.array(feats_7) # (200, 16)
    
    # 4. Calculate Discriminative Power (Cohen's d or T-test)
    # Effect Size = |Mean1 - Mean2| / Pooled_Std
    results = []
    for i, name in enumerate(FEATURE_NAMES):
        m1, s1 = np.mean(feats_1[:, i]), np.std(feats_1[:, i])
        m2, s2 = np.mean(feats_7[:, i]), np.std(feats_7[:, i])
        
        pooled_std = np.sqrt((s1**2 + s2**2) / 2) + 1e-9
        effect_size = np.abs(m1 - m2) / pooled_std
        
        results.append({
            "Feature": name,
            "Effect_Size": effect_size,
            "Mean_1": m1,
            "Mean_7": m2
        })
        
    df = pd.DataFrame(results).sort_values(by="Effect_Size", ascending=False)
    print("\n[Top Features for Distinguishing 1 vs 7]")
    print(df.head(5))
    
    # 5. Visualize Top Feature Distribution
    top_feat = df.iloc[0]["Feature"]
    top_idx = FEATURE_NAMES.index(top_feat)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(feats_1[:, top_idx], label='Digit 1', fill=True, color='blue', alpha=0.3)
    sns.kdeplot(feats_7[:, top_idx], label='Digit 7', fill=True, color='red', alpha=0.3)
    plt.title(f"Distribution of '{top_feat}' (Best Separator for 1 vs 7)", fontsize=14)
    plt.xlabel(f"{top_feat} Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("pairwise_1vs7_analysis.png", dpi=150)
    print(f"[Done] Pairwise plot saved to 'pairwise_1vs7_analysis.png'")

if __name__ == "__main__":
    # analyze_cvae() # Skip full analysis for now
    analyze_pairwise_1vs7()
