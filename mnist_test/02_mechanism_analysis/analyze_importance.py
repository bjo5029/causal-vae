
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CONFIG, FEATURE_NAMES
from train import train_model

def analyze_importance():
    device = CONFIG["DEVICE"]
    print(f"[Importance Analysis] Device: {device}")
    
    # 1. Load Model (Re-train for consistency as before)
    print("[Importance Analysis] Loading Model...")
    vae = train_model()
    vae.eval()
    
    # 2. Perform Intervention (Sensitivity Analysis)
    # We want to see how M changes as we change T (0 to 9)
    # Since specific z doesn't matter for T->M predictor (it only takes T),
    # we just pass one-hot vectors for each digit.
    
    print("[Importance Analysis] Calculating feature sensitivity...")
    
    t_input = torch.eye(10).to(device) # (10, 10) for digits 0-9
    
    with torch.no_grad():
        # Shape: (10, M_DIM)
        m_pred = vae.morph_predictor(t_input).cpu().numpy()
        
    # Scale correction? 
    # Current M values are standardized or raw? In dataset.py, we did some manual normalization (e.g. /784).
    # Ideally, to compare "Sensitivity", we should look at the variance relative to the feature's natural range.
    # However, simply looking at 'Variance' or 'Range' of the predicted values across T is a good proxy.
    
    # 3. Calculate Sensitivity Metrics
    # Metric: Standard Deviation of feature values across the 10 digits
    # Higher StdDev => The feature varies a lot depending on the digit => High Importance
    
    sensitivity = np.std(m_pred, axis=0) # (M_DIM,)
    
    # Create DataFrame for better visualization
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Sensitivity (Std)": sensitivity
    })
    
    # Sort by sensitivity
    df = df.sort_values(by="Sensitivity (Std)", ascending=False)
    
    print("\n[Feature Importance Ranking]")
    print(df)
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Sensitivity (Std)", y="Feature", data=df, palette="viridis")
    plt.title("Morphological Feature Importance (Sensitivity to T)", fontsize=16)
    plt.xlabel("Standard Deviation across Digits (Higher = More Important)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    print("\n[Done] Feature importance plot saved to 'feature_importance.png'")
    
    # Text summary for user
    top_3 = df.head(3)["Feature"].tolist()
    print(f"\n>> Top 3 Key Features: {', '.join(top_3)}")

if __name__ == "__main__":
    analyze_importance()
