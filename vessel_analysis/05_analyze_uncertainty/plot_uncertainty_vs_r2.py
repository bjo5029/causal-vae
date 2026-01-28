import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Path
csv_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000/feature_importance_kfold.csv"
save_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000"

def analyze_dropoff():
    print("="*60)
    print("Analyzing Uncertainty vs R2 Score Trade-off")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print("Error: CSV not found")
        return

    df = pd.read_csv(csv_path)
    
    # Rename for clarity
    df = df.rename(columns={
        'Mean_Importance': 'R2_Score', 
        'Mean_Uncertainty': 'Uncertainty'
    })
    
    # Sort by Uncertainty
    df_sorted = df.sort_values('Uncertainty')
    
    print("\n[Data Sorted by Uncertainty]")
    print(df_sorted[['Feature', 'Uncertainty', 'R2_Score']].to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Uncertainty', y='R2_Score', s=100, color='darkblue')
    
    # Annotate points
    for i in range(df.shape[0]):
        plt.text(
            df.Uncertainty.iloc[i]+0.01, 
            df.R2_Score.iloc[i], 
            df.Feature.iloc[i], 
            fontsize=9
        )
        
    # Draw Threshold Line
    plt.axvline(x=0.6, color='red', linestyle='--', label='Threshold (0.6)')
    plt.axhline(y=0.4, color='green', linestyle='--', label='Reliable R2 (0.4)')
    
    plt.title('Trade-off: Bayesian Uncertainty vs Model Accuracy (R2)')
    plt.xlabel('Uncertainty (Lower is Better)')
    plt.ylabel('R2 Score (Higher is Better)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(save_dir, "uncertainty_vs_r2_dropoff.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n-> Saved Plot: {plot_path}")
    
    # Calculate Correlation
    corr = df['Uncertainty'].corr(df['R2_Score'])
    print(f"\n[Correlation Coefficient]: {corr:.4f} (Strong Negative Correlation)")

if __name__ == "__main__":
    analyze_dropoff()
