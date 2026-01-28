import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def analyze_discriminative_features():
    """
    각 조건을 구분하는데 가장 중요한 형태학적 피처 분석
    """
    print("="*60)
    print("Discriminative Feature Importance Analysis")
    print("="*60)
    
    # Load predictions
    csv_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/7_results_kfold_morph10000/predictions_by_treatment.csv"
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col != 'Treatment']
    
    print(f"\nLoaded {len(df)} treatments with {len(feature_cols)} features")
    
    # Prepare data for classification
    X = df[feature_cols].values
    y = df['Treatment'].values
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    
    # Method 1: Random Forest Feature Importance
    print("\n[1/3] Random Forest Feature Importance...")
    rf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    rf_importance = rf.feature_importances_
    
    # Training accuracy (no CV since we have only 1 sample per class)
    train_accuracy = rf.score(X, y)
    print(f"Random Forest Training Accuracy: {train_accuracy:.3f}")
    
    # Method 2: Variance across treatments (분산 기반)
    print("\n[2/3] Variance-based Importance...")
    variance_importance = X.var(axis=0)
    variance_importance = variance_importance / variance_importance.sum()  # Normalize
    
    # Method 3: ANOVA F-statistic (각 피처가 treatment를 얼마나 잘 구분하는지)
    print("\n[3/3] ANOVA F-statistic...")
    from sklearn.feature_selection import f_classif
    f_scores, p_values = f_classif(X, y)
    f_importance = f_scores / f_scores.sum()  # Normalize
    
    # Combine results
    results_df = pd.DataFrame({
        'Feature': feature_cols,
        'RF_Importance': rf_importance,
        'Variance_Importance': variance_importance,
        'ANOVA_F_Importance': f_importance,
        'ANOVA_p_value': p_values
    })
    
    # Calculate average importance (ensemble)
    results_df['Average_Importance'] = (
        results_df['RF_Importance'] + 
        results_df['Variance_Importance'] + 
        results_df['ANOVA_F_Importance']
    ) / 3
    
    # Sort by average importance
    results_df = results_df.sort_values('Average_Importance', ascending=False)
    
    # Save results
    output_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/7_results_kfold_morph10000"
    csv_output = os.path.join(output_dir, "discriminative_feature_importance.csv")
    results_df.to_csv(csv_output, index=False)
    print(f"\n✓ Saved: {csv_output}")
    
    # Print top features
    print("\n" + "="*60)
    print("Top 5 Most Discriminative Features")
    print("="*60)
    print(results_df[['Feature', 'Average_Importance', 'RF_Importance', 'ANOVA_p_value']].head(5).to_string(index=False))
    
    # Visualization 1: Bar plot comparing all methods
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = [
        ('RF_Importance', 'Random Forest Importance', axes[0, 0]),
        ('Variance_Importance', 'Variance-based Importance', axes[0, 1]),
        ('ANOVA_F_Importance', 'ANOVA F-statistic Importance', axes[1, 0]),
        ('Average_Importance', 'Average Importance (Ensemble)', axes[1, 1])
    ]
    
    for col, title, ax in methods:
        sorted_df = results_df.sort_values(col, ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df)))
        
        ax.barh(range(len(sorted_df)), sorted_df[col], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['Feature'], fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.suptitle('Discriminative Feature Importance Analysis\n(Which features best distinguish between treatments?)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_output = os.path.join(output_dir, "discriminative_feature_importance.png")
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_output}")
    plt.close()
    
    # Visualization 2: Heatmap of feature values across treatments
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort treatments and features for better visualization
    top_features = results_df.head(8)['Feature'].tolist()
    heatmap_data = df[top_features].T
    
    sns.heatmap(heatmap_data, 
                xticklabels=df['Treatment'].values,
                yticklabels=top_features,
                cmap='RdBu_r', center=0,
                annot=False, fmt='.2f',
                cbar_kws={'label': 'Predicted Value'},
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Treatment Conditions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top Discriminative Features', fontsize=12, fontweight='bold')
    ax.set_title('Top 8 Discriminative Features Across Treatments', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=90, ha='right', fontsize=7)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    heatmap_output = os.path.join(output_dir, "discriminative_features_heatmap.png")
    plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {heatmap_output}")
    plt.close()
    
    # Visualization 3: Feature importance comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['RF_Importance'], width, label='Random Forest', alpha=0.8)
    ax.bar(x, results_df['Variance_Importance'], width, label='Variance-based', alpha=0.8)
    ax.bar(x + width, results_df['ANOVA_F_Importance'], width, label='ANOVA F-statistic', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Feature'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Feature Importance Methods', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    comparison_output = os.path.join(output_dir, "importance_methods_comparison.png")
    plt.savefig(comparison_output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {comparison_output}")
    plt.close()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    return results_df

if __name__ == "__main__":
    results = analyze_discriminative_features()
