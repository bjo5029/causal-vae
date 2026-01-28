import sys
import os
import torch
import numpy as np
import pandas as pd
import itertools

# Add core path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

from dataset import VesselDataset
from config import CONFIG

def calculate_stats_and_snr():
    print("="*60)
    print("1. Calculating Dataset Statistics (Real Mean/Std)")
    print("="*60)
    
    # 1. Load Dataset
    dataset = VesselDataset(mode='all')
    
    # Access the scaler fitted in __init__
    scaler = dataset.scaler
    feature_names = dataset.feature_cols
    
    # Dictionary to store scale (std) for each feature
    feature_scales = {}
    
    stats_data = []
    print("\n[Feature Statistics]")
    print(f"{'Feature':<30} | {'Mean':<10} | {'Std (Scale)':<10}")
    print("-" * 60)
    
    for i, name in enumerate(feature_names):
        mean_val = scaler.mean_[i]
        std_val = scaler.scale_[i]
        feature_scales[name] = std_val
        
        print(f"{name:<30} | {mean_val:.2f}     | {std_val:.2f}")
        
        stats_data.append({
            'Feature': name,
            'Mean': mean_val,
            'Std': std_val
        })
        
    # Save stats
    base_result_dir = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000"
    os.makedirs(base_result_dir, exist_ok=True)
    
    df_stats = pd.DataFrame(stats_data)
    stats_path = os.path.join(base_result_dir, "feature_stats.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"\n✓ Saved feature stats to: {stats_path}")

    print("\n" + "="*60)
    print("2. Calculating Pairwise SNR & Real Differences")
    print("="*60)
    
    # Load predictions and uncertainty
    pred_path = os.path.join(base_result_dir, "predictions_by_treatment.csv")
    unc_path = os.path.join(base_result_dir, "uncertainty_by_treatment.csv")
    
    # If not exist, we warn user
    if not os.path.exists(pred_path) or not os.path.exists(unc_path):
        print(f"[Error] Required files not found:\n {pred_path}\n {unc_path}")
        print("Please run 'extract_uncertainty_by_treatment.py' and ensure 'predictions_by_treatment.csv' exists.")
        return

    df_pred = pd.read_csv(pred_path)
    df_unc = pd.read_csv(unc_path)
    
    treatments = df_pred['Treatment'].unique()
    features = [c for c in df_pred.columns if c != 'Treatment']
    
    # Lists to store report rows
    snr_summary_rows = []    # One row per pair (Max SNR feature)
    detailed_report_rows = [] # One row per pair per feature
    
    for t1, t2 in itertools.combinations(treatments, 2):
        # Get vectors
        mu1 = df_pred[df_pred['Treatment'] == t1][features].values[0] # (12,)
        mu2 = df_pred[df_pred['Treatment'] == t2][features].values[0] # (12,)
        
        sigma1 = df_unc[df_unc['Treatment'] == t1][features].values[0]
        sigma2 = df_unc[df_unc['Treatment'] == t2][features].values[0]
        
        # Calculate for all features
        for i, feat in enumerate(features):
            # Change Calculation: Target (B) - Reference (A)
            # This represents the change FROM t1 TO t2
            diff_z = mu2[i] - mu1[i] 
            
            val1 = mu1[i]
            val2 = mu2[i]
            unc1 = sigma1[i]
            unc2 = sigma2[i]
            
            # SNR (Direction agnostic for magnitude)
            noise = np.sqrt(unc1**2 + unc2**2)
            snr = np.abs(diff_z) / (noise + 1e-8)
            
            # Map short names (from CSV) to long names (from Dataset)
            name_mapping = {
                "Subnetwork Count": "Subnetwork Count(edge count >= 3)",
                "Total Vessel Length": "Total Vessel Length (μm)",
                "Total Vessel Volume": "Total Vessel Volume (μm^3)",
                "Average Vessel Radius": "Average Vessel Radius (μm)"
            }
            long_name = name_mapping.get(feat, feat)
            
            # Real Difference
            real_std = feature_scales[long_name]
            diff_real = diff_z * real_std
            
            detailed_report_rows.append({
                'From_Treatment': t1,
                'To_Treatment': t2,
                'Feature': long_name, 
                'Diff_Z': diff_z,
                'Real_Std': real_std,
                'Diff_Real': diff_real,
                'SNR': snr
            })
            
        # Summary (Max SNR logic from before)
        diff_vec = np.abs(mu2 - mu1)
        noise_vec = np.sqrt(sigma1**2 + sigma2**2)
        snr_vec = diff_vec / (noise_vec + 1e-8)
        
        max_idx = np.argmax(snr_vec)
        
        snr_summary_rows.append({
            'From_Treatment': t1,
            'To_Treatment': t2,
            'Max_SNR': snr_vec[max_idx],
            'Max_SNR_Feature': features[max_idx],
            'Mean_SNR': np.mean(snr_vec)
        })
        
    # Save Summary SNR (pairwise_snr.csv)
    df_snr = pd.DataFrame(snr_summary_rows)
    df_snr = df_snr.sort_values('Max_SNR', ascending=False)
    snr_path = os.path.join(base_result_dir, "pairwise_snr.csv")
    df_snr.to_csv(snr_path, index=False)
    print(f"✓ Saved Pairwise SNR summary to: {snr_path}")
    
    # Save Detailed Report (all_pairwise_report.csv)
    df_report = pd.DataFrame(detailed_report_rows)
    # Sort by From -> To -> Feature
    df_report = df_report.sort_values(['From_Treatment', 'To_Treatment', 'Feature'])
    
    report_path = os.path.join(base_result_dir, "all_pairwise_report.csv")
    df_report.to_csv(report_path, index=False)
    print(f"✓ Saved Detailed Pairwise Report to: {report_path}")
    
    print("\nTop 5 Rows (Sorted by Treatment):")
    cols = ['From_Treatment', 'To_Treatment', 'Feature', 'Diff_Real', 'Real_Std', 'SNR']
    print(df_report[cols].head().to_string())
    
    # 3. Save Matrix Format (User Request)
    print("\n" + "="*60)
    print("3. Generating Matrix Format")
    print("="*60)
    
    # Pivot
    df_matrix = df_report.pivot_table(
        index=['From_Treatment', 'To_Treatment'], 
        columns='Feature', 
        values='Diff_Real'
    )
    
    existing_cols = [c for c in feature_names if c in df_matrix.columns]
    df_matrix = df_matrix[existing_cols]
    
    matrix_path = os.path.join(base_result_dir, "pairwise_diff_real_matrix.csv")
    print(f"✓ Saved Matrix Report to: {matrix_path}")
    print("(Rows: From/To Pairs, Columns: Features, Values: Real Difference = To - From)")
    
    # 4. Generate Enhanced Matrix (Interleaved: Diff | SNR Info)
    print("\n" + "="*60)
    print("4. Generating Enhanced Matrix (Diff + SNR/Rank)")
    print("="*60)
    
    # Calculate Rank per Pair (Row) based on SNR
    df_report['Rank'] = df_report.groupby(['From_Treatment', 'To_Treatment'])['SNR'].rank(method='min', ascending=False)
    
    # Construct "Info" string: "[Rank #] SNR: X.XX"
    df_report['SNR_Info'] = df_report.apply(
        lambda x: f"[{int(x['Rank'])}] SNR: {x['SNR']:.2f}", axis=1
    )
    
    # Pivot for Diff
    df_diff = df_report.pivot_table(
        index=['From_Treatment', 'To_Treatment'], 
        columns='Feature', 
        values='Diff_Real'
    )
    
    # Pivot for Info
    df_info = df_report.pivot_table(
        index=['From_Treatment', 'To_Treatment'], 
        columns='Feature', 
        values='SNR_Info',
        aggfunc='first' # String
    )
    
    # Interleave columns
    # Get feature names that exist in the pivot
    active_features = [f for f in feature_names if f in df_diff.columns]
    
    final_df = pd.DataFrame(index=df_diff.index)
    
    for feat in active_features:
        final_df[f"{feat}"] = df_diff[feat]
        final_df[f"{feat} (Info)"] = df_info[feat]
        
    enhanced_path = os.path.join(base_result_dir, "pairwise_report_formatted.csv")
    final_df.to_csv(enhanced_path)
    print(f"✓ Saved Enhanced Matrix to: {enhanced_path}")
    print("  Structure: [Feature Diff] | [Feature Info (Rank, SNR)] ...")

if __name__ == "__main__":
    calculate_stats_and_snr()
