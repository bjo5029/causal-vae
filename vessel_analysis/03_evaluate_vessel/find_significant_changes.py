import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import pandas as pd
import numpy as np
import os
import math

# Paths
uncertainty_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000/uncertainty_by_treatment.csv"
pairwise_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000/all_pairwise_report.csv"

def find_significant_changes():
    print("="*60)
    print("Finding Significant Changes (High Diff AND High Confidence)")
    print("="*60)
    
    # 1. Load Data
    if not os.path.exists(uncertainty_path):
        print("Error: Uncertainty file not found")
        return
    if not os.path.exists(pairwise_path):
        print("Error: Pairwise diff file not found")
        return
        
    df_unc = pd.read_csv(uncertainty_path).set_index("Treatment")
    df_pair = pd.read_csv(pairwise_path)
    
    print(f"Loaded Uncertainty: {df_unc.shape}")
    print(f"Loaded Pairwise Diffs: {df_pair.shape}")
    
    # Get treatment map (index to name)
    # The pairwise report uses indices (0, 1, ...), we need to map them to names to look up uncertainty
    # We can infer the mapping because df_unc is sorted alphabetically usually?
    # Let's verify mapping from pairwise report if possible, or assume alphabetical order of unique names
    
    # In kfold_pairwise.py, unique_groups = np.unique(t_labels)
    # In extract_uncertainty.py, group_names = sorted(full_dataset.df['group_name'].dropna().unique())
    # Both are sorted unique names. So index i -> i-th name in sorted list.
    
    treatment_names = sorted(df_unc.index.tolist())
    
    feature_names = [
        "Node count", "Extremity Count", "Junction Count", "Edge count",
        "Segment Count", "Branch Count", "Isolated Edge Count",
        "Subnetwork Count", "Total Vessel Length", "Mean Tortuosity",
        "Total Vessel Volume", "Average Vessel Radius"
    ]
    
    results = []
    
    # 2. Iterate Pairwise Diffs
    print("\nCalculating Signal-to-Noise Ratio (SNR)...")
    
    for idx, row in df_pair.iterrows():
        t_from_idx = int(row['Treatment_From'])
        t_to_idx = int(row['Treatment_To'])
        
        t_from_name = treatment_names[t_from_idx]
        t_to_name = treatment_names[t_to_idx]
        
        # Identity comparison (same treatment) -> skip
        if t_from_idx == t_to_idx:
            continue
            
        for feat in feature_names:
            diff_val = row[feat]
            
            # Get uncertainties
            sig_from = df_unc.loc[t_from_name, feat]
            sig_to = df_unc.loc[t_to_name, feat]
            
            # Combined Uncertainty (Error Propagation)
            combined_sig = math.sqrt(sig_from**2 + sig_to**2)
            
            # Avoid zero division
            if combined_sig < 1e-6:
                combined_sig = 1e-6
            
            # "High Confidence" metric: We want LOW combined_sig
            # "Large Diff" metric: We want HIGH abs(diff_val)
            # Combined metric (SNR): abs(diff) / combined_sig
            snr = abs(diff_val) / combined_sig
            
            results.append({
                'From': t_from_name,
                'To': t_to_name,
                'Feature': feat,
                'Diff': diff_val,
                'Uncertainty': combined_sig,
                'SNR': snr
            })
            
    # 3. Sort and Report
    df_res = pd.DataFrame(results)
    
    # Sort by SNR descending
    df_top = df_res.sort_values('SNR', ascending=False)
    
    print("\n[Top 10 Most Significant Changes (High Diff / Low Uncertainty)]")
    # Format for better display
    print(df_top[['From', 'To', 'Feature', 'Diff', 'Uncertainty', 'SNR']].head(10).to_string(index=False))
    
    # Also find examples where Abs(Diff) is high but Uncertainty is ALSO high (Noisy Large Diffs)
    # Just to contrast
    # df_res['Product'] = df_res['Diff'].abs() * df_res['Uncertainty'] 
    
    save_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000/significant_changes.csv"
    df_top.to_csv(save_path, index=False)
    print(f"\n✓ Saved detailed report to: {save_path}")

if __name__ == "__main__":
    find_significant_changes()
