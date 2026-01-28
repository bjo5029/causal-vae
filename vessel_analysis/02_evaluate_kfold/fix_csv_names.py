import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import os
import pandas as pd
import numpy as np
from dataset import VesselDataset
from config import CONFIG

def fix_csv_names():
    print("="*60)
    print("Fixing CSV Condition Names")
    print("="*60)
    
    # 1. Load Dataset to get names
    print("\n[1/3] Loading dataset to retrieve group names...")
    # Mode 'train' is enough to load CSV and get group names
    # We don't need to load images, so it should be fast
    full_dataset = VesselDataset(mode='all')
    group_names = full_dataset.group_names
    print(f"Found {len(group_names)} groups: {group_names}")
    
    # 2. Load Target CSV
    # Using the path observed from the user's active file
    target_csv_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000/all_pairwise_report.csv"
    
    if not os.path.exists(target_csv_path):
        print(f"\n[Warning] File not found at: {target_csv_path}")
        # Try checking relative path based on config just in case
        target_csv_path = os.path.join(CONFIG["RESULT_DIR"], "all_pairwise_report.csv")
        print(f"Trying CONFIG path: {target_csv_path}")
        
    if not os.path.exists(target_csv_path):
        print("\n[Error] Target CSV file not found!")
        return

    print(f"\n[2/3] Loading CSV: {target_csv_path}")
    df = pd.read_csv(target_csv_path)
    
    # 3. specific replacements
    print("\n[3/3] Replacing indices with names...")
    
    # Check if columns are numeric
    if pd.api.types.is_numeric_dtype(df['Treatment_From']):
        print("  'Treatment_From' is numeric. Converting to names...")
        df['Treatment_From'] = df['Treatment_From'].apply(lambda x: group_names[int(x)] if 0 <= int(x) < len(group_names) else x)
    else:
        print("  'Treatment_From' is NOT numeric. Skipping...")

    if pd.api.types.is_numeric_dtype(df['Treatment_To']):
        print("  'Treatment_To' is numeric. Converting to names...")
        df['Treatment_To'] = df['Treatment_To'].apply(lambda x: group_names[int(x)] if 0 <= int(x) < len(group_names) else x)
    else:
        print("  'Treatment_To' is NOT numeric. Skipping...")
        
    # Save back
    df.to_csv(target_csv_path, index=False)
    print(f"\n✓ Updated and saved: {target_csv_path}")
    
    # Verify
    print("\nSample (First row):")
    cols = ['Treatment_From', 'Treatment_To'] + [c for c in df.columns if c not in ['Treatment_From', 'Treatment_To']][:3]
    print(df.iloc[[0]][cols].to_string(index=False))

if __name__ == "__main__":
    fix_csv_names()
