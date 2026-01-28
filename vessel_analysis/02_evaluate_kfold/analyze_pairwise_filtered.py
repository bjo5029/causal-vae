import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))


import pandas as pd
import numpy as np
import re
import os

# 1. Load Data CSV to get Treatment Map
data_csv_path = "/home/jeongeun.baek/workspace/causal-vae/data/vessel_analysis_result.csv"
print(f"Loading data from {data_csv_path}")
try:
    df_data = pd.read_csv(data_csv_path)
    group_names = sorted(df_data['group_name'].dropna().unique())
    treatment_map = {i: name for i, name in enumerate(group_names)}
    print(f"Found {len(group_names)} treatments.")
    for i, name in treatment_map.items():
        print(f"{i}: {name}")
except Exception as e:
    print(f"Error loading data csv: {e}")
    exit(1)

# 2. Load Pairwise Report CSV
pairwise_csv_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000/all_pairwise_report.csv"
print(f"Loading pairwise results from {pairwise_csv_path}")
try:
    df_pair = pd.read_csv(pairwise_csv_path)
except Exception as e:
    # Try alternate path if not found (just in case)
    pairwise_csv_path = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/7_results_kfold_morph10000/all_pairwise_report.csv"
    try:
        df_pair = pd.read_csv(pairwise_csv_path)
        print(f"Loaded from alternate path: {pairwise_csv_path}")
    except Exception as e:
        print(f"Error loading pairwise csv: {e}")
        exit(1)

# 3. Helper to parse drug info
def parse_drug_info(name):
    # Rule based parsing
    # "Bispecific 10mg/kg" -> drug="Bispecific", conc=10, unit="mg/kg"
    # "PBS" -> drug="PBS", conc=0
    # "IsotypeControl 10mg/kg"
    
    # Check for concentration pattern
    match = re.search(r'([\d\.]+)\s*mg/kg', name, re.IGNORECASE)
    if match:
        conc = float(match.group(1))
        # Remove concentration from name to get drug type
        drug = re.sub(r'\s*[\d\.]+\s*mg/kg', '', name, flags=re.IGNORECASE).strip()
    else:
        conc = 0.0
        drug = name.strip()
        
    return drug, conc

# 4. Filter and Print Comparisons
print("\n" + "="*80)
print("FILTERED PAIRWISE COMPARISONS (Concentration Matched)")
print("="*80)

# Define features of interest to print concise summary
features_of_interest = ["Extremity Count", "Branch Count", "Subnetwork Count", "Total Vessel Length"]

valid_comparisons = []

for idx, row in df_pair.iterrows():
    t_from_idx = int(row['Treatment_From'])
    t_to_idx = int(row['Treatment_To'])
    
    name_from = treatment_map[t_from_idx]
    name_to = treatment_map[t_to_idx]
    
    drug_from, conc_from = parse_drug_info(name_from)
    drug_to, conc_to = parse_drug_info(name_to)
    
    # Logic for valid comparison:
    # 1. Different Drug, Same Concentration (e.g. Bispecific 10 vs TIE2 10) -> Efficacy
    # 2. Control (PBS) vs High Dose Treatment -> Baseline Check
    # 3. Same Drug, Different Concentration (Low vs High) -> Dose Response (Allow direction Low->High)
    
    is_valid = False
    comp_type = ""
    
    # Case 1: Same Concentration, Different Drug (High dose comparison is most interesting)
    if conc_from > 0 and conc_from == conc_to and drug_from != drug_to:
        is_valid = True
        comp_type = f"Efficacy Comparison ({conc_from} mg/kg)"
    
    # Case 2: PBS vs Treatment (Treatment Effect)
    elif drug_from == "PBS" and conc_to > 0:
        is_valid = True
        comp_type = "Treatment Effect (vs PBS)"
        
    # Case 3: Isotype Control vs Treatment (Specific Effect)
    elif "Isotype" in drug_from and conc_to == conc_from and "Isotype" not in drug_to:
        is_valid = True
        comp_type = "Specific Effect (vs Isotype)"

    # Case 4: Dose Response (Low -> High) within same drug
    elif drug_from == drug_to and conc_from < conc_to:
        is_valid = True
        comp_type = "Dose Response"

    if is_valid:
        print(f"\n[ {comp_type} ]")
        print(f"{name_from} -> {name_to}")
        
        # Print key metrics
        metric_str = []
        for feat in features_of_interest:
            val = row[feat]
            metric_str.append(f"{feat}: {val:.4f}")
        print(", ".join(metric_str))

print("\n" + "="*80)
