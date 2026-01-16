import os
import torch
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../01_baseline_causal_vae')))

import dowhy
from dowhy import CausalModel
from dataset import MorphMNIST12
from config import CONFIG
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("dowhy").setLevel(logging.ERROR)

def analyze_robustness_stress_test():
    print("[1] Loading Dataset...")
    dataset = MorphMNIST12(train=True, limit_count=60000) 
    
    data_list = []
    print("[2] Extracting Features for Digits 1 and 8...")
    for i, (img, m, t) in enumerate(dataset):
        digit = torch.argmax(t).item()
        
        # Filter: Only keep 1 and 8
        if digit not in [1, 8]:
            continue
            
        m_np = m.numpy()
        row = {
            'digit': digit, # Treatment
            'area': m_np[0],
            'perimeter': m_np[1],
            'thickness': m_np[2],
            'major_axis': m_np[3],
            'eccentricity': m_np[4],
            'orientation': m_np[5],
            'solidity': m_np[6],
            'extent': m_np[7],
            'aspect_ratio': m_np[8],
            'euler': m_np[9],
            'h_symmetry': m_np[10],
            'v_symmetry': m_np[11],
        }
        data_list.append(row)
        
    df = pd.DataFrame(data_list)
    
    # Remap digits to binary 0/1 for easier propensity score calculation
    # 1 -> 0, 8 -> 1
    df['treatment'] = df['digit'].apply(lambda x: 1 if x == 8 else 0)
    
    print(f"Data shape (1 vs 8): {df.shape}")
    print(df.head())

    # Add Gaussian Noise to emulate measurement error
    noise_level = 0.5 
    print(f"[Stress Test] Injecting Gaussian Noise (std={noise_level})...")
    features_to_noise = [
        'area', 'perimeter', 'thickness', 'major_axis', 'eccentricity', 
        'orientation', 'solidity', 'extent', 'aspect_ratio', 'euler', 
        'h_symmetry', 'v_symmetry'
    ]
    for f in features_to_noise:
        noise = np.random.normal(0, noise_level, size=len(df))
        df[f] = df[f] + noise

    target_features = features_to_noise
    results = []

    for feature in target_features:
        print(f"\n=============================================")
        print(f"Analyzing Causal Effect (1 vs 8): Treatment -> {feature}")
        print(f"=============================================")
        
        # 1. Define Causal Model
        model = CausalModel(
            data=df,
            treatment='treatment',
            outcome=feature,
            common_causes=[], 
            quiet=True
        )
        
        # 2. Identify Estimand
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # 3. Estimate Effect - Using Linear Regression (IPW requires common causes)
        print("Estimating effect using Linear Regression...")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        
        print(f"Estimated Effect: {estimate.value}")
        
        # 4. Refute Estimate (Stress Tests)
        print("Running Stress Check Refutations...")
        
        # A. Random Common Cause
        refute_rcc = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause"
        )
        
        # B. Placebo Treatment
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        
        # C. Add Unobserved Common Cause
        # Simulates a confounder that affects both treatment and outcome.
        # This checks how sensitive the result is to a potential hidden bias.
        print(" - Checking sensitivity to Unobserved Common Cause...")
        refute_unobserved = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            confounders_effect_on_treatment="binary_flip", 
            confounders_effect_on_outcome="linear",
            effect_strength_on_treatment=0.1,
            effect_strength_on_outcome=0.1
        )
        
        results.append({
            'feature': feature,
            'estimate': estimate.value,
            'rcc_p': refute_rcc.refutation_result['p_value'],
            'placebo_p': refute_placebo.refutation_result['p_value'],
            'unobserved_new_effect': refute_unobserved.new_effect,
            'tipping_point': "Robust (>1.0)" # Default
        })
        
        # D. Find Tipping Point (Sweep)
        print(" - Scanning for Tipping Point (Sensitivity)...")
        strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        original_sign = np.sign(estimate.value)
        
        for s in strengths:
            res_s = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                confounders_effect_on_treatment="binary_flip", 
                confounders_effect_on_outcome="linear",
                effect_strength_on_treatment=s,
                effect_strength_on_outcome=s
            )
            # If sign flips or becomes effectively zero (very small) relative to original
            if np.sign(res_s.new_effect) != original_sign:
                results[-1]['tipping_point'] = f"{s:.1f}"
                break
        
    print("\n\n---------------------------------------------------------------------------------------------------------")
    print("STRESS TEST RESULTS (Noise std=0.5, 1 vs 8, IPW)")
    print("  * Tipping Point: The strength of unobserved confounder required to flip the result.")
    print("  * Unobserved Eff Change: Change in effect with a small confounder (strength=0.1).")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Feature':<15} | {'Effect':<8} | {'RCC (p)':<10} | {'Placebo (p)':<12} | {'Unobs Eff Chg':<15} | {'Tipping Point'}")
    print("-" * 105)
    for res in results:
        # For unobserved confounder, we look at how much the effect changed.
        change = res['estimate'] - res['unobserved_new_effect']
        print(f"{res['feature']:<15} | {res['estimate']:.4f}   | {res['rcc_p']:.4f}     | {res['placebo_p']:.4f}       | {change:.4f}          | {res['tipping_point']}")
    print("-" * 105)



    print("\nInterpretation: If the effect remains positive even at high strength (e.g. 0.5+),")
    print("                it means the causal link is extremely robust.")

if __name__ == "__main__":
    analyze_robustness_stress_test()
