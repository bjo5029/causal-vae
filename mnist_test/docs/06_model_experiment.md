# 6. Model Experiment

## 1. Experiment A: Bayesian Uncertainty

### 1.1 Purpose

The existing model is a Deterministic structure that always predicts the same morphological feature M for a given input condition T. 
However, real data has Stochasticity (probabilistic variability) even under the same conditions. 
The purpose of this experiment is to quantify this and measure the model's Confidence.


### 1.2 Methodology

- **Model Architecture Change**
    
    The MorphPredictor was modified to output both the mean μ and the log variance logσ2 for the features M given an input T.
    
- **Loss Function Change**
    
    The existing MSE loss function was replaced with the **Gaussian Negative Log-Likelihood (NLL)**.
    
    $\mathcal{L}{morph}=\frac{1}{2}\sum{i=1}^{D}\left(\log\sigma_i^2+\frac{(m_i-\mu_i)^2}{\sigma_i^2}\right)$

This loss function

- Penalizes large prediction errors by increasing $\sigma$
- Prevents overestimation of $\sigma$ through the log term

by maintaining a balance between uncertainty estimation and prediction accuracy.


### 1.3 Results

The model learns the uncertainty $\sigma$ for each digit $T$–feature $M$ combination.

The main observations are as follows.

- **High Confidence (Low Uncertainty)**
    - **Horizontal Symmetry (H_Symmetry)**
        
        $\sigma\approx0.03$ (very low) for most numbers. This suggests symmetry is a consistent and inherent geometric trait of digits.


        
- **Low Confidence (High Uncertainty)**
    - **Euler Number (Euler)**
        
        $\sigma\approx0.96$ 
        In handwritten 6s, some have closed loops (Euler=0) while others do not (Euler=1). The high uncertainty correctly reflects this actual data distribution.
    - **숫자 2의 Aspect Ratio**
        
        $\sigma\approx0.58$
        
        High uncertainty was observed because the width-to-height ratio varies greatly depending on individual handwriting styles.
        

## 2. Experiment B: Quantifying the Information Incompleteness of Morphological Features

### 2.1 Purpose

This experiment quantifies the Degree of Incompleteness that occurs when the 12 defined morphological features M pass information from the digit T to the image X.

### 2.2 Methodology

- **Model Architecture**
    1. **Baseline (Model A)**
        
        $M\rightarrow X$
        
    2. **Augmented (Model B)**
        
        $(M,T)\rightarrow X$
        
- If M is a complete mediating variable, knowing T additionally should not improve reconstruction performance.
    
    $MSE_{Baseline}\approx MSE_{Augmented}$
    
### 2.3 Results

The Image Reconstruction Errors (MSE) are as follows:

$MSE_{Baseline}=0.0398$ (Features only)

$MSE_{Augmented}=0.0309$ (Features + Label)

$\Delta MSE\approx0.009$ (Approx. 25% improvement)

### 2.4 Conclusion

- **Explained Region (~78%)**: Since Model A succeeded in significant reconstruction, M explains about 78% of the features captured by the deep learning model.
- **Unexplained Region (~22%)**: The 22.4% error reduction achieved by adding T represents the influence of "Hidden Features" that are not currently defined by M.
