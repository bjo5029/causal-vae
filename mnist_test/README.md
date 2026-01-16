# MNIST Causal VAE Experiments

## Overview
This directory contains experiments validating a Causal VAE model structure using MNIST data before applying it to vessel images. The goal is to identify which morphological features are most important for distinguishing between conditions (drug treatments).

## Directory Structure

### `01_baseline_causal_vae/`
**Objective**: Establish baseline Causal VAE model (T→M→X)

**Key Components**:
- Encoder: Extracts residual style (Z) after removing M and T information
- Decoder: Predicts M' from T, then generates X from M'+Z

### `02_mechanism_analysis/`
**Objective**: Validate M's role via prediction-based analysis (Phase 1: T→M)

**Key Experiments**:
1. **Reconstruction Test**: M'+Z reconstruction quality
2. **Residual Analysis**: Classifier on residuals achieved 96% accuracy → M is incomplete
3. **Z-Permutation Test**: Swapping Z preserves digit identity → M captures core information

**Key Findings**:
- M is **sufficient for classification** (recognizable)
- M is **insufficient for full explanation** (residuals contain information)

### `03_measurement_approach/`
**Objective**: Develop measurement-centric approach (Phase 2: T→X→M)

**Method**:
1. Train CVAE to generate images directly from T (bypassing M)
2. Generate counterfactual images by varying T
3. Re-measure morphological features M from generated images
4. Calculate sensitivity: which features change most when T changes

**Key Findings**:
- **Junctions** identified as most sensitive feature
- Captures features that prediction-based approach might miss

### `04_phase_comparison/`
**Objective**: Compare Phase 1 (Prediction) vs Phase 2 (Measurement)

**Experiments**:
1. **Overall Comparison**: Feature importance rankings across all digits
2. **Pairwise Analysis**: 
   - 1 vs 7 (Geometrical difference)
   - 3 vs 8 (Topological difference)

**Key Insights (Numerical Mean vs Structural Mode)**:
- **Phase 1 (Prediction)**: Represents **"Numerical Mean"** (preserving statistical rates). Good for population trends.
- **Phase 2 (Measurement)**: Represents **"Structural Mode"** (preserving physical constraints). Good for identifying essential shapes and global coherence (hu moments, solidity).

### `05_feature_analysis/`
**Objective**: Causal validation and quantitative feature contribution analysis

**Key Experiments**:
1. **Causal Validation (DoWhy)**: Robustness checks (RCC, Placebo) to verify T→M causality.
2. **Generative Contribution**: Quantifying visual impact of M vs Z using counterfactuals.

**Key Concepts**:
- **Visual Efficiency**: Discrepancy between numerical shift and visual impact (e.g., Extent has high visual impact despite small numerical change).
- **Contribution Decomposition**: Decomposing total image change into Explainable (M) vs Unexplained (Z) parts.

## Documentation

The detailed reports for each experiment phase are organized in `docs/`:

- [01. Baseline Causal VAE](./docs/01_baseline_causal_vae.md)
- [02. Mechanism Analysis](./docs/02_mechanism_analysis.md)
- [03. Measurement Approach](./docs/03_measurement_approach.md)
- [04. Phase Comparison](./docs/04_phase_comparison.md)
- [05. Feature Analysis](./docs/05_feature_analysis.md)
