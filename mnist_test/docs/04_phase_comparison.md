# 04. Phase Comparison
## 1. Background & Objective

This experiment compares two methodologies for analyzing the causal relationship between conditions ($T$) and morphological features ($M$), analyzing the differences in perspective each methodology offers.

*   **Phase 1 (Prediction-based)**: $T \rightarrow M$ Prediction Model (Numerical Approach)
*   **Phase 2 (Measurement-based)**: $T \rightarrow X \rightarrow M$ Generation & Measurement Model (Structural Approach)


## 2. Methodological Differences: Numerical Mean vs Structural Mean

The two methodologies yield results with different characteristics due to differences in their modeling approaches.

| Feature | Phase 1: Prediction ($T \rightarrow M$) | Phase 2: Measurement ($T \rightarrow X \rightarrow M$) |
| :--- | :--- | :--- |
| **Core Mechanism** | **Numerical Averaging** | **Structural Averaging (Mode)** |
| **Working Principle** | Outputs the **statistical expectation** of the data distribution. <br> Example: "50% of 7s have a crossbar" $\rightarrow$ Predicts $0.5$ | Due to the nature of VAEs, generates the **most common (modal/average) shape**. <br> Example: "Draw a 7" $\rightarrow$ Generates the safest 7 (without crossbar) |
| **Resulting Characteristics** | **Preserves Rate**: Reflects the statistical characteristics of the population. <br> **Abstract Values**: Can produce physically impossible intermediate values (e.g., 0.5 crossbars). | **Preserves Constraints**: Generates only topologically valid images. <br> **Smoothed Details**: Unstable features (e.g., crossbars) disappear, while robust features (e.g., Solidity) are emphasized. |


## 3. Experiment Results

### 3.1. Global Feature Importance

Each methodology prioritizes different types of features.

*   **Prediction (Phase 1)**: Sensitive to simple size (Area, Perimeter) or statistical probabilities (Junctions).
*   **Measurement (Phase 2)**:
    *   **Junctions (1.0)**: A key factor representing structural complexity.
    *   **Hu Moments (0.96)**: Most sensitively adjusted factor to balance the overall image during generation.
    *   **Solidity**: A physical constraint determining the convexity of the digit.

### 3.2. Pairwise Case Study: 1 vs 7 (Geometry & Detail)

**"Difference in Interpreting Junctions"**

| Feature | Phase 1 (Prediction) | Phase 2 (Measurement) | Analysis of Difference |
| :--- | :--- | :--- | :--- |
| **Junctions** | **High Score** | **Zero (0)** | **Numerical Mean vs Structural Mode** <br> Phase 1 reflected the **statistical proportion** that "some 7s have crossbars," assigning a high score. <br> Phase 2 generated the most common (uncrossed) form during **structural averaging**, resulting in zero measured junctions. |
| **Solidity** | Low Score | **Top 1** | **Physical Constraint** <br> Phase 2 strongly reflected **'bending'**, a structure essential for distinguishing 7 from 1. |

### 3.3. Pairwise Case Study: 3 vs 8 (Topology & Shape)

**"Weight Difference between Topology and Shape"**

| Feature | Phase 1 (Prediction) | Phase 2 (Measurement) | Analysis of Difference |
| :--- | :--- | :--- | :--- |
| **Junctions** | **1st** | 3rd | Phase 1 focused on the simple numerical difference in the number of junctions. |
| **Hu Moments** | Low Rank | **1st (Hu5, Hu7)** | **Global Shape Coherence** <br> Phase 2 captured the reconstruction of **curvature and center of mass (Hu Moments)** involved in changing a 3 to an 8, beyond simple hole punching (topology). |


## 4. Conclusion

This experiment confirms that the two methodologies explain feature importance from different perspectives.

1.  **Phase 1 (Prediction)**: Useful for identifying the **statistical trends (rates)** of the entire dataset. It summarizes population characteristics through 'average values' that may not appear in individual instances.
2.  **Phase 2 (Measurement)**: Useful for identifying the **structural characteristics (constraints)** of real generates images. It focuses on the **'intrinsic shape (Solidity, Global Balance)'** determines the identity of the digit rather than noisy details.
3.  **Conclusion**: Therefore, it is necessary to select the appropriate methodology according to the purpose of analysis or to interpret both results complementarily.
