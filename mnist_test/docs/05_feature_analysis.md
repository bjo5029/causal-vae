# 5. Causal Relationship and Feature Contribution Analysis

## 5.1. Data Robustness and Causal Validation

**“Did this feature change because the digit changed, or did it just move together by coincidence?”**

To answer this question, **DoWhy** was used.


### Analysis Setup

* **Treatment (Cause):** Digit class (1 vs 7, or 1 vs 8)
* **Outcome (Effect):** Morphological features (Solidity, Eccentricity, etc.)

In other words, the question is:

**“Can we say that this morphological feature changed because the digit changed from 1 to 7?”**

To account for measurement error, **Gaussian noise (std = 0.5)** was added to each feature
(area, solidity, etc.) before running the analysis.


### 1) Random Common Cause (RCC)

* Introduce a completely meaningless random variable as an additional covariate.
* If the estimated causal effect remains unchanged → **stable**.

A high RCC value ≈

“Even after adding a useless variable, the conclusion does not change.”


### 2) Placebo Treatment

* Randomly permute the digit labels (1, 7).
* If the effect still remains → **spurious correlation**.

In a valid causal relationship, the effect should disappear once the true cause
(“being 1”) is destroyed.


### 3) Unobserved Common Cause + Tipping Point

**Assumption**

What if there exists an unobserved hidden variable (U) that affects both the digit and the feature?

We therefore estimate how strong such a hidden confounder must be to overturn the conclusion.

This threshold is called the **tipping point**.


### Interpretation of DoWhy Results (1 → 7)

#### Example: Solidity

| Metric                   | Value   | Interpretation                                                                                                   |
| ------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------- |
| RCC                      | 0.84    | The causal effect on Solidity barely changes even after adding a meaningless random variable                     |
| Placebo                  | 0.86    | When digit labels are shuffled, the effect disappears                                                            |
| Unobserved Effect Change | -0.0161 | Even with a weak hidden confounder (strength 0.1), the estimated effect changes very little                      |
| Tipping Point            | 0.6     | A hidden variable affecting both the digit and Solidity at strength 0.6 is required to invalidate the conclusion |


### Full Feature Results (1 → 7)

| Feature      | Effect  | RCC (p) | Placebo (p) | Unobs Eff Chg | Tipping Point |
| ------------ | ------- | ------- | ----------- | ------------- | ------------- |
| area         | 0.0829  | 0.9600  | 0.7800      | 0.0200        | 0.4           |
| perimeter    | 0.4196  | 0.9600  | 0.9200      | 0.0942        | 0.5           |
| thickness    | 0.0792  | 0.9800  | 1.0000      | 0.0268        | 0.4           |
| major_axis   | 0.0077  | 0.9400  | 0.9200      | 0.0103        | 0.2           |
| eccentricity | -0.1316 | 0.9000  | 0.9400      | -0.0243       | 0.8           |
| orientation  | -0.0017 | 0.9600  | 0.8800      | -0.0016       | 1.0           |
| solidity     | -0.1377 | 0.8400  | 0.8600      | -0.0161       | 0.9           |
| extent       | -0.0040 | 0.9400  | 0.9600      | 0.0071        | >1.0          |
| aspect_ratio | 0.1129  | 0.7200  | 0.9800      | 0.0308        | 0.4           |
| euler        | -0.4694 | 0.9800  | 0.8800      | -0.0917       | 0.6           |
| h_symmetry   | -0.0525 | 1.0000  | 0.9800      | -0.0117       | 1.0           |
| v_symmetry   | -0.0781 | 1.0000  | 0.9800      | -0.0135       | 0.9           |

## 5.2. Counterfactual-Based Generative Contribution Analysis

To verify whether statistically meaningful features are actually used by the model during image generation, we performed a **counterfactual generative contribution analysis**.

**“How much does the image change when only one feature is modified?”**


### Method

1. Compute the mean feature vector for digit 1 (**M₁**) and digit 7 (**M₇**)
2. Fix the same style vector (**Z**)
3. Replace one feature at a time from **M₁** to **M₇**
4. Measure the difference between reconstructed images
5. Repeat **50 times** and compute mean and standard deviation


## Results (1 → 7)

### A. Generative Contribution (Visual Impact)

| Category              | Effect Contrib (%) | Std   |
| --------------------- | ------------------ | ----- |
| Measured Features (M) | 89.72              | 18.22 |
| Unmeasured (Z)        | 64.05              | 18.96 |

#### Feature-wise Contribution

| Feature      | Contrib (Mean) | Std   |
| ------------ | -------------- | ----- |
| Extent       | 39.19          | 9.52  |
| Eccentricity | 32.69          | 10.29 |
| Solidity     | 29.68          | 8.08  |
| Perimeter    | 26.35          | 7.16  |
| AspectRatio  | 25.44          | 6.41  |


### B. Numerical Shift (Raw Feature Difference)

Only the top 6 out of 12 features are listed.
Since the contribution is calculated as
*(Individual Feature Change / Total Feature Change)*,
the sum adds up to **100%**.

| Feature      | Contribution (%) |
| ------------ | ---------------- |
| Perimeter    | 19.30            |
| Solidity     | 12.99            |
| Eccentricity | 12.57            |
| Thickness    | 11.76            |
| AspectRatio  | 10.66            |
| Extent       | 8.48             |

**High efficiency (Extent)**

* Numerical change is small (8.48%, rank 6)
* Visual impact is dominant (39.19%, rank 1)
* A small change is enough to make the image look like a **7**

**Low efficiency (Thickness)**

* Value changes substantially (11.76%)
* Contributes little to visual identity


## Results (1 → 8)

### A. Generative Contribution

| Category              | Effect Contrib (%) | Std   |
| --------------------- | ------------------ | ----- |
| Measured Features (M) | 95.59              | 18.16 |
| Unmeasured (Z)        | 68.53              | 20.02 |

#### Feature-wise Contribution

| Feature      | Effect Contrib (%) | Std   |
| ------------ | ------------------ | ----- |
| Perimeter    | 67.79              | 13.46 |
| Euler        | 39.40              | 8.24  |
| AspectRatio  | 27.44              | 9.06  |
| Solidity     | 18.57              | 4.10  |
| Eccentricity | 15.61              | 3.93  |


### B. Numerical Shift

| Feature     | Contribution (%) |
| ----------- | ---------------- |
| Perimeter   | 26.19            |
| Euler       | 23.82            |
| Thickness   | 11.41            |
| AspectRatio | 7.78             |

**Key factors (Perimeter, Euler)**

* Perimeter increase and hole formation clearly separate **1 and 8**
* Both numerically and visually dominant

**High efficiency (AspectRatio)**

* Small numerical change
* Large visual impact → makes **8** look wider

**Low efficiency (Thickness)**

* Changes numerically
* Contributes little to the visual identity of **8**

