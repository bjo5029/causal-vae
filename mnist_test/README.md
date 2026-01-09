# Causal VAE Experiment with MNIST Data

### 1. Background and Objective

* **Ultimate Goal**
  : To analyze the effect of treatment conditions on vessel images and identify which morphological features are most important in distinguishing treatment responses.

* Before using real vessel data, the model structure designed was tested for logical validity using the MNIST dataset.

### 1.1. Data Mapping

| Variable | Role                   | MNIST Experiment (Current)                        | Bio Project (Final Goal)                       |
| -------- | ---------------------- | ------------------------------------------------- | ---------------------------------------------- |
| T        | Treatment              | Digit classes (0–9)                               | Treatment conditions                           |
| M        | Morphological features | Features like thickness, area, etc. (12 features) | Vascular thickness, length, branch count, etc. |
| X        | Image                  | Handwritten digit images                          | Vascular microscopy images                     |
| Z        | Other (Style)          | Handwriting style, noise                          | Imaging noise, background texture              |

### 2. Problem Definition and Solution

### 2.1. Limitations of Existing VAE (Entanglement)

Typical Variational Autoencoders (VAE) compress all image information into a single vector (Z).
In this case, **morphology (M)** and **style (Z)** become entangled.

* **Issue**
  : While images might be reconstructed clearly, the model cannot answer the question:
  *“Which specific features changed when condition T changed?”*

### 2.2. Solution: Element Separation

A strategy was adopted to separate interpretable morphological information (M) and uninterpretable residual style (Z) from the outset of the model structure.

### (1) Encoder: Extract Residual Information

* **Input**: X (image), M (morphology), T (condition)
* **Structure**: `Encoder(X, M, T) → Z`
* **Role**:
  The encoder extracts only the residual style (Z) by subtracting the part of X that can be explained by M and T, isolating the rest as Z.
* **Intent**:
  If M is not provided to the encoder, Z would learn morphological information as well.
  To prevent this, Z is isolated as a pure style unrelated to morphology.

### (2) Decoder: Two-Step Generation Process

The decoder works in two steps.

#### ① Morphology Prediction Stage

* **Input**: T (condition)
* **Output**: M′ (predicted morphology)
* **Meaning**:
  Estimates the “typical form” under the given condition.
  This represents the pure trend of the condition, free from noise.

#### ② Image Generation Stage

* **Input**: [M′, Z] (predicted morphology + isolated style)
* **Output**: X′ (reconstructed image)
* **Meaning**:
  Combines the predicted morphology (M′) with the isolated style (Z) to regenerate the actual image.

### (3) Loss Function

$$
L_{Total} = L_{Recon} + \beta \times L_{KLD} + L_{Morph}
$$

* **Reconstruction Loss**
  Minimizes the difference between input X and the reconstructed X′.
* **KL Divergence**
  Normalizes the distribution of Z.
* **Morphology Regression Loss**
  Minimizes the difference between predicted M′ and actual M
  (learning the relationship between T and M).


### (4) Hyperparameter Strategy: Beta (β)

The hyperparameter β balances image reconstruction quality and Z independence (disentanglement).

#### Trade-off with Beta Values

* **Low β (→ 0)**
  The image is sharp, but Z mixes in morphological information, making causal analysis impossible
  (similar to a regular AE).
* **High β (↑ 10)**
  Z information collapses (posterior collapse), and the image becomes blurry and uniform.

#### Optimal Value Selection

* Based on experimental results, **β = 5.0** was selected.
* Note: β value does not directly affect the morphology prediction (M).

![intervention_grid](./outputs/intervention_grid.png)

As the Beta value increases (from 2 to 5 to 7), the original digit morphology is lost.


### 3. Verification Method and Results Interpretation


### 3.1. Virtual Intervention Experiment

Using the learned model (T → M relationship), a virtual intervention is performed.

* **Method**
  Fix the style (Z) of the original image and force condition (T) to change from 0 to 9.
* **Success Criteria**
  The generated image should change the digit (morphology) but retain the original handwriting style (style).
* **Result Interpretation**
  When changing condition T = A → B, we can observe how the predicted M′ values
  (e.g., thickness, area) change.

<img src="./outputs/intervention_grid_beta5.png" width="400">


### 3.2. Latent Vector (Z) Visualization

* **Method**
  The extracted Z vectors are visualized using t-SNE.
* **Success Criteria**
  Z should not reflect digit information, so the points should be mixed regardless of digit class.
* **Failure Example**
  If the points cluster by digit, it implies that morphological information is still present in Z (entanglement).

<img src="./outputs/z_distribution.png" width="400">


### 3.3. Quantitative Evaluation Using an External Classifier

* **Method**
  The generated images are passed through an external classifier, and the obtained embedding vectors are clustered.

* **Note**
  Since the quality of the images varies significantly based on the β value,
  please consider this clustering result as a reference only.

* Circle: Actual (Real) image distribution

* Triangle: Generated (Fake) image distribution from condition change

<img src="./outputs/external_classifier_distribution.png" width="400">

### 3.4. Morphological Feature Importance

A larger difference (Diff) between predicted (M′) and original morphology (M) indicates key features that are crucial for distinguishing the condition (Top 3).

#### 1 (Source_Digit) → 2 (Target_Digit)

| Feature Name | Prediction | Original | Diff (Change) |
| ------------ | ---------- | -------- | ------------- |
| Perimeter    | 0.86       | 0.40     | 0.4610        |
| Eccentricity | 0.76       | 0.99     | 0.2362        |
| Solidity     | 0.56       | 0.77     | 0.2101        |

#### 1 (Source_Digit) → 8 (Target_Digit)

| Feature Name | Prediction | Original | Diff (Change) |
| ------------ | ---------- | -------- | ------------- |
| Euler Number | 0.26       | 0.75     | 0.4872        |
| Perimeter    | 0.87       | 0.40     | 0.4740        |
| Extent       | 0.50       | 0.34     | 0.1610        |


### 4. Limitations and Conclusion


### 4.1. Limitation: Deterministic Prediction

* **Phenomenon**
  The current structure assumes M = f(T), so when the same condition (e.g., digit 1) is input,
  only the average morphology is output.
* **Constraint**
  Fine morphological variability (variance) within the same digit is excluded from the model.


### 4.2. Conclusion and Expected Impact

Through this MNIST experiment, we validate the causal process where T changes, M changes,
and M combined with style Z generates images.

Once validated, this approach can be directly applied to real vessel data.

Ultimately, this can be used to backtrack and explain which morphological feature (M) changes most significantly when a treatment is administered, helping to elucidate the mechanism of action for the treatment.


### 5. Additional Materials
### 5.1. Selected Morphological Features Information

1. **Area**

   * Formula: `pixels_count / 784.0`
   * Method: Count the total number of white pixels exceeding the threshold (0.2).

2. **Perimeter**

   * Formula: `contour_length / 100.0`
   * Method: Calculate the length of the contour of the shape.

3. **Thickness**

   * Formula: `max(distance_transform) / 5.0`
   * Method: Uses the Distance Transform algorithm.

4. **Major Axis Length**

   * Formula: `ellipse_major_axis / 28.0`
   * Method: The longest diameter of the ellipse that fits the shape.

5. **Eccentricity**

   * Formula: `sqrt(1 - (minor_axis / major_axis)^2)`
   * Method: Measures how much the ellipse is stretched.

6. **Orientation**

   * Formula: `(angle + pi/2) / pi`
   * Method: The angle between the major axis of the shape and the x-axis.

7. **Solidity**

   * Formula: `Area / Convex_Hull_Area`
   * Method: Measures concavity of the shape.

8. **Extent**

   * Formula: `Area / Bounding_Box_Area`
   * Method: Ratio of shape area to bounding box area.

9. **Aspect Ratio**

   * Formula: `(Width / Height) / 3.0`
   * Method: Ratio of bounding box width to height.

10. **Euler Number**

    * Concept: `(Number of blobs) - (Number of holes)`

11. **Horizontal Symmetry**

    * Formula: `1.0 - mean(|Original - Flipped_Left_Right|)`

12. **Vertical Symmetry**

    * Formula: `1.0 - mean(|Original - Flipped_Up_Down|)`

### 5.2. All Intervention Results (10×10)

[All Intervention Results](outputs/all_intervention_results.csv)
