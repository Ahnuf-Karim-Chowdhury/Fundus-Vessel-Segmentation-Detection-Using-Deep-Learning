# Fundus Vessel Segmentation Using Deep Learning

## Project Overview

This project presents an advanced deep learning system for the precise segmentation of retinal blood vessels from fundus images. It integrates innovative architectural designs with specialized training methodologies tailored for medical image analysis.

### Fundus Image Dataset for Vessel Segmentation
High-resolution manually annotated fundus images for vessel segmentation.

DataSet : https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation

### Visualization of Processed Data and Predictions

This visualization phase systematically renders each image-mask pair alongside its composite overlay to support qualitative evaluation. The approach emphasizes clarity, consistency, and ease of interpretation through the following strategies:

* **Sequential Visualization:** Iterates through stored lists of processed originals, masks, and overlays to generate a consistent and repeatable display for each image set.
* **Structured Subplot Layout:** Each figure comprises three horizontally aligned subplots—original image, binary mask, and overlay—to facilitate immediate visual comparison across different representations.
* **Focus on Image Content:** Axes are removed from all subplots to eliminate visual clutter and maintain focus on the image data itself.
* **Consistent Styling:** Each subplot is individually titled to clearly indicate its respective content, aiding in interpretability during result analysis.
* **Layout Optimization:** `tight_layout()` is employed to automatically adjust subplot parameters, ensuring appropriate spacing and preventing label or image overlap, thereby enhancing overall readability and aesthetic quality of the figure.

<p align="center">
    <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Vessel-Segmentation-Detection-Using-Deep-Learning/blob/main/Images/01.png?raw=true" style="max-width: 100%; height: auto;" />
</p>

## Pre-processed Data

The pre-processed datasets are organized as follows:

* **Train Pre-processed Dataset Location:** `DataSet\processed_train\Ground truth` and `DataSet\processed_train\Original`
* **Test Pre-processed Dataset Location:** `DataSet\processed_test\Ground truth` and `DataSet\processed_test\Original`

### Data Augmentation and Preprocessing Pipeline

1.  **Purpose of Data Augmentation and Preprocessing:**
    To enhance the **robustness** and **generalization capabilities** of machine learning models, particularly in medical imaging applications, input images and their corresponding masks undergo a series of transformations for augmentation and standardization.

2.  **Transformations Applied to Original Images:**

    * **Resizing:** Each original image is uniformly resized to $224 \times 224$ pixels. This standardization of dimensions is critical for batch processing in deep learning models.
        $$I_{\text{resized}} = \text{Resize}(I, (224, 224))$$

    * **Random Horizontal Flip:** Images are stochastically flipped horizontally with a probability of $p = 0.5$, introducing variability in orientation.
        $$I' = \begin{cases} \text{FlipHorizontal}(I), & \text{with probability } 0.5 \\ I, & \text{otherwise} \end{cases}$$

    * **Random Rotation:** Images are rotated by a random angle $\theta \in [-15^\circ, 15^\circ]$, enabling the model to learn rotational invariance.
        $$I' = \text{Rotate}(I, \theta), \quad \theta \sim \mathcal{U}(-15^\circ, 15^\circ)$$

    * **Color Jittering:** Minor random perturbations are applied to brightness, contrast, saturation, and hue, each within $\pm 10\%$ of their original values. This simulates diverse lighting and imaging conditions.
        $$I' = \text{ColorJitter}(I; \Delta_{\text{brightness}}=0.1, \Delta_{\text{contrast}}=0.1, \Delta_{\text{saturation}}=0.1, \Delta_{\text{hue}}=0.1)$$

    * **Random Affine Translation:** Images are randomly translated by up to $10\%$ of their width and height, mimicking potential shifts during image acquisition.
        $$I' = \text{AffineTranslate}(I, t_x, t_y), \quad t_x, t_y \sim \mathcal{U}(-0.1, 0.1)$$

    * **Random Resized Crop:** A random crop, scaled between $80\%$ and $100\%$ of the original image size, is extracted and then resized back to $224 \times 224$ pixels. This augments spatial variance.
        $$I' = \text{RandomResizedCrop}(I, \text{scale}=(0.8, 1.0), \text{size}=(224, 224))$$

3.  **Transformations Applied to Masks:**

    * Masks exclusively undergo **geometric transformations** that are spatially congruent with those applied to the original images. This ensures precise pixel-level alignment, which is critical for segmentation tasks.
    * These transformations include resizing, horizontal flips, rotations, and affine translations. However, **color jittering and cropping are excluded**, as masks represent binary or categorical pixel labels rather than continuous color information.
    * This methodology guarantees that for every pixel coordinate $(x,y)$, the mask value $M(x,y)$ precisely corresponds to the transformed original image.

4.  **Synchronization of Random Transformations:**

    * To ensure the spatial transformations applied to an image and its corresponding mask are **identical**, a random seed is generated and set prior to each transformation step.
    * This practice guarantees deterministic and consistent augmentation, thereby preserving the integrity of image-mask pairs essential for supervised learning.

5.  **Loading and Saving:**

    * Images and masks are loaded in RGB format to retain all color channels.
    * Subsequent to the application of their respective transformations, the processed images and masks are saved to their designated output directories.
    * The pipeline iterates through all images in the dataset, with a mechanism to skip any missing files to ensure operational robustness.

6.  **Outcome:**

    * This comprehensive pipeline produces a dataset composed of uniformly sized, augmented image-mask pairs, prepared for model training.
    * The augmentation process improves model **generalization** by exposing it to a broader spectrum of plausible data variations, a crucial factor in medical image analysis where sample sizes are frequently limited.

<p align="center">
    <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Image-Dataset-for-Vessel-Segmentation/blob/main/DataSet/processed_train/Original/100_A.png?raw=true" style="width: 49%; height: auto;" />
    <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Image-Dataset-for-Vessel-Segmentation/blob/main/DataSet/processed_train/Ground%20truth/100_A.png?raw=true" style="width: 49%; height: auto;" />
</p>

## Technical Approach

### Model Architecture Definition & Loading (for Inference)

#### `ResidualConvBlock` Class

This class defines a **residual convolutional block**, which includes:

* Two convolutional layers with kernel size $3 \times 3$
* Batch normalization and ReLU activation after each convolution
* Optional dropout for regularization
* A residual connection that either uses identity or a $1 \times 1$ convolution if dimensions differ:
    $$
    \text{Output} = F(x) + x
    $$

#### `UNetPlus` Class

This class defines the **full segmentation model architecture** used during training and inference. Key components include:

* **Encoder path**: Sequential downsampling through convolution + pooling blocks
* **Bottleneck**: Deepest feature layer (high-level semantics)
* **Decoder path**: Upsampling via transpose convolutions + feature fusion from encoder
* **Skip connections**: Concatenated features from encoder to decoder levels
* **Output layer**: Final $1 \times 1$ convolution for binary segmentation output

### U-Net++ Segmentation Model Training Pipeline

#### Objective

This pipeline trains a deep learning model to perform **binary semantic segmentation** — where each pixel in an image is classified as either **foreground** (e.g., object of interest) or **background**). This is a common task in domains such as:

* Medical imaging (e.g., tumor segmentation)
* Autonomous vehicles (e.g., road/lane segmentation)
* Remote sensing (e.g., land use classification)

#### Why These Components Are Used

| Component           | Why It's Used                                                                                             |
| :------------------ | :-------------------------------------------------------------------------------------------------------- |
| `UNetPlus`          | A deeper, more robust version of U-Net++ that enables precise segmentation via skip connections and residual learning. |
| `ResidualConvBlock` | Allows easier training of deep models by promoting stable gradient flow and avoiding vanishing gradients. |
| `FocalDiceLoss`     | Combines class imbalance handling (Focal Loss) with region-overlap optimization (Dice Loss) — critical in segmentation tasks. |
| `SegmentationDataset` | Custom logic for paired image-mask preprocessing with synchronized augmentation.                          |
| `calculate_metrics` | Measures segmentation performance using domain-relevant metrics like Dice Score and IoU.                  |
| `GradScaler` (Mixed Precision) | Speeds up training and reduces memory usage while preserving numerical accuracy.                          |
| `ReduceLROnPlateau` | Adapts learning rate when progress stalls, helping avoid local minima.                                    |

#### Dataset Handling: `SegmentationDataset`

* Loads image-mask pairs.
* Applies the same geometric/color augmentations to both.
* Normalizes RGB images using ImageNet statistics:
    $$
    x_{\text{norm}} = \frac{x - \mu}{\sigma}
    $$
* Converts grayscale masks to binary using:
    $$
    \text{mask}(x, y) = 
    \begin{cases}
    1, & \text{if pixel} > 0.5 \\
    0, & \text{otherwise}
    \end{cases}
    $$

#### Model Architecture: `UNetPlus` + `ResidualConvBlock`

##### Residual Block

Each block applies:
$$
\text{Output} = F(x) + x
$$
Where:
* $F(x)$ is a nonlinear transformation (conv → BN → ReLU)
* The residual helps maintain identity information and improves convergence.

##### U-Net++ Core Structure

* **Encoder**: Extracts hierarchical features via residual blocks.
* **Decoder**: Reconstructs full-resolution output, fusing encoder features:
    $$
    d_i = \text{DecoderBlock}( \text{Up}(d_{i+1}) \oplus e_i )
    $$
    Where $\oplus$ is channel-wise concatenation.

* **Output**: A 1-channel mask with logits (pre-sigmoid values).

#### Loss Function: `FocalDiceLoss`

##### Why Use This?

Segmentation tasks often suffer from:
* **Class imbalance** (background pixels vastly outnumber foreground)
* **Low overlap** between prediction and ground truth at early stages

Combining **Focal Loss** and **Dice Loss** addresses both.

##### Focal Loss (for class imbalance)

$$
FL(p_t) = -\alpha(1 - p_t)^\gamma \log(p_t)
$$

* Focuses training on **hard examples** (where $p_t$ is low)
* Down-weights **easy examples** that would dominate standard loss

##### Dice Loss (for overlap)

$$
\text{Dice Loss} = 1 - \frac{2|P \cap T| + \epsilon}{|P| + |T| + \epsilon}
$$

* Directly optimizes for segmentation **overlap** between predicted mask $P$ and ground truth $T$
* Robust to class imbalance

##### Combined Loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \mathcal{L}_{\text{dice}}
$$

#### Evaluation Metrics: `calculate_metrics`

Used to monitor both **model correctness** and **segmentation quality** per epoch.

##### Accuracy

Measures correct predictions overall:
$$
\text{Accuracy} = \frac{\text{Number of correct pixels}}{\text{Total pixels}}
$$

> Useful to confirm general learning progress but less informative for imbalanced classes.

##### Dice Score (F1 for segmentation)

$$
\text{Dice} = \frac{2TP}{2TP + FP + FN}
$$

* Sensitive to both false positives and false negatives
* Directly reflects quality of segmentation boundaries

##### IoU (Jaccard Index)

$$
\text{IoU} = \frac{TP}{TP + FP + FN}
$$

* Measures **intersection over union** of predicted and actual mask regions
* More conservative than Dice

#### Training Configuration

* **Optimizer**: `AdamW` — stable updates + weight decay
* **Learning Rate Scheduler**: `ReduceLROnPlateau` reduces LR if Dice score stagnates:
    $$
    \text{LR}_{\text{new}} = \text{LR}_{\text{current}} \times \text{factor}
    $$
* **Gradient Scaling**: Enables efficient mixed precision training on GPU using:
    * `torch.cuda.amp.GradScaler`
    * `autocast`

### Training and Validation Metrics Visualization

Four separate line plots show how the model improves or plateaus:

1.  **Loss**:
    $$
    \text{Loss} = \mathcal{L}_{\text{FocalDice}}
    $$
    Lower values indicate better model fit.
    <p align="center">
        <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Vessel-Segmentation-Detection-Using-Deep-Learning/blob/main/Images/model_01_Loss.png?raw=true" style="max-width: 100%; height: auto;" />
    </p>

2.  **Accuracy**:
    $$
    \text{Accuracy} = \frac{\text{Correct Pixels}}{\text{Total Pixels}} \times 100
    $$
    <p align="center">
        <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Vessel-Segmentation-Detection-Using-Deep-Learning/blob/main/Images/model_01_accuracy.png?raw=true" style="max-width: 100%; height: auto;" />
    </p>

3.  **Dice Score**:
    $$
    \text{Dice} = \frac{2TP}{2TP + FP + FN} \times 100
    $$
    Indicates pixel-wise overlap between predicted and ground truth masks.
    <p align="center">
        <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Vessel-Segmentation-Detection-Using-Deep-Learning/blob/main/Images/model_01_dice.png?raw=true" style="max-width: 100%; height: auto;" />
    </p>

4.  **IoU (Intersection over Union)**:
    $$
    \text{IoU} = \frac{TP}{TP + FP + FN} \times 100
    $$
    Used only for test data to evaluate generalization.
    <p align="center">
        <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Vessel-Segmentation-Detection-Using-Deep-Learning/blob/main/Images/model_01_IoU.png?raw=true" style="max-width: 100%; height: auto;" />
    </p>

#### Final Metric Summary

The final epoch's values are presented in a structured table, summarizing the model's best training and validation performance for each metric.

Final Training Metrics:

| Metric     | Train  | Test   |
| :--------- | :----- | :----- |
| Loss       | 0.8381 | 0.9379 |
| Accuracy   | 84.96% | 89.06% |
| Dice Score | 20.27% | 7.52%  |
| IoU        | N/A    | 3.94%  |

### Confusion Matrix Visualization for Pixel-wise Classification

#### Summary

This section details the generation and visualization of the confusion matrix for binary pixel-wise classification between ground truth (`y_true`) and predicted (`y_pred`) labels.

#### Confusion Matrix Calculation

The confusion matrix $\mathbf{C}$ summarizes classification outcomes:

$$
\mathbf{C} = 
\begin{bmatrix}
TP & FP \\
FN & TN
\end{bmatrix}
$$

where each element represents:

* $TP$: True Positives (correctly predicted vessel pixels)
* $FP$: False Positives (incorrectly predicted vessel pixels)
* $FN$: False Negatives (missed vessel pixels)
* $TN$: True Negatives (correctly predicted background pixels)

Computed via:

$$
\mathbf{C} = \text{confusion\_matrix}(y_{\text{true}}, y_{\text{pred}})
$$

#### Visualization Details

* The matrix is displayed as a heatmap using a blue color scale to indicate frequency.
* Axes are labeled with classes: **Background** and **Vessel**.
* Each cell is annotated with the raw count $c_{ij}$ for clarity.
* Color contrast adjusts dynamically based on a threshold:
    $$
    \text{thresh} = \frac{\max(\mathbf{C})}{2}
    $$
    ensuring readability of text annotations.

<p align="center">
    <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Vessel-Segmentation-Detection-Using-Deep-Learning/blob/main/Images/confusion%20matrix.png?raw=true" style="max-width: 100%; height: auto;" />
</p>

### Evaluation and Visualization of Model Predictions on Test Images

#### Preprocessing and Model Input

Images are transformed via a pipeline including:

* Resizing to $224 \times 224$,
* Conversion to tensor,
* Normalization using ImageNet mean $\mu = [0.485, 0.456, 0.406]$ and std $\sigma = [0.229, 0.224, 0.225]$,

to maintain consistency with model training:

$$
x_{\text{norm}} = \frac{x - \mu}{\sigma}
$$

#### Evaluation Metrics

For each test image, binary masks are predicted and compared to ground truth masks.

* **Accuracy**:
    $$
    \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total pixels}} = \frac{\sum (pred = gt)}{\text{total pixels}}
    $$

* **Dice Coefficient** (measures overlap between prediction and ground truth):
    $$
    \text{Dice} = \frac{2 \times TP}{2 \times TP + FP + FN} = \frac{2 \sum (pred \cap gt)}{\sum pred + \sum gt}
    $$
    Where:
    * $TP =$ True Positives: pixels correctly predicted as vessel,
    * $FP =$ False Positives: pixels incorrectly predicted as vessel,
    * $FN =$ False Negatives: missed vessel pixels.

Small epsilon $1 \times 10^{-8}$ added to denominator for numerical stability.

#### Visualization Components

For each image, three plots are generated:

1.  **Input Image:** Original RGB image resized to $224 \times 224$.
2.  **Predicted Mask:** Binary mask overlaid with performance labels (`GOOD`, `MEDIUM`, `POOR`) based on thresholds for Dice (>0.20) and accuracy (>0.70).
3.  **Error Map:** Color-coded map showing:
    * Green for TP,
    * Red for FP,
    * Blue for FN,
    * Black elsewhere.

This highlights spatial error patterns, guiding model diagnostics.

#### Summary

This step verifies how well the model generalizes to unseen images by combining quantitative metrics with intuitive visual feedback, helping identify strengths and weaknesses in segmentation quality.
<p align="center">
    <img src="https://github.com/Ahnuf-Karim-Chowdhury/Fundus-Vessel-Segmentation-Detection-Using-Deep-Learning/blob/main/Images/model_01_prediction.png?raw=true" style="max-width: 100%; height: auto;" />
</p>
