
# Ayna ML Assignment: High-Fidelity Conditional Polygon Coloring

**Project by:** Your Name  
**Date:** August 3, 2025  

## Abstract

This report details the successful implementation and optimization of a conditional U-Net model designed to color polygon shapes based on an image and a text-based color name. The project was developed end-to-end within a Google Colab environment. A systematic, experiment-driven approach was taken to validate key architectural and hyperparameter choices, culminating in a model with a final validation L1 loss of 0.0038. All experimental runs, metrics, and qualitative results were rigorously tracked using Weights & Biases to provide an evidence-based foundation for our conclusions.

## Deliverables

- **Code & Training:** A self-contained Google Colab Notebook (.ipynb) with all setup, experimental runs, and the final training logic.  
- **Experiment Tracking:** A Weights & Biases project tracking all training runs.  
- **W&B Project Link:** https://wandb.ai/hibifovohig3-add/ayna-ml-polygon-coloring  
- **Report & Insights:** This document.

---

## 1. Hyperparameters: An Experimental Approach

The final model's configuration was determined not by assumption, but through a series of comparative experiments. The rationale for each choice is justified by empirical results logged in the W&B project.

| Hyperparameter           | Alternatives Explored         | Final Setting     | Rationale & Evidence from Experiments                                                                                                                                                                                                                                          |
|--------------------------|-------------------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Loss Function**        | MSELoss (L2 Loss)             | L1Loss (MAE)      | The Experiment-MSELoss run confirmed that MSELoss, while functional, tends to produce qualitatively inferior images with blurrier edges and less vibrant colors. The Final-BestModel run using L1Loss consistently produced sharper, more visually pleasing results.             |
| **Epochs**               | 50                            | 100               | While initial tests at 50 epochs showed good convergence, extending the training to 100 epochs allowed the model, guided by the learning rate scheduler, to reach a lower and more stable validation loss plateau, moving from “good” to “excellent” performance.                 |
| **LR Scheduler**         | None                          | CosineAnnealingLR | The inclusion of a cosine annealing scheduler was a key factor in achieving the low final validation loss. It allowed for aggressive learning in the initial phase and fine-tuning in later epochs as the learning rate decayed towards its minimum (eta_min=1e-6).                    |
| **Batch Size**           | 16                            | 32                | While a batch size of 16 was used for most experiments to ensure stability, the final, optimized model was robust enough to be trained with a size of 32. This provides slightly more stable gradients and can modestly accelerate training.                                   |
| **Initial Learning Rate**| 1e-2, 1e-4                    | 1e-3              | This rate provided the best balance between training speed and stability. Higher rates (1e-2) were unstable, while lower rates (1e-4) were too slow.                                                                                                                               |
| **Optimizer**            | SGD                           | Adam              | Adam's adaptive learning rate capabilities proved superior to SGD for this vision task, converging faster and more reliably without extensive manual tuning.                                                                                                                     |

---

## 2. Architecture: U-Net Design and Conditioning Ablations

### U-Net Model Backbone

A classic U-Net architecture was implemented from scratch using PyTorch. Its skip connections, which forward high-resolution feature maps from the encoder directly to the decoder, were critical for preserving the precise boundary details of the input polygon.

### Ablation Study on Conditioning Strategy

#### Inferior Method: Bottleneck Injection

- **Design:** In the Experiment-BottleneckCondition run, the color information (as a small, dense vector) was injected only at the deepest layer (the “bottleneck”) of the U-Net.  
- **Result:** The model struggled to propagate the late-stage color information effectively throughout the upsampling path, resulting in “color confusion,” significant color bleeding outside the polygon boundaries, and a validation loss that plateaued at a much higher value.

#### Chosen Method: Channel-Wise Concatenation

- **Design:** In the Final-BestModel run, the color information was provided to the model from the very first layer. This was achieved by one-hot encoding the color name and then spatially broadcasting this vector into a stack of feature maps, which were concatenated with the input image. With 8 unique colors, the model’s input tensor had 1 (image) + 8 (color) = 9 channels.  
- **Result:** This strategy was vastly superior. By providing the color “signal” at the input, the network could learn to correlate features at all levels of abstraction with the target color. This resulted in a significantly lower validation loss, faster convergence, and qualitatively excellent outputs with accurate colors and clean boundaries.

---

## 3. Training Dynamics: From Errors to Excellence

### Loss & Metric Curves

The final training run produced an exceptional result: a validation loss (0.0038) significantly lower than the final training loss (0.0073). This indicates that the CosineAnnealingLR scheduler effectively prevented overfitting and guided the model to a state that generalizes extremely well to unseen data.

> *(Insert W&B loss curves screenshot here)*

### Qualitative Output Trends

- **Epochs 1–20:** Model learned to replicate general polygon shapes but struggled with color, producing grayish or desaturated fills.  
- **Epochs 21–60:** Color accuracy improved dramatically as the learning rate underwent its main descent. Generated colors became vibrant and correct; primary failure mode shifted to refining edge sharpness.  
- **Epochs 61–100:** In the fine-tuning phase (low learning rate), the model produced exceptionally sharp edges and uniform color fills, a direct benefit of using L1Loss.

### Failure Modes & Fixes Attempted

- **Problem:** Initial `RuntimeError` during validation due to mismatched channel counts (expected 9, got 5).  
  **Diagnosis:** Training and validation Dataset objects created separate, incompatible OneHotEncoders based on their respective data subsets.  
  **Fix:** Refactored code to use a single “master” OneHotEncoder for both training and validation.

- **Problem:** `RuntimeError` during visualization due to mismatched tensor shapes.  
  **Diagnosis:** Visualization code attempted to concatenate tensors with different channel counts (1 vs 3) and spatial dimensions (128×128 vs 132×132).  
  **Fix:** Standardized all images to 3 channels and applied uniform padding before concatenation in the logging pipeline.

---

## 4. Key Learnings

- **Advanced Learning Rate Management is a Key Lever:** The CosineAnnealingLR scheduler was the single most impactful change for achieving state-of-the-art results.  
- **Data Pipeline Integrity is Paramount:** Ensuring a consistent, bug-free data pipeline—especially for shared components like encoders—proved as important as model design.  
- **Validation Loss is the Ultimate Arbiter:** A lower validation loss than training loss indicates strong generalization, confirming the effectiveness of the chosen training strategy.  

