
---

# 🎨 Conditional Polygon Coloring with a U-Net in PyTorch

This repository contains the complete project for the **Ayna ML Assignment**.
The objective was to build and train a deep learning model that takes **an image of a polygon** and a **color name** as input, and produces an image of that polygon filled with the specified color.

The final model acts as a **general-purpose shape colorizer**, capable of handling **arbitrary polygon shapes**, not just those seen during training. This was achieved using:

* A custom **U-Net architecture**
* A **channel-wise conditioning** strategy
* A **robust, synchronized data augmentation** pipeline

<p align="center">
  <b>Example:</b><br>
  <i>The model takes a polygon shape and a color name to generate a colored output.</i>
  
</p>



---

## 🚀 Features

* **🧠 U-Net Architecture**
  Implemented from scratch in PyTorch for high-fidelity image-to-image translation.

* **🎨 Conditional Generation**
  Conditioned on **text-based color names** using a **channel-wise concatenation** strategy.

* **🧪 Data Augmentation**
  Training includes **synchronized augmentations** (rotation, scaling, translation) for improved generalization.

* **📊 Experiment Tracking**
  All experiments, tuning, and versions tracked via [Weights & Biases](https://wandb.ai/hibifovohig3-add/ayna-ml-polygon-coloring).

* **🧠 Pretrained Model**
  Includes a ready-to-use model (`best_model_augmented.pth`) for immediate inference in **Google Colab**.

---

## 🧑‍💻 Getting Started: Inference with Google Colab

Run the model instantly using the provided [Colab Notebook](https://colab.research.google.com/):

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 🔧 Prerequisites

* A Google account
* No local setup needed!

### 📋 Instructions

1. **Open the Notebook:**
   Click the Colab badge above or open `Ayna_ML_Assignment.ipynb`.

2. **Connect to Runtime:**
   Click **"Connect"** (top right), then switch to GPU via
   `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU`.

3. **Run Inference Cells:**
   Only run cells under the **"FINAL, STREAMLINED INFERENCE CELL"** section:

   * The first cell sets up the environment and loads the model.
   * The second cell runs inference.

4. **Customize Inputs:**

   * 🔷 **New Shape:**

     * Upload your shape image (e.g., `my_star.png`) to Colab.
     * Change the input path:

       ```python
       INPUT_POLYGON_PATH = '/content/circle.png'
       ```

   * 🌈 **New Color:**

     * Choose a color from the "Available colors" list in the notebook.
     * Update the input color:

       ```python
       INPUT_COLOR = 'blue'
       ```

5. **View Results:**
   Your input shape and the final colored output will be displayed automatically.

---

## 📁 Project Structure

```bash
.
├── checkpoints/
│   └── best_model_augmented.pth      # Final pretrained model
├── dataset/
│   ├── training/                     # Training data (images, JSON)
│   └── validation/                   # Validation data (images, JSON)
├── Ayna_ML_Assignment.ipynb         # Main Colab notebook (training + inference)
└── README.md                         # This file
```

---

## 🧪 Experiments & Training

The notebook (`Ayna_ML_Assignment.ipynb`) contains:

* Inferior methods tested (e.g., MSELoss, Bottleneck Injection)
* Final training with:

  * **L1 Loss**
  * **Channel-wise conditioning**
  * **Learning rate scheduling**
  * **Synchronized augmentations**

Everything is tracked and reproducible via [W\&B](https://wandb.ai/hibifovohig3-add/ayna-ml-polygon-coloring).

---

## 📚 Key Learnings

* 🔄 **Synchronized Augmentation is a Game-Changer**
  Prevented overfitting and helped the model generalize to unseen shapes.

* 🔍 **Systematic Experimentation Builds Better Models**
  Comparing multiple loss functions, injection strategies, and data pipelines led to an optimal combination.

* 🔗 **The Data Pipeline *is* Part of the Model**
  Reliable, bug-free preprocessing (e.g., consistent encodings) is crucial for performance.

---

## 📎 Links

* 🔗 [W\&B Dashboard](https://wandb.ai/hibifovohig3-add/ayna-ml-polygon-coloring)
* 🧠 [Open in Google Colab](https://colab.research.google.com/)

---

Feel free to clone, explore, and contribute to the project.
If this project helped you or sparked ideas, consider giving it a ⭐!

---

