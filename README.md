# 🌿 Laboratory Work 4 — Improving CNN Performance
### Using Regularization, Fine-Tuning, and Advanced Evaluation

> **Course:** Computer Science — Machine Learning / Deep Learning
> **Dataset:** Tree & Fruit Image Dataset · 20 Classes · 7,194 Images
> **Framework:** TensorFlow / Keras · Google Colab

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Activity 1 — Evaluation Metrics and Visualization](#activity-1--evaluation-metrics-and-visualization)
- [Activity 2 — Model Interpretability Using Grad-CAM](#activity-2--model-interpretability-using-grad-cam)
- [Activity 3 — Model Enhancement and Performance Optimization](#activity-3--model-enhancement-and-performance-optimization)

---

## Overview

This laboratory work extends a trained CNN image classifier through three activities: full evaluation using standard ML metrics, visual explainability via Grad-CAM, and architecture-level improvements using regularization and fine-tuning techniques. The dataset consists of **20 tree and fruit classes** with a total of **7,194 images**, split 80/20 into training (5,756) and validation (1,438) sets.

---

## Dataset

| Property | Value |
|---|---|
| Total Images | 7,194 |
| Number of Classes | 20 |
| Training Set | 5,756 images (80%) |
| Validation Set | 1,438 images (20%) |
| Image Size | 180 × 180 px |
| Batch Size | 32 |

**Classes:**
`ARATILES` · `AVOCADO` · `Baobab` · `Breadfruit Tree` · `COCOA` · `Cedar of Lebanon` · `DUHAT` · `Dragon Blood Tree` · `Eucalyptus` · `GUAVA` · `KAMIAS` · `Kapur Tree` · `LANGKA` · `LANZONES` · `MANGOSTEEN` · `Monkey Pod Tree` · `PAPAYA` · `RAMBUTAN` · `Soursop Tree` · `Wollemi Pine`

---

## Activity 1 — Evaluation Metrics and Visualization

### Learning Outcomes
- Load a saved Keras model and run predictions on the validation set
- Compute Precision, Recall, and F1-Score per class
- Generate and interpret a Confusion Matrix
- Plot multi-class ROC Curves and compute the overall AUC Score
- Visualize per-class metric comparisons using bar charts

### Workflow

```
Load Model → Collect Predictions → Classification Report → Confusion Matrix → ROC / AUC → Bar Chart
```

### Results

**Overall Validation Accuracy:** `0.9949` (99.49%) on raw model evaluation
**Overall AUC Score:** `0.9627`

#### Classification Report (Baseline Model)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ARATILES | 0.96 | 0.76 | 0.85 | 63 |
| AVOCADO | 0.90 | 0.64 | 0.75 | 72 |
| Baobab (Adansonia digitata) | 0.84 | 0.80 | 0.82 | 59 |
| Breadfruit Tree (Artocarpus altilis) | 0.76 | 0.55 | **0.64** | 51 |
| COCOA | 0.93 | 0.85 | 0.89 | 79 |
| Cedar of Lebanon (Cedrus libani) | 0.84 | 0.86 | 0.85 | 83 |
| DUHAT | 0.78 | 0.81 | 0.79 | 77 |
| Dragon Blood Tree | 0.91 | 0.68 | 0.78 | 44 |
| Eucalyptus | 0.90 | 0.74 | 0.81 | 77 |
| GUAVA | 0.72 | 0.95 | 0.82 | 76 |
| KAMIAS | 0.86 | 0.87 | 0.86 | 76 |
| Kapur Tree (Dryobalanops aromatica) | 0.76 | 0.86 | 0.81 | 80 |
| LANGKA | 0.93 | 0.89 | 0.91 | 71 |
| LANZONES | 0.67 | 0.95 | 0.78 | 65 |
| MANGOSTEEN | 0.94 | 0.66 | 0.78 | 74 |
| Monkey Pod Tree (Samanea saman) | 0.84 | 0.75 | 0.79 | 56 |
| PAPAYA | 0.93 | 0.97 | **0.95** | 66 |
| RAMBUTAN | 0.58 | 1.00 | 0.73 | 68 |
| Soursop Tree (Annona muricata) | 0.89 | 0.71 | 0.79 | 76 |
| Wollemi Pine (Wollemia nobilis) | 0.84 | 0.93 | 0.88 | 61 |
| **Macro Avg** | **0.84** | **0.81** | **0.81** | 1374 |
| **Weighted Avg** | **0.84** | **0.82** | **0.82** | 1374 |

> **Weakest class:** Breadfruit Tree (F1 = 0.64) — lowest recall (0.55), indicating many missed detections.
> **Strongest class:** PAPAYA (F1 = 0.95) — high precision and recall.
> **RAMBUTAN** had the lowest precision (0.58) but perfect recall (1.00), indicating over-prediction.

### Screenshots

> **Confusion Matrix**
> *(Add screenshot: `screenshots/confusion_matrix.png`)*

> **ROC Curve — Multi-class**
> *(Add screenshot: `screenshots/roc_curve.png`)*

> **Precision / Recall / F1 Bar Chart**
> *(Add screenshot: `screenshots/precision_recall_f1_chart.png`)*

---

## Activity 2 — Model Interpretability Using Grad-CAM

### Learning Outcomes
- Understand the concept of Explainable AI (XAI)
- Apply Grad-CAM to visualize which image regions drive CNN predictions
- Identify whether the model focuses on the object or background
- Evaluate and improve trust in model decisions

### How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradient of the predicted class score with respect to the activations of the **last convolutional layer**. These gradients are pooled and used to weight the activation maps, producing a heatmap that highlights the most influential spatial regions.

```
Load Model → Preprocess Image → Identify Last Conv2D Layer → 
Build Grad Model → Compute Gradients → Generate Heatmap → Overlay on Image
```

### Model Layer Used

The last convolutional layer identified for this model:

```
last_conv_layer_name = "conv2d_5"
```

Full layer stack: `sequential_1 → rescaling_1 → conv2d_3 → max_pooling2d_3 → conv2d_4 → max_pooling2d_4 → conv2d_5 → max_pooling2d_5 → dropout → flatten_1 → dense_2 → dropout_1 → dense_3`

### Test Prediction

| Property | Value |
|---|---|
| Test Image | `test.jpg` |
| Predicted Class | **Monkey Pod Tree (Samanea saman)** |
| Confidence | **78.71%** |

### Heatmap Interpretation

| Observation | Meaning |
|---|---|
| 🔴 Heatmap concentrated on the tree/leaves | Model is learning relevant features correctly |
| 🔵 Heatmap on background or surroundings | Model is using spurious correlations |
| ⚪ Scattered with no clear focus | Weak feature learning — model needs improvement |

### Screenshots

> **Grad-CAM Heatmap**
> *(Add screenshot: `screenshots/gradcam_heatmap.png`)*

> **Grad-CAM Overlay on Test Image**
> *(Add screenshot: `screenshots/gradcam_overlay.png`)*

---

## Activity 3 — Model Enhancement and Performance Optimization

### Learning Outcomes
- Identify weaknesses from evaluation metrics and confusion matrix analysis
- Apply regularization techniques to improve model generalization
- Rebuild the CNN with BatchNormalization, Dropout, and data augmentation
- Use Early Stopping and a tuned learning rate to prevent overfitting
- Compare baseline vs. improved model quantitatively

### Identified Weaknesses (from Activity 1)

- **Breadfruit Tree** — lowest F1 (0.64), poorest recall (0.55)
- **RAMBUTAN** — lowest precision (0.58), over-predicting this class
- **AVOCADO, MANGOSTEEN** — low recall (0.64–0.66), many missed detections
- Generalization gap present: training accuracy was significantly higher than validation accuracy, indicating mild overfitting

### Enhancements Applied

#### 1. Advanced Data Augmentation
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])
```
Increases the variety of training data to reduce overfitting.

#### 2. Improved CNN Architecture — BatchNormalization + Dropout
```
Input (180×180×3)
  → Data Augmentation
  → Rescaling (÷255)
  → Conv2D(32) + BatchNorm + MaxPool
  → Conv2D(64) + BatchNorm + MaxPool
  → Conv2D(128) + BatchNorm + MaxPool
  → Dropout(0.4)
  → Flatten → Dense(256) → Dropout(0.5)
  → Dense(20) [output]
```

**Total Parameters:** 11,973,662 (45.68 MB) · Trainable: 3,991,220 (15.23 MB)

#### 3. Optimized Learning Rate
```python
optimizer = Adam(learning_rate=0.0001)
```
Lower learning rate allows finer weight updates and smoother convergence.

#### 4. Early Stopping
```python
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```
Halts training when validation loss stops improving for 3 consecutive epochs, restoring the best-performing weights.

#### 5. Training Configuration
| Parameter | Value |
|---|---|
| Max Epochs | 20 |
| Optimizer | Adam (lr = 0.0001) |
| Loss Function | SparseCategoricalCrossentropy (from_logits=True) |
| Early Stopping | val_loss, patience=3 |
| Saved Model | `my_image_classifier_improved.keras` |

### Comparison Table — Baseline vs. Improved Model

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Training Accuracy | 99.49% (val eval) | *(record after retraining)* |
| Validation Accuracy | 99.49% | *(record after retraining)* |
| Macro Precision | 0.84 | *(record after retraining)* |
| Macro Recall | 0.81 | *(record after retraining)* |
| Macro F1-Score | 0.81 | *(record after retraining)* |
| AUC Score | 0.9627 | *(record after retraining)* |

### Screenshots

> **Accuracy & Loss Training Curves (Improved Model)**
> *(Add screenshot: `screenshots/improved_accuracy_loss.png`)*

---

## 🗂️ Repository Structure

```
LW4/
├── LW4.ipynb                   # Main Colab notebook
├── README.md                   # This file
└── screenshots/
    ├── confusion_matrix.png        # Activity 1
    ├── roc_curve.png               # Activity 1
    ├── precision_recall_f1_chart.png  # Activity 1
    ├── gradcam_heatmap.png         # Activity 2
    ├── gradcam_overlay.png         # Activity 2
    └── improved_accuracy_loss.png  # Activity 3
```

---

## 📦 Dependencies

```
tensorflow >= 2.x
scikit-learn
matplotlib
numpy
opencv-python (cv2)
Pillow
```

---

*Laboratory Work 4 · CNN Evaluation, Explainability, and Optimization*
