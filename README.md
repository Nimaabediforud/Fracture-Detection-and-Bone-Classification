# 🦴 Bone Fracture Detection Using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Paused-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview
This project explores the use of **deep learning** for detecting bone fractures from **X-ray images**.  
The primary goal was to build a **Convolutional Neural Network (CNN)** from scratch (without transfer learning) to classify whether an X-ray image shows a fracture or not.

The workflow covers all key steps of a computer vision project — from **data preprocessing and exploration** to **model training, evaluation, and inference**.  
Although the model achieved promising results on the validation dataset, its generalization to real-world images remains a major challenge.

---

## Objectives
- Build and train a CNN-based image classifier for bone fracture detection.  
- Perform image preprocessing, augmentation, and normalization.  
- Evaluate the model using standard metrics (F1-score, AUC-ROC, Accuracy).  
- Investigate generalization challenges in medical imaging datasets.

---

## Dataset
- The dataset consisted of labeled **X-ray images** categorized into two classes:
  -  **Fractured bones**
  -  **Normal bones**
- Approximate label distribution: **60% fractured / 40% normal**  
- Slight imbalance was observed but acceptable for baseline experiments.  

---

## Model Architecture
A **Convolutional Neural Network (CNN)** was implemented from scratch using **TensorFlow/Keras**.

**Architecture summary:**
- Input: Preprocessed grayscale or RGB X-ray images  
- Convolutional + Pooling layers for feature extraction  
- Flatten + Dense layers for classification  
- Dropout regularization to prevent overfitting  
- **Optimizer:** Adam  
- **Loss function:** Binary cross-entropy  
- **Metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC  

---

## Results

| Metric | Validation Dataset | Real-World Images |
|:-------:|:-----------------:|:-----------------:|
| **F1-score (weighted)** | ~0.75 | Low performance |
| **AUC-ROC** | ~0.80 | — |
| **Accuracy** | ~0.70 | — |

> The model performed reasonably well on the validation dataset.

---

## Experiments & Observations
- **EDA & Preprocessing:** Image normalization, resizing, and data augmentation.  
- **Model Evaluation:** Reliable on internal validation data, weak generalization on external images (e.g. X-ray images from internet).  
- **Key Insight:** Expanding and diversifying the dataset is essential for better real-world performance.

---

## Current Status
This project is currently **paused**.  
Future directions include:
- Integrating **transfer learning** (e.g., ResNet, EfficientNet, DenseNet).  
- Combining multiple open-source medical datasets.  
- Improving interpretability using **Grad-CAM**, **LIME**, or **SHAP**.  

---

## Repository Structure
```
├── Notebooks/ # Jupyter notebooks
├── Models/ # Saved model weights and architectures
├── Src/ # Source code (utilities, preprocessing, etc.)
├── README.md # Project description
└── Data-logs 
```

