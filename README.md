# Kidney CT Scan Classification

Deep learning and machine learning project for classifying kidney CT scan images into four categories: **Cyst**, **Normal**, **Stone**, and **Tumor**.

**Dataset**: [2D CT Kidney for Classification](https://www.kaggle.com/datasets/fizzazaitoonbsds2022/2d-ct-kidney-for-classification)

## Setup Guide

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended) or Apple Silicon Mac (MPS) or CPU
- 8GB+ RAM (64GB recommended for RAM caching)

### Installation

1. **Clone/Download the project**
```bash
git clone https://github.com/youssef-Araby/machine-project-term1.git
cd machine-project-term1
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib seaborn scikit-learn tqdm pillow timm
```

3. **Download and prepare dataset**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fizzazaitoonbsds2022/2d-ct-kidney-for-classification)
   - Extract and place in `data/raw-2D-CT-Kidney/` with the following structure:
   ```
   data/
   └── raw-2D-CT-Kidney/
       ├── Cyst/
       ├── Normal/
       ├── Stone/
       └── Tumor/
   ```

4. **Run notebooks in order**
```
01_data_analysis_processing.ipynb  → Preprocess data, remove duplicates, create splits
02_deep_learning_models.ipynb      → Train all DL models (EfficientNet, ResNet, DenseNet, Custom CNN)
03_ml_models.ipynb                 → Train ML models (SVM, KNN, Random Forest, Logistic Regression) with 5-Fold CV
```

---

## Folder Structure

```
project/
├── data/
│   ├── raw-2D-CT-Kidney/          # Original dataset (download from Kaggle)
│   │   ├── Cyst/
│   │   ├── Normal/
│   │   ├── Stone/
│   │   └── Tumor/
│   ├── processed/                  # After running notebook 01
│   │   ├── train/                  # 70% (9,411 images)
│   │   └── test/                   # 30% (4,034 images)
│   └── dataset_config.json         # Mean, std, paths, class info
│
├── notebooks/
│   ├── 01_data_analysis_processing.ipynb
│   ├── 02_deep_learning_models.ipynb
│   └── 03_ml_models.ipynb
│
├── outputs/
│   ├── dl/                         # Deep learning outputs
│   │   ├── efficientnetb0/
│   │   │   ├── model.pth
│   │   │   ├── results.json
│   │   │   ├── training_curves.png
│   │   │   └── confusion_matrix.png
│   │   ├── resnet50/
│   │   ├── densenet121/
│   │   ├── customcnn/
│   │   ├── dl_results.json         # Combined DL results
│   │   └── dl_model_comparison.png
│   │
│   └── ml/                         # Machine learning outputs
│       ├── svm_linear/
│       │   ├── model.pkl
│       │   ├── results.json
│       │   └── confusion_matrix.png
│       ├── svm_rbf/
│       ├── svm_poly/
│       ├── knn_manual_k5/
│       ├── random_forest/
│       ├── logistic_regression/
│       ├── ml_results.json         # Combined ML results
│       ├── best_model.pkl
│       ├── confusion_matrices.png
│       ├── accuracy_comparison.png
│       └── ml_vs_dl_comparison.png
│
└── README.md
```

---

## Results & Analysis

### Dataset Summary

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 9,411 | 70% |
| Test | 4,034 | 30% |

- **Total**: 13,445 unique images (after duplicate removal via MD5 hashing)
- **Classes**: Cyst, Normal, Stone, Tumor (4 classes)
- **Image Size**: 224×224 pixels
- **Preprocessing**: Stratified splitting, dataset-specific normalization

### Deep Learning Models

| Model | Backbone | Epochs | Description |
|-------|----------|--------|-------------|
| EfficientNet-B0 | Frozen (timm) | 20 | Transfer learning with custom classifier |
| ResNet-50 | Frozen (torchvision) | 20 | Transfer learning with custom classifier |
| DenseNet-121 | Frozen (torchvision) | 20 | Transfer learning with custom classifier |
| Custom CNN | N/A | 100 | 4-stage CNN trained from scratch |

### Machine Learning Models (5-Fold Cross-Validation)

| Model | Features | Description |
|-------|----------|-------------|
| SVM (Linear) | PCA (100 components) | LinearSVC with max_iter=5000 |
| SVM (RBF) | PCA (100 components) | SVC with RBF kernel |
| SVM (Poly) | PCA (100 components) | SVC with polynomial kernel (degree=3) |
| KNN (Manual) | PCA (100 components) | Custom implementation with vectorized distance |
| Random Forest | PCA (100 components) | 100 trees, max_depth=15 |
| Logistic Regression | PCA (100 components) | max_iter=2000 |

**Note**: ML models use PCA for dimensionality reduction (224×224×3 = 150,528 → 100 features) to speed up training.

### Key Findings

1. **Classical ML Performance**: With PCA preprocessing, classical ML models achieve competitive accuracy on this dataset due to the visually distinct classes (Cyst, Normal, Stone, Tumor have clear texture and intensity differences).

2. **Transfer Learning**: Pretrained models (EfficientNet, ResNet, DenseNet) with frozen backbones train efficiently in minutes while leveraging ImageNet features.

3. **5-Fold Cross-Validation**: ML models use stratified 5-fold CV on training data for robust performance estimation.

4. **Data Quality**: Duplicate removal via MD5 hashing ensures unique samples across splits.

5. **GPU Optimization**: RAM caching of preprocessed tensors eliminates I/O bottleneck for DL training.

### Final Results

**Deep Learning Models:**

| Model | Test Accuracy |
|-------|---------------|
| Custom CNN | **99.58%** |
| EfficientNet-B0 | 99.16% |
| ResNet-50 | 97.42% |
| DenseNet-121 | 96.50% |

**Machine Learning Models:**

| Model | CV Mean | CV Std | Test Accuracy |
|-------|---------|--------|---------------|
| Random Forest | 99.92% | ±0.08% | **99.95%** |
| KNN (Manual, k=5) | 99.86% | ±0.06% | 99.90% |
| SVM (RBF) | 99.36% | ±0.14% | 99.48% |
| SVM (Poly) | 98.58% | ±0.32% | 99.03% |
| Logistic Regression | 98.53% | ±0.38% | 98.96% |
| SVM (Linear) | 96.72% | ±0.29% | 97.10% |

**All Models Ranked:**

| Rank | Model | Type | Test Accuracy |
|------|-------|------|---------------|
| 1 | Random Forest | ML | **99.95%** |
| 2 | KNN (Manual) | ML | 99.90% |
| 3 | Custom CNN | DL | 99.58% |
| 4 | SVM (RBF) | ML | 99.48% |
| 5 | EfficientNet-B0 | DL | 99.16% |
| 6 | SVM (Poly) | ML | 99.03% |
| 7 | Logistic Regression | ML | 98.96% |
| 8 | ResNet-50 | DL | 97.42% |
| 9 | SVM (Linear) | ML | 97.10% |
| 10 | DenseNet-121 | DL | 96.50% |

### Why ML Models Match or Outperform Deep Learning

Classical ML models achieved up to 99.95% accuracy, matching or exceeding deep learning. This is due to:

1. **Visually Distinct Classes**: The four kidney conditions (Cyst, Normal, Stone, Tumor) have clear visual differences in texture and intensity that PCA captures effectively.

2. **Effective Dimensionality Reduction**: PCA reduces 150,528 features to 100 components while retaining discriminative information, making classes linearly separable.

3. **Dataset Characteristics**: CT scans from the same patient may appear in both train and test sets (original dataset not split by patient ID), benefiting simpler models.

4. **Transfer Learning Limitation**: Pretrained models (ImageNet) weren't optimized for medical CT scans, while the custom CNN trained from scratch achieved 99.58%.

**Conclusion**: For medical imaging with clearly distinguishable classes, classical ML with proper preprocessing can match deep learning while being faster and more interpretable.

### Training Configuration

**Deep Learning:**
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (transfer learning) / OneCycleLR (custom CNN)
- **Mixed Precision**: FP16 on CUDA
- **Batch Size**: 128
- **Image Size**: 224×224
- **Augmentation**: Horizontal flip, Vertical flip

**Machine Learning:**
- **Preprocessing**: StandardScaler → PCA (100 components)
- **Validation**: 5-Fold Stratified Cross-Validation
- **Image Size**: 224×224 (flattened to 150,528 features, reduced to 100 via PCA)

---

## Hardware Used

- GPU: NVIDIA RTX 3090 Ti (24GB VRAM)
- RAM: 64GB DDR5
- Storage: NVMe SSD

---

## License

This project is for educational purposes (Master's degree coursework).
