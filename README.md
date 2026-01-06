# Kidney CT Scan Classification

Deep learning and machine learning project for classifying kidney CT scan images into four categories: **Cyst**, **Normal**, **Stone**, and **Tumor**.

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

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn scikit-learn tqdm pillow timm
```

4. **Download dataset**
- Place the CT-Kidney dataset in `data/raw-2D-CT-Kidney/` with subfolders: `Cyst/`, `Normal/`, `Stone/`, `Tumor/`

5. **Run notebooks in order**
```
01_data_analysis_processing.ipynb  → Preprocess data, remove duplicates, create splits
02_efficientnet_b0.ipynb           → Train EfficientNet-B0
03_custom_cnn.ipynb                → Train Custom CNN
04_resnet50.ipynb                  → Train ResNet-50
05_densenet121.ipynb               → Train DenseNet-121
06_ml_models.ipynb                 → Train ML models (SVM, KNN, Random Forest, Logistic Regression)
```

---

## Folder Structure

```
project/
├── data/
│   ├── raw-2D-CT-Kidney/          # Original dataset
│   │   ├── Cyst/
│   │   ├── Normal/
│   │   ├── Stone/
│   │   └── Tumor/
│   ├── processed/                  # After preprocessing
│   │   ├── train/                  # 70% stratified
│   │   ├── val/                    # 15% stratified
│   │   └── test/                   # 15% stratified
│   └── dataset_config.json         # Mean, std, paths, class info
│
├── notebooks/
│   ├── 01_data_analysis_processing.ipynb
│   ├── 02_efficientnet_b0.ipynb
│   ├── 03_custom_cnn.ipynb
│   ├── 04_resnet50.ipynb
│   ├── 05_densenet121.ipynb
│   └── 06_ml_models.ipynb
│
├── outputs/
│   ├── figures/                    # Data analysis plots
│   │   └── data_summary.png
│   ├── efficientnet/
│   │   ├── model.pth
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   ├── custom_cnn/
│   │   ├── model.pth
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   ├── resnet50/
│   │   ├── model.pth
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   ├── densenet/
│   │   ├── model.pth
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   └── ml_models/
│       ├── best_model.pkl
│       ├── confusion_matrices.png
│       └── ml_vs_dl_comparison.png
│
└── README.md
```

---

## Results & Analysis

### Purpose

This project evaluates multiple deep learning and classical machine learning approaches for automated kidney disease classification from CT scan images. The goal is to compare:

1. **Transfer Learning** (EfficientNet-B0, ResNet-50, DenseNet-121) - Pretrained on ImageNet with frozen backbones
2. **Custom Architecture** (SimpleCNN) - Trained from scratch on medical imaging data
3. **Classical ML** (SVM, KNN, Random Forest, Logistic Regression) - Traditional machine learning baselines

### Dataset Summary

| Split | Images | Percentage |
|-------|--------|------------|
| Train | ~8,400 | 70% |
| Validation | ~1,800 | 15% |
| Test | ~1,800 | 15% |

- **Classes**: Cyst, Normal, Stone, Tumor (4 classes)
- **Preprocessing**: Duplicate removal via MD5 hashing, stratified splitting
- **Normalization**: Dataset-specific mean/std computed from training set only

### Model Comparison

| Model | Trainable Params | Backbone | Epochs | Expected Accuracy |
|-------|-----------------|----------|--------|-------------------|
| EfficientNet-B0 | ~330K | Frozen | 10 | 95-98% |
| ResNet-50 | ~525K | Frozen | 10 | 94-97% |
| DenseNet-121 | ~263K | Frozen | 10 | 94-97% |
| Custom CNN | ~700K | N/A | 40 | 85-92% |

### Machine Learning Models Comparison

| Model | Type | Features | Expected Accuracy |
|-------|------|----------|-------------------|
| SVM (RBF) | Classical ML | Flattened pixels (64×64) | 70-85% |
| KNN (k=5) | Classical ML | Flattened pixels (64×64) | 65-80% |
| Random Forest | Classical ML | Flattened pixels (64×64) | 70-85% |
| Logistic Regression | Classical ML | Flattened pixels (64×64) | 60-75% |

### Key Findings

1. **Transfer Learning Dominance**: Pretrained models significantly outperform custom CNN due to rich feature representations learned from ImageNet.

2. **Deep Learning vs Classical ML**: Neural networks dramatically outperform classical ML algorithms (SVM, KNN, RF, LR) on this image classification task. Classical ML relies on flattened pixel features which lose spatial information critical for medical imaging.

3. **Frozen Backbone Efficiency**: Training only the classifier head (last layer) is sufficient for this task, reducing training time from hours to minutes.

4. **Data Quality Impact**: Duplicate removal and proper stratification ensure unbiased evaluation across all classes.

5. **GPU Optimization**: RAM caching of preprocessed tensors eliminates I/O bottleneck, achieving near 100% GPU utilization.

### Training Configuration

All models use:
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (transfer learning) / OneCycleLR (custom CNN)
- **Mixed Precision**: FP16 on CUDA for 2x speedup
- **Batch Size**: 128
- **Image Size**: 224×224
- **Augmentation**: Horizontal flip, Vertical flip (on cached tensors)

### Evaluation Metrics

Each model is evaluated on the held-out test set using:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Visual error analysis

### Clinical Relevance

Automated kidney CT classification can assist radiologists by:
- Reducing diagnostic time
- Providing second-opinion screening
- Flagging potential tumors for priority review
- Supporting remote/underserved healthcare facilities

---

## Hardware Used

- GPU: NVIDIA RTX 3090 Ti (24GB VRAM)
- RAM: 64GB DDR5
- Storage: NVMe SSD

Training time per model: ~2-5 minutes (with RAM caching)

---

## License

This project is for educational purposes (Master's degree coursework).
