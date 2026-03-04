readme_content = """# Automated Huntington Disease Diagnosis Using MRI Imaging and Deep Learning

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

# 📋 Project Overview

This capstone project implements an automated deep learning system for diagnosing **Huntington's Disease (HD)** from brain MRI scans.

The system evaluates **three modern convolutional neural network architectures** using transfer learning:

- ResNet50
- DenseNet121
- EfficientNet-B0

These models classify MRI brain scans into:

• Huntington's Disease  
• Healthy Control

The project performs a **comprehensive performance comparison** between the architectures.

---

# 🚀 Key Features

- Transfer Learning using ImageNet pretrained models
- Multi-architecture comparison
- Grad-CAM explainability
- Class imbalance handling
- ROC curve analysis
- Confusion matrix evaluation
- Efficiency vs performance comparison

---

# 🏆 Results Summary

## Test Set Performance

| Metric | ResNet50 | DenseNet121 | EfficientNet-B0 |
|------|------|------|------|
| Accuracy | **0.9495** | 0.8683 | 0.8511 |
| Precision | **0.8885** | 0.6331 | 0.6127 |
| Recall | 0.8893 | **1.0000** | 0.9369 |
| Specificity | **0.9672** | 0.8295 | 0.8258 |
| F1 Score | **0.8889** | 0.7753 | 0.7409 |
| ROC AUC | 0.9776 | **0.9877** | 0.9696 |

### Key Observations

• **ResNet50 achieved the highest overall accuracy and F1 score**  
• **DenseNet121 achieved perfect recall (1.0)**  
• **EfficientNet-B0 is the most lightweight model**

---

# 📊 Confusion Matrix Statistics

### ResNet50
TP: 916  
TN: 3387  
FP: 115  
FN: 114  

### DenseNet121
TP: 1030  
TN: 2905  
FP: 597  
FN: 0  

### EfficientNet-B0
TP: 965  
TN: 2892  
FP: 610  
FN: 65  

---

# 📊 Efficiency vs Performance

| Model | Parameters | Accuracy | F1 Score |
|------|------|------|------|
| ResNet50 | 24.5M | **0.9495** | **0.8889** |
| DenseNet121 | 7.5M | 0.8683 | 0.7753 |
| EfficientNet-B0 | 5.3M | 0.8511 | 0.7409 |

---

# 🗂️ Project Structure

```

HD-MRI-Diagnosis
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model_f1.pth
│   ├── best_model_loss.pth
│   └── final_model.pth
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize_gradcam.py
│   ├── create_dataset.py
│   └── model.py
│
├── results/
│   ├── evaluation/
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   │
│   ├── gradcam/
│   └── training_history.png
│
├── requirements.txt
├── README.md
└── LICENSE

```

---

# 🧠 Methodology

### 1️⃣ Data Collection
MRI scans obtained from the **OASIS-1 dataset**.

### 2️⃣ Preprocessing
- Skull stripping
- Normalization
- Slice extraction
- Data augmentation

### 3️⃣ Model Training
Transfer learning using pretrained CNN architectures.

### 4️⃣ Evaluation
Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Grad-CAM used for model interpretability.

---

# 📄 Dataset Information

Dataset: **OASIS-1 Cross-Sectional MRI Dataset**

Subjects: 436  
MRI Type: T1-weighted  
Age Range: 18-96

Label Distribution

Healthy Control: 336  
Dementia: 100

---

# 💡 Key Findings

Strengths

• High accuracy using ResNet50  
• Perfect recall using DenseNet121  
• Efficient lightweight model using EfficientNet-B0  
• Strong ROC-AUC (>0.96)

Limitations

• Limited dataset size  
• 2D slice approach  
• Potential overfitting risk

---

# 🔮 Future Work

- 3D CNN architectures
- Larger MRI datasets
- Multi-class disease severity prediction
- Attention-based models
- Clinical deployment system

---

# 👨‍💻 Author

Sayed Zabiulla  
Alliance University  

GitHub  
https://github.com/SayedZabiulla

---

# 📝 License

MIT License
"""

import os

PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
readme_path = os.path.join(PROJECT_PATH, "README.md")

with open(readme_path, "w") as f:
    f.write(readme_content)

print(f"README.md created at: {readme_path}")

