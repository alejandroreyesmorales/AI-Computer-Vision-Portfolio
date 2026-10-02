# End-to-End AI Computer Vision Pipeline with Deep Learning and MLOps

## Overview

This project implements an end-to-end computer vision pipeline for cervical cell classification using deep learning and machine learning, with a focus on reproducibility and progressive MLOps integration.

The project uses the **SIPaKMeD** dataset and follows a modular workflow in which a pretrained CNN is used as a feature extractor and a downstream classifier is selected and evaluated using a leakage-aware validation strategy.

The current pipeline includes:

- Group-aware partitioning of the SIPaKMeD dataset.
- Prevention of data leakage by keeping cells from the same parent image within the same subset.
- Deep feature extraction using **ResNet-50 pretrained on ImageNet**.
- Extraction of 2,048-dimensional feature representations.
- Binary classification by grouping the original five cell categories into Normal and Abnormal classes.
- Comparison of MLP, SVM-RBF, and Random Forest classifiers.
- Five-fold cross-validation using `StratifiedGroupKFold` on the development set.
- Selection of the final classifier using development-set performance only.
- Final training of the selected MLP using the complete development set.
- Independent evaluation on a previously unseen test set.
- Preservation of the trained model and feature scaler as inference artifacts.
- Progressive integration of MLOps components into the computer vision pipeline.

The project is designed as a portfolio implementation that demonstrates the transition from a deep learning model to a reproducible machine learning inference pipeline.

---

## Objectives

The main objectives of this project are:

- Build a reproducible computer vision classification pipeline using SIPaKMeD.
- Prevent data leakage through parent-image-level data partitioning.
- Use a pretrained **ResNet-50** model as a deep feature extractor.
- Compare different downstream classifiers using the same feature representation.
- Select the final classifier using only the development data.
- Evaluate the selected model once on an independent test set.
- Preserve the model and preprocessing components required for reproducible inference.
- Extend the classification pipeline toward an end-to-end MLOps workflow.

---

## Dataset

### SIPaKMeD

The project uses the **SIPaKMeD** dataset, which contains five categories of cervical cell images. The dataset includes 966 cell images and 4,049 individual cells.

| Category | Classification | Number of images | Number of cells |
|---|---|---:|---:|
| Normal | Superficial/Intermediate | 126 | 813 |
| Normal | Parabasal | 108 | 787 |
| Abnormal | Koilocytotic | 238 | 825 |
| Abnormal | Metaplastic | 271 | 793 |
| Abnormal | Dyskeratotic | 223 | 813 |
| **Total** | **—** | **966** | **4,049** |

The dataset contains **4,049 cell images** distributed across **271 parent images**.

### Binary Classification

The original five-class labels are preserved during data preparation and feature extraction.

Binary relabeling is performed only during the classification stage:

| Binary class | Original categories |
|---|---|
| Normal | Superficial/Intermediate, Parabasal |
| Abnormal | Koilocytotic, Metaplastic, Dyskeratotic |

This separation allows the extracted feature representation to remain reusable independently of the final classification task.

---

## Data Partitioning

The dataset was partitioned at the **parent-image level** to prevent data leakage.

Since multiple cell patches can originate from the same parent image, assigning individual cells independently to different subsets could result in visually related samples appearing in both development and test data.

A `GroupShuffleSplit` strategy was therefore used, with the parent image identifier serving as the grouping variable.

### Final Partition

| Subset | Cell images | Parent images | Proportion |
|---|---:|---:|---:|
| Development | 3,379 | 230 | 83.45% |
| Independent Test | 670 | 41 | 16.55% |
| **Total** | **4,049** | **271** | **100.00%** |

The target split was approximately 85% development and 15% test. Because partitioning was performed at the parent-image level, the resulting proportions are not exactly 85/15.

### Leakage Prevention

The final partition contains:

```text
Shared parent images between DEV and TEST: 0
```

Therefore, no cell originating from a parent image present in the development subset appears in the independent test subset.

The resulting split files are:

```text
data/splits/
├── dev_data.csv
└── test_data.csv
```

---

## Feature Extraction

A **ResNet-50 pretrained on ImageNet** is used as a frozen feature extractor.

The classification head of ResNet-50 is removed and Global Average Pooling is applied to obtain a compact representation of each cell image.

The resulting feature representation contains:

```text
2048 features per cell image
```

The original metadata are retained together with the extracted features:

| Information | Description |
|---|---|
| `filepath` | Original image path |
| `label` | Original five-class label |
| `group_id` | Parent-image identifier |
| `feature_0` ... `feature_2047` | ResNet-50 feature representation |

The extracted feature files are stored locally in:

```text
data/features/
├── dev_features_resnet50.csv
└── test_features_resnet50.csv
```

The original dataset and generated feature files are excluded from version control.

---

## Classifier Selection

After feature extraction, three classifiers were evaluated using the same 2,048-dimensional representation:

- MLP
- SVM with RBF kernel
- Random Forest

The development set was evaluated using **5-fold StratifiedGroupKFold cross-validation**.

The folds preserve the parent-image grouping so that cells originating from the same parent image are not distributed across training and validation folds.

### Classifier Configurations

The main configurations used for the classifier comparison were:

| Classifier | Key configuration | Standardization |
|---|---|---|
| MLP | `128 → 64 → 1`, ReLU/Sigmoid, Adam, 50 epochs, batch size 32 | Yes |
| SVM-RBF | `kernel=rbf`, `C=1`, `gamma=scale`, `random_state=42` | Yes |
| Random Forest | `n_estimators=200`, `criterion=gini`, `max_depth=None`, `random_state=42`, `n_jobs=-1` | No |

Standardization for the MLP and SVM was performed within each cross-validation fold. The scaler was fitted only on the training portion of each fold and subsequently applied to the corresponding validation portion.

### Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)

### Cross-Validation Results

| Classifier | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---:|---:|---:|---:|
| **MLP** | **0.9834 ± 0.0070** | 0.9827 ± 0.0066 | **0.9828 ± 0.0082** | **0.9827 ± 0.0073** |
| SVM-RBF | 0.9825 ± 0.0089 | **0.9835 ± 0.0075** | 0.9802 ± 0.0109 | 0.9817 ± 0.0093 |
| Random Forest | 0.9591 ± 0.0086 | 0.9605 ± 0.0086 | 0.9547 ± 0.0099 | 0.9573 ± 0.0091 |

The **MLP was selected as the final classifier** because it obtained the highest mean Accuracy and F1-score across the development folds. Precision was used as a secondary criterion when comparing closely performing models.

The classifier-selection results are stored in:

```text
results/classification/
└── model_selection_results.csv
```

---

## Final Classification Model

The selected MLP uses the following architecture:

```text
2048
  ↓
Dense(128, ReLU)
  ↓
Dense(64, ReLU)
  ↓
Dense(1, Sigmoid)
```

### Training Configuration

| Parameter | Configuration |
|---|---|
| Input features | 2,048 |
| Hidden layer 1 | 128 neurons, ReLU |
| Hidden layer 2 | 64 neurons, ReLU |
| Output layer | 1 neuron, Sigmoid |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Epochs | 50 |
| Batch size | 32 |
| Random seed | 42 |
| Classification task | Normal vs. Abnormal |

After classifier selection, the final MLP was retrained using **all 3,379 development samples**.

The scaler was fitted using the complete development set and subsequently applied to the independent test set.

---

## Final Evaluation on the Independent Test Set

The independent test set contains 670 cell images and was not used during classifier selection.

The final model was evaluated on TEST **once**, after the classifier and configuration had already been selected using the development data.

### Results

| Metric | TEST |
|---|---:|
| Accuracy | **0.9806** |
| Precision (macro) | 0.9830 |
| Recall (macro) | 0.9758 |
| Specificity | 0.9563 |
| F1-score (macro) | 0.9792 |

### Confusion Matrix Components

| Component | Value |
|---|---:|
| TN | 241 |
| FP | 11 |
| FN | 2 |
| TP | 416 |

Specificity was calculated as:

$$
\mathrm{Specificity} = \frac{TN}{TN + FP}
$$

The final results are stored in:

```text
results/classification/
├── model_selection_results.csv
└── final_test_results.csv
```

---

## Model Artifacts

The trained model and preprocessing component required for future inference are preserved locally:

```text
models/
└── mlp/
    ├── final_model.keras
    └── scaler.joblib
```

`final_model.keras` contains the trained MLP, while `scaler.joblib` stores the StandardScaler fitted during final training.

Both artifacts are required to reproduce the preprocessing and prediction steps during inference.

---

## Current Project Structure

```text
project-04-ai-computer-vision-mlops/
├── data/
│   ├── splits/
│   │   ├── dev_data.csv
│   │   └── test_data.csv
│   └── features/
│       ├── dev_features_resnet50.csv
│       └── test_features_resnet50.csv
├── models/
│   └── mlp/
│       ├── final_model.keras
│       └── scaler.joblib
├── results/
│   └── classification/
│       ├── model_selection_results.csv
│       └── final_test_results.csv
├── src/
│   ├── Data_Split_lists.py
│   ├── feature_extraction_resnet50.py
│   ├── classification_model_selection.py
│   └── train_final_classifier.py
├── .gitignore
├── README.md
└── requirements.txt
```

Original medical images, generated feature files, trained model artifacts, and other local data are not intended to be uploaded to the public repository unless explicitly required for deployment or reproducibility.

---

## MLOps Roadmap

The project is being developed progressively toward an end-to-end MLOps workflow.

### Completed

- [x] Dataset preparation
- [x] Parent-image-level data partitioning
- [x] Data leakage prevention
- [x] ResNet-50 feature extraction
- [x] Classifier comparison
- [x] Model selection using development data
- [x] Final MLP training
- [x] Independent TEST evaluation
- [x] Model and scaler artifact generation

### Planned

- [ ] FastAPI inference service
- [ ] Docker containerization
- [ ] MLflow experiment tracking and model management
- [ ] CI/CD automation
- [ ] Basic inference monitoring

The MLOps components will be added incrementally while preserving the same trained model and preprocessing pipeline.

---

## Technologies

| Technology | Role |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | ResNet-50 feature extraction and MLP |
| scikit-learn | Data splitting, cross-validation, scaling, and classical classifiers |
| Pandas | Dataset and feature management |
| NumPy | Numerical computation |
| Git | Version control |
| FastAPI | Planned inference API |
| Docker | Planned containerization |
| MLflow | Planned experiment tracking and model management |
| GitHub Actions | Planned CI/CD |

---

## Status

**Current status: Classification pipeline completed.**

The project has successfully progressed from SIPaKMeD data preparation and deep feature extraction to classifier selection and independent final evaluation.

The next development stage is the implementation of the **FastAPI inference service**, which will expose the trained computer vision model through an API and serve as the foundation for the subsequent MLOps components.

---

## Author

**Alejandro Reyes Morales**

AI / Computer Vision / Machine Learning

- LinkedIn: https://www.linkedin.com/in/alejandro-reyes-morales
- GitHub: https://github.com/alejandroreyesmorales
