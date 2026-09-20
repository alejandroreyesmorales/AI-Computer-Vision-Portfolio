
# Project 02 — Local Features for Cell Nuclei Classification

## Overview

This project implements a Machine Learning and Deep Learning pipeline for classifying pixels as either nucleus or background in cervical cytology images.

The approach uses local image features extracted from RGB information surrounding a reference pixel. A Multilayer Perceptron (MLP) is trained to classify pixels using these features.

The project is based on research work involving cervical cell image analysis and was adapted for educational purposes and local reproducibility.

## Objectives

- Develop a pixel-level classification pipeline using local image features.
- Compare different classification algorithms.
- Evaluate model performance using multiple classification metrics.
- Analyze cross-dataset generalization between CRIC and SIPaKMeD.
- Provide a structured and reproducible implementation for educational purposes.

## Datasets

The experiments use data derived from two cervical cytology image datasets:

- CRIC
- SIPaKMeD

The datasets are not included in this repository because of their size and data distribution requirements.

The CSV files used in the experiments must be obtained and configured locally.

## Local Feature Representation

Each sample contains **33 input features**:

| Feature group | Number of features |
|---|---:|
| Reference pixel RGB values | 3 |
| Eight neighboring pixels (RGB) | 24 |
| Mean RGB values | 3 |
| Standard deviation RGB values | 3 |
| **Total** | **33** |

The feature representation is organized as follows:

1. RGB values of the reference pixel.
2. RGB values of eight neighboring pixels.
3. Mean intensity for each RGB channel.
4. Standard deviation for each RGB channel.

The final feature vector contains 33 values, followed by the class label.

## Classification Models

### Multilayer Perceptron (MLP33)

The MLP33 architecture consists of:

- Input layer: 33 features.
- Dense layer: 128 neurons with ReLU activation.
- Dense layer: 64 neurons with ReLU activation.
- Output layer: 2 neurons with softmax activation.

Training configuration:

- Optimizer: Adam.
- Loss function: Sparse categorical crossentropy.
- Epochs: 20.
- Batch size: 2048.
- Train/test split: 80% / 20%.
- Stratified data partitioning.
- Validation split during training: 20%.

The original research experiments were conducted in an HPC environment. This repository contains a simplified local implementation using 20 epochs for educational purposes and practical execution on a local computer.

The implementation should not be interpreted as an exact reproduction of the original HPC experiments.

### Additional Classifiers

The following classifiers were evaluated on the CRIC dataset:

- K-Nearest Neighbors (KNN), with k = 4.
- Decision Tree, using the default configuration of the scikit-learn implementation.
- Random Forest, with 50 trees.

The SVM and Bayesian Gaussian Mixture Model experiments are not included in the curated results tables.

## Experimental Setup

The experiments were conducted using the complete available CSV datasets.

The classification pipeline includes:

1. Loading the feature datasets.
2. Separating the input features and class labels.
3. Performing a stratified train/test split.
4. Training the classification models.
5. Evaluating the predictions.
6. Computing classification metrics.
7. Saving models, training histories, and results.

The following metrics are reported:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score

## Results

### CRIC — Classification Models

| Classifier | Accuracy | Precision | Recall | Specificity | F1-score |
|---|---:|---:|---:|---:|---:|
| MLP33 | 0.9459 | 0.9349 | 0.9585 | 0.9333 | 0.9466 |
| KNN | 0.9434 | 0.9453 | 0.9413 | 0.9456 | 0.9433 |
| Decision Tree | 0.9149 | 0.9173 | 0.9121 | 0.9177 | 0.9147 |
| Random Forest | 0.9482 | 0.9296 | 0.9698 | 0.9265 | 0.9493 |

### SIPaKMeD — MLP33

| Classifier | Accuracy | Precision | Recall | Specificity | F1-score |
|---|---:|---:|---:|---:|---:|
| MLP33 | 0.9289 | 0.9120 | 0.9493 | 0.9084 | 0.9303 |

### Cross-Dataset Evaluation

The MLP33 models were evaluated on the dataset different from the one used during training.

| Training dataset | Evaluation dataset | Accuracy | Precision | Recall | Specificity | F1-score |
|---|---|---:|---:|---:|---:|---:|
| CRIC | SIPaKMeD | 0.8551 | 0.8341 | 0.8866 | 0.8237 | 0.8595 |
| SIPaKMeD | CRIC | 0.9097 | 0.8927 | 0.9313 | 0.8881 | 0.9116 |

The cross-dataset results show a decrease in performance compared with evaluation using data from the same dataset. This experiment provides information about model generalization across different data distributions.

## Repository Structure

```text
project-02-local-features-classification/
│
├── README.md
│
├── data/
│
├── results/
│   ├── cric/
│   ├── sipakmed/
│   ├── cross_dataset/
│   ├── plots/
│   └── tables/
│
└── src/
    ├── summary_datasets.py
    ├── run_summary.py
    ├── train_mlp33_cric.py
    ├── train_mlp33_sipakmed.py
    ├── train_classifiers.py
    ├── evaluate_cross_dataset.py
    ├── plot_training_histories.py
    ├── create_results_tables.py
    └── verify_results_tables.py
```

## Outputs

The repository contains scripts for generating:

- Trained MLP models.
- Training histories.
- Accuracy and loss plots.
- Classification metrics.
- Cross-dataset evaluation results.
- Consolidated CSV tables.

The original datasets are not included in the repository.

## Limitations

- The local implementation uses 20 training epochs and is not an exact reproduction of the original HPC experiments.
- The complete datasets must be configured locally.
- The reported results depend on the data partition, random seed, software environment, and model configuration.
- The additional classifiers were evaluated on CRIC, while MLP33 was evaluated on both datasets.
- Cross-dataset evaluation was performed using the MLP33 models.

## Research Reference

This project is based on the methodology presented in the following
peer-reviewed publication:

Reyes Morales, A., Dalmau Cedeño, O. S., Alarcón Martínez, T. E.,
& Oliva Ibarra, F. E. (2026).

"Multilayer Perceptron for squamous cell nuclei localization in Pap
smear tests using local features."

Revista Mexicana de Ingeniería Biomédica, 47(1), e1537.

DOI: https://doi.org/10.17488/RMIB.47.1.1537

Official publication:
https://rmib.mx/index.php/rmib/article/view/1537

### Implementation Note

The original experiments were conducted in a Linux-based HPC
environment. This repository provides a simplified implementation
for educational purposes and local reproducibility. The training
configuration and results presented here should not be interpreted
as an exact reproduction of the original experiments.

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Research Context

This project is related to research on digital image processing and artificial intelligence for cervical cytology analysis.

The local feature representation is based on RGB information from reference pixels and their surrounding neighborhoods, with the objective of distinguishing nucleus and background regions.

## Author

**Alejandro Reyes Morales**

AI / Computer Vision Engineer

- LinkedIn: https://www.linkedin.com/in/alejandro-reyes-morales
- GitHub: https://github.com/alejandroreyesmorales
```

