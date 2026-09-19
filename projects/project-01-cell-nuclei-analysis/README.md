
# Squamous Cell Nuclei Recognition Using RGB Pixel Intensities

## Overview

This project presents a simple pixel-level classification approach for recognizing squamous cell nuclei in cervical cytology images.

The method uses manually annotated image regions to generate labeled pixel-level data. A Multilayer Perceptron (MLP) is then trained to distinguish between nucleus and background pixels using their RGB intensity values.

This project is based on research presented in the following book chapter:

> Reconocimiento de núcleos de células escamosas en imágenes provenientes de citologías de cuello uterino.

## Objective

To develop a machine learning approach for identifying pixels belonging to squamous cell nuclei in cervical cytology images using RGB color information.

The project explores whether pixel intensity values in the red, green, and blue channels can be used to distinguish nuclear regions from the background.

## Methodology

The methodology consists of the following stages:

### 1. Image Selection

A subset of images is selected from a publicly available cervical cytology image dataset.

The original research used the CRIC dataset as the public image source.

### 2. Manual Mask Generation

Manual masks are created using GIMP.

The nuclear regions are marked in the masks to identify the pixels belonging to the target class.

The masks are used to determine which pixels correspond to nuclear regions and which pixels belong to the background.

### 3. Pixel-Level Data Generation

RGB intensity values are extracted from the original images using the manually created masks.

Each pixel is assigned a binary class label:

- `1`: Nucleus pixel
- `0`: Background pixel

For each image, a number of background pixels is randomly selected to match the number of nucleus pixels.

This procedure generates a balanced dataset containing nucleus and background samples.

### 4. Feature Representation

Each pixel is represented by three input features:

- Red intensity (R)
- Green intensity (G)
- Blue intensity (B)

The intensity values are normalized to the range `[0, 1]`.

No texture descriptors, morphological features, spatial coordinates, or deep learning features are used in this implementation.

### 5. MLP Classification

A Multilayer Perceptron (MLP) is used as a binary classifier.

The model receives the three RGB intensity values of each pixel and predicts whether the pixel belongs to a nucleus or the background.

The documented model configuration includes:

- Input dimension: 3
- Hidden layers: 2
- Hidden layer sizes: 128 and 64 neurons
- Output dimension: 1
- Loss function: Binary cross-entropy
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 2048
- Training epochs: 200

## Project Workflow

```text
Original Cytology Image
          |
          v
Manual Mask Generation (GIMP)
          |
          v
Nucleus Pixel Extraction
          |
          v
Random Background Pixel Sampling
          |
          v
RGB Feature Extraction
          |
          v
Data Normalization
          |
          v
MLP Training
          |
          v
Pixel-Level Classification
```

## Technologies

- Python
- NumPy
- OpenCV
- TensorFlow / Keras
- Matplotlib
- GIMP
- Machine Learning
- Digital Image Processing

## Project Structure

```text
project-01-cell-nuclei-analysis/
│
├── README.md
├── data/
├── notebooks/
├── src/
├── results/
└── docs/
```

## Scope and Limitations

This project is a proof-of-concept for pixel-level classification.

The method relies exclusively on RGB intensity values and does not include:

- Automatic mask generation
- Morphological postprocessing
- Connected-component analysis
- Spatial context modeling
- Texture descriptors
- Comparison with other classifiers
- End-to-end automatic cell segmentation

The approach is intended to demonstrate the construction of a labeled image dataset and the application of a neural network classifier to a medical imaging problem.

## Research Publication

**Chapter:** Reconocimiento de núcleos de células escamosas en imágenes provenientes de citologías de cuello uterino

**Book:** Ciencia y tecnología: una visión de los Cuerpos Académicos del CUValles

**Edition:** 1st edition, 2026

**Publisher:** Universidad de Guadalajara, published in association with Centro Universitario de los Valles (CUValles)

**DOI:** [10.32870/9786076460047](https://doi.org/10.32870/9786076460047)

## Author

**Alejandro Reyes Morales**

AI / Computer Vision Engineer

- LinkedIn: https://www.linkedin.com/in/alejandro-reyes-morales
- GitHub: https://github.com/alejandroreyesmorales