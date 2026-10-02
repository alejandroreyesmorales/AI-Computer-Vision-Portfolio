# End-to-End AI Computer Vision Pipeline with Deep Learning and MLOps

## Overview

This project implements an end-to-end computer vision pipeline for cervical cell classification using deep learning and machine learning, with a focus on reproducibility and progressive MLOps integration.

The project uses the **SIPaKMeD** dataset and follows a modular workflow in which a pretrained CNN is used as a feature extractor and a downstream classifier is selected and evaluated using a leakage-aware validation strategy.

The current pipeline includes:

* Group-aware partitioning of the SIPaKMeD dataset.
* Prevention of data leakage by keeping cells from the same parent image within the same subset.
* Deep feature extraction using **ResNet-50 pretrained on ImageNet**.
* Extraction of 2,048-dimensional feature representations.
* Binary classification by grouping the original five cell categories into Normal and Abnormal classes.
* Comparison of MLP, SVM-RBF, and Random Forest classifiers.
* Five-fold cross-validation using `StratifiedGroupKFold` on the development set.
* Selection of the final classifier using development-set performance only.
* Final training of the selected MLP using the complete development set.
* Independent evaluation on a previously unseen test set.
* Preservation of the trained model and feature scaler as inference artifacts.
* Implementation of a **FastAPI inference service** for model serving.
* HTTP endpoints for service health checking and image-based prediction.
* Automatic API documentation and interactive testing through Swagger UI.
* Progressive integration of MLOps components into the computer vision pipeline.

The project is designed as a portfolio implementation that demonstrates the transition from a deep learning model to a reproducible machine learning inference pipeline and progressively toward an end-to-end MLOps workflow.

---

## Objectives

The main objectives of this project are:

* Build a reproducible computer vision classification pipeline using SIPaKMeD.
* Prevent data leakage through parent-image-level data partitioning.
* Use a pretrained **ResNet-50** model as a deep feature extractor.
* Compare different downstream classifiers using the same feature representation.
* Select the final classifier using only the development data.
* Evaluate the selected model once on an independent test set.
* Preserve the model and preprocessing components required for reproducible inference.
* Expose the trained model through an API-based inference service.
* Prepare the model for containerization and subsequent MLOps integration.
* Extend the classification pipeline toward an end-to-end MLOps workflow.

---

## Dataset

### SIPaKMeD

The project uses the **SIPaKMeD** dataset, which contains five categories of cervical cell images:

| Cell type                |
| ------------------------ |
| Superficial/Intermediate |
| Parabasal                |
| Koilocytotic             |
| Metaplastic              |
| Dyskeratotic             |

The dataset contains **4,049 cell images** distributed across **271 parent images**.

### Binary Classification

The original five-class labels are preserved during data preparation and feature extraction.

Binary relabeling is performed only during the classification stage:

| Binary class | Original categories                     |
| ------------ | --------------------------------------- |
| Normal       | Superficial/Intermediate, Parabasal     |
| Abnormal     | Koilocytotic, Metaplastic, Dyskeratotic |

This separation allows the extracted feature representation to remain reusable independently of the final classification task.

---

## Data Partitioning

The dataset was partitioned at the **parent-image level** to prevent data leakage.

Since multiple cell patches can originate from the same parent image, assigning individual cells independently to different subsets could result in visually related samples appearing in both development and test data.

A `GroupShuffleSplit` strategy was therefore used, with the parent image identifier serving as the grouping variable.

### Final Partition

| Subset           | Cell images | Parent images |  Proportion |
| ---------------- | ----------: | ------------: | ----------: |
| Development      |       3,379 |           230 |      83.45% |
| Independent Test |         670 |            41 |      16.55% |
| **Total**        |   **4,049** |       **271** | **100.00%** |

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

| Information                    | Description                      |
| ------------------------------ | -------------------------------- |
| `filepath`                     | Original image path              |
| `label`                        | Original five-class label        |
| `group_id`                     | Parent-image identifier          |
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

* MLP
* SVM with RBF kernel
* Random Forest

The development set was evaluated using **5-fold StratifiedGroupKFold cross-validation**.

The folds preserve the parent-image grouping so that cells originating from the same parent image are not distributed across training and validation folds.

### Classifier Configurations

The main configurations used for the classifier comparison were:

| Classifier    | Key configuration                                                                      | Standardization |
| ------------- | -------------------------------------------------------------------------------------- | --------------- |
| MLP           | `128 → 64 → 1`, ReLU/Sigmoid, Adam, 50 epochs, batch size 32                           | Yes             |
| SVM-RBF       | `kernel=rbf`, `C=1`, `gamma=scale`, `random_state=42`                                  | Yes             |
| Random Forest | `n_estimators=200`, `criterion=gini`, `max_depth=None`, `random_state=42`, `n_jobs=-1` | No              |

Standardization for the MLP and SVM was performed within each cross-validation fold. The scaler was fitted only on the training portion of each fold and subsequently applied to the corresponding validation portion.

### Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)

### Cross-Validation Results

| Classifier    |            Accuracy |   Precision (macro) |      Recall (macro) |    F1-score (macro) |
| ------------- | ------------------: | ------------------: | ------------------: | ------------------: |
| **MLP**       | **0.9834 ± 0.0070** |     0.9827 ± 0.0066 | **0.9828 ± 0.0082** | **0.9827 ± 0.0073** |
| SVM-RBF       |     0.9825 ± 0.0089 | **0.9835 ± 0.0075** |     0.9802 ± 0.0109 |     0.9817 ± 0.0093 |
| Random Forest |     0.9591 ± 0.0086 |     0.9605 ± 0.0086 |     0.9547 ± 0.0099 |     0.9573 ± 0.0091 |

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

| Parameter           | Configuration        |
| ------------------- | -------------------- |
| Input features      | 2,048                |
| Hidden layer 1      | 128 neurons, ReLU    |
| Hidden layer 2      | 64 neurons, ReLU     |
| Output layer        | 1 neuron, Sigmoid    |
| Optimizer           | Adam                 |
| Loss                | Binary Cross-Entropy |
| Epochs              | 50                   |
| Batch size          | 32                   |
| Random seed         | 42                   |
| Classification task | Normal vs. Abnormal  |

After classifier selection, the final MLP was retrained using **all 3,379 development samples**.

The scaler was fitted using the complete development set and subsequently applied to the independent test set.

---

## Final Evaluation on the Independent Test Set

The independent test set contains 670 cell images and was not used during classifier selection.

The final model was evaluated on TEST **once**, after the classifier and configuration had already been selected using the development data.

### Results

| Metric            |       TEST |
| ----------------- | ---------: |
| Accuracy          | **0.9806** |
| Precision (macro) |     0.9830 |
| Recall (macro)    |     0.9758 |
| Specificity       |     0.9563 |
| F1-score (macro)  |     0.9792 |

### Confusion Matrix Components

| Component | Value |
| --------- | ----: |
| TN        |   241 |
| FP        |    11 |
| FN        |     2 |
| TP        |   416 |

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

`final_model.keras` contains the trained MLP, while `scaler.joblib` stores the `StandardScaler` fitted during final training.

Both artifacts are required to reproduce the preprocessing and prediction steps during inference.

---

# FastAPI Inference Service

The trained classification pipeline is exposed through a REST API using **FastAPI**.

The purpose of this stage is to transform the previously validated machine learning model into a reusable inference service that can receive an image through HTTP and return a prediction.

The API separates the HTTP interface from the internal machine learning inference logic.

The architecture is:

```text
Client
  ↓
POST /predict
  ↓
FastAPI
  ↓
Temporary image file
  ↓
src/inference.py
  ↓
ResNet-50
  ↓
2048-dimensional features
  ↓
StandardScaler
  ↓
Final MLP
  ↓
Prediction
  ↓
JSON response
```

This separation allows other applications, such as web or mobile clients, to consume the model through the API without directly accessing or modifying the internal ML pipeline.

### Inference Logic

The machine learning inference logic is implemented in:

```text
src/inference.py
```

This module is responsible for:

* Loading the pretrained ResNet-50 feature extractor.
* Loading the trained MLP classifier.
* Loading the fitted `StandardScaler`.
* Receiving an image path.
* Converting the image to RGB.
* Resizing the image to `224 × 224`.
* Applying the ResNet-50 preprocessing function.
* Extracting the 2,048-dimensional feature vector.
* Applying the trained scaler.
* Generating the MLP probability.
* Converting the probability into the Normal/Abnormal prediction.

The model components are loaded once when the inference module is initialized rather than being reloaded for every prediction request.

### API Logic

The API implementation is located in:

```text
api/main.py
```

This module is responsible for the HTTP layer rather than the internal ML logic.

The API currently exposes two endpoints.

#### Health Endpoint

```text
GET /health
```

This endpoint verifies that the API service is running.

Example response:

```json
{
  "status": "ok"
}
```

#### Prediction Endpoint

```text
POST /predict
```

This endpoint receives an image using `multipart/form-data` and returns the classification generated by the trained model.

Example response:

```json
{
  "prediction": "Abnormal",
  "class_id": 1,
  "probability": 0.9821
}
```

The `probability` represents the model's predicted probability for the Abnormal class.

### Temporary File Handling

Uploaded images are temporarily written to disk so that they can be passed to the existing inference function, which operates on an image path.

The workflow is:

```text
Uploaded image
      ↓
FastAPI receives file
      ↓
Temporary file created
      ↓
predict(temp_path)
      ↓
Prediction returned
      ↓
Temporary file deleted
```

The original image selected by the client is not modified or moved.

Temporary files are removed after inference using a `finally` block to ensure cleanup even if an error occurs during prediction.

### Input Validation

The `/predict` endpoint performs basic validation of the uploaded file before running inference.

Files whose declared MIME type is not an image are rejected with an HTTP `400 Bad Request` response.

Example:

```json
{
  "detail": "The uploaded file must be an image."
}
```

This prevents invalid file types from being passed unnecessarily to the ML inference pipeline.

### Interactive API Documentation

FastAPI automatically generates an OpenAPI specification and an interactive **Swagger UI** interface.

During development, the API can be accessed through:

```text
http://127.0.0.1:8000/docs
```

Swagger UI provides an interactive interface for:

* inspecting available endpoints;
* uploading an image to `/predict`;
* executing requests;
* inspecting HTTP responses;
* reviewing the generated API contract.

Swagger UI is used as a development and testing interface. It is not part of the internal machine learning inference logic.

A custom web or mobile application could consume the same API without using Swagger.

### Running the API Locally

The API is served during development using **Uvicorn**:

```bash
uvicorn api.main:app --reload
```

The `--reload` option is used during development so that changes to the source code automatically restart the service.

This behavior is a development convenience and is separate from the production deployment strategy that will be addressed in subsequent MLOps stages.

### Dependencies for the API Layer

The API layer uses:

| Dependency       | Role                                            |
| ---------------- | ----------------------------------------------- |
| FastAPI          | API framework and endpoint definition           |
| Uvicorn          | ASGI server used to run the FastAPI application |
| python-multipart | Multipart/form-data file upload support         |
| Pillow           | Image loading and preprocessing support         |

The API was tested locally through Swagger UI using SIPaKMeD images.

The `/predict` endpoint was verified with both:

* a valid `.bmp` image, producing a successful prediction;
* a non-image file, correctly returning HTTP `400 Bad Request`.

---

## Current Project Structure

```text
project-04-ai-computer-vision-mlops/
├── api/
│   └── main.py
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
│   ├── train_final_classifier.py
│   └── inference.py
├── .gitignore
├── README.md
└── requirements.txt
```

Original medical images, generated feature files, trained model artifacts, and other local data are not intended to be uploaded to the public repository unless explicitly required for deployment or reproducibility.

---

## MLOps Roadmap

The project is being developed progressively toward an end-to-end MLOps workflow.

### Completed

* [x] Dataset preparation
* [x] Parent-image-level data partitioning
* [x] Data leakage prevention
* [x] ResNet-50 feature extraction
* [x] Classifier comparison
* [x] Model selection using development data
* [x] Final MLP training
* [x] Independent TEST evaluation
* [x] Model and scaler artifact generation
* [x] FastAPI inference service
* [x] `/health` and `/predict` endpoints
* [x] Image upload handling
* [x] Basic input validation
* [x] Interactive API testing through Swagger UI

### Planned

* [ ] Docker containerization
* [ ] MLflow experiment tracking and model management
* [ ] CI/CD automation
* [ ] Basic inference monitoring

The MLOps components will be added incrementally while preserving the same trained model and preprocessing pipeline.

---

## Technologies

| Technology         | Role                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| Python             | Main programming language                                            |
| TensorFlow / Keras | ResNet-50 feature extraction and MLP                                 |
| scikit-learn       | Data splitting, cross-validation, scaling, and classical classifiers |
| Pandas             | Dataset and feature management                                       |
| NumPy              | Numerical computation                                                |
| FastAPI            | ML inference API                                                     |
| Uvicorn            | ASGI server for local API execution                                  |
| python-multipart   | Multipart file upload support                                        |
| Pillow             | Image loading and preprocessing                                      |
| Git                | Version control                                                      |
| Docker             | Planned containerization                                             |
| MLflow             | Planned experiment tracking and model management                     |
| GitHub Actions     | Planned CI/CD                                                        |

---

## Status

**Current status: Classification pipeline and FastAPI inference service completed.**

The project has progressed from SIPaKMeD data preparation and deep feature extraction to classifier selection, independent final evaluation, and deployment of the trained model through a local FastAPI inference service.

The current API successfully receives image files, executes the complete ResNet-50 → scaler → MLP inference pipeline, returns a JSON prediction, and performs basic input validation and temporary-file cleanup.

The next development stage is **Docker containerization**, which will package the inference service and its software environment into a reproducible container.

---

## Author

**Alejandro Reyes Morales**

AI / Computer Vision / Machine Learning

* LinkedIn: Alejandro Reyes Morales
* GitHub: Alejandro Reyes Morales
