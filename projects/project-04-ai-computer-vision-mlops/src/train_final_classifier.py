import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

DEV_FEATURES_PATH = "data/features/dev_features_resnet50.csv"
TEST_FEATURES_PATH = "data/features/test_features_resnet50.csv"

RESULTS_DIR = "results/classification"
RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "final_test_results.csv"
)

MODEL_DIR = "models/mlp"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.joblib"
)

RANDOM_STATE = 42

EPOCHS = 50
BATCH_SIZE = 32


# ============================================================
# Binary classification mapping
# ============================================================

LABEL_MAP = {
    "im_Superficial-Intermediate cropped": 0,
    "im_Parabasal cropped": 0,
    "im_Koilocytotic cropped": 1,
    "im_Metaplastic cropped": 1,
    "im_Dyskeratotic cropped": 1,
}

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal",
}


# ============================================================
# MLP configuration
# ============================================================

MLP_CONFIG = {
    "architecture": "2048-128-64-1",
    "hidden_layers": "(128, 64)",
    "hidden_activation": "ReLU",
    "output_activation": "Sigmoid",
    "optimizer": "Adam",
    "loss": "BinaryCrossentropy",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "random_state": RANDOM_STATE,
}


# ============================================================
# Reproducibility
# ============================================================

tf.keras.utils.set_random_seed(RANDOM_STATE)


# ============================================================
# Data loading
# ============================================================

def load_features(path):

    df = pd.read_csv(path)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    if len(feature_columns) != 2048:
        raise ValueError(
            f"Expected 2048 features, "
            f"but found {len(feature_columns)}."
        )

    # Binary relabeling occurs only in memory.
    df["binary_label"] = df["label"].map(LABEL_MAP)

    if df["binary_label"].isna().any():

        unknown_labels = df.loc[
            df["binary_label"].isna(),
            "label"
        ].unique()

        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    X = df[feature_columns].values.astype(
        np.float32
    )

    y = df["binary_label"].values.astype(
        np.int32
    )

    return X, y


# ============================================================
# Build MLP
# ============================================================

def build_mlp():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(2048,),
            name="input_features"
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            name="dense_128"
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu",
            name="dense_64"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="binary_output"
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )

    return model


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Confusion matrix:
    #
    # [[TN, FP],
    #  [FN, TP]]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("FINAL CLASSIFIER TRAINING AND TEST EVALUATION")
    print("=" * 70)

    print("Selected classifier: MLP")
    print("Architecture: 2048-128-64-1")
    print("Task: Binary classification")
    print("Training data: ALL DEV")
    print("Evaluation data: TEST")
    print("=" * 70)


    # --------------------------------------------------------
    # Load DEV and TEST
    # --------------------------------------------------------

    print("\nLoading DEV features...")

    X_dev, y_dev = load_features(
        DEV_FEATURES_PATH
    )

    print(f"DEV samples: {len(X_dev)}")

    print("\nLoading TEST features...")

    X_test, y_test = load_features(
        TEST_FEATURES_PATH
    )

    print(f"TEST samples: {len(X_test)}")


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    print("\nFitting StandardScaler on DEV...")

    scaler = StandardScaler()

    X_dev_scaled = scaler.fit_transform(
        X_dev
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Build final MLP
    # --------------------------------------------------------

    print("\nBuilding final MLP...")

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_STATE
    )

    model = build_mlp()

    model.summary()


    # --------------------------------------------------------
    # Train on ALL DEV
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING ON ALL DEVELOPMENT DATA")
    print("=" * 70)

    model.fit(
        X_dev_scaled,
        y_dev,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    print("\nFinal model saved:")
    print(MODEL_PATH)


    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("Scaler saved:")
    print(SCALER_PATH)


    # --------------------------------------------------------
    # TEST prediction
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION ON TEST")
    print("=" * 70)

    probabilities = model.predict(
        X_test_scaled,
        verbose=0
    ).ravel()

    y_pred = (
        probabilities >= 0.5
    ).astype(int)


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\nFinal TEST results:")

    print(
        f"Accuracy:    {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:   {metrics['precision']:.4f}"
    )

    print(
        f"Recall:      {metrics['recall']:.4f}"
    )

    print(
        f"Specificity: {metrics['specificity']:.4f}"
    )

    print(
        f"F1-score:    {metrics['f1']:.4f}"
    )

    print("\nConfusion matrix components:")

    print(
        f"TN: {metrics['tn']}"
    )

    print(
        f"FP: {metrics['fp']}"
    )

    print(
        f"FN: {metrics['fn']}"
    )

    print(
        f"TP: {metrics['tp']}"
    )


    # --------------------------------------------------------
    # Create results record
    # --------------------------------------------------------

    results = {
        "classifier": "MLP",

        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],

        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],

        "feature_extractor": "ResNet-50",
        "feature_dimension": 2048,

        "task": "Binary",

        "train_samples": len(X_dev),
        "test_samples": len(X_test),

        "architecture": MLP_CONFIG["architecture"],
        "hidden_layers": MLP_CONFIG["hidden_layers"],
        "hidden_activation": MLP_CONFIG[
            "hidden_activation"
        ],
        "output_activation": MLP_CONFIG[
            "output_activation"
        ],

        "optimizer": MLP_CONFIG["optimizer"],
        "loss": MLP_CONFIG["loss"],
        "epochs": MLP_CONFIG["epochs"],
        "batch_size": MLP_CONFIG["batch_size"],
        "random_state": MLP_CONFIG["random_state"],

        "threshold": 0.5,
    }


    # --------------------------------------------------------
    # Save final results
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        [results]
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS SAVED")
    print("=" * 70)

    print(RESULTS_PATH)

    print("\nFinal classifier: MLP")
    print("TEST was used only once for final evaluation.")
    print("No model selection was performed using TEST.")


if __name__ == "__main__":
    main()