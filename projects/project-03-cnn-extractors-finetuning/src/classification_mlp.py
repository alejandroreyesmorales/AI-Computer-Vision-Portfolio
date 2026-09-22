
"""
MLP classifier for ResNet-101 extracted features.

The model is trained using DEV features and evaluated on TEST features.
Supports 5-class, 3-class, and binary classification.
"""

import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------
# Label mapping
# ---------------------------------------------------------

label_mapping = {
    "im-superficial-intermediate-cropped": 0,
    "im-parabasal-cropped": 1,
    "im-metaplastic-cropped": 2,
    "im-koilocytotic-cropped": 3,
    "im-dyskeratotic-cropped": 4,
}


def normalize_label(label) -> str:
    """Normalize class labels for consistent mapping."""

    return (
        str(label)
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )


def get_five_class_label(label) -> int:
    """Convert original class labels into five numeric classes."""

    normalized = normalize_label(label)

    label_mapping = {
        "superficial-intermediate": 0,
        "superficial/intermediate": 0,
        "parabasal": 1,
        "metaplastic": 2,
        "koilocytotic": 3,
        "dyskeratotic": 4,
    }

    if normalized not in label_mapping:
        raise ValueError(f"Unknown class label: {label}")

    return label_mapping[normalized]


def map_labels(labels: pd.Series, task: int) -> np.ndarray:
    """Map five original classes to the selected classification task."""

    five_class_labels = labels.apply(get_five_class_label).to_numpy()

    if task == 5:
        return five_class_labels

    if task == 3:
        # Normal: classes 0 and 1
        # Metaplastic: class 2
        # Abnormal: classes 3 and 4
        mapped_labels = np.select(
            [
                np.isin(five_class_labels, [0, 1]),
                five_class_labels == 2,
                np.isin(five_class_labels, [3, 4]),
            ],
            [0, 1, 2],
        )

        return mapped_labels.astype(int)

    if task == 2:
        # Normal: classes 0, 1 and 2
        # Abnormal: classes 3 and 4
        return np.where(five_class_labels <= 2, 0, 1).astype(int)

    raise ValueError("Task must be 2, 3 or 5.")


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_features(file_path: str, task: int):
    """Load extracted features and labels from a CSV file."""

    data = pd.read_csv(file_path)

    possible_label_columns = ["label", "class", "target"]
    label_column = next(
        (
            column
            for column in possible_label_columns
            if column in data.columns
        ),
        None,
    )

    if label_column is None:
        raise ValueError(
            "Could not find the label column. "
            "Expected one of: label, class, target."
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in [label_column, "group_id"]
    ]

    X = data[feature_columns].to_numpy(dtype=np.float32)
    y = map_labels(data[label_column], task)

    return X, y


# ---------------------------------------------------------
# MLP model
# ---------------------------------------------------------

def build_mlp(input_dim: int, number_of_classes: int):
    """Build the MLP classifier."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(
                number_of_classes,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def calculate_specificity(y_true, y_pred, number_of_classes):
    """Calculate macro-averaged specificity."""

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(number_of_classes)),
    )

    specificities = []

    for class_index in range(number_of_classes):
        true_positive = confusion[class_index, class_index]
        false_positive = (
            confusion[:, class_index].sum() - true_positive
        )
        false_negative = (
            confusion[class_index, :].sum() - true_positive
        )
        true_negative = (
            confusion.sum()
            - true_positive
            - false_positive
            - false_negative
        )

        denominator = true_negative + false_positive

        if denominator == 0:
            specificities.append(0.0)
        else:
            specificities.append(true_negative / denominator)

    return float(np.mean(specificities))


def evaluate_model(model, X_test, y_test, number_of_classes):
    """Evaluate the trained model using TEST data."""

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    specificity = calculate_specificity(
        y_test,
        predictions,
        number_of_classes,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score:    {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLP classification using extracted CNN features."
    )

    parser.add_argument(
        "--dev",
        type=str,
        required=True,
        help="Path to the DEV features CSV file.",
    )

    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to the TEST features CSV file.",
    )

    parser.add_argument(
        "--task",
        type=int,
        choices=[2, 3, 5],
        default=5,
        help="Classification task: 2, 3 or 5 classes.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading DEV features...")
    X_dev, y_dev = load_features(args.dev, args.task)

    print("Loading TEST features...")
    X_test, y_test = load_features(args.test, args.task)

    print(f"DEV samples: {X_dev.shape[0]}")
    print(f"TEST samples: {X_test.shape[0]}")
    print(f"Number of features: {X_dev.shape[1]}")
    print(f"Classification task: {args.task} classes")

    # Fit scaler exclusively on DEV data.
    scaler = StandardScaler()

    X_dev = scaler.fit_transform(X_dev)
    X_test = scaler.transform(X_test)

    model = build_mlp(
        input_dim=X_dev.shape[1],
        number_of_classes=args.task,
    )

    print("\nTraining MLP...")

    model.fit(
        X_dev,
        y_dev,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
    )

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        number_of_classes=args.task,
    )


if __name__ == "__main__":
    main()