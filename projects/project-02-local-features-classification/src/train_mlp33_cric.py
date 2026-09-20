
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_balanced_sample(data_path, fraction=1.0):

    print("Loading dataset...")

    start_time = perf_counter()

    data = pd.read_csv(data_path, header=None)

    features = data.iloc[:, :-1].to_numpy(dtype=np.float32)
    labels = data.iloc[:, -1].to_numpy(dtype=np.int32)

    del data

    selected_indices = []

    # Preserve the class distribution through stratified sampling.
    rng = np.random.default_rng(SEED)

    for class_label in np.unique(labels):

        class_indices = np.where(labels == class_label)[0]
        sample_size = int(len(class_indices) * fraction)

        selected = rng.choice(
            class_indices,
            size=sample_size,
            replace=False
        )

        selected_indices.extend(selected)

    selected_indices = np.array(selected_indices)

    rng.shuffle(selected_indices)

    X = features[selected_indices]
    y = labels[selected_indices]

    print(f"Selected samples: {len(y):,}")
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index())

    elapsed_time = perf_counter() - start_time

    print(f"Loading and sampling time: {elapsed_time:.2f} seconds")

    return X, y


def calculate_metrics(y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    # Confusion matrix format: TN, FP, FN, TP.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    }


def train_model(X, y, epochs=20, batch_size=2048):

    # Separate training and testing data using stratification.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(33,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining model...")

    training_start = perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    training_time = perf_counter() - training_start

    print(f"\nTraining time: {training_time:.2f} seconds")

    print("\nEvaluating model...")

    evaluation_start = perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=batch_size,
        verbose=0
    )

    predictions = np.argmax(probabilities, axis=1)

    evaluation_time = perf_counter() - evaluation_start

    metrics = calculate_metrics(y_test, predictions)

    print(f"Evaluation time: {evaluation_time:.2f} seconds")

    print("\n========== RESULTS ==========")

    for metric_name, value in metrics.items():

        if metric_name in [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive"
        ]:
            print(f"{metric_name}: {value}")

        else:
            print(f"{metric_name}: {value:.4f}")

    # Save the trained model for future cross-dataset evaluation.
    model_path = RESULTS_DIR / "mlp33_cric_full.keras"
    model.save(model_path)

    # Save the training history.
    history_path = RESULTS_DIR / "mlp33_cric_history.npz"

    np.savez(
        history_path,
        loss=history.history["loss"],
        accuracy=history.history["accuracy"],
        val_loss=history.history["val_loss"],
        val_accuracy=history.history["val_accuracy"]
    )

    # Save metrics and execution times in a text file.
    metrics_path = RESULTS_DIR / "mlp33_cric_full_metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:

        file.write("MLP33 - CRIC DATASET\n")
        file.write("====================\n\n")

        file.write(f"Number of samples: {len(y):,}\n")
        file.write(f"Training epochs: {epochs}\n")
        file.write(f"Batch size: {batch_size}\n")
        file.write(f"Random seed: {SEED}\n\n")

        file.write("Metrics\n")
        file.write("-------\n")

        for metric_name, value in metrics.items():
            file.write(f"{metric_name}: {value}\n")

        file.write("\nExecution times\n")
        file.write("----------------\n")
        file.write(f"Training time (seconds): {training_time:.2f}\n")
        file.write(f"Evaluation time (seconds): {evaluation_time:.2f}\n")

    print("\nSaved files:")
    print(model_path)
    print(history_path)
    print(metrics_path)

    return model, history, metrics


if __name__ == "__main__":

    data_path = (
        r"C:\Users\asusf\OneDrive\Documentos\Doctorado"
        r"\semestre 4\clasficadores RMIB\BDcric30images.csv"
    )

    total_start = perf_counter()

    X, y = load_balanced_sample(
        data_path,
        fraction=1.0
    )

    model, history, metrics = train_model(
        X,
        y,
        epochs=20,
        batch_size=2048
    )

    total_time = perf_counter() - total_start

    print(f"\nTotal execution time: {total_time:.2f} seconds")