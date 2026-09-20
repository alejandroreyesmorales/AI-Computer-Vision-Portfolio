
from pathlib import Path
from time import time

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


# ============================================================
# CONFIGURATION
# ============================================================

# Local path of the SIPaKMeD dataset
DATA_PATH = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\semestre 4\clasficadores RMIB\BDSIPKMED33.csv"
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results" / "sipakmed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training configuration
DATA_FRACTION = 1.0
EPOCHS = 20
BATCH_SIZE = 2048
RANDOM_SEED = 42


# Output files
MODEL_PATH = RESULTS_DIR / "mlp33_sipakmed_full.keras"
HISTORY_PATH = RESULTS_DIR / "mlp33_sipakmed_history.npz"
METRICS_PATH = RESULTS_DIR / "mlp33_sipakmed_full_metrics.txt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# DATA LOADING
# ============================================================

def load_balanced_sample(
    data_path,
    fraction=1.0,
    random_seed=42
):
    """
    Load the SIPaKMeD dataset and separate features
    from the target labels.

    Expected structure:
    - 33 feature columns
    - 1 final label column

    The dataset is balanced by class.
    """

    print("\nLoading SIPaKMeD dataset...")

    start_time = time()

    data = pd.read_csv(data_path)

    print(f"Original dataset shape: {data.shape}")

    # Optional class-balanced sampling
    if fraction < 1.0:

        label_column = data.columns[-1]

        data = (
            data.groupby(
                label_column,
                group_keys=False
            )
            .apply(
                lambda group: group.sample(
                    frac=fraction,
                    random_state=random_seed
                )
            )
            .reset_index(drop=True)
        )

    # Shuffle the dataset
    data = data.sample(
        frac=1.0,
        random_state=random_seed
    ).reset_index(drop=True)

    # Features: first 33 columns
    X = data.iloc[:, :-1].astype(
        np.float32
    ).values

    # Labels: final column
    y = data.iloc[:, -1].astype(
        np.int32
    ).values

    print(f"Selected dataset shape: {data.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    print(
        f"Class distribution:\n"
        f"{pd.Series(y).value_counts().sort_index()}"
    )

    print(
        f"Loading time: {time() - start_time:.2f} seconds"
    )

    return X, y


# ============================================================
# MODEL
# ============================================================

def create_mlp_model(input_features=33):
    """
    Create the MLP classifier.

    Architecture:
    Input: 33 features
    Hidden layer 1: 128 neurons
    Hidden layer 2: 64 neurons
    Output: 2 classes
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(input_features,)
            ),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                2,
                activation="softmax"
            )
        ],
        name="MLP33_SIPaKMeD"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate binary classification metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# TRAINING
# ============================================================

def train_model(X_train, y_train):
    """
    Train the MLP model and save its training history.
    """

    print("\nCreating MLP model...")

    model = create_mlp_model(
        input_features=X_train.shape[1]
    )

    model.summary()

    print("\nStarting training...")

    training_start = time()

    history = model.fit(
        X_train,
        y_train,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        validation_split=0.20,

        shuffle=True,

        verbose=1
    )

    training_time = time() - training_start

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    # Save trained model
    model.save(MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

    # Save training history
    np.savez(
        HISTORY_PATH,
        accuracy=np.array(
            history.history["accuracy"]
        ),
        val_accuracy=np.array(
            history.history["val_accuracy"]
        ),
        loss=np.array(
            history.history["loss"]
        ),
        val_loss=np.array(
            history.history["val_loss"]
        )
    )

    print(
        f"Training history saved to: "
        f"{HISTORY_PATH}"
    )

    return model, history, training_time


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the independent test set.
    """

    print("\nEvaluating model...")

    evaluation_start = time()

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # Generate class predictions
    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    evaluation_time = time() - evaluation_start

    metrics = calculate_metrics(
        y_test,
        y_pred
    )

    metrics["test_loss"] = test_loss
    metrics["test_accuracy_keras"] = test_accuracy
    metrics["evaluation_time_seconds"] = evaluation_time

    print(
        f"Evaluation time: "
        f"{evaluation_time:.2f} seconds"
    )

    return metrics


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics,
    training_time,
    total_time
):
    """
    Save the evaluation metrics and execution times
    to a text file.
    """

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SIPaKMeD MLP33 CLASSIFICATION RESULTS\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(
            f"Dataset: SIPaKMeD\n"
        )

        file.write(
            f"Data fraction: {DATA_FRACTION}\n"
        )

        file.write(
            f"Epochs: {EPOCHS}\n"
        )

        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        file.write(
            "Architecture: 33 -> 128 -> 64 -> 2\n\n"
        )

        for metric_name, metric_value in metrics.items():

            file.write(
                f"{metric_name}: {metric_value}\n"
            )

        file.write(
            f"\ntraining_time_seconds: "
            f"{training_time}\n"
        )

        file.write(
            f"total_time_seconds: "
            f"{total_time}\n"
        )

    print(
        f"Metrics saved to: {METRICS_PATH}"
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    total_start = time()

    print("=" * 60)
    print("SIPAKMED MLP33 CLASSIFICATION EXPERIMENT")
    print("=" * 60)

    # Load the complete SIPaKMeD dataset
    X, y = load_balanced_sample(
        DATA_PATH,
        fraction=DATA_FRACTION,
        random_seed=RANDOM_SEED
    )

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED
    )

    print(
        f"\nTraining samples: {len(y_train)}"
    )

    print(
        f"Testing samples: {len(y_test)}"
    )

    # Train model
    model, history, training_time = train_model(
        X_train,
        y_train
    )

    # Evaluate model
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    total_time = time() - total_start

    # Display metrics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for metric_name, metric_value in metrics.items():

        print(
            f"{metric_name}: {metric_value}"
        )

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Total execution time: "
        f"{total_time:.2f} seconds"
    )

    # Save metrics
    save_metrics(
        metrics,
        training_time,
        total_time
    )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()