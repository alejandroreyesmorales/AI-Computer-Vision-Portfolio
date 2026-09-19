
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


def train_model(
    data_path,
    epochs=20,
    batch_size=2048,
    random_seed=42
):
    """
    Train an MLP classifier using RGB features.

    Parameters
    ----------
    data_path : str or Path
        Path to the CSV dataset.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    random_seed : int
        Random seed for reproducibility.
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # Reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    print("Loading dataset...")
    data = np.loadtxt(data_path, delimiter=",")

    # Shuffle data
    rng = np.random.default_rng(random_seed)
    rng.shuffle(data)

    # Features and labels
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    print(f"Dataset shape: {data.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Train-test split
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=random_seed,
        stratify=y
    )

    # Train-validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.1,
        random_state=random_seed,
        stratify=y_train_full
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

    # MLP architecture: 3 -> 128 -> 64 -> 2
    model_mlp = Sequential([
        Dense(128, input_dim=3, activation="relu"),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax")
    ])

    model_mlp.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print("Starting training...")

    history = model_mlp.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_val, y_val)
    )

    # Output directories
    project_root = Path(__file__).resolve().parents[1]

    models_dir = project_root / "results" / "models"
    histories_dir = project_root / "results" / "histories"

    models_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / "mlp_128_64.keras"
    model_mlp.save(model_path)

    # Save training history
    history_path = histories_dir / "history_mlp_128_64.npz"

    np.savez(
        history_path,
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        accuracy=history.history["accuracy"],
        val_accuracy=history.history["val_accuracy"]
    )

    # Evaluate model
    test_loss, test_accuracy = model_mlp.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print("\nTraining completed.")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")

    return {
        "model_path": model_path,
        "history_path": history_path,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }