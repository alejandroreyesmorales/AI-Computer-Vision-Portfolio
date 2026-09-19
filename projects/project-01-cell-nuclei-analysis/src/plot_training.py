
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_training_history():

    project_root = Path(__file__).resolve().parents[1]

    history_path = (
        project_root
        / "results"
        / "histories"
        / "history_mlp_128_64.npz"
    )

    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = np.load(history_path)

    epochs = range(1, len(history["loss"]) + 1)

    # Accuracy plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_path = figures_dir / "accuracy_curve.png"
    plt.savefig(accuracy_path, dpi=300)
    plt.show()

    # Loss plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = figures_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    plt.show()

    print(f"Accuracy figure saved to: {accuracy_path}")
    print(f"Loss figure saved to: {loss_path}")


if __name__ == "__main__":
    plot_training_history()