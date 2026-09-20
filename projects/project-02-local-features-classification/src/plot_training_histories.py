
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

CRIC_HISTORY_PATH = RESULTS_DIR / "mlp33_cric_history.npz"

SIPAKMED_HISTORY_PATH = (
    RESULTS_DIR / "sipakmed" / "mlp33_sipakmed_history.npz"
)

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_training_history(history_path, dataset_name):
    print("\n" + "=" * 60)
    print(f"PLOTTING TRAINING HISTORY: {dataset_name}")
    print("=" * 60)

    history = np.load(history_path)

    train_accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    train_loss = history["loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(train_accuracy) + 1)

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.title(f"MLP33 - {dataset_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    accuracy_path = (
        PLOTS_DIR / f"mlp33_{dataset_name.lower()}_accuracy.png"
    )

    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    print(f"Accuracy plot saved to: {accuracy_path}")

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.title(f"MLP33 - {dataset_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    loss_path = (
        PLOTS_DIR / f"mlp33_{dataset_name.lower()}_loss.png"
    )

    plt.savefig(loss_path, dpi=300)
    plt.close()

    print(f"Loss plot saved to: {loss_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("MLP33 TRAINING HISTORY PLOTS")
    print("=" * 60)

    plot_training_history(
        CRIC_HISTORY_PATH,
        "CRIC"
    )

    plot_training_history(
        SIPAKMED_HISTORY_PATH,
        "SIPaKMeD"
    )

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()