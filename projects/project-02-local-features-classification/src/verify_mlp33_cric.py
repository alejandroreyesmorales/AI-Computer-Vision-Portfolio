
from pathlib import Path

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


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado"
    r"\semestre 4\clasficadores RMIB\BDcric30images.csv"
)

MODEL_PATH = PROJECT_ROOT / "results" / "mlp33_cric_full.keras"


print("Loading CRIC dataset...")

data = pd.read_csv(DATA_PATH, header=None)

X = data.iloc[:, :-1].to_numpy(dtype=np.float32)
y = data.iloc[:, -1].to_numpy(dtype=np.int32)

del data


# Reproduce the original train-test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)


print("Loading saved model...")

model = tf.keras.models.load_model(MODEL_PATH)


print("Generating predictions...")

probabilities = model.predict(
    X_test,
    batch_size=2048,
    verbose=0
)

predictions = np.argmax(probabilities, axis=1)


# Calculate evaluation metrics.
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

specificity = tn / (tn + fp)


print("\n========== LOADED MODEL RESULTS ==========")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score:    {f1:.4f}")

print("\nModel loaded and evaluated successfully.")