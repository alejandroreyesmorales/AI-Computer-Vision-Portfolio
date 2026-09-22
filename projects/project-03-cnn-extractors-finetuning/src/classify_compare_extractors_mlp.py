
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_SPLITS = 5
EPOCHS = 30
BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

RESNET_FILE = (
    RESULTS_DIR
    / "resnet101_imagenet"
    / "resnet101_imagenet_dev_features.csv"
)

MOBILENET_FILE = (
    RESULTS_DIR
    / "mobilenetv2_imagenet"
    / "mobilenetv2_imagenet_dev_features.csv"
)

OUTPUT_DIR = RESULTS_DIR / "mlp_extractor_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SUMMARY = OUTPUT_DIR / "mlp_extractor_comparison_5fold.csv"
OUTPUT_FOLD_RESULTS = OUTPUT_DIR / "mlp_extractor_comparison_fold_results.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_features(csv_path):
    df = pd.read_csv(csv_path)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {csv_path}")

    if "group_id" not in df.columns:
        raise ValueError(f"'group_id' column not found in {csv_path}")

    X = df[feature_columns].values.astype(np.float32)
    y = df["label"].values
    groups = df["group_id"].values

    return X, y, groups


# ============================================================
# LABEL MAPPING
# ============================================================

def normalize_label(label):
    label = str(label).lower().strip()

    if "superficial" in label or "intermediate" in label:
        return "Superficial_Intermediate"

    if "parabasal" in label:
        return "Parabasal"

    if "koilocytotic" in label:
        return "Koilocytotic"

    if "metaplastic" in label:
        return "Metaplastic"

    if "dyskeratotic" in label:
        return "Dyskeratotic"

    raise ValueError(f"Unknown label: {label}")


def encode_labels(y):
    normalized_labels = np.array(
        [normalize_label(label) for label in y]
    )

    class_order = [
        "Superficial_Intermediate",
        "Parabasal",
        "Koilocytotic",
        "Metaplastic",
        "Dyskeratotic",
    ]

    label_to_integer = {
        label: index
        for index, label in enumerate(class_order)
    }

    y_encoded = np.array(
        [label_to_integer[label] for label in normalized_labels],
        dtype=np.int32,
    )

    return y_encoded, class_order


# ============================================================
# MLP MODEL
# ============================================================

def build_mlp(input_dim, num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),

            tf.keras.layers.Dense(
                128,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# SPECIFICITY
# ============================================================

def calculate_macro_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )

    specificities = []

    for class_index in range(num_classes):
        true_positive = cm[class_index, class_index]

        false_negative = (
            np.sum(cm[class_index, :]) - true_positive
        )

        false_positive = (
            np.sum(cm[:, class_index]) - true_positive
        )

        true_negative = (
            np.sum(cm)
            - true_positive
            - false_negative
            - false_positive
        )

        denominator = true_negative + false_positive

        if denominator == 0:
            specificity = 0.0
        else:
            specificity = true_negative / denominator

        specificities.append(specificity)

    return float(np.mean(specificities))


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, num_classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "specificity_macro": calculate_macro_specificity(
            y_true,
            y_pred,
            num_classes,
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_extractor(
    extractor_name,
    csv_path,
    folds,
):
    print("\n" + "=" * 70)
    print(f"EXTRACTOR: {extractor_name}")
    print(f"FILE: {csv_path}")
    print("=" * 70)

    X, y_original, groups = load_features(csv_path)
    y, class_names = encode_labels(y_original)

    num_classes = len(class_names)
    input_dim = X.shape[1]

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {input_dim}")
    print(f"Classes: {num_classes}")

    fold_results = []

    for fold_number, (train_index, validation_index) in enumerate(
    folds,
    start=1,
):
        print(f"\nFold {fold_number}/{N_SPLITS}")

        set_seed(SEED + fold_number)

        X_train = X[train_index]
        X_validation = X[validation_index]

        y_train = y[train_index]
        y_validation = y[validation_index]

        # Standardization fitted only on training data
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_validation_scaled = scaler.transform(X_validation)

        model = build_mlp(
            input_dim=input_dim,
            num_classes=num_classes,
        )

        model.fit(
            X_train_scaled,
            y_train,
            validation_data=(
                X_validation_scaled,
                y_validation,
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        probabilities = model.predict(
            X_validation_scaled,
            verbose=0,
        )

        y_pred = np.argmax(
            probabilities,
            axis=1,
        )

        metrics = calculate_metrics(
            y_validation,
            y_pred,
            num_classes,
        )

        metrics["extractor"] = extractor_name
        metrics["fold"] = fold_number

        fold_results.append(metrics)

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision_macro']:.4f} | "
            f"Recall: {metrics['recall_macro']:.4f} | "
            f"Specificity: {metrics['specificity_macro']:.4f} | "
            f"F1: {metrics['f1_macro']:.4f}"
        )

        tf.keras.backend.clear_session()

    return pd.DataFrame(fold_results)


# ============================================================
# SUMMARY
# ============================================================

def create_summary(fold_results):
    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "specificity_macro",
        "f1_macro",
    ]

    summary_rows = []

    for extractor_name, group in fold_results.groupby(
        "extractor"
    ):
        row = {
            "extractor": extractor_name,
            "folds": len(group),
        }

        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(
                ddof=1
            )

            row[f"{metric}_mean_std"] = (
                f"{group[metric].mean():.4f} ± "
                f"{group[metric].std(ddof=1):.4f}"
            )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    print("Loading datasets...")

    X_resnet, y_resnet_original, groups_resnet = load_features(
        RESNET_FILE
    )

    X_mobilenet, y_mobilenet_original, groups_mobilenet = (
        load_features(MOBILENET_FILE)
    )

    y_resnet, class_names_resnet = encode_labels(
        y_resnet_original
    )

    y_mobilenet, class_names_mobilenet = encode_labels(
        y_mobilenet_original
    )

    if not np.array_equal(y_resnet, y_mobilenet):
        raise ValueError(
            "The label order is not identical between extractors."
        )

    if not np.array_equal(groups_resnet, groups_mobilenet):
        raise ValueError(
            "The group order is not identical between extractors."
        )

    print("Creating common StratifiedGroupKFold splits...")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(
        splitter.split(
            X_resnet,
            y_resnet,
            groups_resnet,
        )
    )

    print("Evaluating ResNet-101...")

    resnet_results = evaluate_extractor(
        extractor_name="ResNet-101",
        csv_path=RESNET_FILE,
        folds=folds,
    )

    print("Evaluating MobileNetV2...")

    mobilenet_results = evaluate_extractor(
        extractor_name="MobileNetV2",
        csv_path=MOBILENET_FILE,
        folds=folds,
    )

    fold_results = pd.concat(
        [
            resnet_results,
            mobilenet_results,
        ],
        ignore_index=True,
    )

    summary = create_summary(fold_results)

    fold_results.to_csv(
        OUTPUT_FOLD_RESULTS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary[
            [
                "extractor",
                "accuracy_mean_std",
                "precision_macro_mean_std",
                "recall_macro_mean_std",
                "specificity_macro_mean_std",
                "f1_macro_mean_std",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:")
    print(OUTPUT_FOLD_RESULTS)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()