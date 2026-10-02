import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Configuration
# ============================================================

DEV_FEATURES_PATH = "data/features/dev_features_resnet50.csv"
RESULTS_DIR = "results/classification"
RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "model_selection_results.csv"
)

RANDOM_STATE = 42
N_SPLITS = 5


# Binary classification mapping
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
    "epochs": 50,
    "batch_size": 32,
    "random_state": RANDOM_STATE,
}


# ============================================================
# SVM configuration
# ============================================================

SVM_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "random_state": RANDOM_STATE,
}


# ============================================================
# Random Forest configuration
# ============================================================

RF_CONFIG = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# Utility functions
# ============================================================

def load_data():
    """
    Load DEV features and apply the binary label mapping.

    The original five-class labels remain untouched in the CSV.
    Binary relabeling is performed only in memory.
    """

    print("=" * 70)
    print("LOADING DEVELOPMENT FEATURES")
    print("=" * 70)

    df = pd.read_csv(DEV_FEATURES_PATH)

    print(f"Development samples: {len(df)}")
    print(f"Original classes: {df['label'].nunique()}")

    # --------------------------------------------------------
    # Binary relabeling
    # --------------------------------------------------------

    df["binary_label"] = df["label"].map(LABEL_MAP)

    if df["binary_label"].isna().any():
        unknown_labels = df.loc[
            df["binary_label"].isna(),
            "label"
        ].unique()

        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    X = df[feature_columns].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int32)
    groups = df["group_id"].values

    print(f"Feature dimension: {X.shape[1]}")
    print(f"Normal samples: {(y == 0).sum()}")
    print(f"Abnormal samples: {(y == 1).sum()}")
    print(f"Unique parent images: {len(np.unique(groups))}")

    return X, y, groups


def build_mlp():
    """
    Build the binary MLP:

        2048 -> 128 -> 64 -> 1

    Hidden layers use ReLU.
    Output layer uses Sigmoid.
    """

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


def evaluate_predictions(y_true, y_pred):
    """
    Calculate classification metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
    }


def summarize_metrics(fold_results):
    """
    Calculate mean and standard deviation across folds.
    """

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    summary = {}

    for metric in metrics:
        values = [
            result[metric]
            for result in fold_results
        ]

        summary[f"{metric}_mean"] = np.mean(values)
        summary[f"{metric}_std"] = np.std(
            values,
            ddof=1
        )

    return summary


# ============================================================
# MLP evaluation
# ============================================================

def evaluate_mlp(X, y, groups, cv):
    """
    Evaluate the Keras MLP using 5-fold grouped stratified CV.
    """

    print("\n" + "=" * 70)
    print("MODEL: MLP")
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        print(f"Fold {fold}/{N_SPLITS}")

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_val_scaled = scaler.transform(
            X_val
        )

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(
            RANDOM_STATE
        )

        # ----------------------------------------------------
        # Build and train model
        # ----------------------------------------------------

        model = build_mlp()

        model.fit(
            X_train_scaled,
            y_train,
            epochs=MLP_CONFIG["epochs"],
            batch_size=MLP_CONFIG["batch_size"],
            verbose=0,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probabilities = model.predict(
            X_val_scaled,
            verbose=0
        ).ravel()

        y_pred = (
            probabilities >= 0.5
        ).astype(int)

        metrics = evaluate_predictions(
            y_val,
            y_pred
        )

        fold_results.append(metrics)

        print(
            f"  ACC={metrics['accuracy']:.4f} | "
            f"PREC={metrics['precision']:.4f} | "
            f"REC={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

    return summarize_metrics(fold_results)


# ============================================================
# SVM evaluation
# ============================================================

def evaluate_svm(X, y, groups, cv):
    """
    Evaluate SVM-RBF using 5-fold grouped stratified CV.
    """

    print("\n" + "=" * 70)
    print("MODEL: SVM-RBF")
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        print(f"Fold {fold}/{N_SPLITS}")

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_val_scaled = scaler.transform(
            X_val
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = SVC(
            kernel=SVM_CONFIG["kernel"],
            C=SVM_CONFIG["C"],
            gamma=SVM_CONFIG["gamma"],
            random_state=SVM_CONFIG["random_state"],
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_val_scaled
        )

        metrics = evaluate_predictions(
            y_val,
            y_pred
        )

        fold_results.append(metrics)

        print(
            f"  ACC={metrics['accuracy']:.4f} | "
            f"PREC={metrics['precision']:.4f} | "
            f"REC={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

    return summarize_metrics(fold_results)


# ============================================================
# Random Forest evaluation
# ============================================================

def evaluate_random_forest(X, y, groups, cv):
    """
    Evaluate Random Forest using 5-fold grouped stratified CV.

    Scaling is not applied because tree-based models do not
    require standardized input features.
    """

    print("\n" + "=" * 70)
    print("MODEL: RANDOM FOREST")
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        print(f"Fold {fold}/{N_SPLITS}")

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=RF_CONFIG["n_estimators"],
            criterion=RF_CONFIG["criterion"],
            max_depth=RF_CONFIG["max_depth"],
            random_state=RF_CONFIG["random_state"],
            n_jobs=RF_CONFIG["n_jobs"],
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_val
        )

        metrics = evaluate_predictions(
            y_val,
            y_pred
        )

        fold_results.append(metrics)

        print(
            f"  ACC={metrics['accuracy']:.4f} | "
            f"PREC={metrics['precision']:.4f} | "
            f"REC={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

    return summarize_metrics(fold_results)


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CLASSIFIER MODEL SELECTION")
    print("=" * 70)
    print(f"Cross-validation folds: {N_SPLITS}")
    print(f"Random state: {RANDOM_STATE}")
    print("Dataset: SIPaKMeD")
    print("Task: Binary classification")
    print("Evaluation set: DEV only")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y, groups = load_data()

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Evaluate models
    # --------------------------------------------------------

    all_results = []

    # -------------------------
    # MLP
    # -------------------------

    mlp_metrics = evaluate_mlp(
        X,
        y,
        groups,
        cv
    )

    all_results.append({
        "classifier": "MLP",

        **mlp_metrics,

        "feature_extractor": "ResNet-50",
        "feature_dimension": 2048,
        "task": "Binary",
        "cv_strategy": "StratifiedGroupKFold",
        "cv_folds": N_SPLITS,
        "random_state": RANDOM_STATE,

        "parameters": json.dumps(
            MLP_CONFIG
        ),
    })

    # -------------------------
    # SVM
    # -------------------------

    svm_metrics = evaluate_svm(
        X,
        y,
        groups,
        cv
    )

    all_results.append({
        "classifier": "SVM-RBF",

        **svm_metrics,

        "feature_extractor": "ResNet-50",
        "feature_dimension": 2048,
        "task": "Binary",
        "cv_strategy": "StratifiedGroupKFold",
        "cv_folds": N_SPLITS,
        "random_state": RANDOM_STATE,

        "parameters": json.dumps(
            SVM_CONFIG
        ),
    })

    # -------------------------
    # Random Forest
    # -------------------------

    rf_metrics = evaluate_random_forest(
        X,
        y,
        groups,
        cv
    )

    all_results.append({
        "classifier": "Random Forest",

        **rf_metrics,

        "feature_extractor": "ResNet-50",
        "feature_dimension": 2048,
        "task": "Binary",
        "cv_strategy": "StratifiedGroupKFold",
        "cv_folds": N_SPLITS,
        "random_state": RANDOM_STATE,

        "parameters": json.dumps(
            RF_CONFIG
        ),
    })

    # --------------------------------------------------------
    # Create results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL SELECTION RESULTS")
    print("=" * 70)

    display_columns = [
        "classifier",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n")
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(RESULTS_PATH)

    print("\nNo final classifier was trained.")
    print("Model selection ends at this stage.")


if __name__ == "__main__":
    main()