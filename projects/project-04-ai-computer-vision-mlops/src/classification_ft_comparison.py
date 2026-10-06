import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

FT_FEATURES = {
    "FT-A": "data/features/dev_features_resnet50_ft_a.csv",
    "FT-B": "data/features/dev_features_resnet50_ft_b.csv",
}

RESULTS_DIR = "results/ft_comparison"

RANDOM_STATE = 42
N_SPLITS = 5

EPOCHS = 20
BATCH_SIZE = 32

THRESHOLD = 0.5

MLFLOW_EXPERIMENT = "MLP_FT_Comparison"


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "im_Superficial-Intermediate cropped": 0,
    "im_Parabasal cropped": 0,
    "im_Koilocytotic cropped": 1,
    "im_Metaplastic cropped": 1,
    "im_Dyskeratotic cropped": 1,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

tf.keras.utils.set_random_seed(RANDOM_STATE)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(csv_path, ft_name):
    """
    Load extracted features and convert the original
    5-class labels into binary labels.

    Normal:
        Superficial/Intermediate
        Parabasal

    Abnormal:
        Koilocytotic
        Metaplastic
        Dyskeratotic
    """

    print()
    print("Loading features:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Identify feature columns
    # --------------------------------------------------------
    #
    # FT-A:
    # feature_0, feature_1, ..., feature_2047
    #
    # FT-B:
    # 0, 1, ..., 2047
    #
    # The recovery extraction script used numeric column names
    # for FT-B, so we support both formats.
    # --------------------------------------------------------

    feature_columns = [
        col
        for col in df.columns
        if str(col).startswith("feature_")
    ]

    # FT-B fallback:
    # numeric feature names: 0, 1, ..., 2047
    if len(feature_columns) == 0:

        feature_columns = [
            col
            for col in df.columns
            if str(col).isdigit()
        ]

    # --------------------------------------------------------
    # Validate number of features
    # --------------------------------------------------------

    if len(feature_columns) != 2048:
        raise ValueError(
            f"{ft_name}: Expected 2048 features, "
            f"but found {len(feature_columns)}."
        )

    print(f"Detected feature columns: {len(feature_columns)}")

    # --------------------------------------------------------
    # Binary labels
    # --------------------------------------------------------

    binary_labels = df["label"].map(LABEL_MAP)

    if binary_labels.isna().any():
        unknown_labels = df.loc[
            binary_labels.isna(),
            "label"
        ].unique()

        raise ValueError(
            f"Unknown labels found in {ft_name}: "
            f"{unknown_labels}"
        )

    # Convert to integer
    y = binary_labels.astype(int).to_numpy()

    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    X = df[feature_columns].to_numpy(dtype=np.float32)

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    groups = df["group_id"].to_numpy()

    print(f"Samples: {len(df)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Binary labels shape: {y.shape}")
    print(f"Unique groups: {len(np.unique(groups))}")

    print(
        "Class distribution:",
        {
            0: int(np.sum(y == 0)),
            1: int(np.sum(y == 1)),
        }
    )

    return df, X, y, groups


# ============================================================
# BUILD MLP
# ============================================================

def build_mlp(input_dim=2048):
    """
    MLP architecture:

        2048 input features
              |
        Dense(128, ReLU)
              |
        Dense(64, ReLU)
              |
        Dense(1, Sigmoid)
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid"
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),

        loss=tf.keras.losses.BinaryCrossentropy(),

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


# ============================================================
# METRICS
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# CREATE COMMON FOLDS
# ============================================================

def create_folds(X, y, groups):

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(
        cv.split(
            X,
            y,
            groups
        )
    )

    return folds


# ============================================================
# EVALUATE ONE FINE-TUNED FEATURE SET
# ============================================================

def evaluate_ft(
    ft_name,
    X,
    y,
    groups,
    folds
):

    print()
    print("=" * 70)
    print(f"EVALUATING {ft_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"{ft_name}_MLP"
    ):

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "feature_extractor",
            ft_name
        )

        mlflow.log_param(
            "classifier",
            "MLP"
        )

        mlflow.log_param(
            "input_features",
            2048
        )

        mlflow.log_param(
            "hidden_layer_1",
            128
        )

        mlflow.log_param(
            "hidden_layer_2",
            64
        )

        mlflow.log_param(
            "output_layer",
            "1_sigmoid"
        )

        mlflow.log_param(
            "epochs",
            EPOCHS
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE
        )

        mlflow.log_param(
            "n_splits",
            N_SPLITS
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "threshold",
            THRESHOLD
        )

        # ----------------------------------------------------
        # Store fold results
        # ----------------------------------------------------

        fold_results = []

        # ----------------------------------------------------
        # Cross-validation
        # ----------------------------------------------------

        for fold_number, (
            train_idx,
            val_idx
        ) in enumerate(folds, start=1):

            print()
            print(
                f"{ft_name} - Fold "
                f"{fold_number}/{N_SPLITS}"
            )

            # ------------------------------------------------
            # Split data
            # ------------------------------------------------

            X_train = X[train_idx]
            X_val = X[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            groups_train = groups[train_idx]
            groups_val = groups[val_idx]

            # ------------------------------------------------
            # Verify group integrity
            # ------------------------------------------------

            train_groups = set(
                groups_train
            )

            val_groups = set(
                groups_val
            )

            overlap = train_groups.intersection(
                val_groups
            )

            if overlap:
                raise ValueError(
                    f"Group leakage detected in "
                    f"{ft_name}, fold {fold_number}: "
                    f"{len(overlap)} overlapping groups."
                )

            # ------------------------------------------------
            # Scale features
            #
            # IMPORTANT:
            # scaler is fitted only on training data.
            # ------------------------------------------------

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_val_scaled = scaler.transform(
                X_val
            )

            # ------------------------------------------------
            # Fresh MLP for each fold
            # ------------------------------------------------

            tf.keras.backend.clear_session()

            tf.keras.utils.set_random_seed(
                RANDOM_STATE + fold_number
            )

            model = build_mlp(
                input_dim=2048
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            history = model.fit(
                X_train_scaled,
                y_train,

                validation_data=(
                    X_val_scaled,
                    y_val
                ),

                epochs=EPOCHS,
                batch_size=BATCH_SIZE,

                verbose=0,

                shuffle=True
            )

            # ------------------------------------------------
            # Predict validation set
            # ------------------------------------------------

            probabilities = model.predict(
                X_val_scaled,
                verbose=0
            ).ravel()

            predictions = (
                probabilities >= THRESHOLD
            ).astype(int)

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                y_val,
                predictions
            )

            print(
                f"Accuracy : {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f}"
            )

            print(
                f"Recall   : {metrics['recall']:.4f}"
            )

            print(
                f"F1       : {metrics['f1']:.4f}"
            )

            # ------------------------------------------------
            # Save fold results
            # ------------------------------------------------

            fold_results.append(
                {
                    "feature_extractor": ft_name,
                    "fold": fold_number,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "train_samples": len(train_idx),
                    "validation_samples": len(val_idx),
                    "train_groups": len(
                        np.unique(groups_train)
                    ),
                    "validation_groups": len(
                        np.unique(groups_val)
                    ),
                }
            )

            # ------------------------------------------------
            # Free memory
            # ------------------------------------------------

            del model
            del scaler

            tf.keras.backend.clear_session()

        # ----------------------------------------------------
        # Fold results DataFrame
        # ----------------------------------------------------

        fold_df = pd.DataFrame(
            fold_results
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = {
            "feature_extractor": ft_name,

            "accuracy_mean":
                fold_df["accuracy"].mean(),

            "accuracy_std":
                fold_df["accuracy"].std(),

            "precision_mean":
                fold_df["precision"].mean(),

            "precision_std":
                fold_df["precision"].std(),

            "recall_mean":
                fold_df["recall"].mean(),

            "recall_std":
                fold_df["recall"].std(),

            "f1_mean":
                fold_df["f1"].mean(),

            "f1_std":
                fold_df["f1"].std(),
        }

        # ----------------------------------------------------
        # Log summary metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy_mean",
            summary["accuracy_mean"]
        )

        mlflow.log_metric(
            "accuracy_std",
            summary["accuracy_std"]
        )

        mlflow.log_metric(
            "precision_mean",
            summary["precision_mean"]
        )

        mlflow.log_metric(
            "precision_std",
            summary["precision_std"]
        )

        mlflow.log_metric(
            "recall_mean",
            summary["recall_mean"]
        )

        mlflow.log_metric(
            "recall_std",
            summary["recall_std"]
        )

        mlflow.log_metric(
            "f1_mean",
            summary["f1_mean"]
        )

        mlflow.log_metric(
            "f1_std",
            summary["f1_std"]
        )

        # ----------------------------------------------------
        # Save fold results
        # ----------------------------------------------------

        os.makedirs(
            RESULTS_DIR,
            exist_ok=True
        )

        fold_path = os.path.join(
            RESULTS_DIR,
            f"{ft_name.lower().replace('-', '_')}_fold_results.csv"
        )

        fold_df.to_csv(
            fold_path,
            index=False
        )

        mlflow.log_artifact(
            fold_path
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print()
        print(
            f"{ft_name} SUMMARY"
        )

        print(
            f"Accuracy : "
            f"{summary['accuracy_mean']:.4f} "
            f"± {summary['accuracy_std']:.4f}"
        )

        print(
            f"Precision: "
            f"{summary['precision_mean']:.4f} "
            f"± {summary['precision_std']:.4f}"
        )

        print(
            f"Recall   : "
            f"{summary['recall_mean']:.4f} "
            f"± {summary['recall_std']:.4f}"
        )

        print(
            f"F1       : "
            f"{summary['f1_mean']:.4f} "
            f"± {summary['f1_std']:.4f}"
        )

        return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "FT-A vs FT-B - "
        "MLP FEATURE EXTRACTOR COMPARISON"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    tracking_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print()
    print("MLflow tracking URI:")
    print(tracking_uri)

    # --------------------------------------------------------
    # Load FT-A
    # --------------------------------------------------------

    (
        df_a,
        X_a,
        y_a,
        groups_a
    ) = load_features(
        FT_FEATURES["FT-A"],
        "FT-A"
    )

    # --------------------------------------------------------
    # Load FT-B
    # --------------------------------------------------------

    (
        df_b,
        X_b,
        y_b,
        groups_b
    ) = load_features(
        FT_FEATURES["FT-B"],
        "FT-B"
    )

    # --------------------------------------------------------
    # Verify datasets are aligned
    # --------------------------------------------------------

    print()
    print("Checking FT-A / FT-B consistency...")

    if len(df_a) != len(df_b):
        raise ValueError(
            "FT-A and FT-B have different "
            "numbers of samples."
        )

    if not np.array_equal(
        df_a["filepath"].to_numpy(),
        df_b["filepath"].to_numpy()
    ):
        raise ValueError(
            "FT-A and FT-B filepath order "
            "does not match."
        )

    if not np.array_equal(
        y_a,
        y_b
    ):
        raise ValueError(
            "FT-A and FT-B binary labels "
            "do not match."
        )

    if not np.array_equal(
        groups_a,
        groups_b
    ):
        raise ValueError(
            "FT-A and FT-B group assignments "
            "do not match."
        )

    print("FT-A and FT-B are aligned.")

    # --------------------------------------------------------
    # Create COMMON folds
    # --------------------------------------------------------
    #
    # This is important:
    #
    # Both feature extractors are evaluated using exactly
    # the same train/validation partitions.
    #
    # Therefore the comparison is paired and fair.
    # --------------------------------------------------------

    folds = create_folds(
        X_a,
        y_a,
        groups_a
    )

    print()
    print(
        f"Created {len(folds)} common folds."
    )

    # --------------------------------------------------------
    # Evaluate FT-A
    # --------------------------------------------------------

    summary_a = evaluate_ft(
        ft_name="FT-A",
        X=X_a,
        y=y_a,
        groups=groups_a,
        folds=folds
    )

    # --------------------------------------------------------
    # Evaluate FT-B
    # --------------------------------------------------------

    summary_b = evaluate_ft(
        ft_name="FT-B",
        X=X_b,
        y=y_b,
        groups=groups_b,
        folds=folds
    )

    # --------------------------------------------------------
    # Combine summaries
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            summary_a,
            summary_b
        ]
    )

    # --------------------------------------------------------
    # Ranking
    #
    # Selection criteria:
    #
    # 1. Accuracy mean
    # 2. F1 mean
    # 3. Precision mean
    # 4. Recall mean
    # --------------------------------------------------------

    ranking_df = summary_df.sort_values(
        by=[
            "accuracy_mean",
            "f1_mean",
            "precision_mean",
            "recall_mean"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    ranking_df[
        "rank"
    ] = np.arange(
        1,
        len(ranking_df) + 1
    )

    # --------------------------------------------------------
    # Save comparison summary
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "ft_comparison_summary.csv"
    )

    ranking_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Print final comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL FT-A vs FT-B COMPARISON")
    print("=" * 70)

    print()

    print(
        ranking_df[
            [
                "rank",
                "feature_extractor",
                "accuracy_mean",
                "accuracy_std",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "f1_std",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Selected extractor
    # --------------------------------------------------------

    selected_ft = ranking_df.iloc[0]

    print()
    print("=" * 70)
    print("SELECTED FEATURE EXTRACTOR")
    print("=" * 70)

    print(
        f"Selected: "
        f"{selected_ft['feature_extractor']}"
    )

    print(
        f"Accuracy: "
        f"{selected_ft['accuracy_mean']:.4f}"
    )

    print(
        f"F1: "
        f"{selected_ft['f1_mean']:.4f}"
    )

    print(
        f"Precision: "
        f"{selected_ft['precision_mean']:.4f}"
    )

    print(
        f"Recall: "
        f"{selected_ft['recall_mean']:.4f}"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        summary_path
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()