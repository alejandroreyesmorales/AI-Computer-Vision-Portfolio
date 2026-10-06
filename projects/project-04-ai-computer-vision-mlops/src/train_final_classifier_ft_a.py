import os
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
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

DEV_FEATURES = (
    "data/features/dev_features_resnet50_ft_a.csv"
)

TEST_FEATURES = (
    "data/features/test_features_resnet50_ft_a.csv"
)

RESNET_MODEL = (
    "models/resnet50/ft-a/resnet50_ft-a.keras"
)

MODEL_DIR = "models/mlp"
RESULTS_DIR = "results/final_ft_a"

FINAL_MODEL_PATH = (
    "models/mlp/final_model_ft_a.keras"
)

SCALER_PATH = (
    "models/mlp/scaler_ft_a.joblib"
)

MLFLOW_EXPERIMENT = "Final_Model_FT_A"

RANDOM_STATE = 42

EPOCHS = 20
BATCH_SIZE = 32

THRESHOLD = 0.5


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

tf.keras.utils.set_random_seed(
    RANDOM_STATE
)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(csv_path, dataset_name):

    print()
    print("=" * 70)
    print(f"LOADING {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(csv_path)

    print(f"File: {csv_path}")
    print(f"Samples: {len(df)}")

    # --------------------------------------------------------
    # Detect feature columns
    # --------------------------------------------------------
    #
    # FT-A uses:
    # feature_0 ... feature_2047
    #
    # The fallback also supports numeric feature names.
    # --------------------------------------------------------

    feature_columns = [
        col
        for col in df.columns
        if str(col).startswith("feature_")
    ]

    if len(feature_columns) == 0:

        feature_columns = [
            col
            for col in df.columns
            if str(col).isdigit()
        ]

    if len(feature_columns) != 2048:

        raise ValueError(
            f"{dataset_name}: Expected 2048 features, "
            f"but found {len(feature_columns)}."
        )

    print(
        f"Detected feature columns: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Convert 5 original classes to binary labels
    # --------------------------------------------------------

    binary_labels = df["label"].map(
        LABEL_MAP
    )

    if binary_labels.isna().any():

        unknown_labels = df.loc[
            binary_labels.isna(),
            "label"
        ].unique()

        raise ValueError(
            f"Unknown labels found: "
            f"{unknown_labels}"
        )

    y = binary_labels.astype(
        np.int32
    ).to_numpy()

    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    X = df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    groups = df[
        "group_id"
    ].to_numpy()

    print(
        f"Feature matrix: {X.shape}"
    )

    print(
        f"Unique groups: "
        f"{len(np.unique(groups))}"
    )

    print(
        "Class distribution:"
    )

    print(
        f"  Normal   (0): "
        f"{np.sum(y == 0)}"
    )

    print(
        f"  Abnormal (1): "
        f"{np.sum(y == 1)}"
    )

    return df, X, y, groups


# ============================================================
# BUILD FINAL MLP
# ============================================================

def build_mlp():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(2048,)
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
                1,
                activation="sigmoid"
            ),
        ],
        name="MLP_FT_A"
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
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "FINAL MODEL TRAINING - FT-A + MLP"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    tracking_uri = (
        "sqlite:///mlflow.db"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print()
    print(
        "MLflow tracking URI:"
    )

    print(
        tracking_uri
    )

    # --------------------------------------------------------
    # Load DEV
    # --------------------------------------------------------

    (
        dev_df,
        X_dev,
        y_dev,
        groups_dev
    ) = load_features(
        DEV_FEATURES,
        "DEV"
    )

    # --------------------------------------------------------
    # Load TEST
    # --------------------------------------------------------

    (
        test_df,
        X_test,
        y_test,
        groups_test
    ) = load_features(
        TEST_FEATURES,
        "TEST"
    )

    # --------------------------------------------------------
    # Verify DEV / TEST separation
    # --------------------------------------------------------

    dev_groups = set(
        groups_dev
    )

    test_groups = set(
        groups_test
    )

    group_overlap = (
        dev_groups.intersection(
            test_groups
        )
    )

    if group_overlap:

        raise ValueError(
            "DATA LEAKAGE DETECTED: "
            f"{len(group_overlap)} groups "
            "appear in both DEV and TEST."
        )

    print()
    print(
        "DEV / TEST group integrity: OK"
    )

    print(
        f"DEV groups : "
        f"{len(dev_groups)}"
    )

    print(
        f"TEST groups: "
        f"{len(test_groups)}"
    )

    # --------------------------------------------------------
    # Verify feature dimensions
    # --------------------------------------------------------

    if X_dev.shape[1] != 2048:

        raise ValueError(
            "DEV must contain 2048 features."
        )

    if X_test.shape[1] != 2048:

        raise ValueError(
            "TEST must contain 2048 features."
        )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="Final_FT_A_MLP"
    ):

        # ====================================================
        # LOG PARAMETERS
        # ====================================================

        mlflow.log_param(
            "feature_extractor",
            "ResNet50_FT-A"
        )

        mlflow.log_param(
            "resnet_model",
            RESNET_MODEL
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
            "optimizer",
            "Adam"
        )

        mlflow.log_param(
            "loss",
            "BinaryCrossentropy"
        )

        mlflow.log_param(
            "scaler",
            "StandardScaler"
        )

        mlflow.log_param(
            "threshold",
            THRESHOLD
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "dev_samples",
            len(X_dev)
        )

        mlflow.log_param(
            "test_samples",
            len(X_test)
        )

        mlflow.log_param(
            "dev_groups",
            len(dev_groups)
        )

        mlflow.log_param(
            "test_groups",
            len(test_groups)
        )

        # ====================================================
        # FIT SCALER ONLY ON DEV
        # ====================================================

        print()
        print("=" * 70)
        print(
            "FITTING STANDARD SCALER"
        )
        print("=" * 70)

        scaler = StandardScaler()

        X_dev_scaled = scaler.fit_transform(
            X_dev
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        print(
            "Scaler fitted on DEV only."
        )

        print(
            f"DEV scaled shape: "
            f"{X_dev_scaled.shape}"
        )

        print(
            f"TEST scaled shape: "
            f"{X_test_scaled.shape}"
        )

        # ====================================================
        # BUILD FINAL MLP
        # ====================================================

        print()
        print("=" * 70)
        print(
            "BUILDING FINAL MLP"
        )
        print("=" * 70)

        model = build_mlp()

        model.summary()

        # ====================================================
        # TRAIN ON ALL DEV
        # ====================================================

        print()
        print("=" * 70)
        print(
            "TRAINING FINAL MLP ON ALL DEV"
        )
        print("=" * 70)

        history = model.fit(
            X_dev_scaled,
            y_dev,

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            shuffle=True,

            verbose=1
        )

        # ====================================================
        # LOG TRAINING HISTORY
        # ====================================================

        history_df = pd.DataFrame(
            history.history
        )

        history_path = os.path.join(
            RESULTS_DIR,
            "final_ft_a_training_history.csv"
        )

        history_df.to_csv(
            history_path,
            index=False
        )

        mlflow.log_artifact(
            history_path
        )

        # Log final training metrics
        if "loss" in history.history:

            mlflow.log_metric(
                "final_train_loss",
                history.history["loss"][-1]
            )

        if "accuracy" in history.history:

            mlflow.log_metric(
                "final_train_accuracy",
                history.history["accuracy"][-1]
            )

        # ====================================================
        # SAVE FINAL MLP
        # ====================================================

        print()
        print("=" * 70)
        print(
            "SAVING FINAL MLP"
        )
        print("=" * 70)

        model.save(
            FINAL_MODEL_PATH
        )

        print(
            f"Saved:"
        )

        print(
            FINAL_MODEL_PATH
        )

        # ====================================================
        # SAVE SCALER
        # ====================================================

        print()
        print(
            "Saving StandardScaler..."
        )

        joblib.dump(
            scaler,
            SCALER_PATH
        )

        print(
            f"Saved:"
        )

        print(
            SCALER_PATH
        )

        # ====================================================
        # LOG MODEL + SCALER TO MLFLOW
        # ====================================================

        mlflow.log_artifact(
            FINAL_MODEL_PATH,
            artifact_path="model"
        )

        mlflow.log_artifact(
            SCALER_PATH,
            artifact_path="preprocessing"
        )

        # ====================================================
        # FINAL TEST EVALUATION
        # ====================================================

        print()
        print("=" * 70)
        print(
            "FINAL TEST EVALUATION"
        )
        print("=" * 70)

        probabilities = model.predict(
            X_test_scaled,
            verbose=0
        ).ravel()

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        # ====================================================
        # PRINT FINAL RESULTS
        # ====================================================

        print()
        print(
            f"Accuracy    : {accuracy:.4f}"
        )

        print(
            f"Precision   : {precision:.4f}"
        )

        print(
            f"Recall      : {recall:.4f}"
        )

        print(
            f"Specificity : {specificity:.4f}"
        )

        print(
            f"F1          : {f1:.4f}"
        )

        print()
        print(
            "Confusion Matrix:"
        )

        print(
            cm
        )

        print()
        print(
            f"TN: {tn}"
        )

        print(
            f"FP: {fp}"
        )

        print(
            f"FN: {fn}"
        )

        print(
            f"TP: {tp}"
        )

        # ====================================================
        # LOG TEST METRICS TO MLFLOW
        # ====================================================

        mlflow.log_metric(
            "test_accuracy",
            accuracy
        )

        mlflow.log_metric(
            "test_precision",
            precision
        )

        mlflow.log_metric(
            "test_recall",
            recall
        )

        mlflow.log_metric(
            "test_specificity",
            specificity
        )

        mlflow.log_metric(
            "test_f1",
            f1
        )

        mlflow.log_metric(
            "test_tn",
            float(tn)
        )

        mlflow.log_metric(
            "test_fp",
            float(fp)
        )

        mlflow.log_metric(
            "test_fn",
            float(fn)
        )

        mlflow.log_metric(
            "test_tp",
            float(tp)
        )

        # ====================================================
        # SAVE FINAL RESULTS
        # ====================================================

        results = {
            "feature_extractor": "FT-A",

            "classifier": "MLP",

            "input_features": 2048,

            "hidden_layer_1": 128,

            "hidden_layer_2": 64,

            "epochs": EPOCHS,

            "batch_size": BATCH_SIZE,

            "test_accuracy": accuracy,

            "test_precision": precision,

            "test_recall": recall,

            "test_specificity": specificity,

            "test_f1": f1,

            "TN": int(tn),

            "FP": int(fp),

            "FN": int(fn),

            "TP": int(tp),
        }

        results_path = os.path.join(
            RESULTS_DIR,
            "final_ft_a_test_results.json"
        )

        with open(
            results_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=4
            )

        mlflow.log_artifact(
            results_path
        )

        # ====================================================
        # SAVE CONFUSION MATRIX
        # ====================================================

        cm_df = pd.DataFrame(
            cm,

            index=[
                "Actual_Normal",
                "Actual_Abnormal"
            ],

            columns=[
                "Predicted_Normal",
                "Predicted_Abnormal"
            ]
        )

        cm_path = os.path.join(
            RESULTS_DIR,
            "final_ft_a_confusion_matrix.csv"
        )

        cm_df.to_csv(
            cm_path
        )

        mlflow.log_artifact(
            cm_path
        )

        # ====================================================
        # LOG RESNET MODEL AS ARTIFACT
        # ====================================================

        if os.path.exists(
            RESNET_MODEL
        ):

            mlflow.log_artifact(
                RESNET_MODEL,
                artifact_path="resnet50_ft_a"
            )

        else:

            print()
            print(
                "WARNING: FT-A ResNet model "
                "was not found:"
            )

            print(
                RESNET_MODEL
            )

        # ====================================================
        # RUN SUMMARY
        # ====================================================

        run = mlflow.active_run()

        print()
        print("=" * 70)
        print(
            "MLFLOW RUN"
        )
        print("=" * 70)

        print(
            f"Run ID: {run.info.run_id}"
        )

        print(
            f"Experiment: "
            f"{MLFLOW_EXPERIMENT}"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FINAL FT-A MODEL COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Final ResNet-50 FT-A:"
    )

    print(
        RESNET_MODEL
    )

    print()
    print(
        "Final MLP:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print()
    print(
        "Final scaler:"
    )

    print(
        SCALER_PATH
    )

    print()
    print(
        "Final results:"
    )

    print(
        results_path
    )

    print()
    print(
        "MLflow experiment:"
    )

    print(
        MLFLOW_EXPERIMENT
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()