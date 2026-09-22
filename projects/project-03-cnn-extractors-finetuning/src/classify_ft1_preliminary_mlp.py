
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 101
N_SPLITS = 5
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 5

# Directorio donde están los CSV de características
DATASET_ROOT = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5"
    r"\Experimentos Extractores corregidos"
)

DESTILADO_CSV = DATASET_ROOT / "FT1_KDestilation.csv"

SUPERVISADO_CSV = (
    DATASET_ROOT / "features_FT1" / "FT1_1.csv"
)

# Directorio raíz del proyecto de GitHub
PROJECT_ROOT = Path(
    r"C:\Users\asusf\OneDrive\Documentos"
    r"\AI-Computer-Vision-Portfolio"
    r"\projects\project-03-cnn-extractors-finetuning"
)

# Directorios de salida
RESULTS_DIR = PROJECT_ROOT / "results" / "preliminary_mlp_results"
HISTORY_DIR = PROJECT_ROOT / "results" / "preliminary_mlp_history"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURACIÓN DE SEMILLAS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# MAPEO DE CLASES
# ============================================================

LABEL_MAP = {
    "im_Dyskeratotic cropped": 0,
    "im_Koilocytotic cropped": 1,
    "im_Metaplastic cropped": 2,
    "im_Parabasal cropped": 3,
    "im_Superficial-Intermediate cropped": 4,
}


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    specificities = []

    for class_index in range(NUM_CLASSES):
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        denominator = tn + fp

        if denominator == 0:
            specificity = 0.0
        else:
            specificity = tn / denominator

        specificities.append(specificity)

    return float(np.mean(specificities))


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "specificity_macro": calculate_specificity(
            y_true,
            y_pred
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
    }


# ============================================================
# MLP DESACOPLADO
# Arquitectura fija: Entrada → 128 → 64 → 5
# ============================================================

def build_mlp(input_dim):
    model = tf.keras.Sequential([
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
            NUM_CLASSES,
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# CARGA DE CARACTERÍSTICAS
# ============================================================

def load_features(csv_path):
    print("\n" + "=" * 70)
    print("CARGANDO:")
    print(csv_path)
    print("=" * 70)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {"label", "group_id"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas: {missing_columns}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in ["label", "group_id"]
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)

    unknown_labels = (
        set(df["label"].unique()) - set(LABEL_MAP.keys())
    )

    if unknown_labels:
        raise ValueError(
            f"Etiquetas desconocidas: {unknown_labels}"
        )

    y = (
        df["label"]
        .map(LABEL_MAP)
        .to_numpy(dtype=np.int32)
    )

    groups = df["group_id"].to_numpy()

    print(f"Muestras: {X.shape[0]}")
    print(f"Características de entrada: {X.shape[1]}")
    print(f"Grupos únicos: {len(np.unique(groups))}")
    print(f"Clases: {np.unique(y)}")

    return X, y, groups


# ============================================================
# EVALUACIÓN DE UN MÉTODO
# ============================================================

def evaluate_method(method_name, csv_path):
    start_time = time.time()

    X, y, groups = load_features(csv_path)

    input_dim = X.shape[1]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):

        fold_start = time.time()

        print("\n" + "-" * 70)
        print(f"MÉTODO: {method_name} | FOLD: {fold}/{N_SPLITS}")
        print("-" * 70)

        set_seed(SEED + fold)

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Ajustar scaler únicamente con los datos de entrenamiento
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = build_mlp(input_dim)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            shuffle=True,
            verbose=1
        )

        probabilities = model.predict(
            X_val_scaled,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        y_pred = np.argmax(probabilities, axis=1)

        metrics = calculate_metrics(y_val, y_pred)

        elapsed = time.time() - fold_start

        result = {
            "method": method_name,
            "fold": fold,
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "input_dim": input_dim,
            "architecture": "input-128-64-5",
            "epochs_requested": EPOCHS,
            "epochs_completed": len(history.history["loss"]),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED + fold,
            "training_time_seconds": elapsed,
            **metrics
        }

        fold_results.append(result)

        # ----------------------------------------------------
        # GUARDAR HISTORIAL NPZ
        # ----------------------------------------------------

        history_path = (
            HISTORY_DIR
            / f"{method_name}_fold_{fold}_history.npz"
        )

        np.savez(
            history_path,
            loss=np.array(history.history["loss"]),
            accuracy=np.array(history.history["accuracy"]),
            val_loss=np.array(history.history["val_loss"]),
            val_accuracy=np.array(
                history.history["val_accuracy"]
            ),
            method=method_name,
            fold=fold,
            architecture="input-128-64-5",
            input_dim=input_dim
        )

        print("\nMétricas del fold:")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")

        print(f"Historial guardado en: {history_path}")
        print(f"Tiempo del fold: {elapsed:.2f} segundos")

        del model
        tf.keras.backend.clear_session()

    # --------------------------------------------------------
    # RESULTADOS POR FOLD
    # --------------------------------------------------------

    fold_df = pd.DataFrame(fold_results)

    fold_path = (
        RESULTS_DIR
        / f"{method_name}_fold_results.csv"
    )

    fold_df.to_csv(
        fold_path,
        index=False
    )

    # --------------------------------------------------------
    # RESUMEN ESTADÍSTICO
    # --------------------------------------------------------

    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "specificity_macro",
        "f1_macro"
    ]

    summary_rows = []

    for metric in metric_columns:
        summary_rows.append({
            "method": method_name,
            "metric": metric,
            "mean": fold_df[metric].mean(),
            "std": fold_df[metric].std(ddof=1),
            "min": fold_df[metric].min(),
            "max": fold_df[metric].max()
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        RESULTS_DIR
        / f"{method_name}_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    elapsed_total = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"RESUMEN: {method_name}")
    print("=" * 70)

    print(summary_df.to_string(index=False))

    print(f"\nCSV por fold guardado en:")
    print(fold_path)

    print("\nCSV resumen guardado en:")
    print(summary_path)

    print(f"\nTiempo total del método: {elapsed_total:.2f} segundos")

    return fold_df, summary_df


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    total_start = time.time()

    print("\n" + "#" * 70)
    print("EVALUACIÓN PRELIMINAR DEL MLP DESACOPLADO")
    print("Arquitectura: Entrada → 128 → 64 → 5")
    print("Evaluación: DEV mediante StratifiedGroupKFold")
    print("#" * 70)

    print("\nRUTAS DE GUARDADO:")
    print("PROJECT_ROOT:", PROJECT_ROOT.resolve())
    print("RESULTS_DIR:", RESULTS_DIR.resolve())
    print("HISTORY_DIR:", HISTORY_DIR.resolve())

    methods = [
        ("destilado", DESTILADO_CSV),
        ("supervisado", SUPERVISADO_CSV)
    ]

    all_fold_results = []
    all_summaries = []

    for method_name, csv_path in methods:

        fold_df, summary_df = evaluate_method(
            method_name,
            csv_path
        )

        all_fold_results.append(fold_df)
        all_summaries.append(summary_df)

    # --------------------------------------------------------
    # CSV COMBINADOS
    # --------------------------------------------------------

    combined_fold_df = pd.concat(
        all_fold_results,
        ignore_index=True
    )

    combined_summary_df = pd.concat(
        all_summaries,
        ignore_index=True
    )

    combined_fold_path = (
        RESULTS_DIR
        / "ft1_preliminary_mlp_fold_results.csv"
    )

    combined_summary_path = (
        RESULTS_DIR
        / "ft1_preliminary_mlp_summary.csv"
    )

    combined_fold_df.to_csv(
        combined_fold_path,
        index=False
    )

    combined_summary_df.to_csv(
        combined_summary_path,
        index=False
    )

    total_elapsed = time.time() - total_start

    print("\n" + "#" * 70)
    print("EVALUACIÓN COMPLETADA")
    print("#" * 70)

    print("\nCSV combinado por fold:")
    print(combined_fold_path.resolve())

    print("\nCSV combinado resumen:")
    print(combined_summary_path.resolve())

    print("\nTiempo total:")
    print(f"{total_elapsed:.2f} segundos")


if __name__ == "__main__":
    main()