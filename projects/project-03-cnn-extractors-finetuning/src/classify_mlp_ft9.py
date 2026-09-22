
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Archivos históricos: únicamente FT9_20 y FT9_test_20
DATA_DIR = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5"
    r"\Experimentos Extractores corregidos\FT9_classification"
)

DEV_FILE = DATA_DIR / "FT9_20.csv"
TEST_FILE = DATA_DIR / "FT9_test_20.csv"

# Directorio raíz del proyecto de GitHub
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification_mlp"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURACIÓN DE CLASES
# ============================================================

CLASS_CONFIGS = {
    "5class": {
        "name": "5 clases",
        "mapping": {
            "Superficial-Intermediate": 0,
            "Parabasal": 1,
            "Metaplastic": 2,
            "Dyskeratotic": 3,
            "Koilocytotic": 4
        }
    },

    "3class": {
        "name": "3 clases",
        "mapping": {
            "Superficial-Intermediate": 0,
            "Parabasal": 0,
            "Metaplastic": 1,
            "Dyskeratotic": 2,
            "Koilocytotic": 2
        }
    },

    "2class": {
        "name": "2 clases",
        "mapping": {
            "Superficial-Intermediate": 0,
            "Parabasal": 0,
            "Metaplastic": 1,
            "Dyskeratotic": 1,
            "Koilocytotic": 1
        }
    }
}


# ============================================================
# NORMALIZACIÓN DE ETIQUETAS
# ============================================================

def normalize_label(label):
    """
    Normaliza las etiquetas originales del CSV.

    Ejemplo:
    im_Dyskeratotic cropped
    se convierte en:
    Dyskeratotic
    """

    label = str(label).strip()

    # Eliminar el prefijo im_
    if label.startswith("im_"):
        label = label[3:]

    # Eliminar el sufijo cropped
    label = label.replace(" cropped", "")

    # Normalizar posibles separadores
    label = label.replace("_", "-")

    return label.strip()


def convert_labels(y, mapping):
    """
    Convierte las etiquetas originales al escenario
    de clasificación seleccionado.
    """

    converted_labels = []

    for label in y:

        normalized_label = normalize_label(label)

        if normalized_label not in mapping:
            raise ValueError(
                "Etiqueta no encontrada después de normalizar: "
                f"{label} -> {normalized_label}\n"
                f"Etiquetas disponibles en el mapeo: "
                f"{list(mapping.keys())}"
            )

        converted_labels.append(
            mapping[normalized_label]
        )

    return np.array(
        converted_labels,
        dtype=np.int32
    )


# ============================================================
# CARGA DE DATOS
# ============================================================

def load_data(file_path):
    """
    Carga las características y etiquetas desde un CSV.
    """

    df = pd.read_csv(file_path)

    excluded_columns = [
        "label",
        "group_id"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].values.astype(
        np.float32
    )

    y = df["label"].values

    return X, y


# ============================================================
# MODELO MLP
# ============================================================

def build_mlp(input_dim, num_classes):
    """
    Construye el MLP.

    Arquitectura:
    Entrada -> 128 -> 64 -> número de clases
    """

    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(
            128,
            activation="relu"
        ),

        Dense(
            64,
            activation="relu"
        ),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ============================================================
# ESPECIFICIDAD MACRO
# ============================================================

def calculate_specificity(
    y_true,
    y_pred,
    num_classes
):
    """
    Calcula la especificidad macro
    a partir de la matriz de confusión.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    specificities = []

    for class_index in range(num_classes):

        true_negatives = np.sum(cm) - (
            np.sum(cm[class_index, :])
            + np.sum(cm[:, class_index])
            - cm[class_index, class_index]
        )

        false_positives = (
            np.sum(cm[:, class_index])
            - cm[class_index, class_index]
        )

        denominator = (
            true_negatives
            + false_positives
        )

        if denominator == 0:
            specificity = 0.0
        else:
            specificity = (
                true_negatives
                / denominator
            )

        specificities.append(specificity)

    return float(np.mean(specificities))


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def evaluate_scenario(
    scenario_key,
    scenario_config,
    X_dev,
    y_dev_original,
    X_test,
    y_test_original
):
    """
    Entrena y evalúa un escenario de clasificación.

    Se utiliza:
    - DEV para entrenar.
    - TEST para evaluar.
    - Datos imbalanced.
    - Sin class_weight.
    """

    print("\n" + "=" * 70)
    print(
        f"ESCENARIO: {scenario_config['name']}"
    )
    print("=" * 70)

    mapping = scenario_config["mapping"]

    # Convertir etiquetas
    y_dev = convert_labels(
        y_dev_original,
        mapping
    )

    y_test = convert_labels(
        y_test_original,
        mapping
    )

    num_classes = len(
        np.unique(y_dev)
    )

    # Estandarización:
    # Se ajusta únicamente utilizando DEV
    scaler = StandardScaler()

    X_dev_scaled = scaler.fit_transform(
        X_dev
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Limpiar sesión anterior
    tf.keras.backend.clear_session()

    # Construir modelo
    model = build_mlp(
        input_dim=X_dev_scaled.shape[1],
        num_classes=num_classes
    )

    print(
        f"Características de entrada: "
        f"{X_dev_scaled.shape[1]}"
    )

    print(
        f"Número de clases: {num_classes}"
    )

    print(
        "Datos: imbalanced "
        "(sin pesos de clase)"
    )

    print(
        "Arquitectura: "
        f"{X_dev_scaled.shape[1]}-128-64-{num_classes}"
    )

    print("\nEntrenando modelo...")

    # Entrenamiento
    model.fit(
        X_dev_scaled,
        y_dev,

        epochs=20,
        batch_size=32,

        shuffle=True,
        verbose=1
    )

    # Predicción sobre TEST
    probabilities = model.predict(
        X_test_scaled,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # Métricas
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    specificity = calculate_specificity(
        y_test,
        y_pred,
        num_classes
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nRESULTADOS:")
    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Precision:   {precision:.4f}"
    )

    print(
        f"Recall:      {recall:.4f}"
    )

    print(
        f"Specificity: {specificity:.4f}"
    )

    print(
        f"F1-score:    {f1:.4f}"
    )

    # Guardar métricas
    result = {
        "scenario": scenario_key,
        "scenario_name": scenario_config["name"],

        "dev_file": DEV_FILE.name,
        "test_file": TEST_FILE.name,

        "classes": num_classes,

        "architecture": (
            f"{X_dev_scaled.shape[1]}"
            f"-128-64-{num_classes}"
        ),

        "epochs": 20,
        "batch_size": 32,

        "class_weights": False,

        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "specificity_macro": specificity,
        "f1_macro": f1
    }

    return result, y_test, y_pred


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CLASIFICACIÓN MLP - FT9")
    print("=" * 70)

    print(
        f"\nArchivo DEV:\n{DEV_FILE}"
    )

    print(
        f"\nArchivo TEST:\n{TEST_FILE}"
    )

    # Verificar archivos
    if not DEV_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo DEV:\n{DEV_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo TEST:\n{TEST_FILE}"
        )

    # Cargar únicamente FT9_20 y FT9_test_20
    X_dev, y_dev_original = load_data(
        DEV_FILE
    )

    X_test, y_test_original = load_data(
        TEST_FILE
    )

    print(
        f"\nDimensiones DEV: {X_dev.shape}"
    )

    print(
        f"Dimensiones TEST: {X_test.shape}"
    )

    all_results = []
    all_predictions = []

    # Ejecutar cada escenario una sola vez
    for scenario_key, scenario_config in CLASS_CONFIGS.items():

        result, y_test, y_pred = evaluate_scenario(
            scenario_key,
            scenario_config,

            X_dev,
            y_dev_original,

            X_test,
            y_test_original
        )

        all_results.append(result)

        predictions_df = pd.DataFrame({
            "scenario": scenario_key,
            "y_true": y_test,
            "y_pred": y_pred
        })

        all_predictions.append(
            predictions_df
        )

    # ========================================================
    # GUARDAR RESULTADOS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        OUTPUT_DIR
        / "mlp_ft9_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_path = (
        OUTPUT_DIR
        / "mlp_ft9_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # ========================================================
    # RESUMEN FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    summary_columns = [
        "scenario_name",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "specificity_macro",
        "f1_macro"
    ]

    print(
        results_df[
            summary_columns
        ].to_string(index=False)
    )

    print(
        f"\nResultados guardados en:\n"
        f"{results_path}"
    )

    print(
        f"\nPredicciones guardadas en:\n"
        f"{predictions_path}"
    )

    print("\nProceso finalizado.")