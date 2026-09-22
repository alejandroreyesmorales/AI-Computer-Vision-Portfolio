
import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

SEED = 101

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("SEED:", SEED)


# ============================================================
# RUTAS
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5\Experimentos Extractores corregidos"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_CSV = DATASET_ROOT / "dev_data.csv"
TEST_CSV = DATASET_ROOT / "test_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "features_FT1_supervised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARÁMETROS DEL EXPERIMENTO FT1
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10
FREEZE_LAYER = 324
NUM_CLASSES = 5


# ============================================================
# CARGAR DATASETS
# ============================================================

print("Loading DEV and TEST CSV files...")

df_dev = pd.read_csv(DEV_CSV)
df_test = pd.read_csv(TEST_CSV)

print("DEV samples:", len(df_dev))
print("TEST samples:", len(df_test))


# ============================================================
# RESOLUCIÓN DE RUTAS
# ============================================================

def resolve_filepath(filepath):
    """
    Convierte las rutas relativas del CSV en rutas absolutas
    utilizando la raíz local del dataset.
    """

    filepath = Path(str(filepath))

    if filepath.is_absolute():
        return str(filepath)

    return str(DATASET_ROOT / filepath)


df_dev["filepath"] = df_dev["filepath"].apply(resolve_filepath)
df_test["filepath"] = df_test["filepath"].apply(resolve_filepath)

filepaths_dev = df_dev["filepath"].values
filepaths_test = df_test["filepath"].values

labels_dev = df_dev["label"].values
labels_test = df_test["label"].values

groups_dev = df_dev["group_id"].values
groups_test = df_test["group_id"].values


# ============================================================
# VERIFICACIÓN DE ARCHIVOS
# ============================================================

missing_dev = [
    filepath for filepath in filepaths_dev
    if not Path(filepath).is_file()
]

missing_test = [
    filepath for filepath in filepaths_test
    if not Path(filepath).is_file()
]

if missing_dev or missing_test:
    print("ERROR: There are missing image files.")

    if missing_dev:
        print("Missing DEV files:", len(missing_dev))
        print("Example:", missing_dev[0])

    if missing_test:
        print("Missing TEST files:", len(missing_test))
        print("Example:", missing_test[0])

    raise FileNotFoundError(
        "One or more image files could not be found."
    )

print("All DEV and TEST image paths were found.")


# ============================================================
# LABEL MAP
# ============================================================

label_map = {
    "im_Dyskeratotic cropped": 0,
    "im_Koilocytotic cropped": 1,
    "im_Metaplastic cropped": 2,
    "im_Parabasal cropped": 3,
    "im_Superficial-Intermediate cropped": 4
}


labels_numeric_dev = [
    label_map[label]
    for label in labels_dev
]

labels_numeric_dev = tf.keras.utils.to_categorical(
    labels_numeric_dev,
    NUM_CLASSES
)


# ============================================================
# CARGA DE IMÁGENES
# ============================================================

def load_image(filepath):
    """
    Carga y preprocesa una imagen BMP para ResNet-101.
    """

    image = tf.io.read_file(filepath)

    image = tf.io.decode_bmp(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = preprocess_input(image)

    return image


# ============================================================
# DATASET DE ENTRENAMIENTO
# ============================================================

def build_training_dataset(paths, labels):
    """
    Construye el dataset de entrenamiento con imágenes y etiquetas.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        lambda filepath, label: (
            load_image(filepath),
            label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.shuffle(
        buffer_size=1000,
        seed=SEED,
        reshuffle_each_iteration=True
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


print("Building training dataset...")

training_dataset = build_training_dataset(
    filepaths_dev,
    labels_numeric_dev
)


# ============================================================
# MODELO RESNET-101
# ============================================================

print("Building ResNet-101 model...")

base_model = ResNet101(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ============================================================
# CONGELAMIENTO DE CAPAS
# ============================================================

print(
    "Freezing first",
    FREEZE_LAYER,
    "layers and BatchNormalization layers"
)

for index, layer in enumerate(base_model.layers):

    if index < FREEZE_LAYER:
        layer.trainable = False

    elif isinstance(layer, BatchNormalization):
        layer.trainable = False

    else:
        layer.trainable = True


# ============================================================
# CLASIFICADOR
# ============================================================

features = base_model.output

gap = GlobalAveragePooling2D(
    name="gap"
)(features)

hidden = Dense(
    128,
    activation="relu"
)(gap)

predictions = Dense(
    NUM_CLASSES,
    activation="softmax"
)(hidden)


model = Model(
    inputs=base_model.input,
    outputs=predictions
)


# ============================================================
# COMPILACIÓN
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("Starting FT1 supervised training...")

start_total = time.time()

model.fit(
    training_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)


# ============================================================
# EXTRACTOR DE CARACTERÍSTICAS
# ============================================================

print("Building feature extractor...")

feature_extractor = Model(
    inputs=model.input,
    outputs=model.get_layer("gap").output
)


# ============================================================
# DATASET PARA PREDICCIÓN
# ============================================================

def build_prediction_dataset(paths):
    """
    Construye un dataset únicamente con imágenes.
    """

    dataset = tf.data.Dataset.from_tensor_slices(paths)

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


print("Building DEV prediction dataset...")

prediction_dataset_dev = build_prediction_dataset(
    filepaths_dev
)

print("Building TEST prediction dataset...")

prediction_dataset_test = build_prediction_dataset(
    filepaths_test
)


# ============================================================
# EXTRACCIÓN DE CARACTERÍSTICAS DEV
# ============================================================

print("Extracting DEV features...")

features_dev = feature_extractor.predict(
    prediction_dataset_dev,
    verbose=1
)

print("DEV feature shape:", features_dev.shape)


# ============================================================
# EXTRACCIÓN DE CARACTERÍSTICAS TEST
# ============================================================

print("Extracting TEST features...")

features_test = feature_extractor.predict(
    prediction_dataset_test,
    verbose=1
)

print("TEST feature shape:", features_test.shape)


# ============================================================
# GUARDAR CARACTERÍSTICAS DEV
# ============================================================

features_dev_df = pd.DataFrame(
    features_dev
)

features_dev_df["label"] = labels_dev
features_dev_df["group_id"] = groups_dev

output_file_dev = (
    OUTPUT_DIR / "FT1_supervised_dev_features.csv"
)

features_dev_df.to_csv(
    output_file_dev,
    index=False
)

print(
    "Saved DEV features:",
    output_file_dev
)


# ============================================================
# GUARDAR CARACTERÍSTICAS TEST
# ============================================================

features_test_df = pd.DataFrame(
    features_test
)

features_test_df["label"] = labels_test
features_test_df["group_id"] = groups_test

output_file_test = (
    OUTPUT_DIR / "FT1_supervised_test_features.csv"
)

features_test_df.to_csv(
    output_file_test,
    index=False
)

print(
    "Saved TEST features:",
    output_file_test
)


# ============================================================
# TIEMPO TOTAL
# ============================================================

end_total = time.time()

elapsed_minutes = (
    end_total - start_total
) / 60

print(
    "Total time:",
    round(elapsed_minutes, 2),
    "minutes"
)


# ============================================================
# LIMPIAR SESIÓN
# ============================================================

tf.keras.backend.clear_session()

print("FT1 supervised process completed.")