import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input


# =================================
# CONFIGURACION
# =================================

MASTER_FILE = "Herlev_master.csv"

MODEL_FILE = "ResNet101_FT9_feature_extractor.keras"

OUTPUT_FILE = "Herlev_features.csv"

IMG_SIZE = (224, 224)

BATCH_SIZE = 64


# =================================
# INFORMACION
# =================================

print("=================================")
print("EXTRACCION DE CARACTERISTICAS")
print("HERLEV - RESNET101 FT9")
print("=================================")

print(
    "TensorFlow:",
    tf.__version__
)

print(
    "Modelo:",
    MODEL_FILE
)

print(
    "Lista:",
    MASTER_FILE
)

print(
    "Batch size:",
    BATCH_SIZE
)


# =================================
# CARGAR LISTA MAESTRA
# =================================

print("\n=================================")
print("CARGANDO LISTA MAESTRA")
print("=================================")

df = pd.read_csv(
    MASTER_FILE
)

print(
    "Total de imagenes:",
    len(df)
)

print(
    "\nColumnas:"
)

print(
    df.columns.tolist()
)

print(
    "\nDistribucion de clases:"
)

print(
    df["label"].value_counts()
)


# =================================
# VERIFICAR DATOS
# =================================

if "filepath" not in df.columns:

    raise ValueError(
        "ERROR: no existe la columna filepath."
    )


if "label" not in df.columns:

    raise ValueError(
        "ERROR: no existe la columna label."
    )


if df["filepath"].isnull().any():

    raise ValueError(
        "ERROR: existen rutas vacias."
    )


if df["filepath"].duplicated().any():

    raise ValueError(
        "ERROR: existen rutas duplicadas."
    )


# =================================
# RUTAS
# =================================

filepaths = df["filepath"].values

labels = df["label"].values


# =================================
# VERIFICAR ARCHIVOS
# =================================

print("\n=================================")
print("VERIFICANDO ARCHIVOS")
print("=================================")

missing_files = []

for filepath in filepaths:

    if not os.path.isfile(filepath):

        missing_files.append(
            filepath
        )


print(
    "Archivos faltantes:",
    len(missing_files)
)


if len(missing_files) > 0:

    print(
        "\nPrimeros archivos faltantes:"
    )

    for filepath in missing_files[:10]:

        print(
            filepath
        )

    raise ValueError(
        "ERROR: existen archivos faltantes."
    )


print(
    "OK: todas las imagenes existen."
)


# =================================
# CARGAR MODELO
# =================================

print("\n=================================")
print("CARGANDO FEATURE EXTRACTOR")
print("=================================")

feature_extractor = load_model(
    MODEL_FILE
)

print(
    "Modelo cargado correctamente."
)

print(
    "Input shape:",
    feature_extractor.input_shape
)

print(
    "Output shape:",
    feature_extractor.output_shape
)


# =================================
# FUNCION CARGA IMAGEN
# =================================

def load_image(path):

    img = tf.io.read_file(
        path
    )

    img = tf.image.decode_bmp(
        img,
        channels=3
    )

    img = tf.image.resize(
        img,
        IMG_SIZE
    )

    img = preprocess_input(
        img
    )

    return img


# =================================
# CREAR DATASET
# =================================

print("\n=================================")
print("CREANDO DATASET")
print("=================================")

dataset = tf.data.Dataset.from_tensor_slices(
    filepaths
)

dataset = dataset.map(
    lambda x: load_image(x),
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(
    BATCH_SIZE
)

dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


# =================================
# EXTRAER CARACTERISTICAS
# =================================

print("\n=================================")
print("EXTRAYENDO CARACTERISTICAS")
print("=================================")

start_time = time.time()

features = feature_extractor.predict(
    dataset,
    verbose=1
)

end_time = time.time()


# =================================
# INFORMACION DE FEATURES
# =================================

print("\n=================================")
print("RESULTADO DE LA EXTRACCION")
print("=================================")

print(
    "Shape:",
    features.shape
)

print(
    "Numero de imagenes:",
    features.shape[0]
)

print(
    "Numero de caracteristicas:",
    features.shape[1]
)

print(
    "Tiempo:",
    round(
        (end_time - start_time) / 60,
        2
    ),
    "min"
)


# =================================
# VERIFICAR NUMERO DE IMAGENES
# =================================

if features.shape[0] != len(df):

    raise ValueError(
        "ERROR: el numero de caracteristicas "
        "no coincide con el numero de imagenes."
    )


# =================================
# VERIFICAR DIMENSION
# =================================

if features.shape[1] != 2048:

    raise ValueError(
        "ERROR: se esperaban 2048 caracteristicas."
    )


print(
    "\nOK: 917 imagenes y 2048 caracteristicas."
)


# =================================
# CREAR DATAFRAME
# =================================

print("\n=================================")
print("CREANDO ARCHIVO DE CARACTERISTICAS")
print("=================================")

features_df = pd.DataFrame(
    features
)


# =================================
# AGREGAR INFORMACION
# =================================

features_df["filepath"] = filepaths

features_df["label"] = labels


# =================================
# GUARDAR
# =================================

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =================================
# VERIFICACION FINAL
# =================================

print("\n=================================")
print("ARCHIVO GUARDADO")
print("=================================")

print(
    "Archivo:",
    OUTPUT_FILE
)

print(
    "Filas:",
    len(features_df)
)

print(
    "Columnas:",
    len(features_df.columns)
)

print(
    "\nPrimeras filas:"
)

print(
    features_df.head()
)


# =================================
# DISTRIBUCION FINAL
# =================================

print("\n=================================")
print("DISTRIBUCION FINAL")
print("=================================")

print(
    features_df["label"].value_counts()
)


print("\n=================================")
print("EXTRACCION COMPLETADA")
print("=================================")