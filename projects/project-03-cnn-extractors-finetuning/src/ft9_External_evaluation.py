# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 02:03:46 2026

@author: Asus
"""

import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# =================================
# CONFIGURACION DE SEMILLA
# =================================

SEED = 101

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Seed:", SEED)


# =================================
# CONTROL DE THREADS - CLUSTER
# =================================

N_THREADS = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        1
    )
)

tf.config.threading.set_intra_op_parallelism_threads(
    N_THREADS
)

tf.config.threading.set_inter_op_parallelism_threads(
    4
)

print("CPU threads:", N_THREADS)


# =================================
# CONFIGURACION
# =================================

BASE_DIR = "."

DEV_CSV = os.path.join(
    BASE_DIR,
    "dev_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Validacion_externa"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMG_SIZE = (224, 224)

BATCH_SIZE = 64
EPOCHS = 5

# FT9
FREEZE_LAYER = 122

# SIPaKMeD: 5 clases
NUM_CLASSES = 5


# =================================
# CARGAR DEV
# =================================

print("\nCargando DEV...")

df = pd.read_csv(
    DEV_CSV
)

df["filepath"] = df["filepath"].apply(
    os.path.normpath
)

filepaths = df["filepath"].values
labels = df["label"].values
groups = df["group_id"].values

print(
    "Total imagenes DEV:",
    len(filepaths)
)


# =================================
# LABELS SIPAKMED
# =================================

label_map = {

    "im_Dyskeratotic cropped": 0,

    "im_Koilocytotic cropped": 1,

    "im_Metaplastic cropped": 2,

    "im_Parabasal cropped": 3,

    "im_Superficial-Intermediate cropped": 4
}


y_numeric = [
    label_map[label]
    for label in labels
]

y_numeric = tf.keras.utils.to_categorical(
    y_numeric,
    NUM_CLASSES
)


# =================================
# FUNCION PARA CARGAR IMAGEN
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
# DATASET DEV
# =================================

def build_dataset(
    paths,
    labels
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    ds = ds.map(
        lambda x, y: (
            load_image(x),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(
        1000,
        seed=SEED,
        reshuffle_each_iteration=True
    )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


dataset = build_dataset(
    filepaths,
    y_numeric
)


# =================================
# MODELO BASE
# =================================

print(
    "\nConstruyendo ResNet101..."
)

base_model = ResNet101(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# =================================
# CONFIGURACION FT9
# =================================

print(
    "\nConfiguracion FT9"
)

print(
    "Primeras",
    FREEZE_LAYER,
    "capas congeladas"
)

print(
    "Batch Normalization congeladas"
)


for i, layer in enumerate(
    base_model.layers
):

    if i < FREEZE_LAYER:

        layer.trainable = False

    elif isinstance(
        layer,
        BatchNormalization
    ):

        layer.trainable = False

    else:

        layer.trainable = True


# =================================
# CONTAR CAPAS
# =================================

frozen_layers = sum(
    not layer.trainable
    for layer in base_model.layers
)

trainable_layers = sum(
    layer.trainable
    for layer in base_model.layers
)

print(
    "\nCapas congeladas:",
    frozen_layers
)

print(
    "Capas entrenables:",
    trainable_layers
)


# =================================
# MLP ACOPLADO
# =================================

print(
    "\nConstruyendo MLP..."
)

x = base_model.output

gap = GlobalAveragePooling2D(
    name="gap"
)(x)

dense_128 = Dense(
    128,
    activation="relu",
    name="mlp_dense_128"
)(gap)

pred = Dense(
    NUM_CLASSES,
    activation="softmax",
    name="mlp_classifier"
)(dense_128)

model = Model(
    inputs=base_model.input,
    outputs=pred,
    name="ResNet101_FT9_MLP"
)


# =================================
# COMPILAR
# =================================

model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# =================================
# EARLY STOPPING
# =================================

early_stop = EarlyStopping(

    monitor="loss",

    patience=5,

    restore_best_weights=True
)


# =================================
# RESUMEN
# =================================

print(
    "\nResumen del modelo:"
)

model.summary()


# =================================
# ENTRENAMIENTO
# =================================

print(
    "\n========================================"
)

print(
    "INICIANDO ENTRENAMIENTO FT9"
)

print(
    "ResNet101 + FT9 + BN congeladas"
)

print(
    "========================================\n"
)

start_total = time.time()

history = model.fit(

    dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stop
    ],

    verbose=1
)


# =================================
# GUARDAR MODELO COMPLETO
# =================================

model_file = os.path.join(

    OUTPUT_DIR,

    "ResNet101_FT9_BN_frozen_MLP.keras"
)

model.save(
    model_file
)

print(
    "\nModelo completo guardado:"
)

print(
    model_file
)


# =================================
# GUARDAR HISTORIAL
# =================================

history_df = pd.DataFrame(
    history.history
)

history_file = os.path.join(

    OUTPUT_DIR,

    "training_history_FT9.csv"
)

history_df.to_csv(

    history_file,

    index=False
)

print(
    "\nHistorial guardado:"
)

print(
    history_file
)


# =================================
# CREAR FEATURE EXTRACTOR
# =================================

print(
    "\nCreando feature extractor..."
)

feature_extractor = Model(

    inputs=model.input,

    outputs=model.get_layer(
        "gap"
    ).output,

    name="ResNet101_FT9_feature_extractor"
)


print(
    "Dimension de salida:",
    feature_extractor.output.shape
)


# =================================
# GUARDAR FEATURE EXTRACTOR
# =================================

feature_extractor_file = os.path.join(

    OUTPUT_DIR,

    "ResNet101_FT9_feature_extractor.keras"
)

feature_extractor.save(
    feature_extractor_file
)

print(
    "\nFeature extractor guardado:"
)

print(
    feature_extractor_file
)


# =================================
# CREAR MLP INDEPENDIENTE
# =================================

print(
    "\nCreando MLP independiente..."
)

mlp_input = tf.keras.Input(
    shape=(2048,),
    name="feature_input"
)

mlp_dense_layer = model.get_layer(
    "mlp_dense_128"
)

mlp_classifier_layer = model.get_layer(
    "mlp_classifier"
)

x_mlp = mlp_dense_layer(
    mlp_input
)

output_mlp = mlp_classifier_layer(
    x_mlp
)

mlp_model = Model(

    inputs=mlp_input,

    outputs=output_mlp,

    name="MLP_FT9"
)


# =================================
# GUARDAR MLP
# =================================

mlp_file = os.path.join(

    OUTPUT_DIR,

    "MLP_FT9.keras"
)

mlp_model.save(
    mlp_file
)

print(
    "\nMLP guardado:"
)

print(
    mlp_file
)


# =================================
# CREAR DATASET PARA FEATURES
# =================================

print(
    "\nPreparando DEV para extracción..."
)

predict_ds = tf.data.Dataset.from_tensor_slices(
    filepaths
)

predict_ds = predict_ds.map(

    lambda x: load_image(x),

    num_parallel_calls=tf.data.AUTOTUNE
)

predict_ds = predict_ds.batch(
    BATCH_SIZE
)

predict_ds = predict_ds.prefetch(
    tf.data.AUTOTUNE
)


# =================================
# EXTRAER FEATURES DEV
# =================================

print(
    "\n========================================"
)

print(
    "EXTRAYENDO FEATURES DE SIPAKMED DEV"
)

print(
    "========================================\n"
)

features = feature_extractor.predict(

    predict_ds,

    verbose=1
)

print(
    "\nShape de features DEV:",
    features.shape
)


# =================================
# GUARDAR FEATURES DEV
# =================================

features_df = pd.DataFrame(
    features
)

features_df["label"] = labels

features_df["group_id"] = groups

features_file = os.path.join(

    OUTPUT_DIR,

    "FT9_DEV_features.csv"
)

features_df.to_csv(

    features_file,

    index=False
)

print(
    "\nFeatures DEV guardadas:"
)

print(
    features_file
)


# =================================
# TIEMPO TOTAL
# =================================

end_total = time.time()

print(
    "\n========================================"
)

print(
    "PROCESO FINALIZADO"
)

print(
    "========================================"
)

print(
    "Tiempo TOTAL:",
    round(
        (end_total - start_total) / 60,
        2
    ),
    "minutos"
)


# =================================
# ARCHIVOS GENERADOS
# =================================

print(
    "\nArchivos generados en:"
)

print(
    OUTPUT_DIR
)

print(
    "\n1. ResNet101_FT9_BN_frozen_MLP.keras"
)

print(
    "2. ResNet101_FT9_feature_extractor.keras"
)

print(
    "3. MLP_FT9.keras"
)

print(
    "4. training_history_FT9.csv"
)

print(
    "5. FT9_DEV_features.csv"
)


# =================================
# LIMPIAR SESION
# =================================

tf.keras.backend.clear_session()