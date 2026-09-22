
from pathlib import Path
import time

import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# Configuración
# ============================================================

SEED = 101

tf.keras.utils.set_random_seed(SEED)

DATASET_ROOT = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5"
    r"\Experimentos Extractores corregidos"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_CSV = DATASET_ROOT / "dev_data.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "features_FT1_knowledge_distillation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

FREEZE_LAYER = 324


# ============================================================
# Resolución de rutas
# ============================================================

def resolve_filepath(filepath):
    """
    Convierte las rutas relativas del CSV en rutas absolutas
    utilizando el directorio local del dataset.
    """

    path = Path(str(filepath))

    if path.is_absolute():
        return str(path)

    return str(DATASET_ROOT / path)


# ============================================================
# Cargar información del dataset DEV
# ============================================================

print("Cargando CSV de DEV...")

df = pd.read_csv(DEV_CSV)

filepaths = [
    resolve_filepath(filepath)
    for filepath in df["filepath"]
]

labels = df["label"].values
groups = df["group_id"].values

print("Total de imágenes:", len(filepaths))


# ============================================================
# Verificar que las imágenes existan
# ============================================================

missing_files = [
    filepath
    for filepath in filepaths
    if not Path(filepath).exists()
]

if missing_files:
    print("Archivos no encontrados:")

    for filepath in missing_files[:10]:
        print(filepath)

    raise FileNotFoundError(
        f"No se encontraron {len(missing_files)} archivos."
    )

print("Todas las rutas de imágenes son válidas.")


# ============================================================
# Loader de imágenes
# ============================================================

def load_image(filepath):

    image = tf.io.read_file(filepath)

    image = tf.image.decode_bmp(
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
# Dataset de imágenes
# ============================================================

def build_image_dataset(paths):

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


# ============================================================
# Modelo de referencia: ResNet-101 ImageNet
# ============================================================

print("Construyendo modelo de referencia...")

reference_base = ResNet101(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in reference_base.layers:
    layer.trainable = False

reference_output = reference_base.output

reference_output = GlobalAveragePooling2D(
    name="reference_gap"
)(reference_output)

reference_model = Model(
    inputs=reference_base.input,
    outputs=reference_output
)


# ============================================================
# Generar embeddings de referencia
# ============================================================

print("Generando embeddings de referencia...")

start_time = time.time()

image_dataset = build_image_dataset(filepaths)

reference_embeddings = reference_model.predict(
    image_dataset,
    verbose=1
)

print(
    "Shape de los embeddings de referencia:",
    reference_embeddings.shape
)


# ============================================================
# Modelo de fine-tuning
# ============================================================

print("Construyendo modelo de conocimiento destilado...")

base_model = ResNet101(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for index, layer in enumerate(base_model.layers):

    if index < FREEZE_LAYER:

        layer.trainable = False

    elif isinstance(layer, BatchNormalization):

        layer.trainable = False

    else:

        layer.trainable = True


fine_tune_output = base_model.output

fine_tune_output = GlobalAveragePooling2D(
    name="fine_tune_gap"
)(fine_tune_output)

fine_tune_model = Model(
    inputs=base_model.input,
    outputs=fine_tune_output
)

fine_tune_model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),
    loss="mse"
)


# ============================================================
# Dataset de entrenamiento
# ============================================================

print("Construyendo dataset de entrenamiento...")

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        filepaths,
        reference_embeddings
    )
)

train_dataset = train_dataset.map(
    lambda filepath, reference_embedding: (
        load_image(filepath),
        reference_embedding
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(
    buffer_size=1000,
    seed=SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# Early stopping
# ============================================================

early_stopping = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# Entrenamiento
# ============================================================

print("Iniciando fine-tuning mediante conocimiento destilado...")

fine_tune_model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# Extracción de características de DEV
# ============================================================

print("Extrayendo características de DEV...")

dev_dataset = build_image_dataset(filepaths)

features = fine_tune_model.predict(
    dev_dataset,
    verbose=1
)

print(
    "Shape de las características:",
    features.shape
)


# ============================================================
# Guardar características
# ============================================================

features_df = pd.DataFrame(features)

features_df["label"] = labels
features_df["group_id"] = groups

output_file = (
    OUTPUT_DIR
    / "FT1_knowledge_distillation_dev_features.csv"
)

features_df.to_csv(
    output_file,
    index=False
)

elapsed_time = time.time() - start_time

print("CSV guardado en:")
print(output_file)

print(
    f"Tiempo total: {elapsed_time / 60:.2f} minutos"
)

tf.keras.backend.clear_session()