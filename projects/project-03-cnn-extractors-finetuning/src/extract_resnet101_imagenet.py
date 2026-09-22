
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing the CSV files and image folders
DATASET_ROOT = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5"
    r"\Experimentos Extractores corregidos"
)

DEV_CSV = DATASET_ROOT / "dev_data.csv"

# Output folder inside the public portfolio
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "results" / "resnet101_imagenet"

OUTPUT_FILE = (
    OUTPUT_DIR / "resnet101_imagenet_dev_features.csv"
)

IMG_SIZE = (224, 224)


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(image_path):
    """
    Convert relative image paths from the CSV into absolute paths.
    """

    image_path = Path(str(image_path))

    # If the CSV contains an absolute path, use it directly
    if image_path.is_absolute():
        return image_path

    # Remove the initial './' from relative paths
    relative_path = str(image_path)

    if relative_path.startswith("./"):
        relative_path = relative_path[2:]

    # Resolve the image relative to the dataset folder
    absolute_path = DATASET_ROOT / relative_path

    return absolute_path.resolve()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_and_preprocess_image(image_path):
    """Load and preprocess one image for ResNet-101."""

    image = load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    return image


# ============================================================
# MODEL
# ============================================================

print("Loading ResNet-101 pretrained on ImageNet...")

base_model = ResNet101(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

print("Model loaded successfully.")
print("Feature dimension:", model.output_shape[-1])


# ============================================================
# LOAD DEV DATA
# ============================================================

if not DEV_CSV.exists():
    raise FileNotFoundError(
        f"DEV CSV not found: {DEV_CSV}"
    )

df = pd.read_csv(DEV_CSV)

required_columns = {
    "filepath",
    "label",
    "group_id"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )

print("\nDEV samples:", len(df))

# Convert all relative paths to absolute paths
df["absolute_filepath"] = df["filepath"].apply(
    resolve_image_path
)

print("\nChecking image paths...")

missing_images = [
    path
    for path in df["absolute_filepath"]
    if not path.exists()
]

if missing_images:
    print("\nExample of missing image:")
    print(missing_images[0])

    raise FileNotFoundError(
        f"{len(missing_images)} image paths were not found."
    )

print("All image paths were found successfully.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

features = []

for image_path in tqdm(
    df["absolute_filepath"],
    desc="Extracting ResNet-101 features"
):

    image = load_and_preprocess_image(image_path)

    feature = model.predict(
        image,
        verbose=0
    )

    features.append(feature.flatten())

features = np.asarray(features)

print("\nExtracted feature matrix shape:")
print(features.shape)


# ============================================================
# SAVE FEATURES
# ============================================================

feature_columns = [
    f"feature_{i}"
    for i in range(features.shape[1])
]

features_df = pd.DataFrame(
    features,
    columns=feature_columns
)

metadata_df = df[
    ["label", "group_id"]
].reset_index(drop=True)

final_df = pd.concat(
    [features_df, metadata_df],
    axis=1
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("Feature extraction completed.")
print("========================================")
print("Output file:")
print(OUTPUT_FILE)
print("\nFinal dataframe shape:")
print(final_df.shape)