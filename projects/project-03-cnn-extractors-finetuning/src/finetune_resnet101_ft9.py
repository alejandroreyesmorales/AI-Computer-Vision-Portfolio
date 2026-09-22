"""FT9 fine-tuning example using ResNet-101.

This script documents the FT9 configuration used in the original study.
It is provided for methodological reproducibility and educational purposes.

The historical experiments are not rerun as part of this repository.
FT9 uses 122 initially frozen layers and frozen Batch Normalization layers.
"""

from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


FREEZE_LAYER = 122
NUM_CLASSES = 5
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-5
TRAINABLE_BATCH_NORMALIZATION = False

LABEL_MAP = {
    "im_Dyskeratotic cropped": 0,
    "im_Koilocytotic cropped": 1,
    "im_Metaplastic cropped": 2,
    "im_Parabasal cropped": 3,
    "im_Superficial-Intermediate cropped": 4,
}


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return preprocess_input(image)


def build_training_dataset(filepaths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(
        lambda path, label: (load_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_prediction_dataset(filepaths):
    dataset = tf.data.Dataset.from_tensor_slices(filepaths)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def read_metadata(csv_path):
    dataframe = pd.read_csv(csv_path)
    dataframe["filepath"] = dataframe["filepath"].apply(lambda path: str(Path(path)))

    filepaths = dataframe["filepath"].to_numpy()
    labels_text = dataframe["label"].to_numpy()
    groups = dataframe["group_id"].to_numpy()

    labels_numeric = [LABEL_MAP[label] for label in labels_text]
    labels_one_hot = tf.keras.utils.to_categorical(labels_numeric, NUM_CLASSES)

    return filepaths, labels_text, labels_one_hot, groups


def build_ft9_model():
    base_model = ResNet101(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    for index, layer in enumerate(base_model.layers):
        if index < FREEZE_LAYER:
            layer.trainable = False
        elif isinstance(layer, BatchNormalization):
            layer.trainable = TRAINABLE_BATCH_NORMALIZATION
        else:
            layer.trainable = True

    features = GlobalAveragePooling2D(name="gap")(base_model.output)
    hidden = Dense(128, activation="relu", name="dense_128")(features)
    predictions = Dense(NUM_CLASSES, activation="softmax", name="output")(hidden)

    model = Model(base_model.input, predictions)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def extract_features(model, filepaths):
    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer("gap").output,
    )
    return feature_extractor.predict(build_prediction_dataset(filepaths), verbose=1)


def save_features(features, labels, groups, output_path):
    dataframe = pd.DataFrame(features)
    dataframe["label"] = labels
    dataframe["group_id"] = groups
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Feature table shape: {dataframe.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and extract features using the FT9 configuration."
    )
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ft9_example"),
    )
    args = parser.parse_args()

    start_time = time.time()
    print(f"Frozen layers: {FREEZE_LAYER}")
    print(f"Batch Normalization trainable: {TRAINABLE_BATCH_NORMALIZATION}")

    dev_paths, dev_labels, dev_labels_one_hot, dev_groups = read_metadata(args.dev_csv)
    training_dataset = build_training_dataset(dev_paths, dev_labels_one_hot)
    model = build_ft9_model()

    early_stopping = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
    )

    print("Starting FT9 training...")
    model.fit(
        training_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1,
    )

    print("Extracting development features...")
    dev_features = extract_features(model, dev_paths)
    save_features(
        dev_features,
        dev_labels,
        dev_groups,
        args.output_dir / "FT9_dev_features.csv",
    )

    if args.test_csv is not None:
        test_paths, test_labels, _, test_groups = read_metadata(args.test_csv)
        print("Extracting test features...")
        test_features = extract_features(model, test_paths)
        save_features(
            test_features,
            test_labels,
            test_groups,
            args.output_dir / "FT9_test_features.csv",
        )

    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
