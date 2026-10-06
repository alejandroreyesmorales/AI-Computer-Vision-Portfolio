# src/fine_tuning_resnet50.py

import os
import json
import numpy as np
import pandas as pd
import mlflow
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

IMG_SIZE = (224, 224)
NUM_CLASSES = 5

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEV_CSV = os.path.join(
    BASE_DIR, "data", "splits", "dev_data.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR, "data", "splits", "test_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models", "resnet50"
)

FEATURE_DIR = os.path.join(
    BASE_DIR, "data", "features"
)

MLFLOW_DB = os.path.join(
    BASE_DIR, "mlflow.db"
)


# ============================================================
# 2. CLASS MAPPING
# ============================================================

CLASS_NAMES = sorted([
    "im_Dyskeratotic cropped",
    "im_Koilocytotic cropped",
    "im_Metaplastic cropped",
    "im_Parabasal cropped",
    "im_Superficial-Intermediate cropped"
])

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_split_data():

    dev_df = pd.read_csv(DEV_CSV)
    test_df = pd.read_csv(TEST_CSV)

    dev_df["class_index"] = dev_df["label"].map(CLASS_TO_INDEX)
    test_df["class_index"] = test_df["label"].map(CLASS_TO_INDEX)

    if dev_df["class_index"].isna().any():
        raise ValueError("Unknown class found in DEV dataset.")

    if test_df["class_index"].isna().any():
        raise ValueError("Unknown class found in TEST dataset.")

    dev_df["class_index"] = dev_df["class_index"].astype(int)
    test_df["class_index"] = test_df["class_index"].astype(int)

    return dev_df, test_df


# ============================================================
# 4. GROUP-STRATIFIED VALIDATION SPLIT
# ============================================================

def create_validation_split(dev_df):

    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(
        sgkf.split(
            dev_df,
            y=dev_df["class_index"],
            groups=dev_df["group_id"]
        )
    )

    train_df = dev_df.iloc[train_idx].reset_index(drop=True)
    val_df = dev_df.iloc[val_idx].reset_index(drop=True)

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])

    overlap = train_groups.intersection(val_groups)

    if overlap:
        raise ValueError(
            f"Group leakage detected: {len(overlap)} groups overlap."
        )

    return train_df, val_df


# ============================================================
# 5. IMAGE DATASET
# ============================================================

def load_image(filepath, label):

    image = Image.open(filepath).convert("RGB")
    image = image.resize(IMG_SIZE)

    image = np.asarray(image, dtype=np.float32)
    image = preprocess_input(image)

    return image, label


def create_tf_dataset(df, training=False):

    filepaths = df["filepath"].values
    labels = df["class_index"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels)
    )

    def process(filepath, label):

        image = tf.io.read_file(filepath)

        image = tf.image.decode_bmp(
            image,
            channels=3
        )

        image = tf.image.resize(
            image,
            IMG_SIZE
        )

        image = tf.cast(
            image,
            tf.float32
        )

        image = preprocess_input(image)

        label = tf.one_hot(
            label,
            depth=NUM_CLASSES
        )

        return image, label

    dataset = dataset.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# 6. BUILD FINE-TUNING MODEL
# ============================================================

def build_model(bn_trainable):

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    freeze_until = "conv3_block4_out"

    freeze = True

    for layer in base_model.layers:

        if layer.name == freeze_until:
            freeze = False

        if freeze:
            layer.trainable = False

        else:

            if isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            ):
                layer.trainable = bn_trainable

            else:
                layer.trainable = True

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    output = Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification_head"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 7. TRAIN ONE FINE-TUNING CONFIGURATION
# ============================================================

def train_configuration(
    name,
    bn_trainable,
    train_df,
    val_df
):

    print("\n" + "=" * 70)
    print(f"Starting {name}")
    print("=" * 70)

    model = build_model(
        bn_trainable=bn_trainable
    )

    train_dataset = create_tf_dataset(
        train_df,
        training=True
    )

    val_dataset = create_tf_dataset(
        val_df,
        training=False
    )

    model_dir = os.path.join(
        MODEL_DIR,
        name.lower()
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        model_dir,
        f"resnet50_{name.lower()}.keras"
    )

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping
        ]
    )

    best_epoch = int(
        np.argmax(
            history.history["val_accuracy"]
        ) + 1
    )

    best_val_accuracy = float(
        max(
            history.history["val_accuracy"]
        )
    )

    final_train_accuracy = float(
        history.history["accuracy"][-1]
    )

    final_train_loss = float(
        history.history["loss"][-1]
    )

    final_val_loss = float(
        history.history["val_loss"][-1]
    )

    mlflow.log_params({
        "model": "ResNet50",
        "weights": "ImageNet",
        "freeze_until": "conv3_block4_out",
        "trainable_stages": "conv4_x + conv5_x",
        "bn_trainable": bn_trainable,
        "input_size": "224x224",
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "random_state": RANDOM_STATE
    })

    mlflow.log_metrics({
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "final_train_accuracy": final_train_accuracy,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss
    })

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    history_path = os.path.join(
        model_dir,
        "training_history.json"
    )

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history.history,
            file,
            indent=4
        )

    mlflow.log_artifact(
        history_path,
        artifact_path="training_history"
    )

    return model


# ============================================================
# 8. FEATURE EXTRACTION
# ============================================================

def extract_features(
    model,
    df,
    output_csv
):

    feature_model = Model(
        inputs=model.input,
        outputs=model.get_layer(
            "global_average_pooling2d"
        ).output
    )

    dataset = create_tf_dataset(
        df,
        training=False
    )

    features = feature_model.predict(
        dataset,
        verbose=1
    )

    feature_columns = [
        f"feature_{i}"
        for i in range(features.shape[1])
    ]

    features_df = pd.DataFrame(
        features,
        columns=feature_columns
    )

    metadata_df = df[
        ["filepath", "label", "group_id"]
    ].reset_index(drop=True)

    result_df = pd.concat(
        [
            metadata_df,
            features_df
        ],
        axis=1
    )

    os.makedirs(
        os.path.dirname(output_csv),
        exist_ok=True
    )

    result_df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"Features saved: {output_csv}"
    )

    print(
        f"Feature shape: {features.shape}"
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ResNet50 Fine-Tuning Pipeline")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        FEATURE_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load DEV and TEST
    # --------------------------------------------------------

    dev_df, test_df = load_split_data()

    print(f"DEV samples:  {len(dev_df)}")
    print(f"TEST samples: {len(test_df)}")

    # --------------------------------------------------------
    # Create one fixed group-stratified validation split
    # --------------------------------------------------------

    train_df, val_df = create_validation_split(
        dev_df
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    mlflow.set_experiment(
        "ResNet50_FineTuning"
    )

    # ========================================================
    # FT-A: Batch Normalization trainable
    # ========================================================

    with mlflow.start_run(
        run_name="FT-A_BN_trainable"
    ):

        model_a = train_configuration(
            name="FT-A",
            bn_trainable=True,
            train_df=train_df,
            val_df=val_df
        )

        extract_features(
            model=model_a,
            df=dev_df,
            output_csv=os.path.join(
                FEATURE_DIR,
                "dev_features_resnet50_ft_a.csv"
            )
        )

        extract_features(
            model=model_a,
            df=test_df,
            output_csv=os.path.join(
                FEATURE_DIR,
                "test_features_resnet50_ft_a.csv"
            )
        )

    # ========================================================
    # FT-B: Batch Normalization frozen
    # ========================================================

    with mlflow.start_run(
        run_name="FT-B_BN_frozen"
    ):

        model_b = train_configuration(
            name="FT-B",
            bn_trainable=False,
            train_df=train_df,
            val_df=val_df
        )

        extract_features(
            model=model_b,
            df=dev_df,
            output_csv=os.path.join(
                FEATURE_DIR,
                "dev_features_resnet50_ft_b.csv"
            )
        )

        extract_features(
            model=model_b,
            df=test_df,
            output_csv=os.path.join(
                FEATURE_DIR,
                "test_features_resnet50_ft_b.csv"
            )
        )

    print("\n" + "=" * 70)
    print("Fine-tuning completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()