import numpy as np
import joblib

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D


MODEL_PATH = "models/resnet50/ft-a/resnet50_ft-a.keras"
MLP_PATH = "models/mlp/final_model_ft_a.keras"
SCALER_PATH = "models/mlp/scaler_ft_a.joblib"


def test_model_extractor_loads():
    model = load_model(MODEL_PATH)

    assert model is not None


def test_model_extractor_outputs_2048_features():
    model = load_model(MODEL_PATH)

    gap_layer = next(
        layer
        for layer in model.layers
        if isinstance(layer, GlobalAveragePooling2D)
    )

    feature_extractor = Model(
        inputs=model.input,
        outputs=gap_layer.output
    )

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)

    features = feature_extractor.predict(image, verbose=0)

    assert features.shape == (1, 2048)


def test_scaler_loads():
    scaler = joblib.load(SCALER_PATH)

    assert scaler is not None


def test_classifier_model_loads():
    model = load_model(MLP_PATH)

    assert model is not None


def test_classifier_model_expects_2048_features():
    model = load_model(MLP_PATH)

    assert model.input_shape[-1] == 2048