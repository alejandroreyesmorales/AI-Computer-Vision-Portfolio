
import numpy as np
import joblib
from PIL import Image

from src.classification_utils import classify_probability

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input


RESNET_PATH = "models/resnet50/ft-a/resnet50_ft-a.keras"
MLP_PATH = "models/mlp/final_model_ft_a.keras"
SCALER_PATH = "models/mlp/scaler_ft_a.joblib"


resnet_model = load_model(RESNET_PATH)

gap_layer = next(
    layer
    for layer in resnet_model.layers
    if isinstance(layer, GlobalAveragePooling2D)
)

feature_extractor = Model(
    inputs=resnet_model.input,
    outputs=gap_layer.output
)

mlp_model = load_model(MLP_PATH)
scaler = joblib.load(SCALER_PATH)


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    features_scaled = scaler.transform(features)

    probability_abnormal = float(
        mlp_model.predict(
            features_scaled,
            verbose=0
        )[0][0]
    )

    prediction, class_id = classify_probability(
        probability_abnormal
    )

    return {
        "prediction": prediction,
        "class_id": class_id,
        "probability": probability_abnormal
    }
