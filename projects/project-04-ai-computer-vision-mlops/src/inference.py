from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import numpy as np


# Load ResNet-50 feature extractor
feature_extractor = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# Load final MLP classifier
classifier = load_model(
    "models/mlp/final_model.keras"
)

# Load scaler fitted during training
scaler = joblib.load(
    "models/mlp/scaler.joblib"
)


print("Inference components loaded successfully.")
print("ResNet-50 feature dimension:", feature_extractor.output_shape)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    return image_array


def predict(image_path):
    image_array = preprocess_image(image_path)

    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    scaled_features = scaler.transform(features)

    probability = classifier.predict(
        scaled_features,
        verbose=0
    )

    probability = float(probability[0][0])

    if probability >= 0.5:
        class_id = 1
        prediction = "Abnormal"
    else:
        class_id = 0
        prediction = "Normal"

    return {
        "prediction": prediction,
        "class_id": class_id,
        "probability": probability
    }