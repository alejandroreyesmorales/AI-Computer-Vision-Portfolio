import numpy as np
import joblib

from PIL import Image

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RESNET_PATH = "models/resnet50/ft-a/resnet50_ft-a.keras"
MLP_PATH = "models/mlp/final_model_ft_a.keras"
SCALER_PATH = "models/mlp/scaler_ft_a.joblib"


# ---------------------------------------------------------
# Load FT-A ResNet-50
# ---------------------------------------------------------

resnet_model = load_model(RESNET_PATH)

# Find the GlobalAveragePooling2D layer
gap_layer = next(
    layer
    for layer in resnet_model.layers
    if isinstance(layer, GlobalAveragePooling2D)
)

# Create feature extractor
feature_extractor = Model(
    inputs=resnet_model.input,
    outputs=gap_layer.output
)


# ---------------------------------------------------------
# Load final MLP and scaler
# ---------------------------------------------------------

mlp_model = load_model(MLP_PATH)
scaler = joblib.load(SCALER_PATH)


# ---------------------------------------------------------
# Inference function
# ---------------------------------------------------------

def predict(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize to ResNet-50 input size
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # ResNet-50 preprocessing
    image_array = preprocess_input(image_array)

    # Extract 2048-dimensional features
    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    # Scale features using the scaler fitted during training
    features_scaled = scaler.transform(features)

    # Predict probability of Abnormal class
    probability_abnormal = float(
        mlp_model.predict(features_scaled, verbose=0)[0][0]
    )

    # Binary decision
    class_id = int(probability_abnormal >= 0.5)

    if class_id == 1:
        prediction = "Abnormal"
    else:
        prediction = "Normal"

    return {
        "prediction": prediction,
        "class_id": class_id,
        "probability": probability_abnormal
    }