from src.inference import predict


IMAGE_PATH = (
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5"
    r"\Experimentos Extractores corregidos\im_Dyskeratotic cropped\001_01.bmp"
)


def test_prediction_pipeline():
    result = predict(IMAGE_PATH)

    assert isinstance(result, dict)

    assert "prediction" in result
    assert "class_id" in result
    assert "probability" in result

    assert result["prediction"] in {"Normal", "Abnormal"}
    assert result["class_id"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0


def test_dyskeratotic_image_is_classified_as_abnormal():
    result = predict(IMAGE_PATH)

    assert result["prediction"] == "Abnormal"
    assert result["class_id"] == 1