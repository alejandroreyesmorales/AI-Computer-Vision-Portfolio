from fastapi.testclient import TestClient

from api.main import app


IMAGE_PATH = (
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5"
    r"\Experimentos Extractores corregidos\im_Dyskeratotic cropped\001_01.bmp"
)


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    with open(IMAGE_PATH, "rb") as image:
        response = client.post(
            "/predict",
            files={"file": ("001_01.bmp", image, "image/bmp")}
        )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "class_id" in result
    assert "probability" in result

    assert result["prediction"] in {"Normal", "Abnormal"}
    assert result["class_id"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0