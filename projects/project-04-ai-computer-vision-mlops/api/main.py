from fastapi import FastAPI, UploadFile, File, HTTPException
from src.inference import predict
import tempfile
import os


app = FastAPI(
    title="SIPaKMeD Cell Classification API",
    description="API for cervical cell classification using a ResNet-50 feature extractor and MLP classifier.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be an image."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        result = predict(temp_path)
        return result

    finally:
        os.remove(temp_path)