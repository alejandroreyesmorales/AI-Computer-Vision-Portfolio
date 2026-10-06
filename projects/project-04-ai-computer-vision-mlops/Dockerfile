FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/inference.py ./src/inference.py

COPY models/resnet50/ft-a/resnet50_ft-a.keras ./models/resnet50/ft-a/
COPY models/mlp/final_model_ft_a.keras ./models/mlp/
COPY models/mlp/scaler_ft_a.joblib ./models/mlp/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]