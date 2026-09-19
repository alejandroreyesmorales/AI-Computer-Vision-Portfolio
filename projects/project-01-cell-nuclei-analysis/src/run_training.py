
from pathlib import Path

from train_mlp_cric import train_model


DATA_PATH = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Respaldos\EntrenamientoCRIC30imagenes\3col\Datos_Entrenamiento_BDCRIC.csv"
)


if __name__ == "__main__":

    train_model(
        data_path=DATA_PATH,
        epochs=20,
        batch_size=2048,
        random_seed=42
    )