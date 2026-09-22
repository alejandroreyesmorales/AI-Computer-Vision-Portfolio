import pandas as pd

paths = {
    "Destilado": r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5\Experimentos Extractores corregidos\FT1_KDestilation.csv",
    "Supervisado": r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5\Experimentos Extractores corregidos\features_FT1\FT1_1.csv"
}

for name, path in paths.items():
    df = pd.read_csv(path)

    print(f"\n{'=' * 50}")
    print(name)
    print("Dimensiones:", df.shape)
    print("Primeras columnas:", df.columns[:10].tolist())
    print("Últimas columnas:", df.columns[-5:].tolist())
    print("Etiquetas:", df.iloc[:, -2].unique())
    print(df.head(2))