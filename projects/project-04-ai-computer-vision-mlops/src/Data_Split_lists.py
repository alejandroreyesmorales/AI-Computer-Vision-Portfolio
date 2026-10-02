# -*- coding: utf-8 -*-
"""
Create development and test datasets grouped by parent image.

The split is performed at parent-image level to prevent data leakage:
all patches/cells belonging to the same parent image are assigned
entirely to either DEV or TEST.
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ===============================
# Ruta base del dataset
# ===============================
base_dir = r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5\Experimentos Extractores corregidos"


# ===============================
# Clases del dataset
# ===============================
classes = [
    "im_Dyskeratotic cropped",
    "im_Koilocytotic cropped",
    "im_Metaplastic cropped",
    "im_Parabasal cropped",
    "im_Superficial-Intermediate cropped"
]


data = []


# ===============================
# Construir DataFrame maestro
# ===============================
for cls in classes:

    class_path = os.path.join(base_dir, cls)

    for file in os.listdir(class_path):

        # Solo procesar imágenes BMP
        if file.endswith(".bmp"):

            filepath = os.path.join(class_path, file)

            # Extraer group_id (imagen madre)
            # Ejemplo:
            # 001_02.bmp -> 001
            group_id = file.split("_")[0]

            data.append([
                filepath,
                cls,
                group_id
            ])


df = pd.DataFrame(
    data,
    columns=["filepath", "label", "group_id"]
)


# ===============================
# Información general
# ===============================
print("\n===================================")
print("DATASET")
print("===================================")

print("Total imágenes/células:", len(df))
print("Total grupos únicos:", df["group_id"].nunique())

print("\nDistribución por clase:")
print(df["label"].value_counts())


# ===============================
# División 85 / 15 por grupos
# ===============================
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        df,
        df["label"],
        groups=df["group_id"]
    )
)


dev_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


# ===============================
# Información del split
# ===============================
print("\n===================================")
print("DATA SPLIT")
print("===================================")

print("Development set:", len(dev_df))
print("Test set:", len(test_df))

print(
    "Development percentage:",
    f"{len(dev_df) / len(df) * 100:.2f}%"
)

print(
    "Test percentage:",
    f"{len(test_df) / len(df) * 100:.2f}%"
)

print("\nGrupos en Dev:", dev_df["group_id"].nunique())
print("Grupos en Test:", test_df["group_id"].nunique())


# ===============================
# Distribución de clases
# ===============================
print("\n===================================")
print("CLASS DISTRIBUTION")
print("===================================")

print("\nDEV:")
print(dev_df["label"].value_counts())

print("\nTEST:")
print(test_df["label"].value_counts())


# ===============================
# Verificar data leakage
# ===============================
dev_groups = set(dev_df["group_id"])
test_groups = set(test_df["group_id"])

overlap = dev_groups.intersection(test_groups)


print("\n===================================")
print("DATA LEAKAGE CHECK")
print("===================================")

print("Shared parent images:", len(overlap))

if len(overlap) == 0:
    print("✓ No data leakage detected.")
else:
    print("⚠ WARNING: Parent images appear in both DEV and TEST.")
    print("Shared groups:", sorted(overlap))


# ===============================
# Ruta de salida
# ===============================
project_dir = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

output_dir = os.path.join(
    project_dir,
    "data",
    "splits"
)


# Crear directorio automáticamente
os.makedirs(
    output_dir,
    exist_ok=True
)


# ===============================
# Guardar CSV
# ===============================
dev_path = os.path.join(
    output_dir,
    "dev_data.csv"
)

test_path = os.path.join(
    output_dir,
    "test_data.csv"
)


dev_df.to_csv(
    dev_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ===============================
# Confirmación final
# ===============================
print("\n===================================")
print("FILES SAVED")
print("===================================")

print("DEV:", dev_path)
print("TEST:", test_path)