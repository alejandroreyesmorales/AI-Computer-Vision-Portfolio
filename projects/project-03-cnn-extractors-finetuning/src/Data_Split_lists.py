# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 00:31:53 2026

@author: Asus
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ===============================
# Ruta base
# ===============================
base_dir = r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5\Experimentos Extractores corregidos"

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
        if file.endswith(".bmp"):
            
            filepath = os.path.join(class_path, file)
            
            # Extraer group_id (imagen madre)
            # Ejemplo: 001_02.bmp -> 001
            group_id = file.split("_")[0]
            
            data.append([filepath, cls, group_id])

df = pd.DataFrame(data, columns=["filepath", "label", "group_id"])

print("Total imágenes:", len(df))
print("Total grupos únicos:", df["group_id"].nunique())

# ===============================
# División 85 / 15 por grupos
# ===============================
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

train_idx, test_idx = next(gss.split(df, df["label"], groups=df["group_id"]))

dev_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# ===============================
# Guardar CSV
# ===============================
dev_df.to_csv("dev_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Development set:", len(dev_df))
print("Test set:", len(test_df))
print("Grupos en Dev:", dev_df["group_id"].nunique())
print("Grupos en Test:", test_df["group_id"].nunique())