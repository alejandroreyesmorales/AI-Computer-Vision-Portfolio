import os
import pandas as pd


# =================================
# CONFIGURACION
# =================================

BASE_DIR = "Herlev dataset"

OUTPUT_FILE = "Herlev_master.csv"


# =================================
# CLASIFICACION BINARIA
# =================================

label_map = {

    "normal_superficiel": "Normal",
    "normal_intermediate": "Normal",
    "normal_columnar": "Normal",

    "light_dysplastic": "Abnormal",
    "moderate_dysplastic": "Abnormal",
    "severe_dysplastic": "Abnormal",
    "carcinoma_in_situ": "Abnormal"
}


# =================================
# CREAR LISTA MAESTRA
# =================================

data = []


for folder, label in label_map.items():

    folder_path = os.path.join(
        BASE_DIR,
        folder
    )

    if not os.path.isdir(folder_path):

        print(
            "ADVERTENCIA: no se encontro:",
            folder_path
        )

        continue

    for filename in os.listdir(folder_path):

        filepath = os.path.join(
            folder_path,
            filename
        )

        if os.path.isfile(filepath):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in [
                ".bmp",
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff"
            ]:

                data.append({

                    "filepath": os.path.normpath(
                        filepath
                    ),

                    "label": label
                })


# =================================
# CREAR DATAFRAME
# =================================

df = pd.DataFrame(
    data
)


# =================================
# ORDENAR
# =================================

df = df.sort_values(
    by=[
        "label",
        "filepath"
    ]
).reset_index(
    drop=True
)


# =================================
# GUARDAR
# =================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =================================
# INFORMACION
# =================================

print("\n=================================")
print("LISTA MAESTRA HERLEV")
print("=================================")

print(
    "Total de imagenes:",
    len(df)
)

print(
    "\nDistribucion por clase:"
)

print(
    df["label"].value_counts()
)

print(
    "\nDistribucion por carpeta:"
)

print(
    df["filepath"].str.split(
        os.sep
    ).str[-2].value_counts()
)

print(
    "\nArchivo generado:",
    OUTPUT_FILE
)

print(
    "\nPrimeras filas:"
)

print(
    df.head(10)
)