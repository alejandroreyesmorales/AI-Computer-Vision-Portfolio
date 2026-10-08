# -*- coding: utf-8 -*-
"""
Hierarchical Ensemble - FT9 + IMBALANCED

Clasificación jerárquica mediante ensemble de:
- SVM-RBF
- KNN
- Random Forest
- Logistic Regression
- MLP

Niveles:
C1: Normal vs Abnormal
C4: Superficial-Intermediate vs Parabasal
C2: Metaplastic vs Malignant
C3: Koilocytotic vs Dyskeratotic

Evalúa:
- 5 clases
- 3 clases
- 2 clases

Protocolo:
- Runs 20-29
- Sin class weighting
- StandardScaler
- Votación mayoritaria
- Métricas: ACC, Precision, Recall, Specificity, F1
- Selección del mejor RUN por ACC para cada tarea
- Una matriz de confusión por tarea, correspondiente al mejor RUN
"""

# ============================================================
# IMPORTACIONES
# ============================================================

import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = "FT9_classification"
OUTPUT_DIR = "imbalanced_classifiers"

RUNS = list(range(20, 30))

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# NOMBRES DE CLASES
# ============================================================

CLASS_NAMES_5 = [
    "Superficial-Intermediate",
    "Parabasal",
    "Metaplastic",
    "Koilocytotic",
    "Dyskeratotic"
]

CLASS_NAMES_3 = [
    "Normal",
    "Metaplastic",
    "Abnormal"
]

CLASS_NAMES_2 = [
    "Normal",
    "Abnormal"
]


# ============================================================
# MAPEO A 5 CLASES
# ============================================================

def map_to_5(y):

    out = []

    for label in y:

        if label == 'im_Superficial-Intermediate cropped':
            out.append(0)

        elif label == 'im_Parabasal cropped':
            out.append(1)

        elif label == 'im_Metaplastic cropped':
            out.append(2)

        elif label == 'im_Koilocytotic cropped':
            out.append(3)

        elif label == 'im_Dyskeratotic cropped':
            out.append(4)

        else:
            raise ValueError(
                f"Etiqueta desconocida: {label}"
            )

    return np.array(out)


# ============================================================
# MAPEO DE LOS SUBPROBLEMAS
# ============================================================

# C1:
# Superficial + Parabasal = Normal
# Metaplastic + Koilocytotic + Dyskeratotic = Abnormal

def map_C1(y):

    return np.array([
        0 if i in [0, 1] else 1
        for i in y
    ])


# C2:
# Metaplastic = 0
# Koilocytotic + Dyskeratotic = 1

def map_C2(y):

    return np.array([
        0 if i == 2 else 1
        for i in y
    ])


# C3:
# Koilocytotic = 0
# Dyskeratotic = 1

def map_C3(y):

    return np.array([
        0 if i == 3 else 1
        for i in y
    ])


# C4:
# Superficial-Intermediate = 0
# Parabasal = 1

def map_C4(y):

    return np.array([
        0 if i == 0 else 1
        for i in y
    ])


# ============================================================
# MAPEO FINAL A 3 CLASES
# ============================================================

def map_to_3(y):

    out = []

    for i in y:

        if i in [0, 1]:
            out.append(0)

        elif i == 2:
            out.append(1)

        elif i in [3, 4]:
            out.append(2)

        else:
            raise ValueError(i)

    return np.array(out)


# ============================================================
# MAPEO FINAL A 2 CLASES
# ============================================================

def map_to_2(y):

    return np.array([
        0 if i in [0, 1, 2] else 1
        for i in y
    ])


# ============================================================
# VOTACIÓN MAYORITARIA
# ============================================================

def vote(preds):

    return np.bincount(
        preds
    ).argmax()


# ============================================================
# MLP
# ============================================================

def create_mlp(input_dim):

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy"
    )

    return model


# ============================================================
# ENTRENAMIENTO DEL ENSEMBLE
# ============================================================

def train_ensemble(X, y):

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm = SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        class_weight=None
    )

    svm.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean"
    )

    knn.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        class_weight=None
    )

    lr.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp = create_mlp(
        X.shape[1]
    )

    mlp.fit(
        X,
        to_categorical(y),
        epochs=20,
        batch_size=32,
        verbose=0
    )

    return (
        svm,
        knn,
        rf,
        lr,
        mlp
    )


# ============================================================
# PREDICCIÓN DEL ENSEMBLE
# ============================================================

def predict_ensemble(
    models,
    x
):

    svm, knn, rf, lr, mlp = models

    predictions = [
        svm.predict([x])[0],

        knn.predict([x])[0],

        rf.predict([x])[0],

        lr.predict([x])[0],

        np.argmax(
            mlp.predict(
                np.array([x]),
                verbose=0
            )
        )
    ]

    return vote(
        predictions
    )


# ============================================================
# ESPECIFICIDAD MACRO
# ============================================================

def macro_specificity(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    total = np.sum(cm)

    specificities = []

    for i in range(len(cm)):

        TP = cm[i, i]

        FN = np.sum(cm[i, :]) - TP

        FP = np.sum(cm[:, i]) - TP

        TN = total - (
            TP + FN + FP
        )

        if (TN + FP) == 0:

            specificity = 0.0

        else:

            specificity = (
                TN / (TN + FP)
            )

        specificities.append(
            specificity
        )

    return np.mean(
        specificities
    )


# ============================================================
# FUNCIÓN DE MÉTRICAS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    specificity = macro_specificity(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return (
        acc,
        precision,
        recall,
        specificity,
        f1
    )


# ============================================================
# ARCHIVO DE RESULTADOS
# ============================================================

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "Hierarchical_Ensemble_FT9_imbalanced.txt"
)


# ============================================================
# MEJORES RUNS
# ============================================================

best_5 = None
best_3 = None
best_2 = None


# ============================================================
# PROCESAMIENTO
# ============================================================

with open(
    OUTPUT_FILE,
    "w"
) as f:

    f.write(
        "MODEL: HIERARCHICAL ENSEMBLE - FT9 - IMBALANCED\n"
    )

    f.write(
        "================================================\n\n"
    )

    # ========================================================
    # RUNS
    # ========================================================

    for run in RUNS:

        print("\n" + "=" * 60)
        print(f"RUN {run}")
        print("=" * 60)

        # ----------------------------------------------------
        # DATOS
        # ----------------------------------------------------

        dev_path = os.path.join(
            BASE_DIR,
            f"FT9_{run}.csv"
        )

        test_path = os.path.join(
            BASE_DIR,
            f"FT9_test_{run}.csv"
        )

        dev = pd.read_csv(
            dev_path
        )

        test = pd.read_csv(
            test_path
        )

        X_train = dev.iloc[:, :-2].values

        y_train = map_to_5(
            dev.iloc[:, -2].values
        )

        X_test = test.iloc[:, :-2].values

        y_test = map_to_5(
            test.iloc[:, -2].values
        )

        # ----------------------------------------------------
        # ESCALADO
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # ENTRENAMIENTO
        # ----------------------------------------------------

        start = time.time()

        # ====================================================
        # C1: NORMAL VS ABNORMAL
        # ====================================================

        C1 = train_ensemble(
            X_train,
            map_C1(y_train)
        )

        # ====================================================
        # C4: SUPERFICIAL VS PARABASAL
        # ====================================================

        mask_norm = (
            (y_train == 0) |
            (y_train == 1)
        )

        C4 = train_ensemble(
            X_train[mask_norm],
            map_C4(
                y_train[mask_norm]
            )
        )

        # ====================================================
        # C2: METAPLASTIC VS MALIGNANT
        # ====================================================

        mask_anorm = (
            y_train >= 2
        )

        C2 = train_ensemble(
            X_train[mask_anorm],
            map_C2(
                y_train[mask_anorm]
            )
        )

        # ====================================================
        # C3: KOILOCYTOTIC VS DYSKERATOTIC
        # ====================================================

        mask_malig = (
            (y_train == 3) |
            (y_train == 4)
        )

        C3 = train_ensemble(
            X_train[mask_malig],
            map_C3(
                y_train[mask_malig]
            )
        )

        train_time = (
            time.time() - start
        )

        # ====================================================
        # PREDICCIÓN GLOBAL
        # ====================================================

        y_pred = []

        for x in X_test:

            # ------------------------------------------------
            # C1
            # ------------------------------------------------

            p1 = predict_ensemble(
                C1,
                x
            )

            # ------------------------------------------------
            # NORMAL
            # ------------------------------------------------

            if p1 == 0:

                p4 = predict_ensemble(
                    C4,
                    x
                )

                if p4 == 0:

                    y_pred.append(0)

                else:

                    y_pred.append(1)

            # ------------------------------------------------
            # ABNORMAL
            # ------------------------------------------------

            else:

                p2 = predict_ensemble(
                    C2,
                    x
                )

                # Metaplastic

                if p2 == 0:

                    y_pred.append(2)

                # Koilocytotic / Dyskeratotic

                else:

                    p3 = predict_ensemble(
                        C3,
                        x
                    )

                    if p3 == 0:

                        y_pred.append(3)

                    else:

                        y_pred.append(4)

        y_pred = np.array(
            y_pred
        )

        # ====================================================
        # 5 CLASES
        # ====================================================

        (
            a5,
            p5,
            r5,
            s5,
            f5
        ) = calculate_metrics(
            y_test,
            y_pred
        )

        # ====================================================
        # 3 CLASES
        # ====================================================

        y3_true = map_to_3(
            y_test
        )

        y3_pred = map_to_3(
            y_pred
        )

        (
            a3,
            p3,
            r3,
            s3,
            f3
        ) = calculate_metrics(
            y3_true,
            y3_pred
        )

        # ====================================================
        # 2 CLASES
        # ====================================================

        y2_true = map_to_2(
            y_test
        )

        y2_pred = map_to_2(
            y_pred
        )

        (
            a2,
            p2,
            r2,
            s2,
            f2
        ) = calculate_metrics(
            y2_true,
            y2_pred
        )

        # ====================================================
        # GUARDAR RESULTADOS DEL RUN
        # ====================================================

        f.write(
            f"RUN {run}\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"5C: ACC={a5:.4f} | "
            f"PRE={p5:.4f} | "
            f"REC={r5:.4f} | "
            f"SPE={s5:.4f} | "
            f"F1={f5:.4f}\n"
        )

        f.write(
            f"3C: ACC={a3:.4f} | "
            f"PRE={p3:.4f} | "
            f"REC={r3:.4f} | "
            f"SPE={s3:.4f} | "
            f"F1={f3:.4f}\n"
        )

        f.write(
            f"2C: ACC={a2:.4f} | "
            f"PRE={p2:.4f} | "
            f"REC={r2:.4f} | "
            f"SPE={s2:.4f} | "
            f"F1={f2:.4f}\n"
        )

        f.write(
            f"Training time: {train_time:.4f} s\n\n"
        )

        # ====================================================
        # MEJOR RUN 5 CLASES
        # ====================================================

        if (
            best_5 is None or
            a5 > best_5["acc"]
        ):

            best_5 = {
                "run": run,
                "acc": a5,
                "precision": p5,
                "recall": r5,
                "specificity": s5,
                "f1": f5,
                "y_true": y_test.copy(),
                "y_pred": y_pred.copy()
            }

        # ====================================================
        # MEJOR RUN 3 CLASES
        # ====================================================

        if (
            best_3 is None or
            a3 > best_3["acc"]
        ):

            best_3 = {
                "run": run,
                "acc": a3,
                "precision": p3,
                "recall": r3,
                "specificity": s3,
                "f1": f3,
                "y_true": y3_true.copy(),
                "y_pred": y3_pred.copy()
            }

        # ====================================================
        # MEJOR RUN 2 CLASES
        # ====================================================

        if (
            best_2 is None or
            a2 > best_2["acc"]
        ):

            best_2 = {
                "run": run,
                "acc": a2,
                "precision": p2,
                "recall": r2,
                "specificity": s2,
                "f1": f2,
                "y_true": y2_true.copy(),
                "y_pred": y2_pred.copy()
            }

    # ========================================================
    # RESUMEN DE MEJORES RUNS
    # ========================================================

    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("BEST RUNS BASED ON ACCURACY\n")
    f.write("=" * 60 + "\n\n")

    # --------------------------------------------------------
    # 5 CLASES
    # --------------------------------------------------------

    f.write("5 CLASSES\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"BEST RUN: {best_5['run']}\n"
    )
    f.write(
        f"ACC: {best_5['acc']:.4f}\n"
    )
    f.write(
        f"PRECISION: {best_5['precision']:.4f}\n"
    )
    f.write(
        f"RECALL: {best_5['recall']:.4f}\n"
    )
    f.write(
        f"SPECIFICITY: {best_5['specificity']:.4f}\n"
    )
    f.write(
        f"F1: {best_5['f1']:.4f}\n\n"
    )

    # --------------------------------------------------------
    # 3 CLASES
    # --------------------------------------------------------

    f.write("3 CLASSES\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"BEST RUN: {best_3['run']}\n"
    )
    f.write(
        f"ACC: {best_3['acc']:.4f}\n"
    )
    f.write(
        f"PRECISION: {best_3['precision']:.4f}\n"
    )
    f.write(
        f"RECALL: {best_3['recall']:.4f}\n"
    )
    f.write(
        f"SPECIFICITY: {best_3['specificity']:.4f}\n"
    )
    f.write(
        f"F1: {best_3['f1']:.4f}\n\n"
    )

    # --------------------------------------------------------
    # 2 CLASES
    # --------------------------------------------------------

    f.write("2 CLASSES\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"BEST RUN: {best_2['run']}\n"
    )
    f.write(
        f"ACC: {best_2['acc']:.4f}\n"
    )
    f.write(
        f"PRECISION: {best_2['precision']:.4f}\n"
    )
    f.write(
        f"RECALL: {best_2['recall']:.4f}\n"
    )
    f.write(
        f"SPECIFICITY: {best_2['specificity']:.4f}\n"
    )
    f.write(
        f"F1: {best_2['f1']:.4f}\n\n"
    )

    f.write(
        "Selection criterion: highest Accuracy "
        "for each classification task.\n"
    )


# ============================================================
# FUNCIÓN MATRIZ DE CONFUSIÓN
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    run,
    filename,
    title
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{title} - RUN {run}"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MATRIZ 5 CLASES
# ============================================================

save_confusion_matrix(
    best_5["y_true"],
    best_5["y_pred"],
    CLASS_NAMES_5,
    best_5["run"],
    os.path.join(
        OUTPUT_DIR,
        "Hierarchical_Ensemble_FT9_5classes_imbalanced_CM_best.png"
    ),
    "Hierarchical Ensemble - 5 Classes"
)


# ============================================================
# MATRIZ 3 CLASES
# ============================================================

save_confusion_matrix(
    best_3["y_true"],
    best_3["y_pred"],
    CLASS_NAMES_3,
    best_3["run"],
    os.path.join(
        OUTPUT_DIR,
        "Hierarchical_Ensemble_FT9_3classes_imbalanced_CM_best.png"
    ),
    "Hierarchical Ensemble - 3 Classes"
)


# ============================================================
# MATRIZ 2 CLASES
# ============================================================

save_confusion_matrix(
    best_2["y_true"],
    best_2["y_pred"],
    CLASS_NAMES_2,
    best_2["run"],
    os.path.join(
        OUTPUT_DIR,
        "Hierarchical_Ensemble_FT9_2classes_imbalanced_CM_best.png"
    ),
    "Hierarchical Ensemble - 2 Classes"
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n")
print("=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)

print(
    f"\n5 clases -> RUN {best_5['run']} | "
    f"ACC = {best_5['acc']:.4f}"
)

print(
    f"3 clases -> RUN {best_3['run']} | "
    f"ACC = {best_3['acc']:.4f}"
)

print(
    f"2 clases -> RUN {best_2['run']} | "
    f"ACC = {best_2['acc']:.4f}"
)

print(
    f"\nResultados guardados en:\n{OUTPUT_FILE}"
)