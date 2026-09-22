import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualización PCA de características extraídas con "
            "ResNet-101 ImageNet y ResNet-101 fine-tuned FT9."
        )
    )

    parser.add_argument(
        "--base-csv",
        type=Path,
        default=Path(
            "results/resnet101_imagenet/"
            "resnet101_imagenet_dev_features.csv"
        ),
        help="CSV de características extraídas con pesos ImageNet."
    )

    parser.add_argument(
        "--ft-csv",
        type=Path,
        required=True,
        help="CSV de características extraídas con la configuración FT9."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/PCA_features"),
        help="Directorio donde se guardarán las figuras."
    )

    return parser.parse_args()


# ============================================================
# CARGA DE CARACTERÍSTICAS
# ============================================================

def load_features(csv_path):
    df = pd.read_csv(csv_path)

    # Se excluyen las dos últimas columnas:
    # label y group_id.
    X = df.iloc[:, :-2].to_numpy()

    labels = df["label"].to_numpy()

    return df, X, labels


# ============================================================
# VISUALIZACIÓN PCA
# ============================================================

def plot_pca(X_pca, y_encoded, label_encoder, title, output_path):
    plt.figure(figsize=(8, 6))

    for label_index, label_name in enumerate(label_encoder.classes_):
        indices = y_encoded == label_index

        plt.scatter(
            X_pca[indices, 0],
            X_pca[indices, 1],
            s=18,
            alpha=0.6,
            label=label_name
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)

    plt.legend(
        loc="best",
        fontsize=9,
        ncol=2,
        frameon=True
    )

    plt.tight_layout()

    # PNG para el README y PDF para documentación/publicación.
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close()

    print(f"Figura guardada en: {output_path.with_suffix('.png')}")
    print(f"Figura guardada en: {output_path.with_suffix('.pdf')}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar ambos conjuntos de características.
    df_base, X_base, y_base = load_features(args.base_csv)
    df_ft, X_ft, y_ft = load_features(args.ft_csv)

    # Comprobar que ambos archivos contienen el mismo número de muestras.
    if len(df_base) != len(df_ft):
        raise ValueError(
            "Los CSV no contienen el mismo número de muestras: "
            f"ImageNet={len(df_base)}, FT9={len(df_ft)}."
        )

    # Comprobar que las etiquetas y el orden de las muestras coinciden.
    if not (y_base == y_ft).all():
        raise ValueError(
            "Las etiquetas de ImageNet y FT9 no coinciden o están "
            "en un orden diferente."
        )

    # Codificar etiquetas para colorear las clases.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_base)

    # Combinar las características para que ambas representaciones
    # se proyecten usando el mismo espacio PCA y los mismos ejes.
    X_combined = pd.concat(
        [
            pd.DataFrame(X_base),
            pd.DataFrame(X_ft)
        ],
        axis=0,
        ignore_index=True
    ).to_numpy()

    # PCA común para ambas representaciones.
    pca = PCA(n_components=2)
    X_combined_pca = pca.fit_transform(X_combined)

    n_samples = len(X_base)

    X_base_pca = X_combined_pca[:n_samples]
    X_ft_pca = X_combined_pca[n_samples:]

    explained_variance = pca.explained_variance_ratio_

    print(
        "Varianza explicada por PC1 y PC2: "
        f"{explained_variance[0]:.4f}, "
        f"{explained_variance[1]:.4f}"
    )
    print(
        "Varianza explicada acumulada: "
        f"{explained_variance.sum():.4f}"
    )

    # Generar las figuras.
    plot_pca(
        X_base_pca,
        y_encoded,
        label_encoder,
        "ResNet-101 (ImageNet)",
        args.output_dir / "PCA_resnet101_imagenet"
    )

    plot_pca(
        X_ft_pca,
        y_encoded,
        label_encoder,
        "ResNet-101 Fine-Tuned (FT9)",
        args.output_dir / "PCA_resnet101_FT9"
    )


if __name__ == "__main__":
    main()
