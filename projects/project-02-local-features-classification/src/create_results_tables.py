
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ METRICS FROM TXT
# ============================================================

def read_metrics_txt(filepath):

    metrics = {}

    with open(filepath, "r", encoding="utf-8") as file:

        for line in file:

            if ":" not in line:
                continue

            key, value = line.strip().split(":", 1)

            key = key.strip()
            value = value.strip()

            try:
                metrics[key] = float(value)
            except ValueError:
                continue

    return metrics


# ============================================================
# 1. CRIC CLASSIFIERS + MLP33
# ============================================================

def create_cric_table():

    classifiers_path = (
        RESULTS_DIR
        / "cric"
        / "cric_all_classifiers_metrics.csv"
    )

    mlp_metrics_path = (
        RESULTS_DIR
        / "mlp33_cric_full_metrics.txt"
    )

    if not classifiers_path.exists():
        raise FileNotFoundError(
            f"File not found: {classifiers_path}"
        )

    if not mlp_metrics_path.exists():
        raise FileNotFoundError(
            f"File not found: {mlp_metrics_path}"
        )

    # --------------------------------------------------------
    # Read additional classifiers
    # --------------------------------------------------------

    classifiers_df = pd.read_csv(classifiers_path)

    # Exclude GMM from the curated public table
    if "classifier" in classifiers_df.columns:

        classifiers_df = classifiers_df[
            ~classifiers_df["classifier"]
            .astype(str)
            .str.lower()
            .str.contains("gmm")
        ]

    # --------------------------------------------------------
    # Read MLP33 metrics
    # --------------------------------------------------------

    mlp_metrics = read_metrics_txt(mlp_metrics_path)

    mlp_row = {
        "classifier": "mlp33",
        "accuracy": mlp_metrics.get("accuracy"),
        "precision": mlp_metrics.get("precision"),
        "recall": mlp_metrics.get("recall"),
        "specificity": mlp_metrics.get("specificity"),
        "f1_score": mlp_metrics.get("f1_score"),
    }

    mlp_df = pd.DataFrame([mlp_row])

    # --------------------------------------------------------
    # Select only comparison metrics from classifiers
    # --------------------------------------------------------

    selected_columns = [
        "classifier",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    ]

    classifiers_df = classifiers_df[selected_columns]

    # --------------------------------------------------------
    # Combine MLP33 and additional classifiers
    # --------------------------------------------------------

    final_df = pd.concat(
        [mlp_df, classifiers_df],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save CRIC table
    # --------------------------------------------------------

    output_path = (
        TABLES_DIR / "cric_classifiers_table.csv"
    )

    final_df.to_csv(output_path, index=False)

    print(f"CRIC table saved to: {output_path}")
    print("\nCRIC table:")
    print(final_df.to_string(index=False))

    return final_df


# ============================================================
# 2. SIPAKMED MLP33
# ============================================================

def create_sipakmed_table():

    input_path = (
        RESULTS_DIR
        / "sipakmed"
        / "mlp33_sipakmed_full_metrics.txt"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    metrics = read_metrics_txt(input_path)

    selected_metrics = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    ]

    table_data = {
        metric: [metrics.get(metric)]
        for metric in selected_metrics
    }

    table_data["dataset"] = ["SIPaKMeD"]
    table_data["classifier"] = ["MLP33"]

    df = pd.DataFrame(table_data)

    columns = [
        "dataset",
        "classifier",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    ]

    df = df[columns]

    output_path = (
        TABLES_DIR / "sipakmed_mlp33_table.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"SIPaKMeD table saved to: {output_path}")
    print("\nSIPaKMeD table:")
    print(df.to_string(index=False))

    return df


# ============================================================
# 3. CROSS-DATASET
# ============================================================

def create_cross_dataset_table():

    input_path = (
        RESULTS_DIR
        / "cross_dataset"
        / "cross_dataset_mlp33_metrics.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    selected_columns = [
        "model_trained_on",
        "evaluated_on",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    ]

    df = df[selected_columns]

    output_path = (
        TABLES_DIR / "cross_dataset_mlp33_table.csv"
    )

    df.to_csv(output_path, index=False)

    print(
        f"Cross-dataset table saved to: {output_path}"
    )

    print("\nCross-dataset table:")
    print(df.to_string(index=False))

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATING FINAL RESULTS TABLES")
    print("=" * 60)

    create_cric_table()

    create_sipakmed_table()

    create_cross_dataset_table()

    print("\n" + "=" * 60)
    print("ALL TABLES CREATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()