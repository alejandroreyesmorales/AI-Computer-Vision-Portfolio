
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_dataset"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRIC_DATA_PATH = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\semestre 4\clasficadores RMIB\BDcric30images.csv"
)

SIPAKMED_DATA_PATH = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\semestre 4\clasficadores RMIB\BDSIPKMED33.csv"
)

CRIC_MODEL_PATH = (
    PROJECT_ROOT / "results" / "mlp33_cric_full.keras"
)

SIPAKMED_MODEL_PATH = (
    PROJECT_ROOT / "results" / "sipakmed" / "mlp33_sipakmed_full.keras"
)

N_FEATURES = 33
CHUNK_SIZE = 100_000
BATCH_SIZE = 2048


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "total_samples": total,
    }


# ============================================================
# CROSS-DATASET EVALUATION
# ============================================================

def evaluate_model_on_dataset(model, data_path, model_name, dataset_name):
    print("\n" + "=" * 60)
    print(f"EVALUATION: {model_name} -> {dataset_name}")
    print("=" * 60)

    print(f"Dataset: {data_path}")

    start_time = time.time()

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    total_samples = 0
    chunk_number = 0

    for chunk in pd.read_csv(data_path, chunksize=CHUNK_SIZE):
        chunk_number += 1

        X = chunk.iloc[:, :N_FEATURES].to_numpy(dtype=np.float32)
        y_true = chunk.iloc[:, N_FEATURES].to_numpy(dtype=np.int32)

        predictions = model.predict(
            X,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        y_pred = np.argmax(predictions, axis=1)

        tn += int(np.sum((y_true == 0) & (y_pred == 0)))
        fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        fn += int(np.sum((y_true == 1) & (y_pred == 0)))
        tp += int(np.sum((y_true == 1) & (y_pred == 1)))

        total_samples += len(y_true)

        print(
            f"Processed chunk {chunk_number} | "
            f"Samples: {total_samples:,}"
        )

    evaluation_time = time.time() - start_time

    metrics = calculate_metrics(tn, fp, fn, tp)

    metrics["model_trained_on"] = model_name
    metrics["evaluated_on"] = dataset_name
    metrics["evaluation_time_seconds"] = evaluation_time

    print("\nFINAL RESULTS")
    print("-" * 60)

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    total_start_time = time.time()

    print("=" * 60)
    print("CROSS-DATASET MLP33 EVALUATION")
    print("=" * 60)

    print("\nLoading CRIC model...")
    cric_model = tf.keras.models.load_model(CRIC_MODEL_PATH)

    print("Loading SIPaKMeD model...")
    sipakmed_model = tf.keras.models.load_model(SIPAKMED_MODEL_PATH)

    all_results = []

    # --------------------------------------------------------
    # CRIC MODEL -> SIPAKMED DATASET
    # --------------------------------------------------------

    result_1 = evaluate_model_on_dataset(
        model=cric_model,
        data_path=SIPAKMED_DATA_PATH,
        model_name="CRIC",
        dataset_name="SIPaKMeD"
    )

    all_results.append(result_1)

    # --------------------------------------------------------
    # SIPAKMED MODEL -> CRIC DATASET
    # --------------------------------------------------------

    result_2 = evaluate_model_on_dataset(
        model=sipakmed_model,
        data_path=CRIC_DATA_PATH,
        model_name="SIPaKMeD",
        dataset_name="CRIC"
    )

    all_results.append(result_2)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    output_csv = RESULTS_DIR / "cross_dataset_mlp33_metrics.csv"

    results_df.to_csv(output_csv, index=False)

    output_txt = RESULTS_DIR / "cross_dataset_mlp33_metrics.txt"

    with open(output_txt, "w", encoding="utf-8") as file:
        file.write("CROSS-DATASET MLP33 EVALUATION\n")
        file.write("=" * 60 + "\n\n")

        for result in all_results:
            file.write(
                f"Model trained on: {result['model_trained_on']}\n"
            )
            file.write(
                f"Evaluated on: {result['evaluated_on']}\n"
            )

            for metric, value in result.items():
                if metric not in [
                    "model_trained_on",
                    "evaluated_on"
                ]:
                    if isinstance(value, float):
                        file.write(f"{metric}: {value:.6f}\n")
                    else:
                        file.write(f"{metric}: {value}\n")

            file.write("\n" + "-" * 60 + "\n\n")

    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("CROSS-DATASET EVALUATION COMPLETED")
    print("=" * 60)

    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {output_csv}")
    print(f"Text report saved to: {output_txt}")


if __name__ == "__main__":
    main()