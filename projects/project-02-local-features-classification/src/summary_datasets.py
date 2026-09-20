
from pathlib import Path
import pandas as pd


def summarize_dataset(data_path, chunksize=100_000):

    data_path = Path(data_path)

    total_records = 0
    missing_values = 0
    global_min = float("inf")
    global_max = float("-inf")
    class_counts = {}

    first_chunk = True
    number_of_features = None

    for chunk in pd.read_csv(
        data_path,
        header=None,
        chunksize=chunksize
    ):

        if first_chunk:

            number_of_features = chunk.shape[1] - 1

            print("\nFirst five rows:")
            print(chunk.head())

            print("\nColumn structure:")
            print(f"Total columns: {chunk.shape[1]}")
            print(f"Features: {number_of_features}")
            print("Last column: Class label")

            first_chunk = False

        total_records += len(chunk)

        missing_values += chunk.isnull().sum().sum()

        features = chunk.iloc[:, :-1]
        labels = chunk.iloc[:, -1]

        chunk_min = features.min().min()
        chunk_max = features.max().max()

        global_min = min(global_min, chunk_min)
        global_max = max(global_max, chunk_max)

        counts = labels.value_counts()

        for label, count in counts.items():
            class_counts[label] = class_counts.get(label, 0) + count

    print("\n========== DATASET SUMMARY ==========")
    print(f"File: {data_path.name}")
    print(f"Total records: {total_records}")
    print(f"Number of features: {number_of_features}")
    print(f"Missing values: {missing_values}")
    print(f"Global minimum: {global_min}")
    print(f"Global maximum: {global_max}")

    print("\nClass distribution:")

    for label, count in sorted(class_counts.items()):
        percentage = (count / total_records) * 100
        print(f"Class {label}: {count:,} ({percentage:.2f}%)")

    print("\nAnalysis completed.")