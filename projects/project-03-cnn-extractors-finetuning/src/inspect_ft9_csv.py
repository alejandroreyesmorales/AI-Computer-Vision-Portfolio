
"""
Compact inspection of FT9 feature CSV files.

Reports:
- Dataset dimensions
- Feature count
- Data types summary
- Missing values
- Class distribution
- Group information
- Feature statistics
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\Semestre 5\Experimentos Extractores corregidos\FT9_classification"
)

DEV_FILE = BASE_DIR / "FT9_20.csv"
TEST_FILE = BASE_DIR / "FT9_test_20.csv"


# ---------------------------------------------------------
# Inspection function
# ---------------------------------------------------------

def inspect_csv(file_path: Path) -> None:
    """Print a compact summary of a CSV file."""

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    if not file_path.exists():
        print(f"ERROR: File not found:\n{file_path}")
        return

    data = pd.read_csv(file_path)

    # -----------------------------------------------------
    # General information
    # -----------------------------------------------------

    print("\n--- GENERAL INFORMATION ---")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    label_column = "label"
    group_column = "group_id"

    feature_columns = [
        column
        for column in data.columns
        if column not in [label_column, group_column]
    ]

    print(f"Feature columns: {len(feature_columns)}")

    # -----------------------------------------------------
    # Data types summary
    # -----------------------------------------------------

    print("\n--- DATA TYPES SUMMARY ---")

    print(
        data.dtypes.value_counts().to_string()
    )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print("\n--- MISSING VALUES ---")

    total_missing = int(data.isnull().sum().sum())

    print(f"Total missing values: {total_missing}")

    if total_missing > 0:
        missing_columns = data.isnull().sum()
        missing_columns = missing_columns[
            missing_columns > 0
        ]

        print("\nColumns with missing values:")
        print(missing_columns.to_string())

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print("\n--- CLASS DISTRIBUTION ---")

    if label_column in data.columns:

        class_counts = data[label_column].value_counts()

        print(f"Number of classes: {class_counts.shape[0]}")
        print(f"Class balance:\n{class_counts.to_string()}")

    else:
        print("Label column not found.")

    # -----------------------------------------------------
    # Group information
    # -----------------------------------------------------

    print("\n--- GROUP INFORMATION ---")

    if group_column in data.columns:

        number_of_groups = data[group_column].nunique()

        print(f"Unique groups: {number_of_groups}")
        print(
            f"Minimum samples per group: "
            f"{data[group_column].value_counts().min()}"
        )
        print(
            f"Maximum samples per group: "
            f"{data[group_column].value_counts().max()}"
        )

    else:
        print("Group column not found.")

    # -----------------------------------------------------
    # Feature statistics
    # -----------------------------------------------------

    print("\n--- FEATURE STATISTICS ---")

    numeric_features = data[feature_columns].select_dtypes(
        include="number"
    )

    print(f"Numeric features: {numeric_features.shape[1]}")

    print(
        f"Features with zero variance: "
        f"{(numeric_features.std() == 0).sum()}"
    )

    print(
        f"Minimum feature value: "
        f"{numeric_features.min().min():.6f}"
    )

    print(
        f"Maximum feature value: "
        f"{numeric_features.max().max():.6f}"
    )

    print(
        f"Mean feature value: "
        f"{numeric_features.mean().mean():.6f}"
    )

    print(
        f"Standard deviation mean: "
        f"{numeric_features.std().mean():.6f}"
    )

    # -----------------------------------------------------
    # Sample preview
    # -----------------------------------------------------

    print("\n--- SAMPLE PREVIEW ---")

    preview_columns = feature_columns[:3] + [
        label_column,
        group_column,
    ]

    print(
        data[preview_columns].head(3).to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    inspect_csv(DEV_FILE)
    inspect_csv(TEST_FILE)