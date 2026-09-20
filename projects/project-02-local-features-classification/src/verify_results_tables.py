
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


FILES = [
    "cric_classifiers_table.csv",
    "sipakmed_mlp33_table.csv",
    "cross_dataset_mlp33_table.csv",
]


def main():
    print("=" * 60)
    print("VERIFYING RESULTS TABLES")
    print("=" * 60)

    for filename in FILES:
        filepath = TABLES_DIR / filename

        print("\n" + "-" * 60)
        print(f"FILE: {filename}")
        print("-" * 60)

        if not filepath.exists():
            print("ERROR: File not found")
            continue

        df = pd.read_csv(filepath)

        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nContent:")
        print(df.to_string(index=False))

        if "GMM" in df.to_string().upper():
            print("\nWARNING: GMM reference found")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()