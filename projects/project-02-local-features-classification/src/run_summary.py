
from pathlib import Path

from summary_datasets import summarize_dataset


DATA_DIR = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\semestre 4\clasficadores RMIB"
)

CRIC_PATH = DATA_DIR / "BDcric30images.csv"

SIPAKMED_PATH = DATA_DIR / "BDSIPKMED33.csv"


if __name__ == "__main__":

    print("\n===== CRIC DATASET =====")
    summarize_dataset(CRIC_PATH)

    print("\n===== SIPAKMeD DATASET =====")
    summarize_dataset(SIPAKMED_PATH)