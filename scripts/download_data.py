import shutil
from pathlib import Path

import kagglehub

DATASET = "edumagalhaes/quality-prediction-in-a-mining-process"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    cache_path = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for file in cache_path.glob("*.csv"):
        shutil.copy2(file, RAW_DIR / file.name)
        print(f"Kopierade {file.name} -> {RAW_DIR}")


if __name__ == "__main__":
    main()