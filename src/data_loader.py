import pandas as pd

from .config import RAW_DATA_DIR

def load_csv(filename: str) -> pd.DataFrame:
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found : {file_path}")
    return pd.read_csv(file_path)