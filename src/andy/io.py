import pandas as pd
from pathlib import Path


def get_data_dir() -> Path:
    """Get the absolute path to the data directory.

    Returns:
        Path: Absolute path to the data directory.
    """
    package_root = Path(__file__).parent.parent
    project_root = package_root.parent
    return project_root / "data"

def load_raw_data(path="../data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def load_clean_train_data() -> pd.DataFrame:
    df = load_raw_data()
    
    # ⬇Add any cleaning steps here if needed
    # e.g., dropping nulls, converting types, etc.
    # df.dropna(inplace=True)

    return df

def load_cleaned_train_data() -> pd.DataFrame:
    return pd.read_parquet("data/train_cleaned.parquet")

