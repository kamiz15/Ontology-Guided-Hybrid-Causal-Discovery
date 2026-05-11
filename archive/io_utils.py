import re
from pathlib import Path

import pandas as pd


def _normalise_column_name(name: str) -> str:
    normalised = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return normalised.strip("_")


def _deduplicate_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped = []

    for col in columns:
        count = counts.get(col, 0)
        deduped.append(col if count == 0 else f"{col}_{count + 1}")
        counts[col] = count + 1

    return deduped


def load_tabular_dataset(path: str) -> pd.DataFrame:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(dataset_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(
            f"Unsupported file type for {dataset_path}. Expected .csv, .xlsx, or .xls."
        )

    df.columns = _deduplicate_columns([_normalise_column_name(col) for col in df.columns])
    return df
