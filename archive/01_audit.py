import pandas as pd
import numpy as np
import sys
from config import RAW_DATA_PATH, AUDIT_REPORT_PATH
from io_utils import load_tabular_dataset

df = load_tabular_dataset(RAW_DATA_PATH)

lines = []

# --- Basic shape ---
lines.append(f"Shape: {df.shape}")
lines.append(f"Columns: {df.columns.tolist()}\n")

# --- Data types ---
lines.append("=== Dtypes ===")
lines.append(str(df.dtypes.value_counts()) + "\n")

# --- Missing values ---
missing = df.isnull().mean().sort_values(ascending=False)
lines.append("=== Missing value rate (top 10) ===")
lines.append(str(missing.head(10)) + "\n")

# --- Numeric distributions ---
lines.append("=== Numeric summary ===")
lines.append(str(df.describe().T[["mean", "std", "min", "max"]]) + "\n")

# --- Categorical columns ---
lines.append("=== Categorical columns ===")
cat_cols = df.select_dtypes(include=["object", "bool"]).columns
for col in cat_cols:
    lines.append(f"{col}: {df[col].value_counts().to_dict()}")

report = "\n".join(lines)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

try:
    print(report)
except UnicodeEncodeError:
    print(report.encode("ascii", errors="ignore").decode("ascii"))

with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n[audit] Report saved -> {AUDIT_REPORT_PATH}")
