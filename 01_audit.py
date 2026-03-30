import pandas as pd
import numpy as np
from config import RAW_DATA_PATH

df = pd.read_csv(RAW_DATA_PATH)

# --- Basic shape ---
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# --- Data types ---
print("=== Dtypes ===")
print(df.dtypes.value_counts(), "\n")

# --- Missing values ---
missing = df.isnull().mean().sort_values(ascending=False)
print("=== Missing value rate (top 10) ===")
print(missing.head(10), "\n")

# --- Numeric distributions ---
print("=== Numeric summary ===")
print(df.describe().T[["mean", "std", "min", "max"]], "\n")

# --- Categorical columns ---
cat_cols = df.select_dtypes(include=["object", "bool"]).columns
for col in cat_cols:
    print(f"{col}: {df[col].value_counts().to_dict()}")