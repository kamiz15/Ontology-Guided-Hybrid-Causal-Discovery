# 02_clean.py
# ============================================================
# Step 2 — Dataset cleaning.
# Drops fully empty columns, excludes metadata/admin columns,
# flags and removes highly correlated pairs, imputes moderate
# missingness with column median.
#
# Usage:
#   python 02_clean.py
#   python 02_clean.py --input data/raw/my_file.csv
#
# Output:
#   data/processed/data_clean.csv   (empty cols dropped)
#   data/processed/data_ready.csv   (imputed, causal-ready)
#   reports/high_correlation_pairs.csv
# ============================================================

import argparse
import pandas as pd
import numpy as np
from config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    READY_DATA_PATH,
    HIGH_CORR_PATH,
    EXCLUDE_FROM_CAUSAL,
    MISSING_DROP_THRESHOLD,
    MISSING_FLAG_THRESHOLD,
    IMPUTE_THRESHOLD,
    HIGH_CORR_THRESHOLD,
    NEAR_CONSTANT_STD_RATIO,
)
from io_utils import load_tabular_dataset


# ── Step A: Drop fully empty columns ─────────────────────────
def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_rate = df.isnull().mean()
    fully_empty  = missing_rate[missing_rate >= MISSING_DROP_THRESHOLD].index.tolist()
    print(f"[clean] Dropping {len(fully_empty)} fully empty columns: {fully_empty}")
    return df.drop(columns=fully_empty)


# ── Step B: Exclude metadata / non-causal columns ────────────
def exclude_metadata(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in EXCLUDE_FROM_CAUSAL if c in df.columns]
    not_found = [c for c in EXCLUDE_FROM_CAUSAL if c not in df.columns]
    if not_found:
        print(f"[clean] Note: these exclusion cols not found in data: {not_found}")
    print(f"[clean] Excluding {len(to_drop)} metadata/admin columns.")
    return df.drop(columns=to_drop)


# ── Step C: Flag near-constant columns ───────────────────────
def flag_near_constant(df: pd.DataFrame) -> list:
    numeric = df.select_dtypes(include="number")
    mean_abs = numeric.mean().abs()
    std      = numeric.std()
    # avoid division by zero for columns where mean is 0
    ratio    = std / mean_abs.replace(0, np.nan)
    near_const = ratio[ratio < NEAR_CONSTANT_STD_RATIO].index.tolist()
    if near_const:
        print(f"[clean] Near-constant columns (std < {NEAR_CONSTANT_STD_RATIO} × mean): {near_const}")
    else:
        print("[clean] No near-constant columns found.")
    return near_const


# ── Step D: Find and log highly correlated pairs ─────────────
def find_high_correlations(df: pd.DataFrame, output_path: str) -> list:
    numeric = df.select_dtypes(include="number")
    corr    = numeric.corr().abs()

    pairs = []
    cols  = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = corr.loc[c1, c2]
            if val > HIGH_CORR_THRESHOLD:
                pairs.append({"col_a": c1, "col_b": c2, "correlation": round(val, 4)})

    pairs_df = pd.DataFrame(pairs, columns=["col_a", "col_b", "correlation"]).sort_values("correlation", ascending=False)
    pairs_df.to_csv(output_path, index=False)

    print(f"[clean] High correlation pairs (>{HIGH_CORR_THRESHOLD}): {len(pairs_df)}")
    print(f"[clean] Saved -> {output_path}")
    if not pairs_df.empty:
        print(pairs_df.head(15).to_string(index=False))

    # Return list of columns to consider dropping (second col in each pair)
    # Conservative: only auto-drop if correlation == 1.0 (exact duplicates)
    exact_dupes = [row["col_b"] for _, row in pairs_df.iterrows() if row["correlation"] == 1.0]
    if exact_dupes:
        print(f"[clean] Exact duplicates to drop: {exact_dupes}")
    return exact_dupes


# ── Step E: Impute moderate missingness ──────────────────────
def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols   = df.select_dtypes(include="number").columns
    missing_rate   = df[numeric_cols].isnull().mean()
    to_impute      = missing_rate[(missing_rate > 0) & (missing_rate <= IMPUTE_THRESHOLD)].index.tolist()
    high_remaining = missing_rate[missing_rate > IMPUTE_THRESHOLD].index.tolist()

    if high_remaining:
        print(f"[clean] WARNING: {len(high_remaining)} columns still have >{int(IMPUTE_THRESHOLD*100)}% "
              f"missing after dropping empties — review manually: {high_remaining}")

    df_imputed = df.copy()
    for col in to_impute:
        median_val = df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(median_val)

    print(f"[clean] Imputed {len(to_impute)} columns with column median.")
    return df_imputed


# ── Main ──────────────────────────────────────────────────────
def clean_dataset(input_path: str,
                  clean_path: str,
                  ready_path: str,
                  corr_path: str) -> pd.DataFrame:

    df = load_tabular_dataset(input_path)
    print(f"[clean] Input shape: {df.shape}")

    df = drop_empty_columns(df)
    df.to_csv(clean_path, index=False)
    print(f"[clean] After dropping empty cols: {df.shape} -> saved {clean_path}")

    df = exclude_metadata(df)

    flag_near_constant(df)

    exact_dupes = find_high_correlations(df, corr_path)
    if exact_dupes:
        df = df.drop(columns=exact_dupes)
        print(f"[clean] Dropped {len(exact_dupes)} exact-duplicate columns.")

    df = impute_median(df)
    df.to_csv(ready_path, index=False)
    print(f"[clean] Final causal-ready shape: {df.shape} -> saved {ready_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and prepare dataset for causal discovery.")
    parser.add_argument("--input",  default=RAW_DATA_PATH,   help="Path to raw CSV")
    parser.add_argument("--clean",  default=CLEAN_DATA_PATH, help="Path for clean output")
    parser.add_argument("--ready",  default=READY_DATA_PATH, help="Path for causal-ready output")
    parser.add_argument("--corr",   default=HIGH_CORR_PATH,  help="Path for correlation report")
    args = parser.parse_args()

    clean_dataset(args.input, args.clean, args.ready, args.corr)
