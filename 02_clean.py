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
import re
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
    to_impute      = missing_rate[missing_rate > 0].index.tolist()
    high_remaining = missing_rate[missing_rate > IMPUTE_THRESHOLD].index.tolist()

    if high_remaining:
        print(f"[clean] WARNING: {len(high_remaining)} columns still have >{int(IMPUTE_THRESHOLD*100)}% "
              f"missing after dropping empties; imputing anyway to avoid row loss: {high_remaining}")

    df_imputed = df.copy()
    for col in to_impute:
        median_val = df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(median_val)

    print(f"[clean] Imputed {len(to_impute)} columns with column median.")
    return df_imputed


# ── Step F: Encode known text-based ESG columns ──────────────
def coerce_boolean_and_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = []
    bool_map = {
        True: 1, False: 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "1.0": 1, "0.0": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
    }
    allowed_bool_values = set(bool_map.keys())

    df_out = df.copy()

    for col in df_out.columns:
        series = df_out[col]

        if pd.api.types.is_bool_dtype(series):
            df_out[col] = series.astype(int)
            converted.append(f"{col} (bool -> 0/1)")
            continue

        if series.dtype != object:
            continue

        normalized = series.dropna().astype(str).str.strip().str.lower()
        unique_vals = set(normalized.unique())

        if unique_vals and unique_vals.issubset(allowed_bool_values) and len(unique_vals) <= 4:
            df_out[col] = series.apply(
                lambda v: np.nan if pd.isna(v) else bool_map.get(str(v).strip().lower(), np.nan)
            )
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
            converted.append(f"{col} (boolean-like -> 0/1)")
            continue

        numeric_test = pd.to_numeric(series, errors="coerce")
        non_null_ratio = numeric_test.notna().mean()
        if non_null_ratio > 0.5:
            df_out[col] = numeric_test
            converted.append(f"{col} (mostly numeric object -> numeric)")

    if "systemic_risk_level" in df_out.columns:
        df_out["systemic_risk_level"] = df_out["systemic_risk_level"].map({
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4,
        })
        converted.append("systemic_risk_level (ordinal)")

    if converted:
        print(f"[clean] Coerced {len(converted)} columns before numeric filtering:")
        for item in converted:
            print(f"    {item}")
    else:
        print("[clean] No boolean-like or mostly numeric object columns needed coercion.")

    return df_out


def encode_text_columns(df: pd.DataFrame) -> pd.DataFrame:

    def qualitative_score(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower()
        if any(k in s for k in ['very good', 'third-party verified', 'externally assured',
                                 'high (assured', 'high (verified', 'high (validated',
                                 'high / independent', 'high (external', 'gri-aligned',
                                 'independent limited assurance']):
            return 4.0
        if any(k in s for k in ['documented', 'high', 'qualitative: regulatory',
                                 'qualitative: formalized', 'qualitative: multi',
                                 'qualitative: aligned', 'qualitative: tcfd',
                                 'qualitative: gri', 'qualitative: prb',
                                 'qualitative: detailed', 'qualitative (see',
                                 'comprehensive', 'regulatory compliance']):
            return 3.0
        if 'qualitative' in s:
            return 3.0
        if 'moderate' in s:
            return 2.0
        if 'limited' in s:
            return 1.0
        return np.nan

    def extract_first_number(val, unit_multipliers=None):
        if pd.isna(val):
            return np.nan
        s = str(val).replace(',', '')
        match = re.search(r'\d+(?:\.\d+)?', s)
        if not match:
            return np.nan
        num = float(match.group())
        if unit_multipliers:
            for unit, mult in unit_multipliers.items():
                if unit in s.lower():
                    num *= mult
                    break
        return num

    def extract_percentage(val):
        if pd.isna(val):
            return np.nan
        s = str(val).replace(',', '')
        match = re.search(r'\d+(?:\.\d+)?', s)
        if not match:
            return np.nan
        num = float(match.group())
        if num > 1:
            num = num / 100.0
        return min(num, 1.0)

    def extract_financial_amount(val):
        if pd.isna(val):
            return np.nan

        s = str(val).strip()
        s_lower = s.lower()

        if not s:
            return np.nan

        if not re.search(r'\d', s):
            return qualitative_score(s)

        matches = list(re.finditer(r'\d[\d,]*(?:\.\d+)?', s))
        if not matches:
            return np.nan

        candidates = []
        for match in matches:
            raw = match.group().replace(',', '')
            try:
                num = float(raw)
            except ValueError:
                continue

            tail = s_lower[match.end(): match.end() + 20]
            if "billion" in tail or re.match(r'\s*bn\b', tail):
                mult = 1_000_000_000
            elif "million" in tail or re.match(r'\s*m\b', tail):
                mult = 1_000_000
            elif "thousand" in tail or re.match(r'\s*k\b', tail):
                mult = 1_000
            else:
                mult = 1

            candidates.append(num * mult)

        if not candidates:
            return np.nan

        return max(candidates)

    qualitative_cols = [
        'emission_reduction_policy', 'health_safety',
        'board_strategy_esg_oversight', 'reporting_quality',
    ]
    ghg_units = {'kton': 1000, 'mton': 1_000_000, 'million t': 1_000_000}
    ghg_cols = ['scope_1_ghg_emissions', 'scope_2_ghg_emissions', 'scope_3_ghg_emissions']
    pct_cols = ['renewable_energy_share', 'diversity_women_representation']
    money_cols = ['community_investment', 'sustainable_finance_green_financing', 'total_revenue']

    for col in qualitative_cols:
        if col in df.columns:
            df[col] = df[col].apply(qualitative_score)
            print(f"[encode] {col}: ordinal 1-4 scale")

    for col in ghg_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: extract_first_number(v, ghg_units))
            print(f"[encode] {col}: numeric extracted (tCO2e)")

    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_percentage)
            print(f"[encode] {col}: percentage extracted (0-1)")

    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_financial_amount)
            print(f"[encode] {col}: financial magnitude extracted")

    return df


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

    df = coerce_boolean_and_numeric_columns(df)

    df = encode_text_columns(df)

    # Drop any remaining non-numeric columns — causal discovery requires numeric data
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"[clean] Dropping {len(non_numeric)} non-numeric columns (not usable in causal discovery): {non_numeric}")
        df = df.drop(columns=non_numeric)
    else:
        print("[clean] All remaining columns are numeric.")

    if df.empty or df.shape[1] == 0:
        print("[clean] WARNING: No numeric columns remain after cleaning. Check your input data.")
        return df

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
