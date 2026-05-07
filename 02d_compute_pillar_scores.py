 # 02d_compute_pillar_scores.py
# ============================================================
# Step 02d - Compute ESG pillar scores for real and synthetic data.
#
# Adds environmental, social, governance, and overall ESG composite
# scores to the causal-ready real table and synthetic datasets.
#
# Usage:
#   python 02d_compute_pillar_scores.py
#   python 02d_compute_pillar_scores.py --dry-run
#
# Output:
#   data/processed/data_ready.csv
#   data/synthetic/synthetic_n110.csv
#   data/synthetic/synthetic_n500.csv
#   data/synthetic/synthetic_n2000.csv
#   data/synthetic/ground_truth_adjacency.csv
# ============================================================

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd

from config import READY_DATA_PATH, SYNTHETIC_DIR, SYNTHETIC_GROUND_TRUTH_PATH


PILLAR_COLUMNS = {
    "Environmental": "env_pillar_score",
    "Social": "soc_pillar_score",
    "Governance": "gov_pillar_score",
}
OVERALL_ESG_COLUMN = "overall_esg_score"
PILLAR_SCORE_COLUMNS = list(PILLAR_COLUMNS.values()) + [OVERALL_ESG_COLUMN]

DATASET_PATHS = [
    READY_DATA_PATH,
    os.path.join(SYNTHETIC_DIR, "synthetic_n110.csv"),
    os.path.join(SYNTHETIC_DIR, "synthetic_n500.csv"),
    os.path.join(SYNTHETIC_DIR, "synthetic_n2000.csv"),
]

# Current real/synthetic names are normalized for analysis, while Step 3 keeps
# some workbook-era base names. These aliases preserve Step 3's pillar schema.
COLUMN_ALIASES = {
    "scope_1_emissions_tco2e": "scope_1_ghg_emissions",
    "scope_2_emissions_tco2e": "scope_2_ghg_emissions",
    "scope_3_emissions_tco2e": "scope_3_ghg_emissions",
    "emission_reduction_policy_score": "emission_reduction_policy",
    "community_investment_eur": "community_investment",
    "health_safety_score": "health_safety",
    "board_strategy_esg_oversight_score": "board_strategy_esg_oversight",
    "green_financing_eur": "sustainable_finance_green_financing",
    "total_revenue_eur": "total_revenue",
}


def load_step3_pillar_lookup(script_path: str = "03_build_column_mapping.py") -> dict[str, str]:
    """
    Load variable-to-pillar mappings from Step 3's known schema.

    Parameters
    ----------
    script_path : str, optional
        Path to `03_build_column_mapping.py`.

    Returns
    -------
    dict[str, str]
        Mapping from variable name to one of Environmental, Social,
        Governance, or Financial.

    Raises
    ------
    RuntimeError
        If Step 3 cannot be imported or does not expose KNOWN_MAPPINGS.
    """
    spec = importlib.util.spec_from_file_location("step03_build_column_mapping", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Step 3 mapping script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    known_mappings = getattr(module, "KNOWN_MAPPINGS", None)
    if not isinstance(known_mappings, dict):
        raise RuntimeError("Step 3 mapping script does not expose KNOWN_MAPPINGS.")

    return {
        column_name: values[0]
        for column_name, values in known_mappings.items()
        if values and values[0] in {"Environmental", "Social", "Governance", "Financial"}
    }


def infer_pillar(column: str, pillar_lookup: dict[str, str]) -> str | None:
    """
    Infer the Step 3 pillar for a dataset column.

    Parameters
    ----------
    column : str
        Dataset column name.
    pillar_lookup : dict[str, str]
        Variable-to-pillar lookup loaded from Step 3.

    Returns
    -------
    str or None
        The inferred pillar, or None if the column is not an ESG child.
    """
    if column in PILLAR_SCORE_COLUMNS:
        return None

    candidates = [
        column,
        COLUMN_ALIASES.get(column, ""),
    ]

    if column.endswith("_score"):
        candidates.append(column.removesuffix("_score"))
    if column.endswith("_eur"):
        candidates.append(column.removesuffix("_eur"))
    if column.endswith("_tco2e"):
        candidates.append(column.removesuffix("_tco2e"))

    for candidate in candidates:
        if candidate and candidate in pillar_lookup:
            pillar = pillar_lookup[candidate]
            if pillar in PILLAR_COLUMNS:
                return pillar
            return None
    return None


def zscore_series(series: pd.Series) -> pd.Series:
    """
    Z-score a numeric series with NaN-safe mean and standard deviation.

    Parameters
    ----------
    series : pd.Series
        Input values.

    Returns
    -------
    pd.Series
        Z-scored values. Constant columns return 0.0 for non-missing rows.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean(skipna=True)
    std = numeric.std(skipna=True)

    if pd.isna(std) or std == 0:
        return numeric.where(numeric.isna(), 0.0)
    return (numeric - mean) / std


def minmax_0_100(series: pd.Series) -> pd.Series:
    """
    Min-max scale a series to the [0, 100] interval.

    Parameters
    ----------
    series : pd.Series
        Input values.

    Returns
    -------
    pd.Series
        Scaled values. Constant non-missing inputs return 50.0.
    """
    min_value = series.min(skipna=True)
    max_value = series.max(skipna=True)

    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(np.nan, index=series.index, dtype=float)
    if max_value == min_value:
        return series.where(series.isna(), 50.0).astype(float)
    return ((series - min_value) / (max_value - min_value)) * 100.0


def get_pillar_children(columns: list[str], pillar_lookup: dict[str, str]) -> dict[str, list[str]]:
    """
    Identify ESG child variables present in a dataset.

    Parameters
    ----------
    columns : list[str]
        Dataset columns.
    pillar_lookup : dict[str, str]
        Variable-to-pillar lookup loaded from Step 3.

    Returns
    -------
    dict[str, list[str]]
        Pillar name to child-column list.
    """
    children = {pillar: [] for pillar in PILLAR_COLUMNS}
    for column in columns:
        pillar = infer_pillar(column, pillar_lookup)
        if pillar in children:
            children[pillar].append(column)
    return children


def compute_pillar_scores(
    df: pd.DataFrame,
    pillar_lookup: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Append ESG pillar scores and an overall ESG score to a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    pillar_lookup : dict[str, str]
        Variable-to-pillar lookup loaded from Step 3.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Updated dataset and per-pillar summary metadata.
    """
    out = df.drop(columns=[c for c in PILLAR_SCORE_COLUMNS if c in df.columns]).copy()
    children = get_pillar_children(out.columns.tolist(), pillar_lookup)
    summary: dict[str, Any] = {"children": children, "stats": {}}

    for pillar, output_column in PILLAR_COLUMNS.items():
        child_columns = children[pillar]
        if not child_columns:
            print(f"[step_02d] WARNING: no {pillar.lower()} child columns found.")
            out[output_column] = np.nan
            continue

        zscores = pd.DataFrame({col: zscore_series(out[col]) for col in child_columns})
        raw_score = zscores.mean(axis=1, skipna=True)
        out[output_column] = minmax_0_100(raw_score)

    score_columns_present = [col for col in PILLAR_COLUMNS.values() if col in out.columns]
    out[OVERALL_ESG_COLUMN] = out[score_columns_present].mean(axis=1, skipna=True)

    for column in PILLAR_SCORE_COLUMNS:
        if column in out.columns:
            stats = out[column].agg(["mean", "std", "min", "max"])
            summary["stats"][column] = {
                key: float(value) if not pd.isna(value) else np.nan
                for key, value in stats.items()
            }

    return out, summary


def backup_file(path: str, suffix: str) -> None:
    """
    Back up a file with a fixed suffix.

    Parameters
    ----------
    path : str
        File to back up.
    suffix : str
        Backup suffix, such as `.bak3`.
    """
    if not os.path.exists(path):
        return

    backup_path = f"{path}{suffix}"
    if suffix == ".bak3" and os.path.exists(backup_path):
        print(f"[step_02d] Backup already exists, preserving -> {backup_path}")
        return

    shutil.copy2(path, backup_path)
    print(f"[step_02d] Backup -> {backup_path}")


def process_dataset(
    path: str,
    pillar_lookup: dict[str, str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Compute and optionally write pillar scores for one CSV dataset.

    Parameters
    ----------
    path : str
        Dataset CSV path.
    pillar_lookup : dict[str, str]
        Variable-to-pillar lookup loaded from Step 3.
    dry_run : bool, optional
        If True, do not write output files.

    Returns
    -------
    dict[str, Any]
        Summary metadata for the dataset.
    """
    df = pd.read_csv(path)
    updated, summary = compute_pillar_scores(df, pillar_lookup)

    print(f"[step_02d] Dataset: {path}")
    print(f"[step_02d] Shape: {df.shape} -> {updated.shape}")
    for pillar in PILLAR_COLUMNS:
        child_count = len(summary["children"][pillar])
        print(f"[step_02d] {pillar} children: {child_count} {summary['children'][pillar]}")

    stats_df = pd.DataFrame(summary["stats"]).T[["mean", "std", "min", "max"]]
    print("[step_02d] Pillar score stats:")
    print(stats_df.round(3).to_string())

    if dry_run:
        print(f"[step_02d] Dry run: would write -> {path}")
    else:
        backup_file(path, ".bak3")
        updated.to_csv(path, index=False)
        print(f"[step_02d] Saved -> {path}")

    return summary


def update_synthetic_ground_truth(
    path: str,
    pillar_lookup: dict[str, str],
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Expand the synthetic ground truth with deterministic pillar-score edges.

    Parameters
    ----------
    path : str
        Ground-truth adjacency CSV path.
    pillar_lookup : dict[str, str]
        Variable-to-pillar lookup loaded from Step 3.
    dry_run : bool, optional
        If True, do not write output files.

    Returns
    -------
    pd.DataFrame
        Expanded adjacency matrix.
    """
    adjacency = pd.read_csv(path, index_col=0)
    adjacency = adjacency.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    base_variables = [col for col in adjacency.index.tolist() if col not in PILLAR_SCORE_COLUMNS]
    expanded_variables = base_variables + PILLAR_SCORE_COLUMNS
    expanded = pd.DataFrame(0, index=expanded_variables, columns=expanded_variables, dtype=int)
    expanded.loc[base_variables, base_variables] = adjacency.loc[base_variables, base_variables]

    children = get_pillar_children(base_variables, pillar_lookup)
    for pillar, output_column in PILLAR_COLUMNS.items():
        for child in children[pillar]:
            expanded.loc[child, output_column] = 1

    for pillar_column in PILLAR_COLUMNS.values():
        expanded.loc[pillar_column, OVERALL_ESG_COLUMN] = 1

    print(f"[step_02d] Ground truth: {adjacency.shape} -> {expanded.shape}")
    for pillar, output_column in PILLAR_COLUMNS.items():
        print(f"[step_02d] Ground-truth edges into {output_column}: {len(children[pillar])}")
    print(f"[step_02d] Ground-truth edges into {OVERALL_ESG_COLUMN}: {len(PILLAR_COLUMNS)}")

    if dry_run:
        print(f"[step_02d] Dry run: would write -> {path}")
    else:
        backup_file(path, ".bak")
        expanded.to_csv(path)
        print(f"[step_02d] Saved -> {path}")

    return expanded


def main() -> None:
    """Run Step 02d from the command line."""
    parser = argparse.ArgumentParser(description="Compute ESG pillar scores.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying files.")
    args = parser.parse_args()

    pillar_lookup = load_step3_pillar_lookup()

    print("[step_02d] Computing pillar scores")
    for path in DATASET_PATHS:
        if not os.path.exists(path):
            print(f"[step_02d] WARNING: dataset missing, skipping: {path}")
            continue
        process_dataset(path, pillar_lookup, dry_run=args.dry_run)

    if os.path.exists(SYNTHETIC_GROUND_TRUTH_PATH):
        update_synthetic_ground_truth(
            SYNTHETIC_GROUND_TRUTH_PATH,
            pillar_lookup,
            dry_run=args.dry_run,
        )
    else:
        print(f"[step_02d] WARNING: ground truth missing, skipping: {SYNTHETIC_GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[step_02d] ERROR: pillar score computation failed.")
        traceback.print_exc()
        sys.exit(1)
