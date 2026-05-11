"""Split compound constraint-review rows into individual edge rows."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


LOG_PREFIX = "[split]"
REVIEW_PATH = Path("reports/constraints_for_review.csv")
BACKUP_PATH = Path("reports/constraints_for_review.csv.bak_compound")


def log(message: str) -> None:
    """Print a script-prefixed log message."""

    print(f"{LOG_PREFIX} {message}")


def split_cell(value: Any) -> list[str]:
    """Split one cause/effect cell on semicolons.

    Parameters
    ----------
    value:
        Raw cause or effect cell value.

    Returns
    -------
    list[str]
        Non-empty stripped parts. If no valid parts are found, returns one
        stripped string so the row is still represented.
    """

    text = "" if value is None or pd.isna(value) else str(value).strip()
    parts = [part.strip() for part in text.split(";") if part.strip()]
    return parts if parts else [text]


def merge_paper_ids(values: pd.Series) -> str:
    """Merge semicolon-separated paper ID cells.

    Parameters
    ----------
    values:
        Paper ID cells from duplicate rows.

    Returns
    -------
    str
        Deduplicated semicolon-separated paper IDs in first-seen order.
    """

    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        for paper_id in str(value).split(";"):
            clean = paper_id.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            merged.append(clean)
    return "; ".join(merged)


def numeric_paper_count(value: Any) -> int:
    """Coerce paper_count to an integer for sorting.

    Parameters
    ----------
    value:
        Raw paper_count value.

    Returns
    -------
    int
        Parsed paper count, or zero if parsing fails.
    """

    try:
        if value is None or pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def expand_compounds(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Expand rows with semicolon-separated cause/effect values.

    Parameters
    ----------
    df:
        Constraint-review DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, int, int]
        Expanded DataFrame, number of compound rows expanded, and number of
        replacement edge rows generated from those compound rows.
    """

    expanded_rows: list[dict[str, Any]] = []
    compound_rows = 0
    generated_edges = 0

    for _, row in df.iterrows():
        causes = split_cell(row["cause"])
        effects = split_cell(row["effect"])
        is_compound = len(causes) > 1 or len(effects) > 1

        if is_compound:
            compound_rows += 1
            generated_edges += len(causes) * len(effects)

        for cause in causes:
            for effect in effects:
                new_row = row.to_dict()
                new_row["cause"] = cause
                new_row["effect"] = effect
                expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows, columns=df.columns), compound_rows, generated_edges


def deduplicate_edges(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Deduplicate rows by cause/effect pair.

    For duplicate edge rows, the row with the highest paper_count is kept and
    paper_ids are merged across all duplicate rows.

    Parameters
    ----------
    df:
        Expanded constraint-review DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Deduplicated DataFrame and duplicate row count removed.
    """

    if df.empty:
        return df.copy(), 0

    working = df.copy()
    working["_paper_count_numeric"] = working["paper_count"].map(numeric_paper_count)
    working["_original_order"] = range(len(working))

    deduped_rows: list[dict[str, Any]] = []
    duplicate_rows_removed = 0
    grouped = working.groupby(["cause", "effect"], sort=False, dropna=False)

    for _, group in grouped:
        if len(group) > 1:
            duplicate_rows_removed += len(group) - 1

        best = group.sort_values(
            ["_paper_count_numeric", "_original_order"],
            ascending=[False, True],
        ).iloc[0].copy()
        best["paper_ids"] = merge_paper_ids(group["paper_ids"])
        best = best.drop(labels=["_paper_count_numeric", "_original_order"])
        deduped_rows.append(best.to_dict())

    return pd.DataFrame(deduped_rows, columns=df.columns), duplicate_rows_removed


def split_compound_constraints(path: Path = REVIEW_PATH, dry_run: bool = False) -> pd.DataFrame:
    """Split, deduplicate, and save the constraint review CSV.

    Parameters
    ----------
    path:
        Path to ``constraints_for_review.csv``.
    dry_run:
        If True, print the summary without writing files.

    Returns
    -------
    pd.DataFrame
        Cleaned constraint-review DataFrame.
    """

    df = pd.read_csv(path)
    original_rows = len(df)

    expanded, compound_rows, generated_edges = expand_compounds(df)
    deduped, duplicates_merged = deduplicate_edges(expanded)

    self_loop_mask = deduped["cause"].astype(str).str.strip().eq(
        deduped["effect"].astype(str).str.strip()
    )
    self_loops_dropped = int(self_loop_mask.sum())
    cleaned = deduped.loc[~self_loop_mask].copy()

    log(f"Original rows: {original_rows}")
    log(f"Compound rows expanded: {compound_rows} (into {generated_edges} new edges)")
    log(f"Duplicates merged: {duplicates_merged}")
    log(f"Self-loops dropped: {self_loops_dropped}")
    log(f"Final rows: {len(cleaned)}")

    if dry_run:
        log("Dry run only; no files written.")
        return cleaned

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP_PATH)
    log(f"Backup written -> {BACKUP_PATH}")
    cleaned.to_csv(path, index=False)
    log(f"Saved cleaned constraints -> {path}")
    return cleaned


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Split semicolon-bundled constraint-review rows."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing the cleaned CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the splitter CLI."""

    args = parse_args()
    split_compound_constraints(REVIEW_PATH, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
