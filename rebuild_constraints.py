# rebuild_constraints.py
# ============================================================
# Rebuild the literature-constraint pipeline end to end.
#
# Usage:
#   python rebuild_constraints.py
#   python rebuild_constraints.py --skip-review
#   python rebuild_constraints.py --no-causica-check
# ============================================================

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import CONSTRAINTS_REVIEW_PATH


AUTO_TIER2_NOTE = "auto-approved: tier 2 structural/temporal/impossibility"
AUTO_TIER1_NOTE = "auto-approved: tier 1 high-confidence required edge"
REVIEW_KEY_COLUMNS = ["cause", "effect", "tier"]
REVIEW_PRESERVE_COLUMNS = ["proposed_action", "approved", "notes"]


def log(message: str) -> None:
    print(f"[rebuild] {message}")


def run_script(script_name: str, *args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, script_name, *args]
    log("Running: " + " ".join(command))
    if capture:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result
    return subprocess.run(command, check=True, text=True)


def is_blank(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip() == ""


def append_note(existing: object, note: str) -> str:
    if is_blank(existing):
        return note
    text = str(existing).strip()
    if note in text:
        return text
    return f"{text}; {note}"


def split_constraint_cell(value: Any) -> list[str]:
    """
    Split a constraint endpoint cell into individual variable names.

    Parameters
    ----------
    value : Any
        Raw `cause` or `effect` value from the review CSV.

    Returns
    -------
    list[str]
        Non-empty stripped endpoint names. Unsplittable blank values are
        preserved as a single empty string so validation can handle them.
    """
    text = "" if value is None or pd.isna(value) else str(value).strip()
    parts = [part.strip() for part in text.split(";") if part.strip()]
    return parts if parts else [text]


def merge_paper_ids(values: pd.Series) -> str:
    """
    Merge semicolon-separated paper IDs in first-seen order.

    Parameters
    ----------
    values : pd.Series
        `paper_ids` cells from duplicate edge rows.

    Returns
    -------
    str
        Deduplicated semicolon-separated paper IDs.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        for paper_id in str(value).split(";"):
            clean = paper_id.strip()
            if clean and clean not in seen:
                seen.add(clean)
                merged.append(clean)
    return "; ".join(merged)


def numeric_paper_count(value: Any) -> int:
    """
    Coerce a paper-count cell to an integer.

    Parameters
    ----------
    value : Any
        Raw `paper_count` value.

    Returns
    -------
    int
        Parsed count, or zero when parsing fails.
    """
    try:
        if value is None or pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def confidence_rank(value: Any) -> int:
    """
    Rank confidence labels for duplicate-row merging.

    Parameters
    ----------
    value : Any
        Raw confidence value.

    Returns
    -------
    int
        3 for high, 2 for medium, 1 for low, else 0.
    """
    text = "" if value is None or pd.isna(value) else str(value).strip().lower()
    return {"high": 3, "medium": 2, "low": 1}.get(text, 0)


def split_compound_rows() -> tuple[int, int]:
    """
    Split semicolon-bundled review rows into individual edge rows.

    This runs inside the rebuild pipeline after claim aggregation regenerates
    `constraints_for_review.csv` and before any auto-approval/finalization.
    Rows are expanded by Cartesian product, deduplicated by `(cause, effect)`,
    and self-loops are dropped.

    Returns
    -------
    tuple[int, int]
        Number of compound rows expanded and number of generated edge rows
        from those compound rows before deduplication.
    """
    review_path = Path(CONSTRAINTS_REVIEW_PATH)
    if not review_path.exists():
        log(f"Review CSV not found, skipping compound split: {CONSTRAINTS_REVIEW_PATH}")
        return 0, 0

    df = pd.read_csv(review_path)
    required_columns = {"cause", "effect", "paper_count", "paper_ids"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{CONSTRAINTS_REVIEW_PATH} missing required columns: {sorted(missing)}")

    expanded_rows: list[dict[str, Any]] = []
    compound_rows = 0
    generated_edges = 0

    for _, row in df.iterrows():
        causes = split_constraint_cell(row["cause"])
        effects = split_constraint_cell(row["effect"])
        if len(causes) > 1 or len(effects) > 1:
            compound_rows += 1
            generated_edges += len(causes) * len(effects)

        for cause in causes:
            for effect in effects:
                new_row = row.to_dict()
                new_row["cause"] = cause
                new_row["effect"] = effect
                expanded_rows.append(new_row)

    expanded = pd.DataFrame(expanded_rows, columns=df.columns)
    if expanded.empty:
        expanded.to_csv(review_path, index=False)
        log("Split 0 compound rows into 0 edges")
        return compound_rows, generated_edges

    expanded["_paper_count_numeric"] = expanded["paper_count"].map(numeric_paper_count)
    expanded["_confidence_rank"] = (
        expanded["confidence"].map(confidence_rank)
        if "confidence" in expanded.columns
        else 0
    )
    expanded["_original_order"] = range(len(expanded))

    deduped_rows: list[dict[str, Any]] = []
    duplicates_merged = 0
    for _, group in expanded.groupby(["cause", "effect"], sort=False, dropna=False):
        if len(group) > 1:
            duplicates_merged += len(group) - 1

        best = group.sort_values(
            ["_paper_count_numeric", "_confidence_rank", "_original_order"],
            ascending=[False, False, True],
        ).iloc[0].copy()
        best["paper_count"] = int(group["_paper_count_numeric"].max())
        best["paper_ids"] = merge_paper_ids(group["paper_ids"])
        if "confidence" in expanded.columns:
            ranked = group.sort_values(
                ["_confidence_rank", "_original_order"],
                ascending=[False, True],
            ).iloc[0]
            best["confidence"] = ranked["confidence"]
        best = best.drop(labels=["_paper_count_numeric", "_confidence_rank", "_original_order"])
        deduped_rows.append(best.to_dict())

    cleaned = pd.DataFrame(deduped_rows, columns=df.columns)
    self_loop_mask = cleaned["cause"].astype(str).str.strip().eq(
        cleaned["effect"].astype(str).str.strip()
    )
    self_loops_dropped = int(self_loop_mask.sum())
    cleaned = cleaned.loc[~self_loop_mask].copy()
    cleaned.to_csv(review_path, index=False)

    log(f"Split {compound_rows} compound rows into {generated_edges} edges")
    log(
        "Compound split cleanup: "
        f"duplicates merged={duplicates_merged}, "
        f"self-loops dropped={self_loops_dropped}, "
        f"final rows={len(cleaned)}"
    )
    return compound_rows, generated_edges


def _review_key(row: pd.Series) -> tuple[str, str, str]:
    return tuple(str(row[col]).strip() for col in REVIEW_KEY_COLUMNS)


def load_review_decisions() -> dict[tuple[str, str, str], dict[str, object]]:
    review_path = Path(CONSTRAINTS_REVIEW_PATH)
    if not review_path.exists():
        return {}

    df = pd.read_csv(review_path)
    required_columns = set(REVIEW_KEY_COLUMNS + REVIEW_PRESERVE_COLUMNS)
    if not required_columns.issubset(df.columns):
        return {}

    decisions = {}
    for _, row in df.iterrows():
        approved = row.get("approved")
        notes = row.get("notes")
        proposed_action = row.get("proposed_action")
        if is_blank(approved) and is_blank(notes) and _normalise_action(proposed_action) == "review":
            continue
        decisions[_review_key(row)] = {
            column: row.get(column)
            for column in REVIEW_PRESERVE_COLUMNS
        }
    return decisions


def _normalise_action(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def restore_review_decisions(decisions: dict[tuple[str, str, str], dict[str, object]]) -> int:
    if not decisions:
        return 0

    review_path = Path(CONSTRAINTS_REVIEW_PATH)
    if not review_path.exists():
        return 0

    df = pd.read_csv(review_path)
    required_columns = set(REVIEW_KEY_COLUMNS + REVIEW_PRESERVE_COLUMNS)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{CONSTRAINTS_REVIEW_PATH} missing required columns: {sorted(missing)}")

    for column in REVIEW_PRESERVE_COLUMNS:
        df[column] = df[column].astype("object")

    restored = 0
    for idx, row in df.iterrows():
        key = _review_key(row)
        if key not in decisions:
            continue
        for column, value in decisions[key].items():
            if not is_blank(value):
                df.at[idx, column] = value
        restored += 1

    df.to_csv(review_path, index=False)
    log(f"Restored preserved review decisions: {restored}")
    return restored


def auto_approve_constraints(skip_review: bool) -> tuple[int, int, int]:
    review_path = Path(CONSTRAINTS_REVIEW_PATH)
    if not review_path.exists():
        log(f"Review CSV not found, skipping auto-approval: {CONSTRAINTS_REVIEW_PATH}")
        return 0, 0, 0

    df = pd.read_csv(review_path)
    required_columns = {"tier", "approved", "notes"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{CONSTRAINTS_REVIEW_PATH} missing required columns: {sorted(missing)}")

    approved_blank = df["approved"].map(is_blank)
    df["approved"] = df["approved"].astype("object")
    df["notes"] = df["notes"].astype("object")
    tier2_mask = df["tier"].astype(str).str.strip().eq("2") & approved_blank
    tier1_mask = df["tier"].astype(str).str.strip().eq("1") & approved_blank

    tier2_count = int(tier2_mask.sum())
    tier1_count = int(tier1_mask.sum()) if skip_review else 0

    if tier2_count:
        df.loc[tier2_mask, "approved"] = "yes"
        df.loc[tier2_mask, "notes"] = df.loc[tier2_mask, "notes"].map(
            lambda value: append_note(value, AUTO_TIER2_NOTE)
        )

    if skip_review and tier1_count:
        df.loc[tier1_mask, "approved"] = "yes"
        df.loc[tier1_mask, "notes"] = df.loc[tier1_mask, "notes"].map(
            lambda value: append_note(value, AUTO_TIER1_NOTE)
        )

    df.to_csv(review_path, index=False)

    still_blank_review = int(
        (
            df["tier"].astype(str).str.strip().isin(["1", "3"])
            & df["approved"].map(is_blank)
        ).sum()
    )

    log(f"Auto-approved tier 2 rows: {tier2_count}")
    if skip_review:
        log(f"Auto-approved tier 1 rows due to --skip-review: {tier1_count}")
    return tier2_count, tier1_count, still_blank_review


def wait_for_manual_review(blank_count: int, skip_review: bool) -> None:
    if skip_review:
        log("WARNING: --skip-review used; manual-review pause skipped.")
        return

    if blank_count == 0:
        log("No blank tier 1 or tier 3 rows remain; proceeding without pause.")
        return

    print(
        "[rebuild] Manual review required.\n"
        "Open reports/constraints_for_review.csv in Excel.\n"
        "Fill in the 'approved' column for tier 1 and tier 3 rows\n"
        "(yes / no / modify), save the file, then press Enter to continue."
    )
    input("Press Enter when manual review is complete...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild literature-derived constraints end to end."
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip manual-review pause and auto-approve blank tier 1 rows.",
    )
    parser.add_argument(
        "--no-causica-check",
        action="store_true",
        help="Skip final 14_constraint_adapter.py verification.",
    )
    args = parser.parse_args()

    if args.skip_review:
        log("WARNING: --skip-review will skip the manual-review pause.")

    run_script("02d_compute_pillar_scores.py")
    preserved_decisions = load_review_decisions()
    run_script("10_constraints_from_claims.py")
    restore_review_decisions(preserved_decisions)
    split_compound_rows()
    restore_review_decisions(preserved_decisions)

    _, _, blank_review_count = auto_approve_constraints(args.skip_review)
    wait_for_manual_review(blank_review_count, args.skip_review)

    dry_run = run_script("11_finalize_constraints.py", "--dry-run", capture=True)
    if dry_run.stdout:
        print(dry_run.stdout, end="")
    if dry_run.stderr:
        print(dry_run.stderr, end="", file=sys.stderr)

    response = input("Apply changes? [y/N] ").strip().lower()
    if not response.startswith("y"):
        log("Aborted before applying 11_finalize_constraints.py.")
        return

    run_script("11_finalize_constraints.py")

    if args.no_causica_check:
        log("Skipping 14_constraint_adapter.py due to --no-causica-check.")
        return

    run_script("14_constraint_adapter.py", "--dataset", "real")
    run_script("14_constraint_adapter.py", "--dataset", "synthetic")


if __name__ == "__main__":
    main()
