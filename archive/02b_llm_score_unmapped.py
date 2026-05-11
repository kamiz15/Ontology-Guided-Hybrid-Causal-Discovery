# 02b_llm_score_unmapped.py
# ============================================================
# Step 02b - Use Gemma to score ordinal ESG disclosure cells
# that deterministic parsing could not map in step 02a.
#
# Usage:
#   python 02b_llm_score_unmapped.py --dry-run
#   python 02b_llm_score_unmapped.py --resume --api-key KEY
#
# Output:
#   reports/llm_scoring_log.csv
#   reports/llm_scoring_failures.csv
#   data/processed/data_real_parsed.csv
#   data/processed/data_real_parsed.csv.bak
# ============================================================

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    LIMITATIONS_PATH,
    LLM_SCORING_FAILURES_PATH,
    LLM_SCORING_LOG_PATH,
    ORDINAL_PARSING_UNMAPPED_PATH,
    PARSING_DECISIONS_PATH,
    REAL_PARSED_DATA_PATH,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback keeps the script importable.
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


_gemma_module = importlib.import_module("08_gemma_causal_proposals")
DEFAULT_GOOGLE_MODEL = _gemma_module.DEFAULT_GOOGLE_MODEL
query_google_ai = _gemma_module.query_google_ai
resolve_google_api_key = _gemma_module.resolve_google_api_key


PROMPT_TEMPLATE = """You are coding a qualitative ESG disclosure from a bank's sustainability
report. Assign a score from 1 to 5 based on the rubric below, plus a
confidence value between 0 and 1.

COLUMN: {column_name}
RAW TEXT: "{raw_value}"

RUBRIC:
5 - Excellent / external verification / SBTi-validated / specific
    quantified targets with audit
4 - Very Good / documented commitment with structure / third-party verified /
    GRI/TCFD aligned / specific numeric targets without independent verification
3 - Good / documented framework without verification / general commitments
    with some specificity
2 - Moderate / adequate / generic descriptions of activity without
    commitment language
1 - Limited / weak / vague mention only
NaN - "Qualitative" alone with no further content; cell content not
      applicable to this column

Respond ONLY with a JSON object on a single line, no markdown fences,
no commentary:
{{"score": <integer 1-5 or null>, "confidence": <float 0-1>, "reason": "<one sentence>"}}"""

LOG_COLUMNS = [
    "row_index",
    "column",
    "raw_value",
    "score",
    "confidence",
    "reason",
    "model",
    "timestamp",
]

FAILURE_COLUMNS = [
    "row_index",
    "column",
    "raw_value",
    "error",
    "response",
    "model",
    "timestamp",
]

VALID_SCORES = {1, 2, 3, 4, 5}


class FatalAPIError(RuntimeError):
    """Raised when an API error should stop the whole run immediately."""


def _ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now().date().isoformat()


def _is_non_retryable_api_error(error: Exception) -> bool:
    text = str(error).lower()
    fatal_markers = [
        "api_key_invalid",
        "api key not valid",
        "permission_denied",
        "quota_exceeded",
    ]
    return any(marker in text for marker in fatal_markers)


def load_unmapped(path: str) -> pd.DataFrame:
    """
    Load ordinal cells that step 02a could not parse.

    Parameters
    ----------
    path : str
        CSV path produced by 02a_parse_real_dataset.py.

    Returns
    -------
    pd.DataFrame
        Unmapped rows with row index, target column, and raw text.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if not Path(path).exists():
        print(f"[step_02b] No unmapped ordinal file found: {path}")
        return pd.DataFrame(columns=["row_index", "column", "raw_value", "reason"])

    df = pd.read_csv(path)
    required = {"row_index", "column", "raw_value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df.dropna(subset=["row_index", "column", "raw_value"]).copy()
    df["row_index"] = df["row_index"].astype(int)
    df["column"] = df["column"].astype(str)
    df["raw_value"] = df["raw_value"].astype(str)
    return df


def load_existing_log(path: str) -> pd.DataFrame:
    """
    Load the incremental LLM scoring log if it already exists.

    Parameters
    ----------
    path : str
        Existing log CSV path.

    Returns
    -------
    pd.DataFrame
        Log table with the expected schema, or an empty table.
    """
    if not Path(path).exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    df = pd.read_csv(path)
    missing = set(LOG_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def load_existing_failures(path: str) -> pd.DataFrame:
    """
    Load the LLM scoring failure log if it already exists.

    Parameters
    ----------
    path : str
        Existing failure CSV path.

    Returns
    -------
    pd.DataFrame
        Failure log with the expected schema, or an empty table.
    """
    if not Path(path).exists():
        return pd.DataFrame(columns=FAILURE_COLUMNS)

    df = pd.read_csv(path)
    missing = set(FAILURE_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def build_prompt(column_name: str, raw_value: str) -> str:
    """
    Build the fixed rubric prompt for one qualitative ESG disclosure.

    Parameters
    ----------
    column_name : str
        Parsed target column name.
    raw_value : str
        Original qualitative disclosure text.

    Returns
    -------
    str
        Prompt sent to Gemma.
    """
    safe_value = str(raw_value).replace('"', '\\"')
    return PROMPT_TEMPLATE.format(column_name=column_name, raw_value=safe_value)


def parse_llm_response(response: str) -> dict[str, Any]:
    """
    Parse and validate Gemma's JSON response.

    Parameters
    ----------
    response : str
        Raw model response text.

    Returns
    -------
    dict[str, Any]
        Validated score, confidence, and reason.

    Raises
    ------
    ValueError
        If the response is not a valid single JSON object matching the schema.
    """
    text = response.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object")

    required = {"score", "confidence", "reason"}
    missing = required.difference(obj)
    if missing:
        raise ValueError(f"Response missing required keys: {sorted(missing)}")

    score = obj["score"]
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, int) or score not in VALID_SCORES:
            raise ValueError(f"Invalid score: {score!r}")

    confidence = obj["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError(f"Invalid confidence: {confidence!r}")

    reason = obj["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"Invalid reason: {reason!r}")

    return {
        "score": score,
        "confidence": float(confidence),
        "reason": reason.strip(),
    }


def query_with_retries(
    prompt: str,
    api_key: str,
    model: str,
    retries: int = 3,
    timeout: int = 30,
) -> str:
    """
    Query Google AI Studio with exponential backoff for API errors.

    Parameters
    ----------
    prompt : str
        Prompt to send.
    api_key : str
        Google AI Studio API key.
    model : str
        Google model name.
    retries : int, optional
        Number of retries after the first failed attempt.
    timeout : int, optional
        Per-call timeout in seconds.

    Returns
    -------
    str
        Raw model response.

    Raises
    ------
    Exception
        The final API exception after retries are exhausted.
    """
    for attempt in range(retries + 1):
        try:
            return query_google_ai(
                prompt,
                api_key,
                model=model,
                temperature=0.0,
                timeout=timeout,
                max_output_tokens=512,
            )
        except Exception as exc:
            if _is_non_retryable_api_error(exc):
                raise FatalAPIError(str(exc)) from exc
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Unreachable retry state")


def append_csv_row(path: str, row: dict[str, Any], columns: list[str]) -> None:
    """
    Append one row to a CSV, writing the header when needed.

    Parameters
    ----------
    path : str
        CSV output path.
    row : dict[str, Any]
        Row data.
    columns : list[str]
        Output column order.
    """
    _ensure_output_dir(path)
    csv_path = Path(path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row], columns=columns).to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )


def row_key(row: pd.Series) -> tuple[int, str, str]:
    """
    Build a stable row key for resume matching.

    Parameters
    ----------
    row : pd.Series
        Unmapped or logged row.

    Returns
    -------
    tuple[int, str, str]
        Row index, column, and raw value.
    """
    return (int(row["row_index"]), str(row["column"]), str(row["raw_value"]))


def filter_resume_rows(unmapped: pd.DataFrame, existing_log: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that already have an entry in the scoring log.

    Parameters
    ----------
    unmapped : pd.DataFrame
        Unmapped ordinal cells.
    existing_log : pd.DataFrame
        Existing LLM scoring log.

    Returns
    -------
    pd.DataFrame
        Rows still needing model calls.
    """
    if existing_log.empty:
        return unmapped.copy()

    completed = {row_key(row) for _, row in existing_log.iterrows()}
    keep_mask = [row_key(row) not in completed for _, row in unmapped.iterrows()]
    return unmapped.loc[keep_mask].copy()


def score_unmapped_cells(
    rows: pd.DataFrame,
    api_key: str,
    model: str,
    log_path: str,
    failures_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score unmapped ordinal cells sequentially and save progress after each row.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows to send to Gemma.
    api_key : str
        Google AI Studio API key.
    model : str
        Google model name.
    log_path : str
        Incremental successful-response log path.
    failures_path : str
        Incremental failure log path.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        New successful log rows and new failure rows from this run.
    """
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    iterator = tqdm(rows.iterrows(), total=len(rows), desc="[step_02b] scoring")
    for _, row in iterator:
        prompt = build_prompt(str(row["column"]), str(row["raw_value"]))
        response = ""
        timestamp = _timestamp()
        try:
            response = query_with_retries(prompt, api_key, model=model)
            parsed = parse_llm_response(response)
            log_row = {
                "row_index": int(row["row_index"]),
                "column": str(row["column"]),
                "raw_value": str(row["raw_value"]),
                "score": parsed["score"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "model": model,
                "timestamp": timestamp,
            }
            append_csv_row(log_path, log_row, LOG_COLUMNS)
            successes.append(log_row)
        except FatalAPIError as exc:
            failure_row = {
                "row_index": int(row["row_index"]),
                "column": str(row["column"]),
                "raw_value": str(row["raw_value"]),
                "error": str(exc),
                "response": response,
                "model": model,
                "timestamp": timestamp,
            }
            append_csv_row(failures_path, failure_row, FAILURE_COLUMNS)
            failures.append(failure_row)
            print(f"[step_02b] Fatal API error: {exc}")
            raise
        except Exception as exc:
            failure_row = {
                "row_index": int(row["row_index"]),
                "column": str(row["column"]),
                "raw_value": str(row["raw_value"]),
                "error": str(exc),
                "response": response,
                "model": model,
                "timestamp": timestamp,
            }
            append_csv_row(failures_path, failure_row, FAILURE_COLUMNS)
            failures.append(failure_row)
            print(
                f"[step_02b] Failed row={row['row_index']} "
                f"column={row['column']}: {exc}"
            )
        finally:
            time.sleep(0.5)

    return (
        pd.DataFrame(successes, columns=LOG_COLUMNS),
        pd.DataFrame(failures, columns=FAILURE_COLUMNS),
    )


def validate_logged_scores(
    log_df: pd.DataFrame,
    failures_path: str,
    model: str,
) -> pd.DataFrame:
    """
    Validate logged scores before merging them into the parsed dataset.

    Parameters
    ----------
    log_df : pd.DataFrame
        Incremental LLM scoring log.
    failures_path : str
        Failure log path for malformed logged rows.
    model : str
        Model name to include in any generated failure rows.

    Returns
    -------
    pd.DataFrame
        Valid log rows.
    """
    valid_rows: list[dict[str, Any]] = []
    for _, row in log_df.iterrows():
        score = row["score"]
        confidence = row["confidence"]
        error = ""

        if pd.notna(score):
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                error = f"Invalid logged score: {score!r}"
            else:
                if not numeric_score.is_integer() or int(numeric_score) not in VALID_SCORES:
                    error = f"Invalid logged score: {score!r}"

        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            error = f"Invalid logged confidence: {confidence!r}"
        else:
            if not 0 <= numeric_confidence <= 1:
                error = f"Invalid logged confidence: {confidence!r}"

        if error:
            append_csv_row(
                failures_path,
                {
                    "row_index": row.get("row_index", ""),
                    "column": row.get("column", ""),
                    "raw_value": row.get("raw_value", ""),
                    "error": error,
                    "response": "",
                    "model": model,
                    "timestamp": _timestamp(),
                },
                FAILURE_COLUMNS,
            )
            continue

        clean_row = row.to_dict()
        clean_row["row_index"] = int(clean_row["row_index"])
        clean_row["confidence"] = numeric_confidence
        if pd.notna(score):
            clean_row["score"] = int(float(score))
        else:
            clean_row["score"] = None
        valid_rows.append(clean_row)

    return pd.DataFrame(valid_rows, columns=LOG_COLUMNS)


def merge_scores(parsed_path: str, scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge non-null LLM ordinal scores into the parsed dataset.

    Parameters
    ----------
    parsed_path : str
        Path to data/processed/data_real_parsed.csv.
    scored_df : pd.DataFrame
        Valid scoring log rows.

    Returns
    -------
    pd.DataFrame
        Updated parsed dataset.

    Raises
    ------
    FileNotFoundError
        If the parsed dataset does not exist.
    ValueError
        If any row index or column is not present in the parsed dataset.
    """
    if not Path(parsed_path).exists():
        raise FileNotFoundError(f"Parsed dataset not found: {parsed_path}")

    parsed = pd.read_csv(parsed_path)
    backup_path = f"{parsed_path}.bak"
    shutil.copy2(parsed_path, backup_path)
    print(f"[step_02b] Backup written -> {backup_path}")

    updates = 0
    for _, row in scored_df.iterrows():
        if pd.isna(row["score"]):
            continue

        row_index = int(row["row_index"])
        column = str(row["column"])
        if column not in parsed.columns:
            raise ValueError(f"Column not found in parsed dataset: {column}")
        if row_index < 0 or row_index >= len(parsed):
            raise ValueError(f"Row index outside parsed dataset: {row_index}")

        parsed.loc[row_index, column] = int(row["score"])
        updates += 1

    parsed.to_csv(parsed_path, index=False)
    print(f"[step_02b] Merged {updates} non-null LLM scores -> {parsed_path}")
    return parsed


def append_limitations_entry(
    path: str,
    low_confidence_count: int,
    confidence_threshold: float,
) -> None:
    """
    Append a run note under the limitations action checklist.

    Parameters
    ----------
    path : str
        LIMITATIONS_TO_REVIEW.md path.
    low_confidence_count : int
        Count of cells below the confidence threshold.
    confidence_threshold : float
        Threshold used to flag low-confidence cells.
    """
    if low_confidence_count <= 0:
        return

    entry = (
        f"- [ ] Run completed {_today()}: {low_confidence_count} cells flagged "
        f"with confidence < {confidence_threshold:g} \u2014 see reports/llm_scoring_log.csv"
    )
    md_path = Path(path)
    if not md_path.exists():
        md_path.write_text(entry + "\n", encoding="utf-8")
        return

    text = md_path.read_text(encoding="utf-8")
    marker = "**Rubric used:**"
    if marker in text:
        before, after = text.split(marker, 1)
        text = before.rstrip() + "\n" + entry + "\n\n" + marker + after
    else:
        text = text.rstrip() + "\n" + entry + "\n"
    md_path.write_text(text, encoding="utf-8")


def append_parsing_decisions_note(path: str, model: str) -> None:
    """
    Append a note to parsing_decisions.md documenting LLM coding.

    Parameters
    ----------
    path : str
        Parsing decisions markdown path.
    model : str
        Model used for LLM scoring.
    """
    _ensure_output_dir(path)
    note = f"""

## LLM-Assisted Ordinal Coding

Date: {_today()}
Model: {model}
Source: reports/ordinal_parsing_unmapped.csv
Output log: reports/llm_scoring_log.csv

Gemma via Google AI Studio was used to score ordinal ESG disclosure cells that
were not covered by deterministic keyword parsing. Low-confidence scores are
flagged for manual review in LIMITATIONS_TO_REVIEW.md.
"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(note)


def print_summary(
    attempted: int,
    successes: pd.DataFrame,
    failures: pd.DataFrame,
    confidence_threshold: float,
    skipped: int | None = None,
) -> None:
    """
    Print the required scoring summary block.

    Parameters
    ----------
    attempted : int
        Cells processed in this run only.
    successes : pd.DataFrame
        Valid model responses from this run only. Rows with null scores are
        valid responses, but count as unscored in the summary.
    failures : pd.DataFrame
        Failed cells from this run only.
    confidence_threshold : float
        Threshold used to flag low-confidence cells.
    skipped : int | None, optional
        Resume-mode cells skipped because they were already logged.
    """
    assigned = successes[successes["score"].notna()].copy()
    unscored = successes[successes["score"].isna()].copy()
    successful = len(assigned)
    failed = len(failures) + len(unscored)
    confidence = pd.to_numeric(assigned["confidence"], errors="coerce")
    high_confidence = int((confidence >= confidence_threshold).sum())
    low_confidence = int((confidence < confidence_threshold).sum())

    assert attempted == successful + failed, (
        f"Counter mismatch: attempted={attempted}, "
        f"successful={successful}, failed={failed}"
    )
    assert high_confidence + low_confidence == successful, (
        f"Confidence buckets don't sum to successful: "
        f"high={high_confidence}, low={low_confidence}, "
        f"successful={successful}"
    )

    print("[step_02b] Summary")
    if skipped is not None:
        print(f"[step_02b]   Resume mode: skipped {skipped} cells already in log")
    print(f"[step_02b]   Cells attempted: {attempted}")
    print(f"[step_02b]   Cells scored successfully: {successful}")
    print(f"[step_02b]   Cells with confidence >= {confidence_threshold:g}: {high_confidence}")
    print(f"[step_02b]   Cells with confidence < {confidence_threshold:g}: {low_confidence}")
    print(f"[step_02b]   Cells failed: {failed}")
    print("[step_02b]   Per-column distribution of assigned scores:")

    if assigned.empty:
        print("[step_02b]     No assigned scores.")
        return

    assigned["score"] = assigned["score"].astype(int)
    distribution = (
        assigned.groupby(["column", "score"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=sorted(VALID_SCORES), fill_value=0)
    )
    for column, counts in distribution.iterrows():
        parts = ", ".join(f"{score}={int(counts[score])}" for score in sorted(VALID_SCORES))
        print(f"[step_02b]     {column}: {parts}")


def run_llm_scoring(
    unmapped_path: str,
    parsed_path: str,
    log_path: str,
    failures_path: str,
    limitations_path: str,
    decisions_path: str,
    model: str,
    api_key: str | None,
    dry_run: bool,
    resume: bool,
    confidence_threshold: float,
) -> None:
    """
    Run the full LLM scoring and merge workflow.

    Parameters
    ----------
    unmapped_path : str
        CSV of ordinal cells that deterministic parsing did not score.
    parsed_path : str
        Parsed data CSV to update after successful scoring.
    log_path : str
        Incremental scoring log path.
    failures_path : str
        Malformed response and API failure log path.
    limitations_path : str
        LIMITATIONS_TO_REVIEW.md path.
    decisions_path : str
        parsing_decisions.md path.
    model : str
        Google model name.
    api_key : str | None
        Optional API key override.
    dry_run : bool
        If True, print prompts without API calls or writes.
    resume : bool
        If True, skip rows already present in the scoring log.
    confidence_threshold : float
        Threshold for low-confidence manual-review flags.
    """
    unmapped = load_unmapped(unmapped_path)
    if unmapped.empty:
        print(f"[step_02b] No unmapped ordinal cells to score in {unmapped_path}.")
        return

    existing_log = load_existing_log(log_path)
    rows_to_process = filter_resume_rows(unmapped, existing_log) if resume else unmapped.copy()
    skipped = 0
    if resume:
        skipped = len(unmapped) - len(rows_to_process)
        print(f"[step_02b] Resume mode: skipped {skipped} cells already in log")

    print(f"[step_02b] Cells queued for scoring: {len(rows_to_process)}")
    print(f"[step_02b] Model: {model}")

    if dry_run:
        print("[step_02b] DRY RUN: prompts below would be sent to the API.")
        for _, row in rows_to_process.iterrows():
            print(
                f"\n[step_02b] Prompt row_index={row['row_index']} "
                f"column={row['column']}"
            )
            print(build_prompt(str(row["column"]), str(row["raw_value"])))
        print("[step_02b] DRY RUN complete. No API calls or writes were made.")
        return

    try:
        resolved_key = resolve_google_api_key(api_key)
    except ValueError as exc:
        print(f"[step_02b] ERROR: {exc}")
        return

    attempted = len(rows_to_process)
    new_successes = pd.DataFrame(columns=LOG_COLUMNS)
    new_failures = pd.DataFrame(columns=FAILURE_COLUMNS)

    if not rows_to_process.empty:
        try:
            new_successes, new_failures = score_unmapped_cells(
                rows_to_process,
                api_key=resolved_key,
                model=model,
                log_path=log_path,
                failures_path=failures_path,
            )
        except FatalAPIError:
            print("[step_02b] Aborting run. Fix the API key or account settings, then rerun with --resume.")
            return
    else:
        print("[step_02b] No new rows to score.")

    full_log = load_existing_log(log_path)
    valid_log = validate_logged_scores(full_log, failures_path, model=model)

    if valid_log.empty:
        print("[step_02b] No valid LLM scores to merge.")
    else:
        merge_scores(parsed_path, valid_log)
        run_confidence = pd.to_numeric(new_successes["confidence"], errors="coerce")
        low_count = int((run_confidence < confidence_threshold).sum())
        append_limitations_entry(limitations_path, low_count, confidence_threshold)
        append_parsing_decisions_note(decisions_path, model)

    print_summary(
        attempted=attempted,
        successes=new_successes,
        failures=new_failures,
        confidence_threshold=confidence_threshold,
        skipped=skipped if resume else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use Gemma to score ordinal ESG disclosures left unmapped by step 02a."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without API calls or writes")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the LLM scoring log")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Confidence below this value is flagged for manual review")
    parser.add_argument("--model", default=DEFAULT_GOOGLE_MODEL,
                        help="Google AI Studio model name")
    parser.add_argument("--api-key", default=None,
                        help="Google AI Studio API key; falls back to GOOGLE_AI_KEY")
    parser.add_argument("--unmapped", default=ORDINAL_PARSING_UNMAPPED_PATH,
                        help="CSV of ordinal cells left unmapped by step 02a")
    parser.add_argument("--parsed", default=REAL_PARSED_DATA_PATH,
                        help="Parsed dataset CSV to update")
    parser.add_argument("--log", default=LLM_SCORING_LOG_PATH,
                        help="Incremental successful scoring log")
    parser.add_argument("--failures", default=LLM_SCORING_FAILURES_PATH,
                        help="Malformed response and API failure log")
    parser.add_argument("--limitations", default=LIMITATIONS_PATH,
                        help="Limitations tracker markdown path")
    parser.add_argument("--decisions", default=PARSING_DECISIONS_PATH,
                        help="Parsing decisions markdown path")
    args = parser.parse_args()

    run_llm_scoring(
        unmapped_path=args.unmapped,
        parsed_path=args.parsed,
        log_path=args.log,
        failures_path=args.failures,
        limitations_path=args.limitations,
        decisions_path=args.decisions,
        model=args.model,
        api_key=args.api_key,
        dry_run=args.dry_run,
        resume=args.resume,
        confidence_threshold=args.confidence_threshold,
    )
