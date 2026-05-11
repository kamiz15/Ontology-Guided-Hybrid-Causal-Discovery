"""Surface DECI runtime from DECI-specific outputs.

This script does not run DECI. It reads persisted DECI outputs, summarizes
runtime separately from the main classical-algorithm runtime comparison, and
appends a DECI-specific section to ``runtime_analysis.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "outputs" / "experiments"
RUNTIME_MD = EXP / "runtime_analysis.md"
SUMMARY_CSV = EXP / "deci_runtime_summary.csv"

DECI_FILES = [
    EXP / "deci_diagnostics.csv",
    EXP / "deci_threshold_sweep.csv",
    EXP / "deci_real_selected_config.csv",
]

RUNTIME_COLUMNS = [
    "runtime_seconds",
    "training_runtime_seconds",
    "training_time_seconds",
    "train_runtime_seconds",
    "elapsed_seconds",
    "runtime_s",
    "training_time",
    "fit_time_seconds",
]

SUMMARY_COLUMNS = [
    "variable_set",
    "constraint_mode",
    "epochs",
    "mean_runtime_s",
    "std_runtime_s",
    "n_runs",
]


def log(message: str) -> None:
    """Print with the DECI runtime prefix."""
    print(f"[deci_runtime] {message}", flush=True)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "No rows available."
    render = df.copy()
    for column in render.columns:
        if pd.api.types.is_numeric_dtype(render[column]):
            render[column] = render[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
            )
        else:
            render[column] = render[column].fillna("").astype(str)
    header = "| " + " | ".join(render.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(render.columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in render.columns) + " |"
        for _, row in render.iterrows()
    ]
    return "\n".join([header, separator, *body])


def find_runtime_column(df: pd.DataFrame) -> str | None:
    """Return the first recognized runtime column."""
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in RUNTIME_COLUMNS:
        if candidate in lower_map:
            return lower_map[candidate]
    for column in df.columns:
        lowered = column.lower()
        if "runtime" in lowered and "second" in lowered:
            return column
        if "time" in lowered and "second" in lowered:
            return column
    return None


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to successful rows when a status-like column is available."""
    filtered = df.copy()
    if "status" in filtered.columns:
        filtered = filtered[filtered["status"].astype(str).str.lower().eq("success")]
    elif "training_status" in filtered.columns:
        status = filtered["training_status"].astype(str).str.lower()
        filtered = filtered[status.eq("success") | status.str.startswith("success")]
    return filtered


def locate_deci_runtime() -> tuple[pd.DataFrame, Path, str] | None:
    """Find the first DECI file with usable runtime rows."""
    for path in DECI_FILES:
        if not path.exists():
            log(f"Checked {path.relative_to(ROOT)}: missing")
            continue
        df = pd.read_csv(path)
        runtime_col = find_runtime_column(df)
        if runtime_col is None:
            log(f"Checked {path.relative_to(ROOT)}: no runtime column")
            continue
        df = successful_rows(df)
        df[runtime_col] = pd.to_numeric(df[runtime_col], errors="coerce")
        df = df.dropna(subset=[runtime_col]).copy()
        if df.empty:
            log(f"Checked {path.relative_to(ROOT)}: runtime column exists but no usable rows")
            continue
        log(f"Using {path.relative_to(ROOT)} with runtime column `{runtime_col}`")
        return df, path, runtime_col
    return None


def normalize_constraint_mode(row: pd.Series) -> str:
    """Normalize DECI mode names for reporting."""
    raw = str(row.get("constraint_mode", row.get("mode", row.get("algorithm", "")))).strip().lower()
    if raw in {"constrained", "native_constrained", "deci_native_constrained"}:
        return "native_constrained"
    if raw in {"unconstrained", "native_unconstrained", "deci_native_unconstrained"}:
        return "unconstrained"
    if "constrained" in raw and "unconstrained" not in raw:
        return "native_constrained"
    if "unconstrained" in raw:
        return "unconstrained"
    return raw or "unknown"


def infer_epochs(row: pd.Series) -> str:
    """Return epochs from explicit column or config_id pattern."""
    if "epochs" in row and pd.notna(row["epochs"]):
        try:
            return str(int(float(row["epochs"])))
        except (TypeError, ValueError):
            return str(row["epochs"])
    config_id = str(row.get("config_id", ""))
    match = re.search(r"ep(\d+)", config_id)
    if match:
        return match.group(1)
    return "unknown"


def normalize_data(df: pd.DataFrame, runtime_col: str) -> pd.DataFrame:
    """Normalize DECI runtime rows to common columns."""
    out = df.copy()
    out["runtime_seconds_norm"] = pd.to_numeric(out[runtime_col], errors="coerce")
    out["dataset_norm"] = out["dataset"].astype(str) if "dataset" in out.columns else "unknown"
    out["variable_set_norm"] = out["variable_set"].astype(str) if "variable_set" in out.columns else "unknown"
    out["constraint_mode_norm"] = out.apply(normalize_constraint_mode, axis=1)
    out["epochs_norm"] = out.apply(infer_epochs, axis=1)
    return out.dropna(subset=["runtime_seconds_norm"]).copy()


def summarize_runtime(norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build summary CSV and matched constrained/unconstrained comparison."""
    grouped = (
        norm.groupby(
            ["dataset_norm", "variable_set_norm", "constraint_mode_norm", "epochs_norm"],
            dropna=False,
        )
        .agg(
            mean_runtime_s=("runtime_seconds_norm", "mean"),
            std_runtime_s=("runtime_seconds_norm", "std"),
            n_runs=("runtime_seconds_norm", "count"),
        )
        .reset_index()
    )
    grouped["std_runtime_s"] = grouped["std_runtime_s"].fillna(0.0)

    summary_csv = grouped.rename(
        columns={
            "variable_set_norm": "variable_set",
            "constraint_mode_norm": "constraint_mode",
            "epochs_norm": "epochs",
        }
    )[SUMMARY_COLUMNS].copy()

    rows: list[dict[str, Any]] = []
    for (dataset, variable_set, epochs), sub in grouped.groupby(
        ["dataset_norm", "variable_set_norm", "epochs_norm"],
        dropna=False,
    ):
        modes = {str(row["constraint_mode_norm"]): row for _, row in sub.iterrows()}
        uncon = modes.get("unconstrained")
        constrained = modes.get("native_constrained")
        if uncon is None or constrained is None:
            continue
        uncon_mean = float(uncon["mean_runtime_s"])
        con_mean = float(constrained["mean_runtime_s"])
        overhead_seconds = con_mean - uncon_mean
        multiplier = con_mean / uncon_mean if uncon_mean else np.nan
        if pd.isna(multiplier):
            multiplier_label = ""
        elif multiplier >= 1:
            multiplier_label = f"{multiplier:.2f}x slower"
        else:
            multiplier_label = f"{multiplier:.2f}x of unconstrained (faster)"
        rows.append(
            {
                "dataset": dataset,
                "variable_set": variable_set,
                "epochs": epochs,
                "unconstrained_mean_s": uncon_mean,
                "native_constrained_mean_s": con_mean,
                "overhead_seconds": overhead_seconds,
                "multiplier": multiplier,
                "multiplier_label": multiplier_label,
            }
        )
    comparison = pd.DataFrame(rows)
    return summary_csv, comparison


def build_section(source_path: Path, summary: pd.DataFrame, comparison: pd.DataFrame) -> str:
    """Build the Markdown section appended to runtime_analysis.md."""
    if comparison.empty:
        overhead_sentence = (
            "No matched native_constrained versus unconstrained configurations were "
            "available, so a multiplier could not be computed."
        )
    else:
        parts = []
        for _, row in comparison.iterrows():
            parts.append(
                f"{row['dataset']}/{row['variable_set']}/epochs={row['epochs']}: "
                f"{row['native_constrained_mean_s']:.3f}s vs "
                f"{row['unconstrained_mean_s']:.3f}s, "
                f"overhead {row['overhead_seconds']:.3f}s "
                f"({row['multiplier_label']})"
            )
        overhead_sentence = "; ".join(parts) + "."

    caveat = (
        "No failed or timeout rows are present in the selected-config runtime "
        "file used here. These are still local Windows/runtime observations and "
        "should be treated as configuration-specific rather than a universal DECI "
        "scalability benchmark."
    )

    return f"""## DECI runtime (separate analysis)

Source file: `{source_path.relative_to(ROOT)}`

### Summary

{markdown_table(summary)}

### Matched constrained versus unconstrained comparison

{markdown_table(comparison)}

DECI is reported separately because it follows a different run schedule from
the standard seed loop, uses selected configurations/thresholds, and operates
on a larger and threshold-dependent runtime scale. {overhead_sentence}

{caveat}
"""


def append_runtime_section(section: str) -> None:
    """Append the DECI runtime section, replacing an older copy if present."""
    RUNTIME_MD.parent.mkdir(parents=True, exist_ok=True)
    current = RUNTIME_MD.read_text(encoding="utf-8") if RUNTIME_MD.exists() else "# Runtime Analysis for RQ3\n"
    marker = "## DECI runtime (separate analysis)"
    if marker in current:
        current = current.split(marker)[0].rstrip()
    RUNTIME_MD.write_text(current.rstrip() + "\n\n" + section.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    """Run the DECI runtime summary."""
    found = locate_deci_runtime()
    if found is None:
        log(
            "DECI runtime data not persisted in standard columns. "
            "Recovering from log files would require rerun. Skipping."
        )
        return 0

    df, source_path, runtime_col = found
    norm = normalize_data(df, runtime_col)
    summary, comparison = summarize_runtime(norm)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    log(f"Summary written: {SUMMARY_CSV.relative_to(ROOT)}")

    section = build_section(source_path, summary, comparison)
    append_runtime_section(section)
    log(f"Appended DECI runtime section to {RUNTIME_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
