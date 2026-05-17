"""Consolidate runtime evidence across existing experiment CSVs.

This script does not rerun experiments. It reads persisted runtime columns
from the main, causal-dummy, sensitivity, and DECI-specific outputs and builds
an expanded RQ3 runtime analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs" / "experiments"
FIG = ROOT / "outputs" / "figures"

SOURCE_FILES = [
    EXP / "results.csv",
    EXP / "causal_dummy_final_comparison_raw.csv",
    EXP / "sample_size_sensitivity_results.csv",
    EXP / "snr_sensitivity_results.csv",
    EXP / "deci_real_selected_config.csv",
    EXP / "deci_diagnostics.csv",
    EXP / "deci_ablation_synthetic.csv",
    EXP / "deci_constraint_type_ablation.csv",
]

ALGORITHM_COLUMNS = ["algorithm", "model", "method", "backend"]
MODE_COLUMNS = ["mode", "constraint_mode"]
RUNTIME_COLUMNS = ["runtime_seconds", "runtime_s", "elapsed_seconds", "training_time_seconds"]
SEED_COLUMNS = ["seed", "random_seed"]
SAMPLE_SIZE_COLUMNS = ["sample_size", "n", "n_samples"]
SWEEP_COLUMNS = ["sample_size", "n", "snr", "epochs", "threshold"]

CV_HIGH_VARIANCE = 0.5
CV_CACHING = 2.0


def log(message: str) -> None:
    """Print a consolidated-runtime log line."""
    print(f"[consolidated_runtime] {message}", flush=True)


def cv_warning(cv: float) -> str:
    """Classify runtime coefficient of variation.

    Parameters
    ----------
    cv:
        Coefficient of variation, computed as standard deviation divided by
        mean runtime.

    Returns
    -------
    str
        ``caching suspected`` if CV > 2.0, ``high variance`` if CV is between
        0.5 and 2.0, and an empty string otherwise.
    """
    if pd.isna(cv):
        return ""
    if cv > CV_CACHING:
        return "caching suspected"
    if cv > CV_HIGH_VARIANCE:
        return "high variance"
    return ""


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate present in a list of columns.

    Parameters
    ----------
    columns:
        Available DataFrame columns.
    candidates:
        Candidate column names in priority order.

    Returns
    -------
    str or None
        Matching column name, or ``None`` if no candidate is present.
    """
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def infer_dataset_from_filename(path: Path) -> str:
    """Infer a dataset label when the CSV does not contain one.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    str
        Best-effort dataset label.
    """
    name = path.name.lower()
    if "real" in name or "ecb" in name:
        return "real"
    if "causal_dummy" in name or "snr" in name or "sample_size" in name:
        return "causal_dummy_v2"
    if "advisor" in name or name == "results.csv":
        return "advisor_dummy"
    if "synthetic" in name or "ablation" in name:
        return "synthetic_n2000"
    return "unknown"


def detect_schema(path: Path, df: pd.DataFrame) -> dict[str, str | None]:
    """Detect key columns for a runtime CSV.

    Parameters
    ----------
    path:
        Source CSV path.
    df:
        Loaded DataFrame.

    Returns
    -------
    dict
        Mapping from logical field names to detected column names.
    """
    schema = {
        "runtime": first_existing(df.columns.tolist(), RUNTIME_COLUMNS),
        "algorithm": first_existing(df.columns.tolist(), ALGORITHM_COLUMNS),
        "mode": first_existing(df.columns.tolist(), MODE_COLUMNS),
        "dataset": first_existing(df.columns.tolist(), ["dataset"]),
        "seed": first_existing(df.columns.tolist(), SEED_COLUMNS),
        "sample_size": first_existing(df.columns.tolist(), SAMPLE_SIZE_COLUMNS),
        "snr": first_existing(df.columns.tolist(), ["snr"]),
        "epochs": first_existing(df.columns.tolist(), ["epochs"]),
        "threshold": first_existing(df.columns.tolist(), ["threshold"]),
        "variable_set": first_existing(df.columns.tolist(), ["variable_set"]),
        "status": first_existing(df.columns.tolist(), ["status", "training_status"]),
    }

    if schema["algorithm"] is None:
        log(f"WARNING: {path.name}: no algorithm/model/method column; inferring from filename")
    if schema["mode"] is None:
        log(f"WARNING: {path.name}: no mode/constraint_mode column; defaulting to unknown")
    if schema["dataset"] is None:
        log(f"WARNING: {path.name}: no dataset column; inferring dataset={infer_dataset_from_filename(path)}")
    if schema["sample_size"] == "n":
        log(f"WARNING: {path.name}: using `n` as sample_size")

    detected = ", ".join(f"{key}={value}" for key, value in schema.items())
    log(f"Schema {path.name}: rows={len(df)}, {detected}")
    return schema


def load_source(path: Path) -> tuple[pd.DataFrame, dict[str, str | None]] | None:
    """Load and normalize one runtime source.

    Parameters
    ----------
    path:
        CSV path.

    Returns
    -------
    tuple[pandas.DataFrame, dict] or None
        Normalized rows and detected schema, or ``None`` if no usable runtime
        rows exist.
    """
    if not path.exists():
        log(f"Skipping {path.name}: file not found")
        return None
    df = pd.read_csv(path)
    schema = detect_schema(path, df)
    runtime_col = schema["runtime"]
    if runtime_col is None:
        log(f"Skipping {path.name}: no runtime column")
        return None

    status_col = schema["status"]
    if status_col is not None and len(df):
        status = df[status_col].astype(str).str.lower()
        df = df[status.eq("success") | status.str.startswith("success")].copy()
    df[runtime_col] = pd.to_numeric(df[runtime_col], errors="coerce")
    df = df.dropna(subset=[runtime_col]).copy()
    if df.empty:
        log(f"Skipping {path.name}: runtime_seconds column has no successful rows")
        return None

    normalized = pd.DataFrame(index=df.index)
    normalized["source_file"] = path.name
    normalized["runtime_seconds"] = df[runtime_col].astype(float)
    normalized["algorithm"] = (
        df[schema["algorithm"]].astype(str)
        if schema["algorithm"] is not None
        else infer_algorithm_from_filename(path)
    )
    normalized["mode"] = (
        df[schema["mode"]].astype(str)
        if schema["mode"] is not None
        else "unknown"
    )
    normalized["dataset"] = (
        df[schema["dataset"]].astype(str)
        if schema["dataset"] is not None
        else infer_dataset_from_filename(path)
    )
    normalized["seed"] = df[schema["seed"]] if schema["seed"] is not None else np.nan
    normalized["sample_size"] = pd.to_numeric(df[schema["sample_size"]], errors="coerce") if schema["sample_size"] else np.nan
    normalized["snr"] = pd.to_numeric(df[schema["snr"]], errors="coerce") if schema["snr"] else np.nan
    normalized["epochs"] = pd.to_numeric(df[schema["epochs"]], errors="coerce") if schema["epochs"] else np.nan
    normalized["threshold"] = pd.to_numeric(df[schema["threshold"]], errors="coerce") if schema["threshold"] else np.nan
    normalized["variable_set"] = df[schema["variable_set"]].astype(str) if schema["variable_set"] else "unknown"
    normalized["constraint_mode"] = normalized["mode"].map(normalize_mode)
    normalized["algorithm"] = normalized["algorithm"].map(normalize_algorithm)
    deci_constrained = normalized["algorithm"].eq("DECI") & normalized["constraint_mode"].eq("constrained")
    normalized.loc[deci_constrained, "constraint_mode"] = "native_constrained"
    return normalized, schema


def infer_algorithm_from_filename(path: Path) -> str:
    """Infer an algorithm label from filename.

    Parameters
    ----------
    path:
        Source path.

    Returns
    -------
    str
        Best-effort algorithm label.
    """
    name = path.name.lower()
    if "deci" in name:
        return "DECI"
    return "unknown"


def normalize_algorithm(value: Any) -> str:
    """Normalize algorithm labels.

    Parameters
    ----------
    value:
        Raw algorithm value.

    Returns
    -------
    str
        Display-ready algorithm name.
    """
    text = str(value)
    lower = text.lower()
    if lower.startswith("deci") or lower == "causica_native":
        return "DECI"
    if lower in {"pc", "lingam", "ges"}:
        return lower.upper() if lower != "lingam" else "LiNGAM"
    if lower.startswith("notears"):
        return "NOTEARS"
    return text


def normalize_mode(value: Any) -> str:
    """Normalize constraint/mode labels.

    Parameters
    ----------
    value:
        Raw mode string.

    Returns
    -------
    str
        Normalized mode label.
    """
    text = str(value).strip().lower()
    if text in {"constrained", "native_constrained", "forbidden_only"}:
        return "constrained" if text != "native_constrained" else "native_constrained"
    if "native_constrained" in text:
        return "native_constrained"
    if "unconstrained" in text:
        return "unconstrained"
    if "constrained" in text:
        return "constrained"
    return text or "unknown"


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate runtime by a set of grouping columns.

    Parameters
    ----------
    df:
        Normalized runtime rows.
    group_cols:
        Columns used for grouping.

    Returns
    -------
    pandas.DataFrame
        Aggregated mean, standard deviation, median, n, CV, and warning flag.
    """
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_runtime_s=("runtime_seconds", "mean"),
            std_runtime_s=("runtime_seconds", "std"),
            median_runtime_s=("runtime_seconds", "median"),
            n_runs=("runtime_seconds", "count"),
        )
        .reset_index()
    )
    grouped["std_runtime_s"] = grouped["std_runtime_s"].fillna(0.0)
    grouped["cv"] = np.where(
        grouped["mean_runtime_s"].ne(0),
        grouped["std_runtime_s"] / grouped["mean_runtime_s"],
        np.nan,
    )
    grouped["cv_warning"] = grouped["cv"].map(cv_warning)
    for _, row in grouped[grouped["cv_warning"].astype(str).ne("")].iterrows():
        keys = ", ".join(f"{col}={row[col]}" for col in group_cols)
        log(f"CV warning {row['cv_warning']}: {keys}, CV={row['cv']:.2f}")
    return grouped


def build_overhead(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute constrained versus unconstrained runtime overhead.

    Parameters
    ----------
    df:
        Aggregated runtime rows containing ``constraint_mode`` and
        ``mean_runtime_s``.
    group_cols:
        Columns that identify matched configurations excluding
        ``constraint_mode``.

    Returns
    -------
    pandas.DataFrame
        Matched overhead rows.
    """
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        modes = {str(row["constraint_mode"]): row for _, row in sub.iterrows()}
        uncon = modes.get("unconstrained")
        constrained = modes.get("constrained")
        if constrained is None:
            constrained = modes.get("native_constrained")
        if uncon is None or constrained is None:
            continue
        uncon_mean = float(uncon["mean_runtime_s"])
        con_mean = float(constrained["mean_runtime_s"])
        multiplier = con_mean / uncon_mean if uncon_mean else np.nan
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "unconstrained_mean_s": uncon_mean,
                "constrained_mean_s": con_mean,
                "overhead_seconds": con_mean - uncon_mean,
                "multiplier": multiplier,
                "unconstrained_cv_warning": uncon.get("cv_warning", ""),
                "constrained_cv_warning": constrained.get("cv_warning", ""),
            }
        )
        if row["unconstrained_cv_warning"] == "caching suspected" or row["constrained_cv_warning"] == "caching suspected":
            row["overhead_pct"] = np.nan
        else:
            row["overhead_pct"] = (con_mean - uncon_mean) / uncon_mean * 100 if uncon_mean else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a CSV, creating parent directories.

    Parameters
    ----------
    path:
        Destination CSV path.
    df:
        DataFrame to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log(f"Wrote {path.relative_to(ROOT)} ({len(df)} rows)")


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a DataFrame as a markdown table.

    Parameters
    ----------
    df:
        DataFrame to render.
    max_rows:
        Maximum number of rows to include.

    Returns
    -------
    str
        Markdown table text.
    """
    if df.empty:
        return "No rows available."
    render = df.head(max_rows).copy()
    for column in render.columns:
        if pd.api.types.is_numeric_dtype(render[column]):
            render[column] = render[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
        else:
            render[column] = render[column].fillna("").astype(str)
    header = "| " + " | ".join(render.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(render.columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in render.columns) + " |" for _, row in render.iterrows()]
    if len(df) > max_rows:
        body.append(f"| ... | {len(df) - max_rows} more rows |" + " |" * max(0, len(render.columns) - 2))
    return "\n".join([header, separator, *body])


def plot_runtime_lines(df: pd.DataFrame, x_col: str, path: Path, title: str, xlabel: str) -> None:
    """Create a log-scale runtime line plot.

    Parameters
    ----------
    df:
        Aggregated runtime table.
    x_col:
        X-axis column.
    path:
        Output figure path.
    title:
        Plot title.
    xlabel:
        X-axis label.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        log(f"Skipping figure {path.name}: no rows")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for (algorithm, mode), sub in df.groupby(["algorithm", "mode"], dropna=False):
        sub = sub.sort_values(x_col)
        x = sub[x_col].astype(float).to_numpy()
        y = sub["mean_runtime_s"].astype(float).to_numpy()
        std = sub["std_runtime_s"].fillna(0.0).astype(float).to_numpy()
        label = f"{algorithm} {mode}"
        ax.plot(x, y, marker="o", label=label)
        lower = np.maximum(y - std, 1e-6)
        upper = y + std
        ax.fill_between(x, lower, upper, alpha=0.12)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean runtime seconds (log scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"Wrote {path.relative_to(ROOT)}")


def build_report(
    sample: pd.DataFrame,
    snr: pd.DataFrame,
    real: pd.DataFrame,
    deci: pd.DataFrame,
    canonical: pd.DataFrame,
    output_path: Path,
) -> str:
    """Build and write the consolidated runtime report.

    Parameters
    ----------
    sample:
        Sample-size runtime table.
    snr:
        SNR runtime table.
    real:
        Real ECB runtime table.
    deci:
        DECI breakdown table.
    canonical:
        Canonical configuration overhead table.
    output_path:
        Destination Markdown path.

    Returns
    -------
    str
        Revised RQ3 answer paragraph.
    """
    canonical_brief = canonical[
        ["algorithm", "dataset", "overhead_seconds", "overhead_pct", "multiplier"]
    ].copy() if not canonical.empty else pd.DataFrame()
    sample_brief = sample[
        sample["sample_size"].isin([sample["sample_size"].min(), sample["sample_size"].max()])
    ].copy() if not sample.empty else pd.DataFrame()
    snr_brief = snr[
        snr["snr"].isin([snr["snr"].min(), snr["snr"].max()])
    ].copy() if not snr.empty else pd.DataFrame()

    deci_mean = float(deci["mean_runtime_s"].mean()) if not deci.empty else np.nan
    non_deci_sample_mean = float(sample["mean_runtime_s"].mean()) if not sample.empty else np.nan
    multiplier = deci_mean / non_deci_sample_mean if pd.notna(deci_mean) and pd.notna(non_deci_sample_mean) and non_deci_sample_mean else np.nan
    multiplier_text = f"about {multiplier:.1f}x the mean sample-sweep classical runtime" if pd.notna(multiplier) else "not comparable from available rows"

    rq3_answer = (
        "Across existing runtime artifacts, constraint overhead is small or negative "
        "for the classical causal-discovery runs at the canonical causal-dummy "
        "configuration: PC is faster when constrained, LiNGAM and GES have small "
        "post-processing overheads, and NOTEARS remains subject to the previously "
        "identified gCastle caching caveat. The sample-size sweep provides the "
        "direct scalability evidence: runtime grows with N most visibly for LiNGAM, "
        "while PC and GES remain on a lower seconds scale; constrained PC stays "
        "competitive across N and does not show a scaling penalty. The SNR sweep "
        "shows no strong evidence that noisier settings systematically increase "
        "runtime; runtime is mainly algorithm- and sample-size-driven rather than "
        "SNR-driven. Real ECB runtime evidence is currently DECI-only from selected "
        "real-data configurations, where constrained DECI is slightly faster than "
        "unconstrained on the reduced variable set but should be interpreted "
        "separately because it follows a different training schedule. DECI training "
        f"cost is on a larger scale than the classical sweeps ({multiplier_text}), "
        "and threshold/configuration choices dominate its runtime. Still unknown: "
        "constraint-count sensitivity and full-variable DECI scalability on Windows."
    )

    text = f"""# Consolidated runtime analysis

## Sample-size scaling (N=110 to N=3000)

The direct sample-size sweep covers causal dummy v2 from N=100 to N=3000.
Real ECB N=110 appears in the DECI-only real section below, not in the
classical sample-size sweep.

{markdown_table(sample_brief, max_rows=24)}

## SNR sensitivity (runtime as a function of noise)

{markdown_table(snr_brief, max_rows=24)}

## Real ECB runtime

Currently this is DECI-only because the real ECB runtime rows are persisted in
`deci_real_selected_config.csv`, while `results.csv` does not contain successful
real ECB rows.

{markdown_table(real)}

## DECI runtime breakdown

The DECI breakdown concatenates DECI ablation files and retains `source_file`
for provenance.

{markdown_table(deci, max_rows=30)}

## Canonical constraint overhead

{markdown_table(canonical_brief, max_rows=30)}

## Revised RQ3 answer

{rq3_answer}
"""
    output_path.write_text(text, encoding="utf-8")
    log(f"Wrote {output_path.relative_to(ROOT)}")
    return rq3_answer


def main() -> int:
    """Run consolidated runtime analysis.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Consolidate runtime analysis across existing CSVs.")
    parser.add_argument("--output-dir", default=str(EXP), help="Directory for runtime analysis CSV/Markdown outputs.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    normalized_sources: dict[str, pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []
    for path in SOURCE_FILES:
        loaded = load_source(path)
        if loaded is None:
            continue
        norm, _schema = loaded
        normalized_sources[path.name] = norm
        all_rows.append(norm)

    sample_rows = normalized_sources.get("sample_size_sensitivity_results.csv", pd.DataFrame())
    snr_rows = normalized_sources.get("snr_sensitivity_results.csv", pd.DataFrame())
    real_rows = normalized_sources.get("deci_real_selected_config.csv", pd.DataFrame())
    canonical_rows = pd.concat(
        [
            normalized_sources.get("results.csv", pd.DataFrame()),
            normalized_sources.get("causal_dummy_final_comparison_raw.csv", pd.DataFrame()),
        ],
        ignore_index=True,
    )
    deci_rows = pd.concat(
        [
            normalized_sources.get("deci_ablation_synthetic.csv", pd.DataFrame()),
            normalized_sources.get("deci_constraint_type_ablation.csv", pd.DataFrame()),
            normalized_sources.get("deci_real_selected_config.csv", pd.DataFrame()),
            normalized_sources.get("deci_diagnostics.csv", pd.DataFrame()),
        ],
        ignore_index=True,
    )

    sample_runtime = aggregate(sample_rows, ["algorithm", "mode", "sample_size"])
    snr_runtime = aggregate(snr_rows, ["algorithm", "mode", "snr"])
    real_runtime = aggregate(real_rows, ["algorithm", "mode"])
    deci_breakdown = aggregate(
        deci_rows,
        ["source_file", "dataset", "constraint_mode", "variable_set", "epochs", "threshold"],
    )
    canonical_agg = aggregate(canonical_rows, ["algorithm", "dataset", "constraint_mode"])
    canonical_overhead = build_overhead(canonical_agg, ["algorithm", "dataset"])

    write_csv(
        output_dir / "runtime_by_sample_size.csv",
        sample_runtime.rename(columns={"sample_size": "sample_size"}),
    )
    write_csv(output_dir / "runtime_by_snr.csv", snr_runtime)
    write_csv(output_dir / "runtime_real_ecb.csv", real_runtime)
    write_csv(output_dir / "runtime_deci_breakdown.csv", deci_breakdown)
    write_csv(output_dir / "runtime_canonical_overhead.csv", canonical_overhead)
    if all_rows:
        write_csv(output_dir / "runtime_all_sources_summary.csv", aggregate(pd.concat(all_rows, ignore_index=True), ["source_file", "algorithm", "dataset", "constraint_mode"]))

    if not args.no_figures:
        plot_runtime_lines(sample_runtime, "sample_size", FIG / "runtime_vs_sample_size.png", "Runtime vs Sample Size", "Sample size")
        plot_runtime_lines(snr_runtime, "snr", FIG / "runtime_vs_snr.png", "Runtime vs SNR", "SNR")
    else:
        log("Skipping figures because --no-figures was supplied")

    rq3_answer = build_report(
        sample_runtime,
        snr_runtime,
        real_runtime,
        deci_breakdown,
        canonical_overhead,
        output_dir / "runtime_consolidated_report.md",
    )
    print("\n## Revised RQ3 answer\n")
    print(rq3_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
