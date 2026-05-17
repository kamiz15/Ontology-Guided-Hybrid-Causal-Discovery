"""Create final experiment tables from generated quantitative outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CORE_DIR = REPO_ROOT / "scripts" / "core"
for _path in (SCRIPT_DIR, CORE_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from causal_dummy_experiment_utils import EXP_DIR, ROOT, ensure_dirs, log_command, markdown_table


CAUSAL_SUMMARY = EXP_DIR / "causal_dummy_final_comparison_summary.csv"
SNR_SUMMARY = EXP_DIR / "snr_sensitivity_summary.csv"
SAMPLE_SUMMARY = EXP_DIR / "sample_size_sensitivity_summary.csv"
ADVISOR_ABLATION = EXP_DIR / "advisor_dummy_constraint_ablation_summary.csv"
ADVISOR_FALLBACK = EXP_DIR / "advisor_dummy_results_summary.csv"
REAL_FINAL_OLD = EXP_DIR / "final_algorithm_comparison_real_ecb.csv"
REAL_SUMMARY = EXP_DIR / "results_summary.csv"

FINAL_CAUSAL = EXP_DIR / "final_causal_dummy_comparison.csv"
FINAL_SNR = EXP_DIR / "final_snr_sensitivity.csv"
FINAL_SAMPLE = EXP_DIR / "final_sample_size_sensitivity.csv"
FINAL_ADVISOR = EXP_DIR / "final_advisor_dummy_constraint_compliance.csv"
FINAL_REAL = EXP_DIR / "final_real_ecb_case_study.csv"
FINAL_MD = EXP_DIR / "final_experiment_summary.md"


DISPLAY = {
    "pc": "PC",
    "lingam": "LiNGAM",
    "notears": "NOTEARS",
    "notears_postproc": "NOTEARS",
    "ges": "GES",
    "ges_postproc": "GES",
    "PC": "PC",
    "LiNGAM": "LiNGAM",
    "NOTEARS": "NOTEARS",
    "GES": "GES",
    "DECI": "DECI",
}


def display_algorithm(value: Any) -> str:
    """Normalize algorithm labels for final tables."""
    text = str(value)
    if text.startswith("deci"):
        return "DECI"
    return DISPLAY.get(text, text)


def fmt_mean_std(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return a string column formatted as mean +/- std."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    std = df[std_col] if std_col in df.columns else 0.0
    return [
        "" if pd.isna(mean) else f"{float(mean):.4f} +/- {float(0.0 if pd.isna(s) else s):.4f}"
        for mean, s in zip(df[mean_col], std)
    ]


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV when available."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_causal_table() -> pd.DataFrame:
    """Build the final causal-dummy true-DAG comparison table."""
    df = read_csv_or_empty(CAUSAL_SUMMARY)
    if df.empty:
        return df
    out = df.copy()
    out["algorithm"] = out["algorithm"].map(display_algorithm)
    rename = {
        "f1_mean": "F1_mean",
        "f1_std": "F1_std",
        "shd_mean": "SHD_mean",
        "shd_std": "SHD_std",
        "precision_mean": "precision_mean",
        "precision_std": "precision_std",
        "recall_mean": "recall_mean",
        "recall_std": "recall_std",
        "violations_mean": "violations_mean",
        "violations_std": "violations_std",
        "runtime_seconds_mean": "runtime_seconds_mean",
        "runtime_seconds_std": "runtime_seconds_std",
    }
    out = out.rename(columns=rename)
    columns = [
        "algorithm",
        "constraint_mode",
        "F1_mean",
        "F1_std",
        "SHD_mean",
        "SHD_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "violations_mean",
        "violations_std",
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "successful_runs",
        "failed_runs",
    ]
    return out[[column for column in columns if column in out.columns]]


def build_sensitivity_table(path: Path, group_col: str) -> pd.DataFrame:
    """Build a final sensitivity table with mean +/- std columns."""
    df = read_csv_or_empty(path)
    if df.empty:
        return df
    out = df.copy()
    out["algorithm"] = out["algorithm"].map(display_algorithm)
    for metric in ["f1", "shd", "precision", "recall", "runtime_seconds"]:
        out[f"{metric}_mean_std"] = fmt_mean_std(out, metric)
    columns = [
        "algorithm",
        "constraint_mode",
        group_col,
        "f1_mean_std",
        "shd_mean_std",
        "precision_mean_std",
        "recall_mean_std",
        "runtime_seconds_mean_std",
        "successful_runs",
        "failed_runs",
    ]
    return out[[column for column in columns if column in out.columns]]


def build_advisor_table() -> pd.DataFrame:
    """Build advisor-dummy reference-DAG alignment and violation table."""
    source = ADVISOR_ABLATION if ADVISOR_ABLATION.exists() else ADVISOR_FALLBACK
    df = read_csv_or_empty(source)
    if df.empty:
        return df
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).eq("advisor_dummy")].copy()
    if df.empty:
        return df
    out = pd.DataFrame({
        "algorithm": df["algorithm"].map(display_algorithm),
        "mode": df.get("mode", ""),
        "constraint_mode": df.get("constraint_mode", ""),
        "reference_dag_f1_mean": df.get("f1_directed_mean", np.nan),
        "reference_dag_f1_std": df.get("f1_directed_std", np.nan),
        "reference_dag_shd_mean": df.get("shd_mean", np.nan),
        "reference_dag_shd_std": df.get("shd_std", np.nan),
        "violations_mean": df.get("literature_violation_count_mean", np.nan),
        "violations_std": df.get("literature_violation_count_std", np.nan),
        "edge_count_mean": df.get("edge_count_predicted_mean", np.nan),
        "runtime_seconds_mean": df.get("runtime_seconds_mean", np.nan),
        "successful_runs": df.get("successful_runs", np.nan),
    })
    out = out.sort_values(["algorithm", "constraint_mode", "mode"], kind="stable")
    return out


def build_real_table() -> pd.DataFrame:
    """Build real ECB alignment and violation table without F1/SHD."""
    if REAL_FINAL_OLD.exists():
        df = pd.read_csv(REAL_FINAL_OLD)
        columns = [
            "algorithm",
            "constraint_mode",
            "constraint_handling",
            "alignment_mean",
            "alignment_std",
            "edge_count_mean",
            "violations_mean",
            "stable_edges_60",
            "stable_edges_80",
        ]
        return df[[column for column in columns if column in df.columns]].copy()

    df = read_csv_or_empty(REAL_SUMMARY)
    if df.empty or "dataset" not in df.columns:
        return pd.DataFrame()
    real = df[df["dataset"].astype(str).eq("real")].copy()
    if real.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "algorithm": real["algorithm"].map(display_algorithm),
        "constraint_mode": real.get("constraint_mode", ""),
        "alignment_mean": real.get("literature_alignment_score_mean", np.nan),
        "alignment_std": real.get("literature_alignment_score_std", np.nan),
        "edge_count_mean": real.get("edge_count_predicted_mean", np.nan),
        "violations_mean": real.get("literature_violation_count_mean", np.nan),
        "successful_runs": real.get("successful_runs", np.nan),
    })


def write_summary(
    causal: pd.DataFrame,
    snr: pd.DataFrame,
    sample: pd.DataFrame,
    advisor: pd.DataFrame,
    real: pd.DataFrame,
) -> None:
    """Write a concise final experiment summary."""
    def best_causal() -> str:
        if causal.empty or "F1_mean" not in causal.columns:
            return "No causal-dummy final comparison rows available."
        ok = causal.sort_values(["F1_mean", "SHD_mean"], ascending=[False, True]).head(5)
        return markdown_table(ok[["algorithm", "constraint_mode", "F1_mean", "SHD_mean", "precision_mean", "recall_mean", "violations_mean"]])

    text = f"""# Final Experiment Summary

## Generated Tables

- `{FINAL_CAUSAL.relative_to(ROOT)}`
- `{FINAL_SNR.relative_to(ROOT)}`
- `{FINAL_SAMPLE.relative_to(ROOT)}`
- `{FINAL_ADVISOR.relative_to(ROOT)}`
- `{FINAL_REAL.relative_to(ROOT)}`

## Causal Dummy v2

Main causal-recovery metrics are computed against the generated ground-truth DAG.

{best_causal()}

## SNR Sensitivity

{markdown_table(snr.head(10))}

## Sample-Size Sensitivity

{markdown_table(sample.head(10))}

## Advisor Dummy

Advisor-dummy rows are reference-DAG alignment and constraint-compliance outputs only.

{markdown_table(advisor.head(10))}

## Real ECB Case Study

Real ECB rows are literature-alignment and violation-reduction outputs only. No F1 or SHD is reported here.

{markdown_table(real.head(10))}
"""
    FINAL_MD.write_text(text, encoding="utf-8")


def main() -> int:
    """Build final tables."""
    ensure_dirs()
    log_command("20_create_final_result_tables.py", {})

    causal = build_causal_table()
    snr = build_sensitivity_table(SNR_SUMMARY, "snr")
    sample = build_sensitivity_table(SAMPLE_SUMMARY, "n")
    advisor = build_advisor_table()
    real = build_real_table()

    causal.to_csv(FINAL_CAUSAL, index=False)
    snr.to_csv(FINAL_SNR, index=False)
    sample.to_csv(FINAL_SAMPLE, index=False)
    advisor.to_csv(FINAL_ADVISOR, index=False)
    real.to_csv(FINAL_REAL, index=False)
    write_summary(causal, snr, sample, advisor, real)

    print(f"[tables] Causal dummy -> {FINAL_CAUSAL}")
    print(f"[tables] SNR -> {FINAL_SNR}")
    print(f"[tables] Sample size -> {FINAL_SAMPLE}")
    print(f"[tables] Advisor dummy -> {FINAL_ADVISOR}")
    print(f"[tables] Real ECB -> {FINAL_REAL}")
    print(f"[tables] Summary -> {FINAL_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
