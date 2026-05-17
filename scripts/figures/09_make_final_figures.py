"""Generate final experiment figures from current output tables."""

from __future__ import annotations

from pathlib import Path

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
from causal_dummy_experiment_utils import (
    ALGORITHMS_CORE,
    ALGORITHMS_FINAL,
    EXP_DIR,
    FIG_DIR,
    ensure_dirs,
    log_command,
    plot_grouped_metric,
    plot_line_metric,
)


CAUSAL_SUMMARY = EXP_DIR / "causal_dummy_final_comparison_summary.csv"
SNR_SUMMARY = EXP_DIR / "snr_sensitivity_summary.csv"
SAMPLE_SUMMARY = EXP_DIR / "sample_size_sensitivity_summary.csv"
ADVISOR_TABLE = EXP_DIR / "final_advisor_dummy_constraint_compliance.csv"
REAL_TABLE = EXP_DIR / "final_real_ecb_case_study.csv"
STABILITY_CAUSAL = EXP_DIR / "stability_summary_causal_dummy.csv"


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV or return an empty table."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def placeholder(path: Path, message: str) -> None:
    """Create a placeholder figure when source data are unavailable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def causal_dummy_figures() -> list[Path]:
    """Create causal-dummy comparison figures."""
    summary = read_csv(CAUSAL_SUMMARY)
    outputs = [
        (FIG_DIR / "causal_dummy_f1_comparison.png", "f1", "F1", "Causal Dummy v2: F1"),
        (FIG_DIR / "causal_dummy_final_f1.png", "f1", "F1", "Causal Dummy v2: F1"),
        (FIG_DIR / "causal_dummy_shd_comparison.png", "shd", "SHD", "Causal Dummy v2: SHD"),
        (FIG_DIR / "causal_dummy_final_shd.png", "shd", "SHD", "Causal Dummy v2: SHD"),
        (FIG_DIR / "causal_dummy_violations.png", "violations", "Forbidden-edge violations", "Causal Dummy v2: Violations"),
        (FIG_DIR / "causal_dummy_final_violations.png", "violations", "Forbidden-edge violations", "Causal Dummy v2: Violations"),
        (FIG_DIR / "causal_dummy_runtime.png", "runtime_seconds", "Runtime seconds", "Causal Dummy v2: Runtime"),
        (FIG_DIR / "causal_dummy_final_runtime.png", "runtime_seconds", "Runtime seconds", "Causal Dummy v2: Runtime"),
    ]
    if summary.empty:
        for path, _metric, _ylabel, _title in outputs:
            placeholder(path, "Missing causal dummy summary")
        placeholder(FIG_DIR / "causal_dummy_precision_recall.png", "Missing causal dummy summary")
        return [path for path, _metric, _ylabel, _title in outputs] + [FIG_DIR / "causal_dummy_precision_recall.png"]
    for path, metric, ylabel, title in outputs:
        plot_grouped_metric(summary, metric, ylabel, title, path, ALGORITHMS_FINAL)
    precision_recall(summary, FIG_DIR / "causal_dummy_precision_recall.png")
    return [path for path, _metric, _ylabel, _title in outputs] + [FIG_DIR / "causal_dummy_precision_recall.png"]


def precision_recall(summary: pd.DataFrame, path: Path) -> None:
    """Plot precision and recall side by side for final comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty:
        placeholder(path, "Missing causal dummy summary")
        return
    data = summary.copy()
    data = data[data["algorithm"].isin(ALGORITHMS_FINAL)]
    data["label"] = data["algorithm"].astype(str) + "\n" + data["constraint_mode"].astype(str)
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.8), 5.2), dpi=160)
    ax.bar(x - width / 2, data["precision_mean"].astype(float), width, label="precision", color="#60A5FA")
    ax.bar(x + width / 2, data["recall_mean"].astype(float), width, label="recall", color="#34D399")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("Causal Dummy v2: Precision and Recall")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def sensitivity_figures() -> list[Path]:
    """Create SNR and sample-size figures."""
    paths: list[Path] = []
    snr = read_csv(SNR_SUMMARY)
    for path, metric, ylabel, title in [
        (FIG_DIR / "snr_f1_sweep.png", "f1", "F1", "Causal Dummy v2: F1 vs SNR"),
        (FIG_DIR / "snr_shd_sweep.png", "shd", "SHD", "Causal Dummy v2: SHD vs SNR"),
    ]:
        if snr.empty:
            placeholder(path, "Missing SNR summary")
        else:
            plot_line_metric(snr, "snr", metric, ylabel, title, path, ALGORITHMS_CORE)
        paths.append(path)

    sample = read_csv(SAMPLE_SUMMARY)
    for path, metric, ylabel, title in [
        (FIG_DIR / "sample_size_f1.png", "f1", "F1", "Causal Dummy v2: F1 vs Sample Size"),
        (FIG_DIR / "sample_size_shd.png", "shd", "SHD", "Causal Dummy v2: SHD vs Sample Size"),
    ]:
        if sample.empty:
            placeholder(path, "Missing sample-size summary")
        else:
            plot_line_metric(sample, "n", metric, ylabel, title, path, ALGORITHMS_CORE)
        paths.append(path)
    return paths


def stability_figure() -> Path:
    """Create causal-dummy stability figure."""
    path = FIG_DIR / "stability_causal_dummy.png"
    summary = read_csv(STABILITY_CAUSAL)
    if summary.empty:
        placeholder(path, "Missing stability summary")
        return path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = summary.copy()
    data["label"] = data["algorithm"].astype(str) + "\n" + data["constraint_mode"].astype(str)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.8), 5), dpi=160)
    ax.bar(x, data["stable_edges_60"].astype(float), color="#60A5FA")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=20, ha="right")
    ax.set_ylabel("Edges stable in at least 60% of seeds")
    ax.set_title("Causal Dummy v2: Edge Stability")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def advisor_figure() -> Path:
    """Create advisor-dummy violation figure."""
    path = FIG_DIR / "advisor_dummy_violations.png"
    table = read_csv(ADVISOR_TABLE)
    if table.empty or "violations_mean" not in table.columns:
        placeholder(path, "Missing advisor dummy table")
        return path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = table.copy()
    data = data[data["constraint_mode"].astype(str).isin(["none", "forbidden_only", "required_light"])]
    data["label"] = data["algorithm"].astype(str) + "\n" + data["constraint_mode"].astype(str)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(max(9, len(data) * 0.55), 5), dpi=160)
    ax.bar(x, data["violations_mean"].fillna(0).astype(float), color="#F59E0B")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=35, ha="right")
    ax.set_ylabel("Mean forbidden-edge violations")
    ax.set_title("Advisor Dummy: Constraint Violations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def real_figure() -> Path:
    """Create real ECB alignment and violation figure."""
    path = FIG_DIR / "real_ecb_alignment_violations.png"
    table = read_csv(REAL_TABLE)
    if table.empty:
        placeholder(path, "Missing real ECB table")
        return path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = table.copy()
    data["label"] = data["algorithm"].astype(str) + "\n" + data["constraint_mode"].astype(str)
    x = np.arange(len(data))
    width = 0.36
    alignment = data["alignment_mean"].fillna(0).astype(float) if "alignment_mean" in data else np.zeros(len(data))
    violations = data["violations_mean"].fillna(0).astype(float) if "violations_mean" in data else np.zeros(len(data))
    fig, ax = plt.subplots(figsize=(max(9, len(data) * 0.65), 5), dpi=160)
    ax.bar(x - width / 2, alignment, width, label="literature alignment", color="#60A5FA")
    ax.bar(x + width / 2, violations, width, label="violations", color="#F87171")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=35, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Real ECB: Alignment and Violations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    """Generate all final requested figures."""
    ensure_dirs()
    log_command("09_make_final_figures.py", {})
    outputs = []
    outputs.extend(causal_dummy_figures())
    outputs.extend(sensitivity_figures())
    outputs.append(stability_figure())
    outputs.append(advisor_figure())
    outputs.append(real_figure())
    print("[figures] Wrote:")
    for path in outputs:
        print(f"[figures] - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
