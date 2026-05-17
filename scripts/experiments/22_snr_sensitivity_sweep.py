"""Run the causal-dummy v2 SNR sensitivity experiment.

This script uses the fixed ground-truth DAG from ``generate_causal_dummy.py``
and evaluates PC, LiNGAM, and GES under unconstrained and forbidden-constrained
settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    CONSTRAINT_MODES,
    DEFAULT_N,
    DEFAULT_SEEDS,
    DEFAULT_SNR_GRID,
    EXP_DIR,
    FIG_DIR,
    RESULT_COLUMNS,
    ensure_dirs,
    log_command,
    parse_float_list,
    parse_int_list,
    plot_line_metric,
    run_experiment_cell,
    run_experiment_pair,
    script_command,
    summarize_results,
    write_csv,
    write_experiment_report,
)


RESULTS_PATH = EXP_DIR / "snr_sensitivity_results.csv"
SUMMARY_PATH = EXP_DIR / "snr_sensitivity_summary.csv"
REPORT_PATH = EXP_DIR / "snr_sensitivity_report.md"


def run_sweep(snr_grid: list[float], seeds: list[int], n: int) -> pd.DataFrame:
    """Run all SNR experiment cells."""
    rows: list[dict] = []
    total = len(snr_grid) * len(seeds) * len(ALGORITHMS_CORE)
    done = 0
    for snr in snr_grid:
        for seed in seeds:
            for algorithm in ALGORITHMS_CORE:
                done += 1
                print(
                    f"[snr] {done}/{total}: snr={snr:g} seed={seed} {algorithm} "
                    "[unconstrained+constrained]",
                    flush=True,
                )
                rows.extend(
                    run_experiment_pair(
                        algorithm=algorithm,
                        seed=seed,
                        n=n,
                        snr=snr,
                        experiment="snr_sensitivity",
                    )
                )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame) -> list[Path]:
    """Create all SNR sensitivity figures."""
    outputs = [
        (FIG_DIR / "snr_f1_sweep.png", "f1", "F1", "Causal Dummy v2: F1 vs SNR"),
        (FIG_DIR / "snr_shd_sweep.png", "shd", "SHD", "Causal Dummy v2: SHD vs SNR"),
        (FIG_DIR / "snr_precision_sweep.png", "precision", "Precision", "Causal Dummy v2: Precision vs SNR"),
        (FIG_DIR / "snr_recall_sweep.png", "recall", "Recall", "Causal Dummy v2: Recall vs SNR"),
        (FIG_DIR / "snr_runtime_sweep.png", "runtime_seconds", "Runtime seconds", "Causal Dummy v2: Runtime vs SNR"),
    ]
    for path, metric, ylabel, title in outputs:
        plot_line_metric(
            summary=summary,
            x_col="snr",
            metric=metric,
            ylabel=ylabel,
            title=title,
            path=path,
            algorithms=ALGORITHMS_CORE,
        )
    return [path for path, _metric, _ylabel, _title in outputs]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SNR sensitivity on causal dummy v2.")
    parser.add_argument("--snr-grid", default=",".join(str(x) for x in DEFAULT_SNR_GRID))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()

    ensure_dirs()
    snr_grid = parse_float_list(args.snr_grid)
    seeds = parse_int_list(args.seeds)
    log_command("22_snr_sensitivity_sweep.py", vars(args))

    results = run_sweep(snr_grid=snr_grid, seeds=seeds, n=args.n)
    summary = summarize_results(results, ["algorithm", "constraint_mode", "snr"])
    write_csv(RESULTS_PATH, results, RESULT_COLUMNS)
    write_csv(SUMMARY_PATH, summary)
    figure_paths = make_figures(summary)

    headline_cols = [
        "algorithm",
        "constraint_mode",
        "snr",
        "f1_mean",
        "f1_std",
        "shd_mean",
        "shd_std",
        "precision_mean",
        "recall_mean",
        "runtime_seconds_mean",
        "successful_runs",
        "failed_runs",
    ]
    write_experiment_report(
        path=REPORT_PATH,
        title="SNR Sensitivity Experiment",
        command=script_command(),
        design_lines=[
            "Dataset: causal dummy v2 generated from a fixed structural DAG.",
            f"Sample size per run: n={args.n}.",
            f"SNR grid: {', '.join(str(x) for x in snr_grid)}.",
            f"Seeds: {', '.join(str(x) for x in seeds)}.",
            "Algorithms: PC, LiNGAM, GES.",
            "Constraint modes: unconstrained and constrained.",
            "Metrics are computed against the generated ground-truth DAG.",
        ],
        outputs=[RESULTS_PATH, SUMMARY_PATH, REPORT_PATH, *figure_paths],
        results=results,
        summary=summary,
        headline_cols=[column for column in headline_cols if column in summary.columns],
    )

    print(f"[snr] Results -> {RESULTS_PATH}")
    print(f"[snr] Summary -> {SUMMARY_PATH}")
    print(f"[snr] Report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
