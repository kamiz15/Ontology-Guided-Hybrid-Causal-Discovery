"""Run the causal-dummy v2 sample-size sensitivity experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_dummy_experiment_utils import (
    ALGORITHMS_CORE,
    CONSTRAINT_MODES,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_SEEDS,
    DEFAULT_SNR,
    EXP_DIR,
    FIG_DIR,
    RESULT_COLUMNS,
    ensure_dirs,
    log_command,
    parse_int_list,
    plot_line_metric,
    run_experiment_pair,
    script_command,
    summarize_results,
    write_csv,
    write_experiment_report,
)


RESULTS_PATH = EXP_DIR / "sample_size_sensitivity_results.csv"
SUMMARY_PATH = EXP_DIR / "sample_size_sensitivity_summary.csv"
REPORT_PATH = EXP_DIR / "sample_size_sensitivity_report.md"


def run_sweep(sample_sizes: list[int], seeds: list[int], snr: float) -> pd.DataFrame:
    """Run all sample-size experiment cells."""
    rows: list[dict] = []
    total = len(sample_sizes) * len(seeds) * len(ALGORITHMS_CORE)
    done = 0
    for n in sample_sizes:
        for seed in seeds:
            for algorithm in ALGORITHMS_CORE:
                done += 1
                print(
                    f"[sample-size] {done}/{total}: n={n} seed={seed} {algorithm} "
                    "[unconstrained+constrained]",
                    flush=True,
                )
                rows.extend(
                    run_experiment_pair(
                        algorithm=algorithm,
                        seed=seed,
                        n=n,
                        snr=snr,
                        experiment="sample_size_sensitivity",
                    )
                )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame) -> list[Path]:
    """Create all sample-size figures."""
    outputs = [
        (FIG_DIR / "sample_size_f1.png", "f1", "F1", "Causal Dummy v2: F1 vs Sample Size"),
        (FIG_DIR / "sample_size_shd.png", "shd", "SHD", "Causal Dummy v2: SHD vs Sample Size"),
        (FIG_DIR / "sample_size_precision.png", "precision", "Precision", "Causal Dummy v2: Precision vs Sample Size"),
        (FIG_DIR / "sample_size_recall.png", "recall", "Recall", "Causal Dummy v2: Recall vs Sample Size"),
        (FIG_DIR / "sample_size_runtime.png", "runtime_seconds", "Runtime seconds", "Causal Dummy v2: Runtime vs Sample Size"),
    ]
    for path, metric, ylabel, title in outputs:
        plot_line_metric(
            summary=summary,
            x_col="n",
            metric=metric,
            ylabel=ylabel,
            title=title,
            path=path,
            algorithms=ALGORITHMS_CORE,
        )
    return [path for path, _metric, _ylabel, _title in outputs]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sample-size sensitivity on causal dummy v2.")
    parser.add_argument("--sample-sizes", default=",".join(str(x) for x in DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR)
    args = parser.parse_args()

    ensure_dirs()
    sample_sizes = parse_int_list(args.sample_sizes)
    seeds = parse_int_list(args.seeds)
    log_command("23_sample_size_sensitivity.py", vars(args))

    results = run_sweep(sample_sizes=sample_sizes, seeds=seeds, snr=args.snr)
    summary = summarize_results(results, ["algorithm", "constraint_mode", "n"])
    write_csv(RESULTS_PATH, results, RESULT_COLUMNS)
    write_csv(SUMMARY_PATH, summary)
    figure_paths = make_figures(summary)

    headline_cols = [
        "algorithm",
        "constraint_mode",
        "n",
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
        title="Sample-Size Sensitivity Experiment",
        command=script_command(),
        design_lines=[
            "Dataset: causal dummy v2 generated from the same fixed DAG at every sample size.",
            f"Sample sizes: {', '.join(str(x) for x in sample_sizes)}.",
            f"Seeds: {', '.join(str(x) for x in seeds)}.",
            f"SNR: {args.snr:g}.",
            "Algorithms: PC, LiNGAM, GES.",
            "Constraint modes: unconstrained and constrained.",
            "The design tests whether constraints help more when sample size is limited.",
        ],
        outputs=[RESULTS_PATH, SUMMARY_PATH, REPORT_PATH, *figure_paths],
        results=results,
        summary=summary,
        headline_cols=[column for column in headline_cols if column in summary.columns],
    )

    print(f"[sample-size] Results -> {RESULTS_PATH}")
    print(f"[sample-size] Summary -> {SUMMARY_PATH}")
    print(f"[sample-size] Report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
