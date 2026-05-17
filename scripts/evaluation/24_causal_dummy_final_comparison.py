"""Run the final causal-dummy v2 algorithm comparison."""

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
    ALGORITHMS_FINAL,
    CONSTRAINT_MODES,
    DEFAULT_N,
    DEFAULT_SEEDS,
    DEFAULT_SNR,
    EXP_DIR,
    FIG_DIR,
    RESULT_COLUMNS,
    ensure_dirs,
    log_command,
    parse_int_list,
    plot_grouped_metric,
    run_experiment_pair,
    script_command,
    summarize_results,
    write_csv,
    write_experiment_report,
)


RAW_PATH = EXP_DIR / "causal_dummy_final_comparison_raw.csv"
SUMMARY_PATH = EXP_DIR / "causal_dummy_final_comparison_summary.csv"
REPORT_PATH = EXP_DIR / "causal_dummy_final_comparison_report.md"
DECI_EXPLORATORY_PATH = EXP_DIR / "deci_exploratory_results.csv"


def run_comparison(seeds: list[int], n: int, snr: float, include_deci: bool) -> pd.DataFrame:
    """Run all final-comparison cells."""
    algorithms = list(ALGORITHMS_FINAL)
    if include_deci:
        algorithms.append("DECI")
    rows: list[dict] = []
    total = len(seeds) * len(algorithms)
    done = 0
    for seed in seeds:
        for algorithm in algorithms:
            done += 1
            print(
                f"[causal-final] {done}/{total}: seed={seed} {algorithm} "
                "[unconstrained+constrained]",
                flush=True,
            )
            if algorithm == "DECI":
                for constraint_mode in CONSTRAINT_MODES:
                    row = {
                        column: pd.NA for column in RESULT_COLUMNS
                    }
                    row.update({
                        "algorithm": "DECI",
                        "dataset": "causal_dummy_v2",
                        "constraint_mode": constraint_mode,
                        "seed": seed,
                        "n": n,
                        "snr": snr,
                        "status": "failed",
                        "runtime_seconds": 0.0,
                        "error_type": "SkippedExploratory",
                        "error_message": "DECI is exploratory and was not run by this lightweight final comparison driver.",
                    })
                    rows.append(row)
                continue
            rows.extend(
                run_experiment_pair(
                    algorithm=algorithm,
                    seed=seed,
                    n=n,
                    snr=snr,
                    experiment="causal_dummy_final",
                )
            )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame) -> list[Path]:
    """Create final comparison figures."""
    outputs = [
        (FIG_DIR / "causal_dummy_final_f1.png", "f1", "F1", "Causal Dummy v2: Final F1"),
        (FIG_DIR / "causal_dummy_final_shd.png", "shd", "SHD", "Causal Dummy v2: Final SHD"),
        (FIG_DIR / "causal_dummy_final_violations.png", "violations", "Forbidden-edge violations", "Causal Dummy v2: Violations"),
        (FIG_DIR / "causal_dummy_final_runtime.png", "runtime_seconds", "Runtime seconds", "Causal Dummy v2: Runtime"),
    ]
    for path, metric, ylabel, title in outputs:
        plot_grouped_metric(
            summary=summary,
            metric=metric,
            ylabel=ylabel,
            title=title,
            path=path,
            algorithms=ALGORITHMS_FINAL,
        )
    return [path for path, _metric, _ylabel, _title in outputs]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Final causal dummy v2 algorithm comparison.")
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR)
    parser.add_argument("--include-deci", action="store_true", help="Write exploratory skipped DECI rows.")
    args = parser.parse_args()

    ensure_dirs()
    seeds = parse_int_list(args.seeds)
    log_command("24_causal_dummy_final_comparison.py", vars(args))

    results = run_comparison(seeds=seeds, n=args.n, snr=args.snr, include_deci=args.include_deci)
    main_results = results[~results["algorithm"].astype(str).eq("DECI")].copy()
    deci_results = results[results["algorithm"].astype(str).eq("DECI")].copy()
    summary = summarize_results(main_results, ["algorithm", "constraint_mode"])

    write_csv(RAW_PATH, results, RESULT_COLUMNS)
    write_csv(SUMMARY_PATH, summary)
    if not deci_results.empty:
        write_csv(DECI_EXPLORATORY_PATH, deci_results, RESULT_COLUMNS)
    figure_paths = make_figures(summary)

    headline_cols = [
        "algorithm",
        "constraint_mode",
        "f1_mean",
        "f1_std",
        "shd_mean",
        "shd_std",
        "precision_mean",
        "recall_mean",
        "violations_mean",
        "runtime_seconds_mean",
        "successful_runs",
        "failed_runs",
    ]
    write_experiment_report(
        path=REPORT_PATH,
        title="Causal Dummy v2 Final Algorithm Comparison",
        command=script_command(),
        design_lines=[
            "Dataset: causal dummy v2.",
            f"Sample size: n={args.n}.",
            f"SNR: {args.snr:g}.",
            f"Seeds: {', '.join(str(x) for x in seeds)}.",
            "Algorithms: PC, LiNGAM, NOTEARS, GES.",
            "Constraint modes: unconstrained and constrained.",
            "Metrics are computed against the generated ground-truth DAG.",
            "DECI is excluded from the main table unless explicitly run and complete.",
        ],
        outputs=[RAW_PATH, SUMMARY_PATH, REPORT_PATH, *figure_paths],
        results=results,
        summary=summary,
        headline_cols=[column for column in headline_cols if column in summary.columns],
    )

    print(f"[causal-final] Raw -> {RAW_PATH}")
    print(f"[causal-final] Summary -> {SUMMARY_PATH}")
    print(f"[causal-final] Report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
