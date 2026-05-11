# 13_check_synthetic_correlations.py
# ============================================================
# Quick diagnostic for synthetic-data edge-direction sanity checks.
#
# This script is intentionally lightweight: it checks marginal Pearson
# correlations for a small set of expected positive, negative, and non-edge
# pairs. Marginal correlations are not a formal DAG validation test because
# common causes and indirect paths can make non-adjacent variables correlate.
#
# Usage:
#   python 13_check_synthetic_correlations.py
#   python 13_check_synthetic_correlations.py --n-samples 110,500,2000
# ============================================================

from __future__ import annotations

import argparse
import os

import pandas as pd


EDGES_TO_CHECK = [
    ("total_asset", "total_revenue_eur", "+"),
    ("total_energy_consumption", "scope_1_emissions_tco2e", "+"),
    ("roa_eat", "tobins_q", "+"),
    ("emission_reduction_policy_score", "renewable_energy_share", "+"),
    ("board_strategy_esg_oversight_score", "corruption_cases", "-"),
    ("training_hours", "turnover_rate", "-"),
    ("emission_reduction_policy_score", "environmental_fines", "-"),
    ("iso_14001_exists", "diversity_representation", "0"),
    ("renewable_energy_share", "corruption_cases", "0"),
]


def parse_n_samples(value: str) -> list[int]:
    """
    Parse a comma-separated sample-size argument.

    Parameters
    ----------
    value : str
        Comma-separated sample sizes, for example ``"110,500,2000"``.

    Returns
    -------
    list[int]
        Parsed sample sizes.
    """
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def correlation_flag(
    correlation: float,
    expected: str,
    weak_threshold: float,
    leak_threshold: float,
) -> str:
    """
    Assign a simple diagnostic flag to a marginal correlation.

    Parameters
    ----------
    correlation : float
        Pearson correlation between the pair.
    expected : str
        Expected sign: ``"+"``, ``"-"``, or ``"0"``.
    weak_threshold : float
        Minimum absolute sign-consistent correlation for expected edges.
    leak_threshold : float
        Maximum absolute marginal correlation for expected zero pairs.

    Returns
    -------
    str
        Empty string, ``" <-- WEAK"``, or ``" <-- LEAKING"``.
    """
    if expected == "+" and correlation < weak_threshold:
        return " <-- WEAK"
    if expected == "-" and correlation > -weak_threshold:
        return " <-- WEAK"
    if expected == "0" and abs(correlation) > leak_threshold:
        return " <-- LEAKING"
    return ""


def run_checks(
    n_samples: list[int],
    data_dir: str,
    weak_threshold: float,
    leak_threshold: float,
) -> None:
    """
    Run marginal-correlation diagnostics for generated synthetic datasets.

    Parameters
    ----------
    n_samples : list[int]
        Synthetic sample sizes to inspect.
    data_dir : str
        Directory containing ``synthetic_n{N}.csv`` files.
    weak_threshold : float
        Minimum absolute sign-consistent correlation for expected edges.
    leak_threshold : float
        Maximum absolute marginal correlation for expected zero pairs.
    """
    for n in n_samples:
        path = os.path.join(data_dir, f"synthetic_n{n}.csv")
        df = pd.read_csv(path)

        print(f"=== N={n} ===")
        for parent, child, expected in EDGES_TO_CHECK:
            correlation = df[[parent, child]].corr().iloc[0, 1]
            flag = correlation_flag(correlation, expected, weak_threshold, leak_threshold)
            print(
                f"  {parent[:30]:30s} -> {child[:30]:30s}  "
                f"expected {expected}  got {correlation:+.3f}{flag}"
            )
        print()

    print("Note:")
    print(
        "  A LEAKING flag here means marginal correlation, not necessarily a direct DAG error."
    )
    print(
        "  For example, renewable_energy_share and corruption_cases share an upstream"
    )
    print(
        "  governance/policy path, so they can be correlated even without a direct edge."
    )


def main() -> None:
    """
    Parse CLI arguments and run the synthetic correlation diagnostic.
    """
    parser = argparse.ArgumentParser(
        description="Check simple marginal correlations in synthetic ESG datasets."
    )
    parser.add_argument("--n-samples", default="110,500,2000")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--weak-threshold", type=float, default=0.15)
    parser.add_argument("--leak-threshold", type=float, default=0.20)
    args = parser.parse_args()

    run_checks(
        n_samples=parse_n_samples(args.n_samples),
        data_dir=args.data_dir,
        weak_threshold=args.weak_threshold,
        leak_threshold=args.leak_threshold,
    )


if __name__ == "__main__":
    main()
