# 14_check_synthetic_ranges.py
# ============================================================
# Reality check for synthetic variables that previously had
# unrealistic extreme values after exponentiation.
#
# Usage:
#   python 14_check_synthetic_ranges.py
#   python 14_check_synthetic_ranges.py --input data/synthetic/synthetic_n2000.csv
# ============================================================

from __future__ import annotations

import argparse

import pandas as pd


RANGE_CHECKS = [
    ("tobins_q", 8.0),
    ("debt_to_equity_ratio", 25.0),
    ("corruption_cases", 50.0),
]


def check_ranges(input_path: str) -> None:
    """
    Print max-value checks for selected synthetic variables.

    Parameters
    ----------
    input_path : str
        Path to the synthetic dataset CSV to inspect.
    """
    df = pd.read_csv(input_path)

    print("Reality check on previously-broken variables:")
    for column, threshold in RANGE_CHECKS:
        actual = df[column].max()
        flag = "OK" if actual < threshold else "STILL BROKEN"
        print(f"  {column}: max = {actual:.2f}, threshold = {threshold:g} -> {flag}")


def main() -> None:
    """
    Parse CLI arguments and run the synthetic range check.
    """
    parser = argparse.ArgumentParser(
        description="Check max values for selected synthetic ESG variables."
    )
    parser.add_argument("--input", default="data/synthetic/synthetic_n2000.csv")
    args = parser.parse_args()

    check_ranges(args.input)


if __name__ == "__main__":
    main()
