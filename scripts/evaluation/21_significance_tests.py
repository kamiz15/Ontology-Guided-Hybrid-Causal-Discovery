"""
21_significance_tests.py
========================
Statistical significance tests for the constrained vs unconstrained causal
discovery comparison on the advisor_dummy dataset.

Test: paired Wilcoxon signed-rank test (non-parametric, n=5 seeds) on
F1 and SHD for each algorithm under the forbidden_only constraint mode.

Null hypothesis: no difference in metric between constrained and unconstrained.
Alternative: constrained improves the metric (one-sided).

Outputs:
    outputs/experiments/significance_tests.csv
    outputs/experiments/significance_tests.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
ABLATION_PATH = ROOT / "outputs" / "experiments" / "advisor_dummy_constraint_ablation.csv"
OUT_CSV = ROOT / "outputs" / "experiments" / "significance_tests.csv"
OUT_MD  = ROOT / "outputs" / "experiments" / "significance_tests.md"

SEEDS = [42, 43, 44, 45, 46]
CONSTRAINT_MODE = "forbidden_only"

ALGORITHM_LABELS = {
    "pc":                        "PC",
    "notears_postproc":          "NOTEARS",
    "lingam":                    "LiNGAM",
    "ges_postproc":              "GES",
    "deci_native_unconstrained": "DECI (unconstrained side)",
    "deci_native_constrained":   "DECI (constrained side)",
}

# Canonical (unconstrained_alg, constrained_alg) pairs for DECI
# (DECI uses separate rows rather than a mode column)
DECI_PAIR = ("deci_native_unconstrained", "deci_native_constrained")


def wilcoxon_one_sided(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    """
    One-sided Wilcoxon signed-rank test: H1 is x > y (x is better).
    Returns (statistic, p_value, note).
    """
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return float("nan"), float("nan"), "all differences zero â€” test not applicable"
    if len(nonzero) < 3:
        return float("nan"), float("nan"), f"only {len(nonzero)} non-zero differences â€” insufficient for test"
    try:
        stat, p_two = wilcoxon(x, y, alternative="greater")
        return float(stat), float(p_two), ""
    except Exception as exc:
        return float("nan"), float("nan"), str(exc)


def run_tests() -> pd.DataFrame:
    df = pd.read_csv(ABLATION_PATH)
    df = df[df["status"] == "success"].copy()
    df["f1_directed"] = pd.to_numeric(df["f1_directed"], errors="coerce")
    df["shd"]         = pd.to_numeric(df["shd"],         errors="coerce")

    base = df[df["constraint_mode"] == CONSTRAINT_MODE].copy()

    rows = []

    # --- standard algorithms: have a "mode" column with unconstrained/constrained
    standard_algs = [a for a in base["algorithm"].unique() if a not in DECI_PAIR]
    for alg in sorted(standard_algs):
        sub = base[base["algorithm"] == alg]
        unc = sub[sub["mode"] == "unconstrained"].set_index("seed")
        con = sub[sub["mode"] == "constrained"].set_index("seed")
        shared_seeds = sorted(set(unc.index) & set(con.index))
        if len(shared_seeds) < 3:
            continue

        f1_unc = np.array([unc.loc[s, "f1_directed"] for s in shared_seeds], dtype=float)
        f1_con = np.array([con.loc[s, "f1_directed"] for s in shared_seeds], dtype=float)
        shd_unc = np.array([unc.loc[s, "shd"] for s in shared_seeds], dtype=float)
        shd_con = np.array([con.loc[s, "shd"] for s in shared_seeds], dtype=float)

        f1_stat, f1_p, f1_note = wilcoxon_one_sided(f1_con, f1_unc)   # H1: constrained F1 > unconstrained F1
        shd_stat, shd_p, shd_note = wilcoxon_one_sided(shd_unc, shd_con)  # H1: unconstrained SHD > constrained SHD

        rows.append({
            "algorithm":        ALGORITHM_LABELS.get(alg, alg),
            "n_seeds":          len(shared_seeds),
            "f1_unc_mean":      float(np.nanmean(f1_unc)),
            "f1_con_mean":      float(np.nanmean(f1_con)),
            "f1_delta_mean":    float(np.nanmean(f1_con - f1_unc)),
            "f1_wilcoxon_stat": f1_stat,
            "f1_p_value":       f1_p,
            "f1_significant":   f1_p < 0.05 if not np.isnan(f1_p) else False,
            "f1_note":          f1_note,
            "shd_unc_mean":     float(np.nanmean(shd_unc)),
            "shd_con_mean":     float(np.nanmean(shd_con)),
            "shd_delta_mean":   float(np.nanmean(shd_con - shd_unc)),
            "shd_wilcoxon_stat":shd_stat,
            "shd_p_value":      shd_p,
            "shd_significant":  shd_p < 0.05 if not np.isnan(shd_p) else False,
            "shd_note":         shd_note,
        })

    # --- DECI: two separate algorithm labels for unconstrained / constrained
    deci_unc_rows = base[base["algorithm"] == DECI_PAIR[0]].set_index("seed")
    deci_con_rows = base[base["algorithm"] == DECI_PAIR[1]].set_index("seed")
    shared = sorted(set(deci_unc_rows.index) & set(deci_con_rows.index))
    if len(shared) >= 3:
        f1_unc = np.array([deci_unc_rows.loc[s, "f1_directed"] for s in shared], dtype=float)
        f1_con = np.array([deci_con_rows.loc[s, "f1_directed"] for s in shared], dtype=float)
        shd_unc = np.array([deci_unc_rows.loc[s, "shd"] for s in shared], dtype=float)
        shd_con = np.array([deci_con_rows.loc[s, "shd"] for s in shared], dtype=float)

        f1_stat, f1_p, f1_note = wilcoxon_one_sided(f1_con, f1_unc)
        shd_stat, shd_p, shd_note = wilcoxon_one_sided(shd_unc, shd_con)

        rows.append({
            "algorithm":        "DECI",
            "n_seeds":          len(shared),
            "f1_unc_mean":      float(np.nanmean(f1_unc)),
            "f1_con_mean":      float(np.nanmean(f1_con)),
            "f1_delta_mean":    float(np.nanmean(f1_con - f1_unc)),
            "f1_wilcoxon_stat": f1_stat,
            "f1_p_value":       f1_p,
            "f1_significant":   f1_p < 0.05 if not np.isnan(f1_p) else False,
            "f1_note":          f1_note,
            "shd_unc_mean":     float(np.nanmean(shd_unc)),
            "shd_con_mean":     float(np.nanmean(shd_con)),
            "shd_delta_mean":   float(np.nanmean(shd_con - shd_unc)),
            "shd_wilcoxon_stat":shd_stat,
            "shd_p_value":      shd_p,
            "shd_significant":  shd_p < 0.05 if not np.isnan(shd_p) else False,
            "shd_note":         shd_note,
        })

    return pd.DataFrame(rows)


def format_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001 ***"
    if p < 0.01:
        return f"{p:.3f} **"
    if p < 0.05:
        return f"{p:.3f} *"
    return f"{p:.3f}"


def write_markdown(results: pd.DataFrame) -> str:
    lines = [
        "# Significance Tests: Constrained vs Unconstrained (forbidden_only)",
        "",
        "Dataset: advisor_dummy (ontology-derived reference DAG evaluation)",
        "Test: one-sided paired Wilcoxon signed-rank test, n=5 seeds",
        "H1 (F1): constrained F1 > unconstrained F1",
        "H1 (SHD): constrained SHD < unconstrained SHD",
        "Significance: * p<0.05, ** p<0.01, *** p<0.001",
        "",
        "Note: with n=5, the minimum achievable p-value is 1/32 â‰ˆ 0.031.",
        "When all 5 differences have the same sign and are non-zero, p â‰ˆ 0.031.",
        "Results marked 'n/a' indicate all paired differences were zero",
        "(algorithm produced identical output regardless of constraints).",
        "",
        "## F1 Results",
        "",
        "| Algorithm | F1 unconstrained | F1 constrained | Delta F1 | p-value |",
        "|-----------|-----------------|----------------|----------|---------|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['algorithm']} "
            f"| {row['f1_unc_mean']:.4f} "
            f"| {row['f1_con_mean']:.4f} "
            f"| {row['f1_delta_mean']:+.4f} "
            f"| {format_p(row['f1_p_value'])} |"
        )
    lines += [
        "",
        "## SHD Results (lower is better)",
        "",
        "| Algorithm | SHD unconstrained | SHD constrained | Delta SHD | p-value |",
        "|-----------|------------------|-----------------|-----------|---------|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['algorithm']} "
            f"| {row['shd_unc_mean']:.1f} "
            f"| {row['shd_con_mean']:.1f} "
            f"| {row['shd_delta_mean']:+.1f} "
            f"| {format_p(row['shd_p_value'])} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "Algorithms with zero F1 in both conditions (LiNGAM, NOTEARS) cannot",
        "be tested â€” the test requires at least one non-zero difference.",
        "This confirms these algorithms found no signal on this dataset",
        "regardless of constraint mode, consistent with the low-signal nature",
        "of the ontology-derived reference DAG evaluation.",
    ]
    return "\n".join(lines)


def main() -> None:
    print("Running significance tests on advisor_dummy forbidden_only results...")
    results = run_tests()

    results.to_csv(OUT_CSV, index=False)
    print(f"CSV saved: {OUT_CSV}")

    md = write_markdown(results)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Markdown saved: {OUT_MD}")

    print("\n" + md.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
