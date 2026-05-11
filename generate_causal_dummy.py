"""
generate_causal_dummy.py
========================
Generates ESG-Finance_dummy_data_causal_v2.csv — a version of the advisor
dummy dataset where 46 variables are generated from structural equations that
match the ontology-derived reference DAG used in the thesis evaluation.

Compared to the original Dummy_dataset_ESG.txt:
  - Root variables (no parents in the reference DAG) are still drawn randomly.
  - Variables that have parents in the reference DAG are generated as linear
    combinations of their parents plus Gaussian noise, then clipped to their
    valid range.
  - Signal-to-noise is moderate (SNR ≈ 0.5–0.7 per parent) so algorithms
    have real structure to recover without it being trivially obvious.
  - The same data-quality problems are injected afterward: 5–20% missing
    values, extreme outliers, wrong data types, and invalid categories.
  - Variables not in the reference DAG (the remaining advisor columns) are
    still generated randomly.

The ground-truth DAG adjacency matrix is saved alongside the CSV so that
run_all.py --dataset causal_dummy can load it directly instead of using an
ontology-derived reference.

Usage:
    python generate_causal_dummy.py
    python generate_causal_dummy.py --n 3000 --seed 42 --snr 0.6 --out data/advisor_dummy
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_DEFAULT = 3000
SEED_DEFAULT = 42
SNR_DEFAULT = 0.6   # coefficient weight per parent (in normalised [0,1] space)
NOISE_STD = 0.25    # additive noise std in normalised space (before rescaling)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "advisor_dummy"

# ---------------------------------------------------------------------------
# Variable specs — dtype, range, and sign of each parent coefficient
# ---------------------------------------------------------------------------

# Each entry:  variable_name -> (dtype, lo, hi)
# dtype: "float" | "bool" | "int"
# lo/hi: valid range for clipping (ignored for bool)
VAR_SPECS: dict[str, tuple[str, float, float]] = {
    # Environmental
    "co2_ch4_n2o_scope_1_3":         ("float",  0,  100),
    "carbon_intensity":               ("float",  0,  100),
    "emission_reduction_policy":      ("bool",   0,    1),
    "carbon_neutral_commitment":      ("bool",   0,    1),
    "air_emissions_sox_nox_pm":       ("float",  0,  100),
    "land_area_affected":             ("float",  0,  100),
    "biodiversity_protection_actions":("bool",   0,    1),
    "fsc_pefc_certified_sourcing":    ("float",  0,  100),
    "climate_risk_assessment_done":   ("bool",   0,    1),
    "resilience_score":               ("float",  0,  100),
    "environmental_fines":            ("float",  0,  100),
    "iso_14001_exists":               ("bool",   0,    1),
    "reporting_quality_score":        ("float",  0,  100),
    "green_product_revenue":          ("float",  0,  100),
    "green_buildings_area":           ("float",  0,  100),
    "resource_efficiency_index":      ("float",  0,  100),
    "total_energy_consumption":       ("float",  0,  100),
    "renewable_energy_share":         ("float",  0,  100),
    "hazardous_waste_generated":      ("float",  0,  100),
    "recyclable_packaging_share":     ("float",  0,  100),
    "toxic_spills":                   ("int",    0,   50),
    "waste_recycled_share":           ("float",  0,  100),
    "water_withdrawal":               ("float",  0,  100),
    "emf_exposure":                   ("float",  0,  100),
    "gmo_products":                   ("bool",   0,    1),
    "ods_emissions":                  ("float",  0,  100),
    # Social
    "safety_transparency_trials":     ("bool",   0,    1),
    "collective_bargaining_coverage": ("float",  0,  100),
    "fair_wage_gap":                  ("float",  0,  100),
    "community_investment":           ("float",  0,  100),
    "health_impact_score":            ("float",  0,  100),
    "csr_contribution":               ("float",  0,  100),
    "customer_satisfaction_score":    ("float",  0,  100),
    "hiv_program_coverage":           ("float",  0,  100),
    "human_rights_violations":        ("int",    0,   50),
    "diversity_representation":       ("float",  0,  100),
    "child_labor_compliance":         ("bool",   0,    1),
    "indigenous_consent_verification":("bool",   0,    1),
    "access_to_services":             ("float",  0,  100),
    "healthcare_access_employees":    ("float",  0,  100),
    "animal_welfare_compliance":      ("bool",   0,    1),
    "training_hours":                 ("float",  0,  100),
    "turnover_rate":                  ("float",  0,  100),
    "injury_frequency_rate":          ("float",  0,  100),
    "contract_compliance":            ("bool",   0,    1),
    "union_membership":               ("float",  0,  100),
    "product_safety_compliance":      ("float",  0,  100),
    "responsible_marketing_compliance":("bool",  0,    1),
    "supplier_audits":                ("bool",   0,    1),
    # Governance
    "esg_oversight_policy":           ("bool",   0,    1),
    "board_diversity":                ("float",  0,  100),
    "governance_compliance_score":    ("float",  0,  100),
    "esg_incentive_bonus":            ("bool",   0,    1),
    "assurance_score":                ("float",  0,  100),
    "green_financing":                ("float",  0,  100),
    "ceo_chair_split":                ("bool",   0,    1),
    "auditor_independence_score":     ("float",  0,  100),
    "ethical_breaches":               ("int",    0,   50),
    "anti_competitive_violations":    ("int",    0,   50),
    "corruption_cases":               ("int",    0,   50),
    "privacy_compliance":             ("bool",   0,    1),
    "financial_inclusion":            ("float",  0,  100),
    "global_compact_membership":      ("bool",   0,    1),
    "shareholder_rights_score":       ("float",  0,  100),
    "site_closure_plan":              ("bool",   0,    1),
    "tax_transparency_reporting":     ("bool",   0,    1),
    "lobby_spending":                 ("float",  0,  100),
    "systemic_risk_level":            ("str",    0,    0),
    # Financial variables present in the advisor CSV but not in the ESG spec
    # These remain randomly generated (not in the reference DAG)
    "net_profit_margin":              ("float", -50,  50),
    "solvency_ratio":                 ("float",  0,   10),
    "debt_to_equity_ratio":           ("float",  0,   10),
    "market_value_equity":            ("float",  0, 1e6),
    "roa_eat":                        ("float", -20,  40),
    "roe_eat":                        ("float", -20,  40),
}

# ---------------------------------------------------------------------------
# Structural equations (reference DAG edges)
# Each entry: target -> list of (source, coefficient_sign)
#   +1  = positive relationship (parent increase → child increase)
#   -1  = negative relationship (parent increase → child decrease)
# All arithmetic is done in normalised [0,1] space.
# ---------------------------------------------------------------------------

STRUCTURAL_EDGES: dict[str, list[tuple[str, int]]] = {
    # --- emission / energy block ---
    "emission_reduction_policy": [
        ("esg_oversight_policy",   +1),
        ("esg_incentive_bonus",    +1),
        ("carbon_neutral_commitment", +1),
    ],
    "total_energy_consumption": [
        ("emission_reduction_policy", -1),   # policy reduces energy use
    ],
    "renewable_energy_share": [
        ("emission_reduction_policy",   +1),
        ("carbon_neutral_commitment",   +1),
    ],
    "co2_ch4_n2o_scope_1_3": [
        ("emission_reduction_policy",   -1),
        ("total_energy_consumption",    +1),
    ],
    "carbon_intensity": [
        ("emission_reduction_policy",   -1),
        ("renewable_energy_share",      -1),
        ("total_energy_consumption",    +1),
    ],

    # --- environmental outcomes ---
    "resilience_score": [
        ("climate_risk_assessment_done", +1),
    ],
    "hazardous_waste_generated": [
        ("waste_recycled_share",        -1),
    ],
    "environmental_fines": [
        ("co2_ch4_n2o_scope_1_3",       +1),
        ("carbon_intensity",            +1),
        ("hazardous_waste_generated",   +1),
        ("water_withdrawal",            +1),
        ("iso_14001_exists",            -1),   # certification reduces fines
    ],

    # --- reporting / assurance ---
    "reporting_quality_score": [
        ("esg_oversight_policy",  +1),
        ("board_diversity",       +1),
        ("iso_14001_exists",      +1),
    ],
    "assurance_score": [
        ("esg_oversight_policy",       +1),
        ("auditor_independence_score", +1),
    ],

    # --- governance compliance ---
    "ethical_breaches": [
        ("corruption_cases",       +1),
    ],
    "governance_compliance_score": [
        ("esg_oversight_policy",        +1),
        ("board_diversity",             +1),
        ("ceo_chair_split",             +1),
        ("auditor_independence_score",  +1),
        ("ethical_breaches",            -1),
        ("corruption_cases",            -1),
        ("anti_competitive_violations", -1),
    ],

    # --- social outcomes ---
    "injury_frequency_rate": [
        ("training_hours",             -1),
    ],
    "turnover_rate": [
        ("training_hours",             -1),
        ("customer_satisfaction_score", -1),
    ],
    "fair_wage_gap": [
        ("collective_bargaining_coverage", -1),
    ],
    "human_rights_violations": [
        ("diversity_representation",   -1),
    ],

    # --- ESG → financial ---
    "net_profit_margin": [
        ("customer_satisfaction_score", +1),
        ("injury_frequency_rate",       -1),
        ("iso_14001_exists",            +1),
    ],
    "solvency_ratio": [
        ("governance_compliance_score", +1),
    ],
    "debt_to_equity_ratio": [
        ("governance_compliance_score", -1),
        ("environmental_fines",         +1),
    ],
    "market_value_equity": [
        ("environmental_fines",         -1),
        ("carbon_intensity",            -1),
    ],
    "roa_eat": [
        ("emission_reduction_policy",   +1),
    ],
    "roe_eat": [
        ("emission_reduction_policy",   +1),
    ],
}

# Topological order (level 0 = roots, processed first)
TOPO_ORDER = [
    # roots — generated randomly, no parents in the DAG
    "esg_oversight_policy",
    "board_diversity",
    "ceo_chair_split",
    "auditor_independence_score",
    "esg_incentive_bonus",
    "carbon_neutral_commitment",
    "climate_risk_assessment_done",
    "iso_14001_exists",
    "waste_recycled_share",
    "water_withdrawal",
    "training_hours",
    "collective_bargaining_coverage",
    "diversity_representation",
    "customer_satisfaction_score",
    "anti_competitive_violations",
    "corruption_cases",
    # level 1
    "emission_reduction_policy",
    "ethical_breaches",
    "reporting_quality_score",
    "assurance_score",
    "resilience_score",
    "hazardous_waste_generated",
    "injury_frequency_rate",
    "turnover_rate",
    "fair_wage_gap",
    "human_rights_violations",
    # level 2
    "total_energy_consumption",
    "renewable_energy_share",
    "governance_compliance_score",
    "net_profit_margin",
    "roa_eat",
    "roe_eat",
    # level 3
    "co2_ch4_n2o_scope_1_3",
    "carbon_intensity",
    "solvency_ratio",
    # level 4
    "environmental_fines",
    # level 5
    "debt_to_equity_ratio",
    "market_value_equity",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalise a continuous array to [0, 1] given its range."""
    span = hi - lo
    if span == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / span


def _rescale(arr_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a [0,1] array back to [lo, hi]."""
    return arr_norm * (hi - lo) + lo


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_root(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    dtype, lo, hi = VAR_SPECS[name]
    if dtype == "float":
        return np.round(rng.uniform(lo, hi, n), 2)
    if dtype == "bool":
        return rng.choice([True, False], n)
    if dtype == "int":
        return rng.integers(int(lo), int(hi) + 1, n)
    if dtype == "str":
        return rng.choice(["Low", "Medium", "High"], n)
    raise ValueError(f"Unknown dtype {dtype} for {name}")


def _generate_child(
    name: str,
    data: dict[str, np.ndarray],
    n: int,
    snr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a child variable from its parents using a linear SCM."""
    dtype, lo, hi = VAR_SPECS[name]
    parents = STRUCTURAL_EDGES[name]

    # Compute the structural signal in normalised space
    signal = np.zeros(n, dtype=float)
    for parent_name, sign in parents:
        p_dtype, p_lo, p_hi = VAR_SPECS[parent_name]
        raw = data[parent_name].astype(float)
        if p_dtype == "bool":
            normed = raw           # already 0/1
        elif p_dtype == "int":
            normed = _norm(raw, p_lo, p_hi)
        else:
            normed = _norm(raw, p_lo, p_hi)
        signal += sign * snr * normed

    noise = rng.normal(0, NOISE_STD, n)
    latent = signal + noise  # in roughly [-k*snr - 3*noise_std, k*snr + 3*noise_std]

    if dtype == "bool":
        # Centre signal around 0.5 probability for roots, use sigmoid
        prob = _sigmoid(latent * 4)   # scale so moderate signal gives clear split
        return rng.uniform(0, 1, n) < prob

    if dtype == "int":
        # Map latent to a mean count, sample from Poisson
        base = (lo + hi) / 2.0
        span = hi - lo
        mean_count = np.clip(base + latent * span * 0.3, lo, hi)
        return np.round(mean_count + rng.normal(0, span * 0.05, n)).clip(lo, hi).astype(int)

    if dtype == "float":
        # Normalised latent → clip to [0,1] → rescale to [lo, hi]
        mid = 0.5
        normed = np.clip(mid + latent * 0.5, 0, 1)
        return np.round(_rescale(normed, lo, hi), 2)

    raise ValueError(f"Unknown dtype {dtype} for {name}")


# ---------------------------------------------------------------------------
# Ground-truth adjacency matrix
# ---------------------------------------------------------------------------

def _build_adjacency(variables: list[str]) -> pd.DataFrame:
    """Return a DataFrame adjacency matrix (row=source, col=target, 1=edge)."""
    adj = pd.DataFrame(0, index=variables, columns=variables)
    for target, parents in STRUCTURAL_EDGES.items():
        for parent_name, _ in parents:
            if parent_name in adj.index and target in adj.columns:
                adj.loc[parent_name, target] = 1
    return adj


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(n: int, seed: int, snr: float,
             skip_quality_issues: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    data: dict[str, np.ndarray] = {}

    # 1. Generate all variables in topological order
    for name in TOPO_ORDER:
        if name not in STRUCTURAL_EDGES:
            data[name] = _generate_root(name, n, rng)
        else:
            data[name] = _generate_child(name, data, n, snr, rng)

    # 2. Generate all remaining variables randomly
    for name, (dtype, lo, hi) in VAR_SPECS.items():
        if name in data:
            continue
        if dtype == "float":
            data[name] = np.round(rng.uniform(lo, hi, n), 2)
        elif dtype == "bool":
            data[name] = rng.choice([True, False], n)
        elif dtype == "int":
            data[name] = rng.integers(int(lo), int(hi) + 1, n)
        elif dtype == "str":
            data[name] = rng.choice(["Low", "Medium", "High"], n)

    df = pd.DataFrame(data)

    # 3. Inject data quality problems (same as original advisor script)
    if skip_quality_issues:
        adj = _build_adjacency(list(TOPO_ORDER))
        return df, adj

    num_cols = list(df.select_dtypes(include=[float, int, np.number]).columns)
    df = df.astype(object)

    # 5–20% missing values per column
    for col in df.columns:
        missing_count = random.randint(int(0.05 * n), int(0.20 * n))
        idx = random.sample(range(n), missing_count)
        df.loc[idx, col] = None

    # Extreme outliers on first 10 numeric columns (×10–×100)
    for col in list(num_cols)[:10]:
        col_max = pd.to_numeric(df[col], errors="coerce").dropna().max()
        if pd.notna(col_max):
            df.loc[random.sample(range(n), 5), col] = col_max * random.choice([10, 50, 100])

    # Wrong data types (string "error" in numeric fields)
    for col in list(num_cols)[:8]:
        df.loc[random.sample(range(n), 5), col] = "error"

    # Invalid category
    if "systemic_risk_level" in df.columns:
        df.loc[random.sample(range(n), 6), "systemic_risk_level"] = "Very High"

    # 4. Build ground-truth adjacency matrix
    structural_vars = list(TOPO_ORDER)
    adj = _build_adjacency(structural_vars)

    return df, adj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate causal ESG dummy data.")
    parser.add_argument("--n",    type=int,   default=N_DEFAULT, help="Number of rows")
    parser.add_argument("--seed", type=int,   default=SEED_DEFAULT)
    parser.add_argument("--snr",  type=float, default=SNR_DEFAULT,
                        help="Signal weight per parent (normalised space, default 0.6)")
    parser.add_argument("--out",  type=str,   default=str(OUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    try:
        from causal_dummy_experiment_utils import log_command

        log_command("generate_causal_dummy.py", vars(args))
    except Exception:
        pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating causal dummy data: n={args.n}, seed={args.seed}, snr={args.snr}")
    df, adj = generate(args.n, args.seed, args.snr)

    csv_path = out_dir / "ESG-Finance_dummy_data_causal_v2.csv"
    adj_path = out_dir / "causal_dummy_ground_truth_dag.csv"
    edges_path = out_dir / "causal_dummy_ground_truth_edges.csv"

    df.to_csv(csv_path, index=False)
    adj.to_csv(adj_path)

    # Also write a flat edge list (easier to load in run_all.py)
    edge_rows = []
    for target, parents in STRUCTURAL_EDGES.items():
        for parent_name, sign in parents:
            edge_rows.append({"source": parent_name, "target": target,
                               "coefficient_sign": sign})
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)

    print(f"Data CSV   : {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")
    print(f"Adj matrix : {adj_path}  ({adj.shape[0]} × {adj.shape[1]})")
    print(f"Edge list  : {edges_path}  ({len(edge_rows)} edges)")

    # Quick sanity check: report correlation between each parent and child
    print("\nSanity check — Pearson r between parent and child (numeric only):")
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    for target, parents in STRUCTURAL_EDGES.items():
        if target not in df_numeric.columns:
            continue
        y = df_numeric[target].dropna()
        for pname, sign in parents:
            if pname not in df_numeric.columns:
                continue
            common = df_numeric[[pname, target]].dropna()
            if len(common) < 30:
                continue
            r = common[pname].corr(common[target])
            expected_sign = "+" if sign == 1 else "-"
            ok = "OK" if (sign == 1 and r > 0) or (sign == -1 and r < 0) else "WRONG SIGN"
            print(f"  {pname} -> {target}: r={r:.3f} (expected {expected_sign})  [{ok}]")


if __name__ == "__main__":
    main()
