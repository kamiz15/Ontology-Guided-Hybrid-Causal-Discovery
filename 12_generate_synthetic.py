# 12_generate_synthetic.py
# ============================================================
# Step 12 - Generate synthetic ESG data from a known causal DAG.
#
# The synthetic data is sampled from a structural causal model (SCM)
# so downstream discovery algorithms have a real ground truth to recover.
#
# Usage:
#   python 12_generate_synthetic.py
#   python 12_generate_synthetic.py --n-samples 110,500,2000 --seed 42
#
# Output:
#   data/synthetic/synthetic_n{N}.csv
#   data/synthetic/synthetic_n{N}_metadata.json
#   data/synthetic/ground_truth_adjacency.csv
#   data/synthetic/ground_truth_edges.csv
#   reports/synthetic_generation_summary.md
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from config import (
    SYNTHETIC_DIR,
    SYNTHETIC_EDGES_PATH,
    SYNTHETIC_GENERATION_SUMMARY,
    SYNTHETIC_GROUND_TRUTH_PATH,
)


VARIABLES = [
    # Environmental
    "emission_reduction_policy_score",
    "renewable_energy_share",
    "scope_1_emissions_tco2e",
    "scope_2_emissions_tco2e",
    "scope_3_emissions_tco2e",
    "total_energy_consumption",
    "environmental_fines",
    "iso_14001_exists",

    # Social
    "training_hours",
    "injury_frequency_rate",
    "turnover_rate",
    "diversity_representation",
    "community_investment_eur",
    "customer_satisfaction_score",

    # Governance
    "board_strategy_esg_oversight_score",
    "board_diversity",
    "ceo_chair_split",
    "auditor_independence_score",
    "corruption_cases",

    # Financial
    "total_asset",
    "total_revenue_eur",
    "roa_eat",
    "debt_to_equity_ratio",
    "tobins_q",
    "green_financing_eur",
    "pe_ratio",
    "roe_eat",
    "asset_growth_pct",
]

PILLARS = {
    "Environmental": [
        "emission_reduction_policy_score",
        "renewable_energy_share",
        "scope_1_emissions_tco2e",
        "scope_2_emissions_tco2e",
        "scope_3_emissions_tco2e",
        "total_energy_consumption",
        "environmental_fines",
        "iso_14001_exists",
    ],
    "Social": [
        "training_hours",
        "injury_frequency_rate",
        "turnover_rate",
        "diversity_representation",
        "community_investment_eur",
        "customer_satisfaction_score",
    ],
    "Governance": [
        "board_strategy_esg_oversight_score",
        "board_diversity",
        "ceo_chair_split",
        "auditor_independence_score",
        "corruption_cases",
    ],
    "Financial": [
        "total_asset",
        "total_revenue_eur",
        "roa_eat",
        "debt_to_equity_ratio",
        "tobins_q",
        "green_financing_eur",
        "pe_ratio",
        "roe_eat",
        "asset_growth_pct",
    ],
}

VARIABLE_DTYPES = {
    "emission_reduction_policy_score": "ordinal",
    "renewable_energy_share": "continuous_0_1",
    "scope_1_emissions_tco2e": "log_normal",
    "scope_2_emissions_tco2e": "log_normal",
    "scope_3_emissions_tco2e": "log_normal",
    "total_energy_consumption": "log_normal",
    "environmental_fines": "log_normal",
    "iso_14001_exists": "binary",
    "training_hours": "log_normal",
    "injury_frequency_rate": "continuous",
    "turnover_rate": "continuous",
    "diversity_representation": "continuous_0_1",
    "community_investment_eur": "log_normal",
    "customer_satisfaction_score": "continuous",
    "board_strategy_esg_oversight_score": "ordinal",
    "board_diversity": "continuous_0_1",
    "ceo_chair_split": "binary",
    "auditor_independence_score": "continuous_0_100",
    "corruption_cases": "count",
    "total_asset": "log_normal_root",
    "total_revenue_eur": "log_normal",
    "roa_eat": "continuous",
    "debt_to_equity_ratio": "positive",
    "tobins_q": "positive",
    "green_financing_eur": "log_normal",
    "pe_ratio": "positive",
    "roe_eat": "continuous",
    "asset_growth_pct": "continuous",
}

GROUND_TRUTH_EDGES = [
    # === Firm size cascades (total_asset is exogenous root) ===
    ("total_asset", "total_revenue_eur"),                       # +
    ("total_asset", "scope_1_emissions_tco2e"),                 # +
    ("total_asset", "scope_2_emissions_tco2e"),                 # +
    ("total_asset", "total_energy_consumption"),                # +
    ("total_asset", "training_hours"),                          # +
    ("total_asset", "community_investment_eur"),                # +
    ("total_asset", "debt_to_equity_ratio"),                    # +

    # === Governance drives ESG actions ===
    ("ceo_chair_split", "board_strategy_esg_oversight_score"),  # +
    ("board_diversity", "board_strategy_esg_oversight_score"),  # +
    ("board_strategy_esg_oversight_score", "emission_reduction_policy_score"),  # +
    ("board_strategy_esg_oversight_score", "iso_14001_exists"),                 # +
    ("board_strategy_esg_oversight_score", "auditor_independence_score"),       # +
    ("board_strategy_esg_oversight_score", "green_financing_eur"),              # +
    ("board_strategy_esg_oversight_score", "corruption_cases"),                 # -

    # === Environmental policy reduces emissions, increases renewables ===
    ("emission_reduction_policy_score", "renewable_energy_share"),     # +
    ("emission_reduction_policy_score", "scope_1_emissions_tco2e"),    # -
    ("emission_reduction_policy_score", "scope_2_emissions_tco2e"),    # -
    ("emission_reduction_policy_score", "environmental_fines"),        # -
    ("renewable_energy_share", "scope_2_emissions_tco2e"),             # -
    ("iso_14001_exists", "environmental_fines"),                       # -

    # === Energy drives scope 1/2 emissions (physical relationship) ===
    ("total_energy_consumption", "scope_1_emissions_tco2e"),    # +
    ("total_energy_consumption", "scope_2_emissions_tco2e"),    # +

    # === Social ===
    ("training_hours", "injury_frequency_rate"),                # -
    ("training_hours", "turnover_rate"),                        # -
    ("diversity_representation", "customer_satisfaction_score"),# +
    ("turnover_rate", "customer_satisfaction_score"),           # -

    # === Governance integrity ===
    ("auditor_independence_score", "corruption_cases"),         # -

    # === Financial cascades ===
    ("total_revenue_eur", "roa_eat"),                           # +
    ("roa_eat", "tobins_q"),                                    # +
    ("debt_to_equity_ratio", "tobins_q"),                       # -

    # === ESG -> financial (the central thesis question) ===
    ("emission_reduction_policy_score", "tobins_q"),            # +
    ("environmental_fines", "roa_eat"),                         # -
    ("corruption_cases", "tobins_q"),                           # -
    ("renewable_energy_share", "green_financing_eur"),          # +

    # === Income / valuation cascade ===
    ("roa_eat", "roe_eat"),                                      # +
    ("debt_to_equity_ratio", "roe_eat"),                         # +
    ("tobins_q", "pe_ratio"),                                    # +
    ("roa_eat", "pe_ratio"),                                     # +
    ("roa_eat", "asset_growth_pct"),                             # +
    ("total_revenue_eur", "asset_growth_pct"),                   # +

    # === ESG -> new financial outcomes (the central thesis question) ===
    ("emission_reduction_policy_score", "pe_ratio"),             # +
    ("board_strategy_esg_oversight_score", "asset_growth_pct"),  # +
    ("environmental_fines", "asset_growth_pct"),                 # -
]

EDGE_SIGNS = {
    ("total_asset", "total_revenue_eur"): "+",
    ("total_asset", "scope_1_emissions_tco2e"): "+",
    ("total_asset", "scope_2_emissions_tco2e"): "+",
    ("total_asset", "total_energy_consumption"): "+",
    ("total_asset", "training_hours"): "+",
    ("total_asset", "community_investment_eur"): "+",
    ("total_asset", "debt_to_equity_ratio"): "+",
    ("ceo_chair_split", "board_strategy_esg_oversight_score"): "+",
    ("board_diversity", "board_strategy_esg_oversight_score"): "+",
    ("board_strategy_esg_oversight_score", "emission_reduction_policy_score"): "+",
    ("board_strategy_esg_oversight_score", "iso_14001_exists"): "+",
    ("board_strategy_esg_oversight_score", "auditor_independence_score"): "+",
    ("board_strategy_esg_oversight_score", "green_financing_eur"): "+",
    ("board_strategy_esg_oversight_score", "corruption_cases"): "-",
    ("emission_reduction_policy_score", "renewable_energy_share"): "+",
    ("emission_reduction_policy_score", "scope_1_emissions_tco2e"): "-",
    ("emission_reduction_policy_score", "scope_2_emissions_tco2e"): "-",
    ("emission_reduction_policy_score", "environmental_fines"): "-",
    ("renewable_energy_share", "scope_2_emissions_tco2e"): "-",
    ("iso_14001_exists", "environmental_fines"): "-",
    ("total_energy_consumption", "scope_1_emissions_tco2e"): "+",
    ("total_energy_consumption", "scope_2_emissions_tco2e"): "+",
    ("training_hours", "injury_frequency_rate"): "-",
    ("training_hours", "turnover_rate"): "-",
    ("diversity_representation", "customer_satisfaction_score"): "+",
    ("turnover_rate", "customer_satisfaction_score"): "-",
    ("auditor_independence_score", "corruption_cases"): "-",
    ("total_revenue_eur", "roa_eat"): "+",
    ("roa_eat", "tobins_q"): "+",
    ("debt_to_equity_ratio", "tobins_q"): "-",
    ("emission_reduction_policy_score", "tobins_q"): "+",
    ("environmental_fines", "roa_eat"): "-",
    ("corruption_cases", "tobins_q"): "-",
    ("renewable_energy_share", "green_financing_eur"): "+",
    ("roa_eat", "roe_eat"): "+",
    ("debt_to_equity_ratio", "roe_eat"): "+",
    ("tobins_q", "pe_ratio"): "+",
    ("roa_eat", "pe_ratio"): "+",
    ("roa_eat", "asset_growth_pct"): "+",
    ("total_revenue_eur", "asset_growth_pct"): "+",
    ("emission_reduction_policy_score", "pe_ratio"): "+",
    ("board_strategy_esg_oversight_score", "asset_growth_pct"): "+",
    ("environmental_fines", "asset_growth_pct"): "-",
}

LOG_NORMAL_VARS = {
    "scope_1_emissions_tco2e",
    "scope_2_emissions_tco2e",
    "scope_3_emissions_tco2e",
    "total_energy_consumption",
    "environmental_fines",
    "training_hours",
    "community_investment_eur",
    "total_revenue_eur",
    "green_financing_eur",
}

ORDINAL_VARS = {
    "emission_reduction_policy_score",
    "board_strategy_esg_oversight_score",
}

CLIPPED_01_VARS = {
    "renewable_energy_share",
    "diversity_representation",
    "board_diversity",
}

BINARY_VARS = {
    "iso_14001_exists",
    "ceo_chair_split",
}

STRICTLY_POSITIVE_VARS = {
    "debt_to_equity_ratio",
    "tobins_q",
    "pe_ratio",
}

BASE_INTERCEPTS = {
    "scope_1_emissions_tco2e": 10.5,
    "scope_2_emissions_tco2e": 10.3,
    "scope_3_emissions_tco2e": 12.5,
    "total_energy_consumption": 13.5,
    "environmental_fines": 2.0,
    "training_hours": 3.4,
    "community_investment_eur": 14.5,
    "total_revenue_eur": 20.0,
    "green_financing_eur": 17.5,
    "debt_to_equity_ratio": 0.0,
    "tobins_q": 0.2,
    "pe_ratio": 2.4,
    "corruption_cases": -1.0,
}

FIGURES_DIR = "reports/figures"


def validate_dag() -> nx.DiGraph:
    """
    Validate that the ground-truth edge list forms an acyclic graph.

    Returns
    -------
    nx.DiGraph
        Directed acyclic graph containing all synthetic variables.

    Raises
    ------
    ValueError
        If the edge list contains unknown variables or a directed cycle.
    """
    unknown = sorted({node for edge in GROUND_TRUTH_EDGES for node in edge} - set(VARIABLES))
    if unknown:
        raise ValueError(f"Ground-truth edges reference unknown variables: {unknown}")

    missing_signs = sorted(set(GROUND_TRUTH_EDGES) - set(EDGE_SIGNS))
    extra_signs = sorted(set(EDGE_SIGNS) - set(GROUND_TRUTH_EDGES))
    if missing_signs or extra_signs:
        raise ValueError(
            "EDGE_SIGNS must match GROUND_TRUTH_EDGES exactly: "
            f"missing={missing_signs}, extra={extra_signs}"
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(VARIABLES)
    graph.add_edges_from(GROUND_TRUTH_EDGES)

    try:
        cycle = nx.find_cycle(graph, orientation="original")
    except nx.NetworkXNoCycle:
        return graph

    cycle_nodes = [edge[0] for edge in cycle]
    if cycle:
        cycle_nodes.append(cycle[0][0])
    print(f"[step_12] ERROR: Ground-truth DAG contains a cycle: {cycle_nodes}")
    raise ValueError(f"Ground-truth DAG contains a cycle: {cycle_nodes}")


def sample_edge_coefficients(
    rng: np.random.Generator,
) -> dict[tuple[str, str], float]:
    """
    Sample one signed coefficient for every ground-truth edge.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator seeded once by the caller.

    Returns
    -------
    dict[tuple[str, str], float]
        Mapping from (parent, child) edge to sampled coefficient.
    """
    coefficients: dict[tuple[str, str], float] = {}
    for edge in GROUND_TRUTH_EDGES:
        sign = EDGE_SIGNS[edge]
        magnitude = rng.uniform(0.5, 1.5)
        coefficients[edge] = magnitude if sign == "+" else -magnitude
    return coefficients


def generate_dataset(
    n_samples: int,
    seed: int,
    noise_scale: float,
) -> pd.DataFrame:
    """
    Generate one synthetic ESG dataset from the configured SCM.

    Parameters
    ----------
    n_samples : int
        Number of rows to generate.
    seed : int
        Random seed used for coefficient sampling and data sampling.
    noise_scale : float
        Standard deviation multiplier for structural noise.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with the configured variables.
    """
    graph = validate_dag()
    rng = np.random.default_rng(seed)
    coefficients = sample_edge_coefficients(rng)
    return _generate_dataset_with_coefficients(
        n_samples=n_samples,
        rng=rng,
        noise_scale=noise_scale,
        coefficients=coefficients,
        graph=graph,
    )


def parse_n_samples(value: str) -> list[int]:
    """
    Parse the comma-separated --n-samples CLI value.

    Parameters
    ----------
    value : str
        Comma-separated positive integer sample sizes.

    Returns
    -------
    list[int]
        Parsed sample sizes.

    Raises
    ------
    ValueError
        If any sample size is invalid.
    """
    sizes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError(f"Sample sizes must be positive: {size}")
        sizes.append(size)
    if not sizes:
        raise ValueError("At least one sample size is required.")
    return sizes


def run_generation(
    n_samples_list: list[int],
    seed: int,
    noise_scale: float,
    output_dir: str,
) -> None:
    """
    Generate all requested synthetic datasets and ground-truth outputs.

    Parameters
    ----------
    n_samples_list : list[int]
        Sample sizes to generate.
    seed : int
        Random seed for coefficients and structural noise.
    noise_scale : float
        Standard deviation multiplier for structural noise.
    output_dir : str
        Directory where synthetic CSV files are written.
    """
    graph = validate_dag()
    rng = np.random.default_rng(seed)
    coefficients = sample_edge_coefficients(rng)

    _ensure_dir(output_dir)
    _ensure_dir(os.path.dirname(SYNTHETIC_GROUND_TRUTH_PATH))
    _ensure_dir(os.path.dirname(SYNTHETIC_GENERATION_SUMMARY))
    _ensure_dir(FIGURES_DIR)

    adjacency = _build_ground_truth_adjacency(graph)
    _backup_existing(SYNTHETIC_GROUND_TRUTH_PATH)
    adjacency.to_csv(SYNTHETIC_GROUND_TRUTH_PATH)
    edges_df = _build_edges_dataframe(coefficients)
    _backup_existing(SYNTHETIC_EDGES_PATH)
    edges_df.to_csv(SYNTHETIC_EDGES_PATH, index=False)
    print(f"[step_12] Ground-truth adjacency -> {SYNTHETIC_GROUND_TRUTH_PATH}")
    print(f"[step_12] Ground-truth edges -> {SYNTHETIC_EDGES_PATH}")

    generated: list[dict[str, Any]] = []
    for n_samples in n_samples_list:
        df = _generate_dataset_with_coefficients(
            n_samples=n_samples,
            rng=rng,
            noise_scale=noise_scale,
            coefficients=coefficients,
            graph=graph,
        )
        csv_path = os.path.join(output_dir, f"synthetic_n{n_samples}.csv")
        metadata_path = os.path.join(output_dir, f"synthetic_n{n_samples}_metadata.json")
        figure_path = _write_histograms(df, n_samples, FIGURES_DIR)

        _backup_existing(csv_path)
        df.to_csv(csv_path, index=False)
        metadata = _build_metadata(df, n_samples, seed, noise_scale)
        _backup_existing(metadata_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        stats = _summary_stats(df)
        generated.append({
            "n_samples": n_samples,
            "csv_path": csv_path,
            "metadata_path": metadata_path,
            "figure_path": figure_path,
            "stats": stats,
        })
        print(f"[step_12] Synthetic data N={n_samples} -> {csv_path}")
        print(f"[step_12] Metadata N={n_samples} -> {metadata_path}")
        print(f"[step_12] Histograms N={n_samples} -> {figure_path}")

    _write_generation_summary(
        generated=generated,
        coefficients=coefficients,
        output_path=SYNTHETIC_GENERATION_SUMMARY,
        seed=seed,
        noise_scale=noise_scale,
    )
    print(f"[step_12] Summary -> {SYNTHETIC_GENERATION_SUMMARY}")


def main() -> None:
    """
    Parse CLI arguments and run synthetic data generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic ESG data from a known causal DAG."
    )
    parser.add_argument("--n-samples", default="110,500,2000",
                        help="Comma-separated sample sizes to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for coefficients and data")
    parser.add_argument("--output-dir", default=SYNTHETIC_DIR,
                        help="Directory for synthetic datasets")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="Structural noise scale")
    args = parser.parse_args()

    n_samples_list = parse_n_samples(args.n_samples)
    print(f"[step_12] Sample sizes: {n_samples_list}")
    print(f"[step_12] Seed: {args.seed}")
    print(f"[step_12] Noise scale: {args.noise_scale}")

    run_generation(
        n_samples_list=n_samples_list,
        seed=args.seed,
        noise_scale=args.noise_scale,
        output_dir=args.output_dir,
    )


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _backup_existing(path: str) -> None:
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak")


def _parents_by_child(graph: nx.DiGraph) -> dict[str, list[str]]:
    return {node: list(graph.predecessors(node)) for node in graph.nodes}


def _build_ground_truth_adjacency(graph: nx.DiGraph) -> pd.DataFrame:
    adjacency = pd.DataFrame(0, index=VARIABLES, columns=VARIABLES, dtype=int)
    for parent, child in graph.edges:
        adjacency.loc[parent, child] = 1
    return adjacency


def _build_edges_dataframe(
    coefficients: dict[tuple[str, str], float],
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "parent": parent,
            "child": child,
            "expected_sign": EDGE_SIGNS[(parent, child)],
            "coefficient": coefficients[(parent, child)],
        }
        for parent, child in GROUND_TRUTH_EDGES
    ])


def _generate_dataset_with_coefficients(
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float,
    coefficients: dict[tuple[str, str], float],
    graph: nx.DiGraph,
) -> pd.DataFrame:
    parents = _parents_by_child(graph)
    data: dict[str, np.ndarray] = {}

    for variable in nx.topological_sort(graph):
        data[variable] = _sample_variable(
            variable=variable,
            parents=parents[variable],
            data=data,
            coefficients=coefficients,
            n_samples=n_samples,
            rng=rng,
            noise_scale=noise_scale,
        )

    df = pd.DataFrame({name: data[name] for name in VARIABLES})
    return _cast_output_dtypes(df)


def _sample_variable(
    variable: str,
    parents: list[str],
    data: dict[str, np.ndarray],
    coefficients: dict[tuple[str, str], float],
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    if not parents:
        return _sample_root(variable, n_samples, rng, noise_scale)

    linear = _linear_combo(variable, parents, data, coefficients, n_samples, rng, noise_scale)

    if variable in ORDINAL_VARS:
        return _to_ordinal_quintiles(linear)
    if variable in BINARY_VARS:
        probabilities = _sigmoid(linear)
        return rng.binomial(1, probabilities, size=n_samples)
    if variable == "corruption_cases":
        lam = np.exp(np.clip(linear, -3.0, 3.0))
        return rng.poisson(lam)
    if variable in CLIPPED_01_VARS:
        return np.clip(_sigmoid(linear), 0.0, 1.0)
    if variable == "auditor_independence_score":
        return np.clip(50.0 + 15.0 * linear, 0.0, 100.0)
    if variable in LOG_NORMAL_VARS:
        return np.exp(np.clip(linear, -20.0, 25.0))
    if variable == "pe_ratio":
        return np.exp(np.clip(linear, 0.0, 4.0))
    if variable == "tobins_q":
        return np.exp(np.clip(linear, -2.0, 2.0))
    if variable == "debt_to_equity_ratio":
        return np.exp(np.clip(linear, -1.0, 3.0))
    if variable == "injury_frequency_rate":
        return np.clip(3.0 + linear, 0.0, None)
    if variable == "turnover_rate":
        return np.clip(0.18 + 0.06 * linear, 0.0, 1.0)
    if variable == "customer_satisfaction_score":
        return np.clip(70.0 + 10.0 * linear, 0.0, 100.0)
    if variable == "roa_eat":
        return np.clip(0.04 + 0.025 * linear, -0.25, 0.25)
    if variable == "roe_eat":
        return np.clip(0.02 + 0.07 * linear, -0.30, 0.40)
    if variable == "asset_growth_pct":
        return np.clip(0.04 + 0.08 * linear, -0.30, 0.50)

    return linear


def _sample_root(
    variable: str,
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    if variable == "total_asset":
        return rng.lognormal(mean=20.0, sigma=1.5, size=n_samples)
    if variable == "scope_3_emissions_tco2e":
        return rng.lognormal(mean=12.5, sigma=1.2, size=n_samples)
    if variable in CLIPPED_01_VARS:
        base = -0.4 if variable == "board_diversity" else 0.0
        raw = base + rng.normal(0.0, noise_scale, size=n_samples)
        return np.clip(_sigmoid(raw), 0.0, 1.0)
    if variable in BINARY_VARS:
        raw = 0.3 + rng.normal(0.0, noise_scale, size=n_samples)
        probabilities = _sigmoid(raw)
        return rng.binomial(1, probabilities, size=n_samples)
    if variable in LOG_NORMAL_VARS:
        base = BASE_INTERCEPTS.get(variable, 1.0)
        return np.exp(rng.normal(base, noise_scale, size=n_samples))
    if variable in ORDINAL_VARS:
        raw = rng.normal(0.0, noise_scale, size=n_samples)
        return _to_ordinal_quintiles(raw)

    return rng.normal(0.0, noise_scale, size=n_samples)


def _linear_combo(
    variable: str,
    parents: list[str],
    data: dict[str, np.ndarray],
    coefficients: dict[tuple[str, str], float],
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    linear = np.full(n_samples, BASE_INTERCEPTS.get(variable, 0.0), dtype=float)
    for parent in parents:
        coefficient = coefficients[(parent, variable)]
        linear += coefficient * _parent_signal(parent, data[parent])
    linear += rng.normal(0.0, noise_scale, size=n_samples)
    return linear


def _parent_signal(variable: str, values: np.ndarray) -> np.ndarray:
    x = values.astype(float)
    if variable == "total_asset":
        return (np.log1p(x) - 20.0) / 1.5
    if variable in LOG_NORMAL_VARS or variable in STRICTLY_POSITIVE_VARS:
        return _zscore(np.log1p(np.maximum(x, 0.0)))
    if variable in ORDINAL_VARS:
        return (x - 3.0) / 1.414
    if variable in CLIPPED_01_VARS:
        return 4.0 * (x - 0.5)
    if variable == "auditor_independence_score":
        return (x - 50.0) / 20.0
    if variable in BINARY_VARS:
        return x - 0.5
    if variable == "corruption_cases":
        return _zscore(np.log1p(x))
    return _zscore(x)


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _to_ordinal_quintiles(values: np.ndarray) -> np.ndarray:
    cutpoints = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return (np.digitize(values, cutpoints) + 1).astype(int)


def _cast_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ORDINAL_VARS | BINARY_VARS | {"corruption_cases"}:
        out[column] = out[column].astype(int)
    return out


def _summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.agg(["mean", "std", "min", "median", "max"]).T
    stats = stats.rename_axis("variable").reset_index()
    return stats


def _build_metadata(
    df: pd.DataFrame,
    n_samples: int,
    seed: int,
    noise_scale: float,
) -> dict[str, Any]:
    stats = _summary_stats(df)
    return {
        "n_samples": n_samples,
        "seed": seed,
        "noise_scale": noise_scale,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "variables": VARIABLES,
        "edge_count": len(GROUND_TRUTH_EDGES),
        "summary_stats": {
            row["variable"]: {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "min": float(row["min"]),
                "median": float(row["median"]),
                "max": float(row["max"]),
            }
            for _, row in stats.iterrows()
        },
    }


def _write_histograms(
    df: pd.DataFrame,
    n_samples: int,
    figures_dir: str,
) -> str:
    _ensure_dir(figures_dir)
    n_cols = 5
    n_rows = int(np.ceil(len(VARIABLES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    axes_flat = axes.flatten()

    for ax, variable in zip(axes_flat, VARIABLES):
        series = df[variable]
        if variable in LOG_NORMAL_VARS or variable == "total_asset":
            plot_values = np.log1p(series)
            title = f"log1p({variable})"
        else:
            plot_values = series
            title = variable
        ax.hist(plot_values, bins=30, color="#4c78a8", edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    for ax in axes_flat[len(VARIABLES):]:
        ax.axis("off")

    fig.suptitle(f"Synthetic ESG distributions (N={n_samples})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(figures_dir, f"synthetic_n{n_samples}_histograms.png")
    _backup_existing(path)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _write_generation_summary(
    generated: list[dict[str, Any]],
    coefficients: dict[tuple[str, str], float],
    output_path: str,
    seed: int,
    noise_scale: float,
) -> None:
    lines = [
        "# Synthetic Data Generation Summary",
        "",
        "## DAG Description",
        "",
        "The synthetic dataset is generated from a fixed structural causal model",
        "with known directed edges. Edge coefficients are sampled once per run",
        "and reused across all requested sample sizes, so the data-generating",
        "process is unchanged when only N changes.",
        "",
        f"- Variables: {len(VARIABLES)}",
        f"- Edges: {len(GROUND_TRUTH_EDGES)}",
        f"- Seed: {seed}",
        f"- Noise scale: {noise_scale}",
        "",
        "## Realism Clips",
        "",
        "The SCM clips selected log-scale intermediates before exponentiation",
        "to prevent unrealistic synthetic outliers while preserving the causal",
        "signal carried by their parent variables.",
        "",
        "- `tobins_q`: log value clipped to [-2.0, 2.0], giving a loose range of roughly [0.13, 7.39]",
        "- `debt_to_equity_ratio`: log value clipped to [-1.0, 3.0], giving a loose range of roughly [0.37, 20.09]",
        "- `corruption_cases`: Poisson log-rate clipped to [-3.0, 3.0], giving lambda in roughly [0.05, 20.09]",
        "",
        "Real-world banking ranges informed these clip choices, but the clips",
        "are deliberately loose to preserve recoverable synthetic signal.",
        "",
        "## Variables",
        "",
    ]

    for pillar, variables in PILLARS.items():
        lines.append(f"### {pillar}")
        lines.append("")
        lines.extend(f"- `{variable}` ({VARIABLE_DTYPES[variable]})" for variable in variables)
        lines.append("")

    lines.extend([
        "## Ground-Truth Edges",
        "",
        _markdown_table_from_df(_build_edges_dataframe(coefficients).round({"coefficient": 4})),
        "",
        "## Sample Statistics",
        "",
    ])

    for item in generated:
        n_samples = item["n_samples"]
        stats = item["stats"].round(3)
        lines.extend([
            f"### N = {n_samples}",
            "",
            f"- Data: `{item['csv_path']}`",
            f"- Metadata: `{item['metadata_path']}`",
            f"- Histogram figure: `{item['figure_path']}`",
            "",
            _markdown_table_from_df(stats),
            "",
        ])

    _ensure_dir(os.path.dirname(output_path))
    _backup_existing(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _markdown_table_from_df(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]

    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[step_12] ERROR: synthetic data generation failed.")
        traceback.print_exc()
        sys.exit(1)
