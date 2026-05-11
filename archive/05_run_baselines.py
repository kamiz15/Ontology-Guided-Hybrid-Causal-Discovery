# 05_run_baselines.py
# ============================================================
# Week 2 — Baseline + Hybrid Pipeline
# Runs three causal discovery algorithms on the cleaned ESG
# dataset and saves outputs for comparison:
#
#   1. Unconstrained PC   (causal-learn)
#   2. Unconstrained LiNGAM (lingam)
#   3. Constrained PC     (causal-learn + ontology forbidden edges)
#
# Each run is logged with algorithm name, hyperparameters,
# timestamp, and edge count for reproducibility.
#
# Outputs (per algorithm):
#   outputs/graphs/<algo>_adjacency.csv
#   outputs/graphs/<algo>_graph.gml
#   outputs/metrics/run_log.csv        (appended)
#
# Usage:
#   python 05_run_baselines.py
#   python 05_run_baselines.py --input data/processed/data_ready.csv
#   python 05_run_baselines.py --alpha 0.05
#
# Requirements:
#   pip install causal-learn lingam networkx pandas numpy
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────
from config import READY_DATA_PATH

# Import forbidden/required edges and builder from Step 4
from forbidden_edges import (
    FORBIDDEN_EDGES,
    REQUIRED_EDGES,
    build_background_knowledge,
)


# ================================================================
#  UTILITIES
# ================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the causal-ready dataset. Expects all-numeric columns."""
    df = pd.read_csv(path)
    print(f"[data] Loaded {path}: {df.shape[0]} rows × {df.shape[1]} columns")

    # Drop any remaining non-numeric columns (e.g. systemic_risk_level)
    numeric_df = df.select_dtypes(include="number")
    dropped = set(df.columns) - set(numeric_df.columns)
    if dropped:
        print(f"[data] Dropped non-numeric columns: {sorted(dropped)}")

    # Drop rows with any NaN (causal-learn requires complete data)
    before = len(numeric_df)
    numeric_df = numeric_df.dropna()
    after = len(numeric_df)
    if before != after:
        print(f"[data] Dropped {before - after} rows with NaN -> {after} rows remain")

    print(f"[data] Final shape: {numeric_df.shape}")
    return numeric_df


def adjacency_to_dataframe(adj_matrix: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Convert a numpy adjacency matrix to a labelled DataFrame."""
    return pd.DataFrame(adj_matrix, index=columns, columns=columns)


def adjacency_to_gml(adj_df: pd.DataFrame, path: str):
    """Save an adjacency DataFrame as a directed .gml graph file."""
    G = nx.DiGraph()
    cols = list(adj_df.columns)
    G.add_nodes_from(cols)
    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            weight = adj_df.iloc[i, j]
            if weight != 0:
                G.add_edge(src, tgt, weight=float(weight))
    nx.write_gml(G, path)
    return G


def count_edges(adj_matrix: np.ndarray) -> int:
    """Count directed edges (non-zero entries)."""
    return int(np.count_nonzero(adj_matrix))


def log_run(log_path: str, entry: dict):
    """Append a run entry to the CSV log."""
    df_new = pd.DataFrame([entry])
    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(log_path, index=False)


def save_outputs(name: str, adj_matrix: np.ndarray, columns: list[str],
                 hyperparams: dict, elapsed: float, log_path: str):
    """Save adjacency CSV, .gml, and append to run log."""
    adj_df = adjacency_to_dataframe(adj_matrix, columns)

    csv_path = f"outputs/graphs/{name}_adjacency.csv"
    gml_path = f"outputs/graphs/{name}_graph.gml"

    adj_df.to_csv(csv_path)
    G = adjacency_to_gml(adj_df, gml_path)

    n_edges = count_edges(adj_matrix)
    n_nodes = len(columns)

    print(f"  -> Saved: {csv_path}")
    print(f"  -> Saved: {gml_path}")
    print(f"  -> Nodes: {n_nodes}, Directed edges: {n_edges}")
    print(f"  -> Runtime: {elapsed:.2f}s")

    log_run(log_path, {
        "timestamp":      datetime.now().isoformat(),
        "algorithm":      name,
        "n_rows":         "see_data",  # filled by caller
        "n_columns":      n_nodes,
        "n_edges":        n_edges,
        "density":        round(n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0, 4),
        "runtime_sec":    round(elapsed, 2),
        "hyperparams":    json.dumps(hyperparams),
        "csv_path":       csv_path,
        "gml_path":       gml_path,
    })

    return G


# ================================================================
#  ALGORITHM 1: Unconstrained PC (causal-learn)
# ================================================================

def run_unconstrained_pc(data: np.ndarray, columns: list[str],
                         alpha: float = 0.05, log_path: str = "") -> np.ndarray:
    """
    Run the PC algorithm without any background knowledge.
    Uses Fisher-Z conditional independence test.
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    print("\n" + "=" * 60)
    print("  ALGORITHM 1: Unconstrained PC")
    print("=" * 60)

    hyperparams = {"alpha": alpha, "indep_test": "fisherz", "stable": True,
                   "uc_rule": 0, "uc_priority": 2, "background_knowledge": None}
    print(f"  Hyperparams: {hyperparams}")

    t0 = time.time()
    cg = pc(data, alpha=alpha, indep_test=fisherz, stable=True,
            uc_rule=0, uc_priority=2)
    elapsed = time.time() - t0

    # Extract adjacency matrix from the CausalGraph object
    adj_matrix = cg.G.graph  # numpy array
    # causal-learn uses: 1 = tail, -1 = arrowhead, 0 = no edge
    # Convert to standard adjacency: adj[i][j]=1 means i->j
    n = adj_matrix.shape[0]
    directed_adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            # i -> j means: adj_matrix[i,j] = -1 and adj_matrix[j,i] = 1
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                directed_adj[j, i] = 1  # j causes i? No — let's be precise
            # In causal-learn's encoding:
            # graph[i,j] = -1 and graph[j,i] = 1  means  i --> j
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                directed_adj[i, j] = 1
            # i - j (undirected): graph[i,j] = -1 and graph[j,i] = -1
            elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                directed_adj[i, j] = 1
                directed_adj[j, i] = 1

    save_outputs("unconstrained_pc", directed_adj, columns, hyperparams, elapsed, log_path)
    return directed_adj


# ================================================================
#  ALGORITHM 2: Unconstrained LiNGAM (lingam)
# ================================================================

def run_unconstrained_lingam(data: np.ndarray, columns: list[str],
                             log_path: str = "") -> np.ndarray:
    """
    Run DirectLiNGAM without any prior knowledge.
    Assumes non-Gaussian, continuous data with linear relationships.
    """
    import lingam

    print("\n" + "=" * 60)
    print("  ALGORITHM 2: Unconstrained LiNGAM (DirectLiNGAM)")
    print("=" * 60)

    hyperparams = {"method": "DirectLiNGAM", "prior_knowledge": None,
                   "measure": "pwling"}
    print(f"  Hyperparams: {hyperparams}")

    t0 = time.time()
    model = lingam.DirectLiNGAM()
    model.fit(data)
    elapsed = time.time() - t0

    # adjacency_matrix_: B[i,j] != 0 means j -> i  (effect <- cause)
    # We want adj[i,j] = 1 meaning i -> j, so transpose
    raw_adj = model.adjacency_matrix_  # shape (n, n)

    # Threshold small coefficients (noise) — keep edges |coeff| > 0.01
    threshold = 0.01
    directed_adj = (np.abs(raw_adj) > threshold).astype(int)

    # Print causal order
    print(f"  Causal order: {[columns[i] for i in model.causal_order_]}")

    save_outputs("unconstrained_lingam", directed_adj, columns, hyperparams, elapsed, log_path)

    # Also save the weighted adjacency (actual coefficients)
    weight_df = adjacency_to_dataframe(np.round(raw_adj, 4), columns)
    weight_path = "outputs/graphs/unconstrained_lingam_weights.csv"
    weight_df.to_csv(weight_path)
    print(f"  -> Saved weighted adjacency: {weight_path}")

    return directed_adj


# ================================================================
#  ALGORITHM 3: Constrained PC (with ontology forbidden edges)
# ================================================================

def run_constrained_pc(data: np.ndarray, columns: list[str],
                       alpha: float = 0.05, log_path: str = "") -> np.ndarray:
    """
    Run the PC algorithm WITH ontology-derived background knowledge
    (forbidden + required edges from 04_forbidden_edges.py).
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    print("\n" + "=" * 60)
    print("  ALGORITHM 3: Constrained PC (ontology-guided)")
    print("=" * 60)

    # Build background knowledge from forbidden_edges module
    bk, n_forbidden, n_required = build_background_knowledge(columns)

    hyperparams = {
        "alpha": alpha, "indep_test": "fisherz", "stable": True,
        "uc_rule": 0, "uc_priority": 2,
        "forbidden_edges_applied": n_forbidden,
        "required_edges_applied": n_required,
        "forbidden_edges_total": len(FORBIDDEN_EDGES),
        "required_edges_total": len(REQUIRED_EDGES),
    }
    print(f"  Hyperparams: {hyperparams}")

    t0 = time.time()
    cg = pc(data, alpha=alpha, indep_test=fisherz, stable=True,
            uc_rule=0, uc_priority=2, background_knowledge=bk)
    elapsed = time.time() - t0

    # Same adjacency extraction as unconstrained PC
    adj_matrix = cg.G.graph
    n = adj_matrix.shape[0]
    directed_adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                directed_adj[i, j] = 1
            elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                directed_adj[i, j] = 1
                directed_adj[j, i] = 1

    save_outputs("constrained_pc", directed_adj, columns, hyperparams, elapsed, log_path)
    return directed_adj


# ================================================================
#  COMPARISON SUMMARY
# ================================================================

def print_comparison(results: dict[str, np.ndarray], columns: list[str]):
    """Print a side-by-side comparison of the three graphs."""
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)

    header = f"{'Metric':<35}"
    for name in results:
        header += f"{name:<25}"
    print(header)
    print("-" * len(header))

    # Edge counts
    row = f"{'Directed edges':<35}"
    for name, adj in results.items():
        row += f"{count_edges(adj):<25}"
    print(row)

    # Density
    n = len(columns)
    max_edges = n * (n - 1)
    row = f"{'Density':<35}"
    for name, adj in results.items():
        d = count_edges(adj) / max_edges if max_edges > 0 else 0
        row += f"{d:.4f}{'':<20}"
    print(row)

    # Edges unique to constrained vs unconstrained PC
    if "unconstrained_pc" in results and "constrained_pc" in results:
        upc = results["unconstrained_pc"]
        cpc = results["constrained_pc"]
        only_unconstrained = int(np.sum((upc == 1) & (cpc == 0)))
        only_constrained = int(np.sum((upc == 0) & (cpc == 1)))
        shared = int(np.sum((upc == 1) & (cpc == 1)))
        print(f"\n  PC unconstrained vs constrained:")
        print(f"    Shared edges:              {shared}")
        print(f"    Only in unconstrained PC:  {only_unconstrained}")
        print(f"    Only in constrained PC:    {only_constrained}")
        print(f"    Edges removed by ontology: {only_unconstrained - only_constrained}")

    # Jaccard similarity between each pair
    names = list(results.keys())
    print(f"\n  Jaccard similarity (edge overlap):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = (results[names[i]] != 0).flatten()
            b = (results[names[j]] != 0).flatten()
            intersection = int(np.sum(a & b))
            union = int(np.sum(a | b))
            jaccard = intersection / union if union > 0 else 0
            print(f"    {names[i]} <-> {names[j]}: {jaccard:.4f}")


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Run baseline causal discovery algorithms")
    parser.add_argument("--input", default=READY_DATA_PATH,
                        help="Path to causal-ready CSV (default: from config.py)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for PC algorithm (default: 0.05)")
    args = parser.parse_args()

    log_path = "outputs/metrics/run_log.csv"

    # ── Load data ────────────────────────────────────────────
    df = load_data(args.input)
    data = df.values
    columns = list(df.columns)

    print(f"\n[info] Variables entering causal discovery: {len(columns)}")
    print(f"[info] Sample size: {len(df)}")
    print(f"[info] Alpha (PC): {args.alpha}")
    print(f"[info] Timestamp: {datetime.now().isoformat()}")

    # ── Run all three algorithms ─────────────────────────────
    results = {}

    adj_pc = run_unconstrained_pc(data, columns, alpha=args.alpha, log_path=log_path)
    results["unconstrained_pc"] = adj_pc

    adj_lingam = run_unconstrained_lingam(data, columns, log_path=log_path)
    results["unconstrained_lingam"] = adj_lingam

    adj_cpc = run_constrained_pc(data, columns, alpha=args.alpha, log_path=log_path)
    results["constrained_pc"] = adj_cpc

    # ── Comparison ───────────────────────────────────────────
    print_comparison(results, columns)

    # ── Update n_rows in log (we know it now) ────────────────
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df.loc[log_df["n_rows"] == "see_data", "n_rows"] = len(df)
        log_df.to_csv(log_path, index=False)

    print(f"\n[done] All outputs in outputs/graphs/ and outputs/metrics/")
    print(f"[done] Run log: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
