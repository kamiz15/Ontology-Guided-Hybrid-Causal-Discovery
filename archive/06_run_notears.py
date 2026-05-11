# 06_run_notears.py
# ============================================================
# Week 2 (optional) — NOTEARS baseline via gCastle
# A continuous-optimization approach to causal discovery.
# Runs independently and appends to the same run_log.csv.
#
# NOTEARS treats structure learning as a continuous optimization
# problem with an acyclicity constraint, producing a DAG
# directly without conditional independence tests.
#
# Usage:
#   python 06_run_notears.py
#   python 06_run_notears.py --lambda1 0.1 --w-threshold 0.3
#
# Requirements:
#   pip install gcastle pandas numpy networkx
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

from config import READY_DATA_PATH

# ── Shared utilities (duplicated for standalone use) ──────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[data] Loaded {path}: {df.shape[0]} rows × {df.shape[1]} columns")
    numeric_df = df.select_dtypes(include="number")
    dropped = set(df.columns) - set(numeric_df.columns)
    if dropped:
        print(f"[data] Dropped non-numeric columns: {sorted(dropped)}")
    before = len(numeric_df)
    numeric_df = numeric_df.dropna()
    after = len(numeric_df)
    if before != after:
        print(f"[data] Dropped {before - after} rows with NaN -> {after} rows remain")
    return numeric_df


def adjacency_to_dataframe(adj: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(adj, index=cols, columns=cols)


def adjacency_to_gml(adj_df: pd.DataFrame, path: str):
    G = nx.DiGraph()
    cols = list(adj_df.columns)
    G.add_nodes_from(cols)
    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            w = adj_df.iloc[i, j]
            if w != 0:
                G.add_edge(src, tgt, weight=float(w))
    nx.write_gml(G, path)


def count_edges(adj: np.ndarray) -> int:
    return int(np.count_nonzero(adj))


def log_run(log_path: str, entry: dict):
    df_new = pd.DataFrame([entry])
    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(log_path, index=False)


def run_notears(data: np.ndarray, columns: list[str],
                lambda1: float = 0.1, w_threshold: float = 0.3,
                log_path: str = "") -> np.ndarray:
    """
    Run NOTEARS via the gCastle library.
    
    Parameters
    ----------
    lambda1 : float
        L1 regularization strength (sparsity).
    w_threshold : float
        Threshold for pruning small edge weights.
    """
    from castle.algorithms import GOLEM, Notears

    print("\n" + "=" * 60)
    print("  ALGORITHM 4: NOTEARS (gCastle)")
    print("=" * 60)

    hyperparams = {
        "method": "NOTEARS",
        "library": "gCastle",
        "lambda1": lambda1,
        "w_threshold": w_threshold,
        "loss_type": "l2",
    }
    print(f"  Hyperparams: {hyperparams}")

    t0 = time.time()
    nt = Notears(lambda1=lambda1, loss_type="l2", w_threshold=w_threshold)
    nt.learn(data)
    elapsed = time.time() - t0

    # gCastle stores the weighted adjacency in causal_matrix
    raw_adj = nt.causal_matrix  # shape (n, n)

    # Binarize: adj[i,j] = 1 means i -> j
    directed_adj = (np.abs(raw_adj) > 0).astype(int)

    n_edges = count_edges(directed_adj)
    n = len(columns)
    print(f"  -> Nodes: {n}, Directed edges: {n_edges}")
    print(f"  -> Runtime: {elapsed:.2f}s")

    # Save outputs
    adj_df = adjacency_to_dataframe(directed_adj, columns)
    csv_path = "outputs/graphs/notears_adjacency.csv"
    gml_path = "outputs/graphs/notears_graph.gml"

    adj_df.to_csv(csv_path)
    adjacency_to_gml(adj_df, gml_path)
    print(f"  -> Saved: {csv_path}")
    print(f"  -> Saved: {gml_path}")

    # Save weighted adjacency too
    weight_df = adjacency_to_dataframe(np.round(raw_adj, 4), columns)
    weight_path = "outputs/graphs/notears_weights.csv"
    weight_df.to_csv(weight_path)
    print(f"  -> Saved weighted adjacency: {weight_path}")

    # Log
    log_run(log_path, {
        "timestamp":   datetime.now().isoformat(),
        "algorithm":   "notears_gcastle",
        "n_rows":      data.shape[0],
        "n_columns":   n,
        "n_edges":     n_edges,
        "density":     round(n_edges / (n * (n - 1)) if n > 1 else 0, 4),
        "runtime_sec": round(elapsed, 2),
        "hyperparams": json.dumps(hyperparams),
        "csv_path":    csv_path,
        "gml_path":    gml_path,
    })

    return directed_adj


def main():
    parser = argparse.ArgumentParser(description="Run NOTEARS baseline via gCastle")
    parser.add_argument("--input", default=READY_DATA_PATH)
    parser.add_argument("--lambda1", type=float, default=0.1,
                        help="L1 penalty (sparsity)")
    parser.add_argument("--w-threshold", type=float, default=0.3,
                        help="Edge weight threshold for pruning")
    args = parser.parse_args()

    log_path = "outputs/metrics/run_log.csv"
    df = load_data(args.input)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)
    print(f"[data] StandardScaler applied — all columns now mean=0, std=1")

    run_notears(data_scaled, list(df.columns),
                lambda1=args.lambda1, w_threshold=args.w_threshold,
                log_path=log_path)

    print(f"\n[done] NOTEARS output in outputs/graphs/")


if __name__ == "__main__":
    main()
