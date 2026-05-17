#!/usr/bin/env python3
"""
Generate PC-constrained DAG with synthetic data and create figure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import networkx as nx

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "synthetic"
OUTPUT_DIR = ROOT / "outputs" / "figures"
EXP_DIR = ROOT / "outputs" / "experiments"

DATA_PATH = DATA_DIR / "synthetic_n2000.csv"
CONSTRAINTS_PATH = DATA_DIR / "ground_truth_constraints_synthetic.csv"
OUTPUT_FIGURE_PATH = OUTPUT_DIR / "pc_constrained_synthetic.png"
ADJACENCY_OUTPUT = EXP_DIR / "pc_constrained_synthetic_adjacency.csv"

# ============================================================
# LOAD DATA & CONSTRAINTS
# ============================================================
def load_data():
    """Load synthetic data."""
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Loaded data shape: {df.shape}")
    return df, df.columns.tolist()

def load_constraints():
    """Load forbidden edge constraints from constraint matrix."""
    constraint_matrix = pd.read_csv(CONSTRAINTS_PATH, index_col=0)
    columns = constraint_matrix.index.tolist()
    forbidden = []

    # -1 indicates forbidden edge from row to column
    for i, source in enumerate(columns):
        for j, target in enumerate(columns):
            if constraint_matrix.iloc[i, j] == -1:
                forbidden.append((source, target))

    print(f"Loaded {len(forbidden)} forbidden edges")
    return forbidden, columns

def run_pc_constrained(data, columns, forbidden):
    """Run PC algorithm with forbidden edge constraints."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc as causallearn_pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    # Create background knowledge with forbidden edges
    node_objects = [GraphNode(column) for column in columns]
    node_map = {column: node for column, node in zip(columns, node_objects)}
    background = BackgroundKnowledge()

    for source, target in forbidden:
        if source in node_map and target in node_map:
            background.add_forbidden_by_node(node_map[source], node_map[target])

    print("Running PC algorithm with constraints...")
    graph = causallearn_pc(
        data,
        alpha=0.05,
        indep_test="fisherz",
        stable=True,
        background_knowledge=background,
        verbose=False,
        show_progress=False,
        node_names=columns,
    )

    # Extract adjacency matrix
    index_map = {column: idx for idx, column in enumerate(columns)}
    adj = np.zeros((len(columns), len(columns)), dtype=int)

    for edge in graph.G.get_graph_edges():
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()

        if n1 not in index_map or n2 not in index_map:
            continue

        ep1 = str(edge.get_endpoint1())
        ep2 = str(edge.get_endpoint2())
        i, j = index_map[n1], index_map[n2]

        if "TAIL" in ep1 and "ARROW" in ep2:
            adj[i, j] = 1
        elif "ARROW" in ep1 and "TAIL" in ep2:
            adj[j, i] = 1

    np.fill_diagonal(adj, 0)
    return adj, graph

def visualize_dag(adj, columns, output_path):
    """Visualize DAG as a network graph."""
    # Create directed graph from adjacency matrix
    G = nx.DiGraph()
    G.add_nodes_from(columns)

    edges = []
    for i, source in enumerate(columns):
        for j, target in enumerate(columns):
            if adj[i, j] == 1:
                edges.append((source, target))
                G.add_edge(source, target)

    print(f"DAG has {len(edges)} edges")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), dpi=160)

    # Use spring layout with increased k for better spacing
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color="#60A5FA",
        node_size=800,
        ax=ax,
        alpha=0.9
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#4B5563",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        width=1.5,
        ax=ax,
        connectionstyle="arc3,rad=0.1",
        alpha=0.7
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight="bold",
        ax=ax
    )

    ax.set_title("PC-Constrained DAG (Synthetic Data, n=2000)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.close(fig)

def main():
    """Main execution."""
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data, columns = load_data()

    # Load constraints
    forbidden, _ = load_constraints()

    # Run PC algorithm
    adj, graph = run_pc_constrained(data.values, columns, forbidden)

    # Save adjacency matrix
    adj_df = pd.DataFrame(adj, index=columns, columns=columns)
    adj_df.to_csv(ADJACENCY_OUTPUT)
    print(f"Adjacency matrix saved to {ADJACENCY_OUTPUT}")

    # Visualize
    visualize_dag(adj, columns, OUTPUT_FIGURE_PATH)

    # Print statistics
    edge_count = np.sum(adj)
    print(f"\nPC-Constrained Synthetic DAG Statistics:")
    print(f"  Variables: {len(columns)}")
    print(f"  Edges: {edge_count}")
    print(f"  Density: {edge_count / (len(columns) * (len(columns) - 1)):.4f}")

if __name__ == "__main__":
    main()
