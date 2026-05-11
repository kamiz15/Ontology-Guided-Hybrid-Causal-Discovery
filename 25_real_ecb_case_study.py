"""Create the real ECB case-study graph artifact."""

from __future__ import annotations

import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "data_ready.csv"
REVIEW_PATH = ROOT / "reports" / "constraints_for_review.csv"
FIGURE_PATH = ROOT / "outputs" / "figures" / "real_ecb_case_study_graph.png"
ADAPTER_PATH = ROOT / "14_constraint_adapter.py"
FORBIDDEN_MODULE_CANDIDATES = [
    ROOT / "04_forbidden_edges_real.py",
    ROOT / "archive" / "04_forbidden_edges_real.py",
    ROOT / "04_forbidden_edges_real.py.bak",
]

Constraint = tuple[str, str]


def log(message: str) -> None:
    """Print a case-study log message."""
    print(f"[casestudy] {message}")


def load_module(path: Path, module_name: str) -> Any:
    """Load a Python module from an explicit path.

    Parameters
    ----------
    path : Path
        Module path to load.
    module_name : str
    Runtime module name.

    Returns
    -------
    Any
        Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        loader = SourceFileLoader(module_name, str(path))
        spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_real_constraints(variable_names: list[str]) -> tuple[list[Constraint], list[Constraint]]:
    """Load real-data constraints, using archived fallback if needed.

    Parameters
    ----------
    variable_names : list of str
        Active real-data variables.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
        Forbidden and required directed constraints.
    """
    adapter = load_module(ADAPTER_PATH, "constraint_adapter_step_14_case_study")
    try:
        forbidden, required, _ = adapter.load_constraints_for_dataset("real")
        log("Loaded real constraints through 14_constraint_adapter.py")
    except Exception as exc:
        log(f"WARNING: adapter real-constraint load failed ({exc}); trying archived module")
        forbidden_module = None
        for candidate in FORBIDDEN_MODULE_CANDIDATES:
            if candidate.exists():
                forbidden_module = load_module(candidate, "forbidden_edges_real_case_study")
                log(f"Loaded real constraints from {candidate.relative_to(ROOT)}")
                break
        if forbidden_module is None:
            raise FileNotFoundError("No real forbidden-edge module found") from exc
        forbidden = list(getattr(forbidden_module, "FORBIDDEN_EDGES", []))
        required = list(getattr(forbidden_module, "REQUIRED_EDGES", []))

    variable_set = set(variable_names)
    filtered_forbidden = [
        (str(source), str(target))
        for source, target in forbidden
        if str(source) in variable_set and str(target) in variable_set
    ]
    filtered_required = [
        (str(source), str(target))
        for source, target in required
        if str(source) in variable_set and str(target) in variable_set
    ]
    log(
        "Applicable real constraints: "
        f"{len(filtered_forbidden)} forbidden, {len(filtered_required)} required"
    )
    return filtered_forbidden, filtered_required


def build_background_knowledge(
    variable_names: list[str],
    forbidden: list[Constraint],
    required: list[Constraint],
) -> Any:
    """Build causal-learn background knowledge with the existing adapter.

    Parameters
    ----------
    variable_names : list of str
        Data columns in causal-learn order.
    forbidden : list[tuple[str, str]]
        Directed forbidden constraints.
    required : list[tuple[str, str]]
        Directed required constraints.

    Returns
    -------
    Any
        causal-learn BackgroundKnowledge object.
    """
    from causallearn.graph.GraphNode import GraphNode

    adapter = load_module(ADAPTER_PATH, "constraint_adapter_step_14_case_study_bk")
    node_objects = [GraphNode(name) for name in variable_names]
    return adapter.build_causal_learn_bk(
        variable_names=variable_names,
        forbidden=forbidden,
        required=required,
        node_objects=node_objects,
    )


def load_real_data(path: Path) -> pd.DataFrame:
    """Load and standardize the real ECB numeric dataset.

    Parameters
    ----------
    path : Path
        Input CSV path.

    Returns
    -------
    pandas.DataFrame
        Numeric data with missing rows removed.
    """
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    dropped_columns = sorted(set(df.columns) - set(numeric.columns))
    if dropped_columns:
        log(f"Dropped non-numeric columns: {dropped_columns}")
    before = len(numeric)
    numeric = numeric.dropna(axis=0)
    if len(numeric) != before:
        log(f"Dropped {before - len(numeric)} rows with missing values")
    log(f"Loaded real ECB data: N={len(numeric)}, variables={numeric.shape[1]}")
    return numeric


def standardized_array(df: pd.DataFrame) -> np.ndarray:
    """Return a z-scored NumPy array.

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric input data.

    Returns
    -------
    numpy.ndarray
        Standardized array with zero-variance columns protected.
    """
    values = df.to_numpy(dtype=float)
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=0)
    stds[stds == 0] = 1.0
    return (values - means) / stds


def endpoint_name(endpoint: Any) -> str:
    """Normalize a causal-learn edge endpoint name."""
    name = getattr(endpoint, "name", None)
    if name is None:
        name = str(endpoint)
    return str(name).split(".")[-1].upper()


def graph_node_name(node: Any) -> str:
    """Extract a stable variable name from a causal-learn graph node."""
    if hasattr(node, "get_name"):
        return str(node.get_name())
    return str(node)


def extract_cpdag_edges(causal_graph: Any) -> pd.DataFrame:
    """Extract directed and undirected edges from a causal-learn CPDAG.

    Parameters
    ----------
    causal_graph : Any
        Result object returned by causal-learn PC.

    Returns
    -------
    pandas.DataFrame
        Edge table with cause, effect, and edge_type columns.
    """
    graph = causal_graph.G
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for edge in graph.get_graph_edges():
        node1 = graph_node_name(edge.get_node1())
        node2 = graph_node_name(edge.get_node2())
        endpoint1 = endpoint_name(edge.get_endpoint1())
        endpoint2 = endpoint_name(edge.get_endpoint2())

        if endpoint1 == "TAIL" and endpoint2 == "ARROW":
            source, target, edge_type = node1, node2, "directed"
        elif endpoint1 == "ARROW" and endpoint2 == "TAIL":
            source, target, edge_type = node2, node1, "directed"
        else:
            source, target = sorted([node1, node2])
            edge_type = "undirected"

        key = (source, target, edge_type)
        if key not in seen:
            seen.add(key)
            rows.append({"cause": source, "effect": target, "edge_type": edge_type})

    return pd.DataFrame(rows, columns=["cause", "effect", "edge_type"])


def load_review_support(variable_names: list[str]) -> tuple[dict[Constraint, dict[str, Any]], int, int]:
    """Load approved literature-supported directions from the review CSV.

    Parameters
    ----------
    variable_names : list of str
        Variables in the real ECB case-study data.

    Returns
    -------
    tuple[dict[tuple[str, str], dict[str, Any]], int, int]
        Support metadata by directed edge, count mapped to real schema, and
        total approved audited constraints.
    """
    review = pd.read_csv(REVIEW_PATH)
    approved = review[review["approved"].astype(str).str.lower().eq("yes")].copy()
    approved_count = int(len(approved))
    variable_set = set(variable_names)
    mapped = approved[
        approved["cause"].astype(str).isin(variable_set)
        & approved["effect"].astype(str).isin(variable_set)
    ].copy()

    support: dict[Constraint, dict[str, Any]] = {}
    for _, row in mapped.iterrows():
        action = str(row.get("proposed_action", "")).lower()
        if "forbid_reverse" not in action and "require" not in action and "support" not in action:
            continue
        pair = (str(row["cause"]), str(row["effect"]))
        support[pair] = {
            "paper_ids": str(row.get("paper_ids", "")).strip(),
            "paper_count": int(row.get("paper_count", 0) or 0),
            "proposed_action": str(row.get("proposed_action", "")),
        }

    return support, int(len(mapped)), approved_count


def annotate_edges(
    edges: pd.DataFrame,
    support: dict[Constraint, dict[str, Any]],
    forbidden: list[Constraint],
) -> pd.DataFrame:
    """Annotate discovered edges against literature review constraints.

    Parameters
    ----------
    edges : pandas.DataFrame
        Discovered edge table.
    support : dict[tuple[str, str], dict[str, Any]]
        Approved literature-supported directions.
    forbidden : list[tuple[str, str]]
        Explicit forbidden directions.

    Returns
    -------
    pandas.DataFrame
        Annotated edge table.
    """
    forbidden_set = set(forbidden)
    rows: list[dict[str, str]] = []
    forbidden_hits: list[tuple[str, str]] = []

    for _, row in edges.iterrows():
        source = str(row["cause"])
        target = str(row["effect"])
        edge_type = str(row["edge_type"])
        pair = (source, target)

        annotation = "literature_silent"
        papers = ""
        if edge_type == "directed" and pair in forbidden_set:
            annotation = "literature_forbidden"
            forbidden_hits.append(pair)
        elif pair in support:
            annotation = "literature_supported"
            papers = support[pair]["paper_ids"]
        elif edge_type == "undirected" and (target, source) in support:
            annotation = "literature_supported"
            papers = support[(target, source)]["paper_ids"]

        rows.append({
            "cause": source,
            "effect": target,
            "edge_type": edge_type,
            "annotation": annotation,
            "supporting_paper_ids": papers,
        })

    for source, target in forbidden_hits:
        log(f"WARNING: discovered literature-forbidden edge {source} -> {target}")

    return pd.DataFrame(rows)


def run_pc(data: pd.DataFrame, alpha: float, forbidden: list[Constraint], required: list[Constraint]) -> pd.DataFrame:
    """Run constrained causal-learn PC on the real ECB dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric real ECB data.
    alpha : float
        PC conditional-independence threshold.
    forbidden : list[tuple[str, str]]
        Forbidden directed constraints.
    required : list[tuple[str, str]]
        Required directed constraints.

    Returns
    -------
    pandas.DataFrame
        Extracted CPDAG edge table.
    """
    from causallearn.search.ConstraintBased.PC import pc

    variable_names = list(data.columns)
    background_knowledge = build_background_knowledge(variable_names, forbidden, required)
    log(f"Running PC with alpha={alpha}")
    result = pc(
        standardized_array(data),
        alpha=alpha,
        indep_test="fisherz",
        stable=True,
        background_knowledge=background_knowledge,
        verbose=False,
        show_progress=False,
        node_names=variable_names,
    )
    edges = extract_cpdag_edges(result)
    log(f"Extracted {len(edges)} CPDAG edges from PC result")
    return edges


def render_graph(edges: pd.DataFrame, variable_names: list[str], path: Path, alpha: float, n_rows: int) -> None:
    """Render the annotated case-study graph.

    Parameters
    ----------
    edges : pandas.DataFrame
        Annotated edge table.
    variable_names : list of str
        Nodes to include in the graph.
    path : Path
        Output PNG path.
    alpha : float
        PC alpha used in the title.
    n_rows : int
        Dataset row count used in the title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path.parent.mkdir(parents=True, exist_ok=True)
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(variable_names)
    for _, row in edges.iterrows():
        layout_graph.add_edge(str(row["cause"]), str(row["effect"]))
    positions = nx.spring_layout(layout_graph, seed=42, k=1.0)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.set_title(
        f"Discovered causal graph: real ECB sample (N={n_rows}, "
        f"PC with literature constraints, alpha={alpha})",
        fontsize=12,
        pad=18,
    )

    nx.draw_networkx_nodes(
        layout_graph,
        positions,
        node_color="#f5f7fb",
        edgecolors="#334155",
        linewidths=1.0,
        node_size=1900,
        ax=ax,
    )
    nx.draw_networkx_labels(layout_graph, positions, font_size=7, font_color="#111827", ax=ax)

    directed_styles = {
        "literature_supported": {"edge_color": "#15803d", "width": 2.0},
        "literature_forbidden": {"edge_color": "#b91c1c", "width": 2.0},
        "literature_silent": {"edge_color": "#6b7280", "width": 1.0},
    }
    for annotation, style in directed_styles.items():
        selected = edges[
            edges["edge_type"].eq("directed") & edges["annotation"].eq(annotation)
        ]
        edge_list = [(str(row["cause"]), str(row["effect"])) for _, row in selected.iterrows()]
        if edge_list:
            nx.draw_networkx_edges(
                layout_graph,
                positions,
                edgelist=edge_list,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=14,
                connectionstyle="arc3,rad=0.05",
                ax=ax,
                **style,
            )

    undirected = edges[edges["edge_type"].eq("undirected")]
    undirected_edges = [(str(row["cause"]), str(row["effect"])) for _, row in undirected.iterrows()]
    if undirected_edges:
        nx.draw_networkx_edges(
            layout_graph,
            positions,
            edgelist=undirected_edges,
            arrows=False,
            edge_color="#6b7280",
            width=1.0,
            style="dashed",
            ax=ax,
        )

    legend_handles = [
        Line2D([0], [0], color="#15803d", lw=2.0, label="literature_supported"),
        Line2D([0], [0], color="#b91c1c", lw=2.0, label="literature_forbidden"),
        Line2D([0], [0], color="#6b7280", lw=1.0, label="literature_silent"),
        Line2D([0], [0], color="#6b7280", lw=1.0, linestyle="--", label="undirected (CPDAG)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"Figure written: {path.relative_to(ROOT)}")


def write_summary(
    edges: pd.DataFrame,
    support: dict[Constraint, dict[str, Any]],
    mapped_constraints: int,
    approved_constraints: int,
    path: Path,
) -> None:
    """Write the case-study markdown summary.

    Parameters
    ----------
    edges : pandas.DataFrame
        Annotated edge table.
    support : dict[tuple[str, str], dict[str, Any]]
        Literature-supported directions and metadata.
    mapped_constraints : int
        Number of approved audited constraints mapping to the real schema.
    approved_constraints : int
        Number of approved audited constraints.
    path : Path
        Summary output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    directed_count = int(edges["edge_type"].eq("directed").sum())
    undirected_count = int(edges["edge_type"].eq("undirected").sum())
    total_count = int(len(edges))
    annotation_counts = edges["annotation"].value_counts().to_dict()
    supported_count = int(annotation_counts.get("literature_supported", 0))
    silent_count = int(annotation_counts.get("literature_silent", 0))
    forbidden_count = int(annotation_counts.get("literature_forbidden", 0))

    supported_edges = edges[edges["annotation"].eq("literature_supported")].copy()
    strongest_lines: list[str] = []
    for _, row in supported_edges.iterrows():
        pair = (str(row["cause"]), str(row["effect"]))
        info = support.get(pair) or support.get((pair[1], pair[0]), {})
        strongest_lines.append(
            f"- {row['cause']} -> {row['effect']} "
            f"({row['edge_type']}): {info.get('paper_ids', row['supporting_paper_ids'])}"
        )
    strongest_lines = strongest_lines[:5] or ["- None detected in the discovered graph."]

    forbidden_edges = edges[edges["annotation"].eq("literature_forbidden")]
    if forbidden_edges.empty:
        forbidden_text = "No literature-forbidden directed edges were discovered."
    else:
        forbidden_text = "\n".join(
            f"- {row['cause']} -> {row['effect']}" for _, row in forbidden_edges.iterrows()
        )

    summary = f"""# Real ECB Case Study Graph Summary

## Edge counts

- Total discovered edges: {total_count}
- Directed edges: {directed_count}
- Undirected CPDAG edges: {undirected_count}

## Annotation distribution

- literature_supported: {supported_count}
- literature_silent: {silent_count}
- literature_forbidden: {forbidden_count}

## Strongest literature-supported edges

{chr(10).join(strongest_lines)}

## Literature-forbidden edges

{forbidden_text}

## Template interpretation

The real ECB case-study graph is a qualitative constraint-guided discovery artifact rather than a ground-truth recovery test. The discovered CPDAG respects the audited literature constraints in the fitted graph: no literature-forbidden directed edge is present, while supported edges are explicitly marked when the discovered relation aligns with the reviewed evidence. The literature-constraint footprint on the real 16-variable schema is small: only {mapped_constraints} of {approved_constraints} audited constraints map to the available real-data variables, so most discovered edges remain literature-silent rather than contradicted or confirmed. This makes the case study useful for inspecting plausible ESG-finance structure and constraint compliance, but not for estimating causal recovery accuracy. The quantitative evidence base for recovery and scalability remains the causal-dummy, SNR-sensitivity, and sample-size-sensitivity experiments.
"""
    path.write_text(summary, encoding="utf-8")
    log(f"Summary written: {path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create the real ECB case-study graph.")
    parser.add_argument("--alpha", type=float, default=0.05, help="PC alpha threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "casestudy",
        help="Directory for case-study CSV and markdown outputs.",
    )
    parser.add_argument("--no-figure", action="store_true", help="Skip PNG figure generation.")
    return parser.parse_args()


def main() -> None:
    """Run the real ECB case-study graph pipeline."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_real_data(DATA_PATH)
    variable_names = list(data.columns)
    forbidden, required = load_real_constraints(variable_names)
    support, mapped_constraints, approved_constraints = load_review_support(variable_names)
    log(
        "Approved review constraints mapped to real schema: "
        f"{mapped_constraints} of {approved_constraints}"
    )

    raw_edges = run_pc(data, args.alpha, forbidden, required)
    annotated_edges = annotate_edges(raw_edges, support, forbidden)

    edge_path = output_dir / "real_ecb_edges.csv"
    annotated_edges.to_csv(edge_path, index=False)
    log(f"Edges written: {edge_path.relative_to(ROOT)}")

    if not args.no_figure:
        render_graph(annotated_edges, variable_names, FIGURE_PATH, args.alpha, len(data))

    summary_path = output_dir / "real_ecb_case_study_summary.md"
    write_summary(
        annotated_edges,
        support,
        mapped_constraints,
        approved_constraints,
        summary_path,
    )

    forbidden_count = int(annotated_edges["annotation"].eq("literature_forbidden").sum())
    if forbidden_count == 0:
        log("No literature-forbidden directed edges found in the discovered graph")


if __name__ == "__main__":
    main()
