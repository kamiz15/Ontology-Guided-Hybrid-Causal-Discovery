"""Run GES for ESG causal-discovery experiments.

This module wraps the available score-based GES implementation and keeps
constraint handling explicit. Causal-learn's GES implementation does not
accept background knowledge, so constrained variants are post-processed and
reported as ``ges_postproc`` by ``run_all.py``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _causal_learn_graph_to_adj(graph: Any) -> np.ndarray:
    """
    Convert a causal-learn graph object to directed adjacency.

    Parameters
    ----------
    graph : Any
        causal-learn ``GeneralGraph`` object.

    Returns
    -------
    np.ndarray
        Binary adjacency where ``A[i, j] = 1`` means variable ``i`` points to
        variable ``j``.
    """
    matrix = np.asarray(graph.graph)
    n_vars = matrix.shape[0]
    adjacency = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars):
        for j in range(n_vars):
            if matrix[i, j] == -1 and matrix[j, i] == 1:
                adjacency[i, j] = 1
    np.fill_diagonal(adjacency, 0)
    return adjacency


def _orient_cpdag_to_dag(graph: Any) -> Any:
    """
    Orient a causal-learn CPDAG to one compatible DAG when possible.

    Parameters
    ----------
    graph : Any
        causal-learn graph object returned by GES.

    Returns
    -------
    Any
        Directed graph object. If orientation fails, the original graph is
        returned and only already-directed edges will be evaluated.
    """
    try:
        from causallearn.utils.PDAG2DAG import pdag2dag

        return pdag2dag(copy.deepcopy(graph))
    except Exception:
        return graph


def _apply_postprocess_constraints(
    adjacency: np.ndarray,
    columns: list[str],
    forbidden: list[tuple[str, str]],
    required: list[tuple[str, str]],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Apply forbidden and required constraints to a binary adjacency matrix.

    Parameters
    ----------
    adjacency : np.ndarray
        Binary adjacency before post-processing.
    columns : list[str]
        Variable order.
    forbidden : list[tuple[str, str]]
        Forbidden directed edges.
    required : list[tuple[str, str]]
        Required directed edges.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        Post-processed adjacency and diagnostic counts.
    """
    index = {name: i for i, name in enumerate(columns)}
    final = np.asarray(adjacency, dtype=int).copy()
    before = final.copy()

    forbidden_present_before = 0
    required_missing_before = 0
    forbidden_removed = 0
    required_added = 0

    for source, target in forbidden:
        if source in index and target in index:
            i, j = index[source], index[target]
            if final[i, j]:
                forbidden_present_before += 1
                forbidden_removed += 1
            final[i, j] = 0

    for source, target in required:
        if source in index and target in index:
            i, j = index[source], index[target]
            if not final[i, j]:
                required_missing_before += 1
                required_added += 1
            final[i, j] = 1
            final[j, i] = 0

    np.fill_diagonal(final, 0)
    return final, {
        "forbidden_edges_predicted_before": forbidden_present_before,
        "required_edges_missing_before": required_missing_before,
        "forbidden_edges_removed": forbidden_removed,
        "required_edges_added": required_added,
        "post_processing_changed": int(np.sum(final != before)),
        "edge_count_before_postprocess": int(before.sum()),
        "edge_count_after_postprocess": int(final.sum()),
    }


def _save_adjacency(
    adjacency: np.ndarray,
    columns: list[str],
    dataset_name: str,
    constraint_label: str,
    seed: int,
    output_dir: Path,
) -> Path:
    """
    Save a learned adjacency matrix as a labeled CSV.

    Parameters
    ----------
    adjacency : np.ndarray
        Binary adjacency.
    columns : list[str]
        Variable order.
    dataset_name : str
        Dataset label.
    constraint_label : str
        File-safe constraint label.
    seed : int
        Random seed.
    output_dir : pathlib.Path
        Directory for graph CSVs.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ges_{dataset_name}_{constraint_label}_seed{seed}.csv"
    pd.DataFrame(adjacency, index=columns, columns=columns).to_csv(path)
    return path


def run_ges(
    data: np.ndarray,
    columns: list[str],
    mode: str,
    dataset_name: str,
    seed: int,
    forbidden: list[tuple[str, str]] | None = None,
    required: list[tuple[str, str]] | None = None,
    constraint_mode: str = "standard",
    output_dir: Path | str = Path("outputs") / "experiments" / "graphs",
    max_parents: int | None = None,
    score_func: str = "local_score_BIC",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Run causal-learn GES and optionally post-process constraints.

    Parameters
    ----------
    data : np.ndarray
        Standardized observations.
    columns : list[str]
        Variable names.
    mode : str
        ``unconstrained`` or ``constrained``.
    dataset_name : str
        Dataset label.
    seed : int
        Bootstrap seed, used only for output naming.
    forbidden : list[tuple[str, str]], optional
        Forbidden directed edges.
    required : list[tuple[str, str]], optional
        Required directed edges.
    constraint_mode : str, default="standard"
        Human-readable constraint mode.
    output_dir : pathlib.Path or str
        Graph output directory.
    max_parents : int, optional
        Maximum parent count passed to GES. Defaults to ``min(3, p - 1)`` to
        keep the score-based search stable on the 40-variable advisor-dummy
        matrix.
    score_func : str, default="local_score_BIC"
        causal-learn local score function.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Binary adjacency and diagnostics.
    """
    from causallearn.search.ScoreBased.GES import ges

    if data.ndim != 2:
        raise ValueError(f"GES expects 2D data, got shape={data.shape}")
    if data.shape[1] != len(columns):
        raise ValueError(
            f"GES data/column mismatch: data has {data.shape[1]} columns, "
            f"but {len(columns)} names were supplied"
        )

    max_p = int(max_parents if max_parents is not None else max(1, min(3, len(columns) - 1)))
    record = ges(
        X=np.asarray(data, dtype=float),
        score_func=score_func,
        maxP=max_p,
        node_names=columns,
    )
    graph = record["G"]
    oriented_graph = _orient_cpdag_to_dag(graph)
    adjacency = _causal_learn_graph_to_adj(oriented_graph)
    np.fill_diagonal(adjacency, 0)

    effective_forbidden = list(forbidden or [])
    effective_required = list(required or [])
    constraint_label = "unconstrained"
    handling = "none"
    postprocess = {
        "forbidden_edges_predicted_before": 0,
        "required_edges_missing_before": 0,
        "forbidden_edges_removed": 0,
        "required_edges_added": 0,
        "post_processing_changed": 0,
        "edge_count_before_postprocess": int(adjacency.sum()),
        "edge_count_after_postprocess": int(adjacency.sum()),
    }

    if mode == "constrained":
        constraint_label = (
            constraint_mode
            if constraint_mode in {"forbidden_only", "required_light", "full_reference_sanity"}
            else "forbidden_only"
        )
        handling = "postprocess"
        adjacency, postprocess = _apply_postprocess_constraints(
            adjacency=adjacency,
            columns=columns,
            forbidden=effective_forbidden,
            required=effective_required,
        )

    graph_path = _save_adjacency(
        adjacency=adjacency,
        columns=columns,
        dataset_name=dataset_name,
        constraint_label=constraint_label,
        seed=seed,
        output_dir=Path(output_dir),
    )

    metadata: dict[str, Any] = {
        "backend_used": "causallearn_ges",
        "constraint_handling": handling,
        "native_constraints_supported": False,
        "constraint_label": constraint_label,
        "graph_path": str(graph_path),
        "score": record.get("score", ""),
        "score_func": score_func,
        "max_parents": max_p,
        "cpdag_oriented_to_dag": True,
        "forbidden_constraints_used": len(effective_forbidden) if mode == "constrained" else 0,
        "required_constraints_used": len(effective_required) if mode == "constrained" else 0,
        **postprocess,
    }
    return adjacency, metadata
