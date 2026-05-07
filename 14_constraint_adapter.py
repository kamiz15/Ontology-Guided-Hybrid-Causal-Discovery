# 14_constraint_adapter.py
# ============================================================
# Step 14 - Constraint adapter for causal discovery algorithms.
#
# Converts shared (cause, effect) constraint lists into the
# algorithm-specific formats expected by causal-learn PC, gCastle NOTEARS,
# Causica DECI, and LiNGAM post-processing.
#
# Usage (standalone sanity check):
#   python 14_constraint_adapter.py
#
# Usage (imported by experiment runner):
#   import importlib
#   adapter = importlib.import_module("14_constraint_adapter")
#   adapter.build_gcastle_prior_matrix(...)
#
# ============================================================

from __future__ import annotations

import argparse
import importlib.util
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import (
    CAUSICA_CONSTRAINT_MATRIX_PATH,
    CONSTRAINT_ADAPTER_LOG_PATH,
    FORBIDDEN_EDGES_REAL_MODULE_PATH,
    FORBIDDEN_EDGES_SYNTHETIC_MODULE_PATH,
    READY_DATA_PATH,
    SYNTHETIC_DIR,
)


Constraint = tuple[str, str]


def build_causal_learn_bk(
    variable_names: list[str],
    forbidden: list[Constraint],
    required: list[Constraint],
    node_objects: list[Any],
) -> Any:
    """
    Build causal-learn background knowledge from shared constraints.

    Parameters
    ----------
    variable_names : list[str]
        Dataset variable names in the same order as ``node_objects``.
    forbidden : list[tuple[str, str]]
        Directed edges that must not exist, encoded as ``(cause, effect)``.
    required : list[tuple[str, str]]
        Directed edges that must exist, encoded as ``(cause, effect)``.
    node_objects : list[Any]
        causal-learn ``Node`` objects from the active ``CausalGraph``.

    Returns
    -------
    Any
        Populated causal-learn ``BackgroundKnowledge`` object.

    Raises
    ------
    ImportError
        If causal-learn is not installed.
    ValueError
        If ``node_objects`` does not align with ``variable_names`` or a
        constraint is both required and forbidden.
    """
    try:
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    except ImportError as exc:
        raise ImportError(
            "causal-learn is required to build PC BackgroundKnowledge. "
            "Install it with `pip install causal-learn`."
        ) from exc

    if len(node_objects) < len(variable_names):
        raise ValueError(
            "node_objects must contain at least one causal-learn Node for "
            "each variable name."
        )

    clean_forbidden, clean_required = _prepare_constraints(
        variable_names=variable_names,
        forbidden=forbidden,
        required=required,
        context="causal_learn",
        raise_on_conflict=True,
    )

    node_map = {
        variable: node_objects[index]
        for index, variable in enumerate(variable_names)
    }
    bk = BackgroundKnowledge()

    for source, target in clean_forbidden:
        _add_background_edge(
            bk=bk,
            method_candidates=["add_forbidden_by_node", "addForbiddenEdge"],
            source_node=node_map[source],
            target_node=node_map[target],
        )

    for source, target in clean_required:
        _add_background_edge(
            bk=bk,
            method_candidates=["add_required_by_node", "addRequiredEdge"],
            source_node=node_map[source],
            target_node=node_map[target],
        )

    print(
        f"[step_14] causal-learn BackgroundKnowledge: "
        f"{len(clean_forbidden)} forbidden, {len(clean_required)} required"
    )
    return bk


def build_gcastle_prior_matrix(
    variable_names: list[str],
    forbidden: list[Constraint],
    required: list[Constraint],
) -> np.ndarray:
    """
    Build a gCastle prior-knowledge matrix.

    Parameters
    ----------
    variable_names : list[str]
        Dataset variable names in matrix order.
    forbidden : list[tuple[str, str]]
        Directed edges that must not exist, encoded as ``(cause, effect)``.
    required : list[tuple[str, str]]
        Directed edges that must exist, encoded as ``(cause, effect)``.

    Returns
    -------
    np.ndarray
        Integer matrix where ``1`` means required, ``-1`` means forbidden,
        and ``0`` means unconstrained. Cell ``M[i, j]`` constrains
        ``variable_names[i] -> variable_names[j]``.

    Raises
    ------
    ValueError
        If the same directed pair is both required and forbidden.
    """
    clean_forbidden, clean_required = _prepare_constraints(
        variable_names=variable_names,
        forbidden=forbidden,
        required=required,
        context="gcastle",
        raise_on_conflict=True,
    )

    index = {name: i for i, name in enumerate(variable_names)}
    matrix = np.zeros((len(variable_names), len(variable_names)), dtype=int)

    for source, target in clean_forbidden:
        matrix[index[source], index[target]] = -1
    for source, target in clean_required:
        matrix[index[source], index[target]] = 1

    print(
        f"[step_14] gCastle prior matrix: shape={matrix.shape}, "
        f"forbidden={int(np.sum(matrix == -1))}, "
        f"required={int(np.sum(matrix == 1))}"
    )
    return matrix


def build_causica_constraint_matrix(
    variable_names: list[str],
    forbidden: list[Constraint],
    required: list[Constraint],
) -> Any:
    """
    Build a real Causica 0.4.x DECI constraint matrix.

    Causica 0.4.x ``DECIModule`` accepts ``constraint_matrix_path`` and loads
    a ``.npy`` matrix whose cells mean ``1 = required edge``,
    ``0 = forbidden edge``, and ``NaN = unconstrained``. This differs from
    the project fallback and gCastle convention, where ``-1`` marks a
    forbidden edge.

    Parameters
    ----------
    variable_names : list[str]
        Dataset variable names in matrix order.
    forbidden : list[tuple[str, str]]
        Directed edges that must not exist.
    required : list[tuple[str, str]]
        Directed edges that must exist.

    Returns
    -------
    Any
        float32 ``torch.Tensor`` of shape ``(N, N)`` using the real Causica
        0.4.x convention: ``1``, ``0``, and ``NaN``.

    Raises
    ------
    ImportError
        If Causica or PyTorch is not installed.
    NotImplementedError
        If an installed Causica version has unknown constraint semantics.
    """
    try:
        import pkg_resources

        causica_version = pkg_resources.get_distribution("causica").version
    except Exception as exc:
        raise ImportError(
            "Causica is not installed, so a DECI constraint matrix cannot be "
            "built for the real Causica API in this Python environment. "
            "In this project, Causica 0.4.5 may be installed in `.venv`, so "
            "try `.\\.venv\\Scripts\\python.exe 14_constraint_adapter.py`. "
            "For a fresh environment, install the project-pinned version, "
            "for example `pip install causica==0.4.5 --no-deps`."
        ) from exc

    if not causica_version.startswith("0.4."):
        raise NotImplementedError(
            f"Causica {causica_version} is installed, but this adapter only "
            "knows the DECI constraint semantics used by Causica 0.4.x."
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to build the Causica constraint tensor."
        ) from exc

    clean_forbidden, clean_required = _prepare_constraints(
        variable_names=variable_names,
        forbidden=forbidden,
        required=required,
        context="causica",
        raise_on_conflict=True,
    )

    index = {name: i for i, name in enumerate(variable_names)}
    matrix = np.full((len(variable_names), len(variable_names)), np.nan, dtype=np.float32)
    for source, target in clean_forbidden:
        matrix[index[source], index[target]] = 0.0
    for source, target in clean_required:
        matrix[index[source], index[target]] = 1.0
    np.fill_diagonal(matrix, np.nan)

    tensor = torch.tensor(matrix, dtype=torch.float32)
    print(
        f"[step_14] Causica {causica_version} constraint tensor: "
        f"shape={tuple(tensor.shape)}, "
        f"forbidden={int(np.sum(matrix == 0.0))}, "
        f"required={int(np.sum(matrix == 1.0))}, "
        f"unconstrained={int(np.isnan(matrix).sum())}"
    )
    return tensor


def write_causica_constraint_matrix_npy(
    variable_names: list[str],
    forbidden: list[Constraint],
    required: list[Constraint],
    output_path: str = CAUSICA_CONSTRAINT_MATRIX_PATH,
) -> str:
    """
    Write the real Causica 0.4.x constraint matrix to ``.npy``.

    ``DECIModule`` takes a ``constraint_matrix_path`` argument and, in Causica
    0.4.x, only loads ``.npy`` files for hard graph constraints. The saved
    matrix uses ``1 = required edge``, ``0 = forbidden edge``, and
    ``NaN = unconstrained``.

    Parameters
    ----------
    variable_names : list[str]
        Dataset variable names in matrix order.
    forbidden : list[tuple[str, str]]
        Directed edges that must not exist.
    required : list[tuple[str, str]]
        Directed edges that must exist.
    output_path : str
        Target ``.npy`` path.

    Returns
    -------
    str
        Path written.
    """
    tensor = build_causica_constraint_matrix(variable_names, forbidden, required)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, tensor.detach().cpu().numpy())
    print(f"[step_14] Causica constraint matrix -> {output_path}")
    return output_path


def apply_lingam_postprocess(
    adjacency_matrix: np.ndarray,
    variable_names: list[str],
    forbidden: list[Constraint],
) -> np.ndarray:
    """
    Zero out forbidden edges after an unconstrained LiNGAM run.

    LiNGAM does not inject constraints natively in this project. This function
    applies a weaker post-processing strategy: it leaves the unconstrained
    fitting untouched, then removes any forbidden directed edges from the
    discovered adjacency matrix. The matrix is assumed to use the project
    convention ``A[i, j] != 0`` means ``variable_names[i] -> variable_names[j]``.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Discovered LiNGAM adjacency matrix.
    variable_names : list[str]
        Dataset variable names in matrix order.
    forbidden : list[tuple[str, str]]
        Directed edges to remove.

    Returns
    -------
    np.ndarray
        Copy of ``adjacency_matrix`` with forbidden entries set to zero.
    """
    matrix = np.asarray(adjacency_matrix).copy()
    expected_shape = (len(variable_names), len(variable_names))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"adjacency_matrix shape {matrix.shape} does not match "
            f"variable_names length {len(variable_names)}."
        )

    clean_forbidden, _ = _prepare_constraints(
        variable_names=variable_names,
        forbidden=forbidden,
        required=[],
        context="lingam",
        raise_on_conflict=False,
    )

    index = {name: i for i, name in enumerate(variable_names)}
    removed = 0
    for source, target in clean_forbidden:
        i, j = index[source], index[target]
        if matrix[i, j] != 0:
            removed += 1
        matrix[i, j] = 0

    print(f"[step_14] LiNGAM post-process removed {removed} forbidden edges")
    return matrix


def load_constraints_from_files(
    forbidden_csv_path: str,
    required_csv_path: str,
) -> tuple[list[Constraint], list[Constraint]]:
    """
    Load approved constraint tuples from draft CSV files.

    Parameters
    ----------
    forbidden_csv_path : str
        Path to the forbidden-edge CSV with ``cause`` and ``effect`` columns.
    required_csv_path : str
        Path to the required-edge CSV with ``cause`` and ``effect`` columns.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
        Forbidden and required constraint lists. Rows with blank cause/effect
        are skipped. If an ``approved`` column is present, only rows where it
        equals ``"yes"`` are loaded.
    """
    forbidden = _read_constraint_csv(forbidden_csv_path, "forbidden")
    required = _read_constraint_csv(required_csv_path, "required")
    print(
        f"[step_14] Loaded from files: {len(forbidden)} forbidden, "
        f"{len(required)} required"
    )
    return forbidden, required


def load_constraints_for_dataset(dataset: str) -> tuple[list[Constraint], list[Constraint], list[str]]:
    """
    Load finalized constraints and variable names for a dataset target.

    Parameters
    ----------
    dataset : str
        Either ``"real"`` or ``"synthetic"``.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]
        Forbidden constraints, required constraints, and variable names.

    Raises
    ------
    ValueError
        If ``dataset`` is not ``"real"`` or ``"synthetic"``.
    FileNotFoundError
        If the dataset-specific module or variable-reference CSV is missing.
    """
    if dataset == "real":
        module_path = FORBIDDEN_EDGES_REAL_MODULE_PATH
        variable_path = READY_DATA_PATH
    elif dataset == "synthetic":
        module_path = FORBIDDEN_EDGES_SYNTHETIC_MODULE_PATH
        variable_path = os.path.join(SYNTHETIC_DIR, "synthetic_n2000.csv")
    else:
        raise ValueError("dataset must be 'real' or 'synthetic'")

    variables = _load_variable_names_from_csv(variable_path)
    forbidden, required = _load_constraints_from_module(module_path)
    print(
        f"[step_14] Dataset={dataset}: loaded {len(forbidden)} forbidden, "
        f"{len(required)} required from {module_path}"
    )
    return forbidden, required, variables


def _prepare_constraints(
    variable_names: list[str],
    forbidden: Iterable[Constraint],
    required: Iterable[Constraint],
    context: str,
    raise_on_conflict: bool,
) -> tuple[list[Constraint], list[Constraint]]:
    variable_set = set(variable_names)
    skipped: list[dict[str, Any]] = []

    clean_forbidden = _clean_constraint_list(
        constraints=forbidden,
        kind="forbidden",
        variable_set=variable_set,
        skipped=skipped,
    )
    clean_required = _clean_constraint_list(
        constraints=required,
        kind="required",
        variable_set=variable_set,
        skipped=skipped,
    )

    conflicts = sorted(set(clean_forbidden) & set(clean_required))
    if conflicts and raise_on_conflict:
        conflict_text = ", ".join(f"{source}->{target}" for source, target in conflicts)
        raise ValueError(f"Constraint conflict: pair is both required and forbidden: {conflict_text}")

    skipped_not_in_vars = sum(1 for item in skipped if item["reason"] == "not_in_vars")
    print(
        f"[step_14] Loaded {len(clean_forbidden)} forbidden, "
        f"{len(clean_required)} required, {skipped_not_in_vars} skipped (not in vars)"
    )
    _write_skipped_log(skipped, context)
    return clean_forbidden, clean_required


def _clean_constraint_list(
    constraints: Iterable[Constraint],
    kind: str,
    variable_set: set[str],
    skipped: list[dict[str, Any]],
) -> list[Constraint]:
    clean: list[Constraint] = []
    seen: set[Constraint] = set()

    for raw_source, raw_target in constraints:
        source = str(raw_source).strip()
        target = str(raw_target).strip()

        if not source or not target:
            continue
        if source == target:
            continue
        if source not in variable_set or target not in variable_set:
            skipped.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "kind": kind,
                "cause": source,
                "effect": target,
                "reason": "not_in_vars",
            })
            print(f"[step_14] Skipped {kind}: {source} -> {target} (not in vars)")
            continue

        pair = (source, target)
        if pair in seen:
            continue
        seen.add(pair)
        clean.append(pair)

    return clean


def _read_constraint_csv(path: str, kind: str) -> list[Constraint]:
    if not os.path.exists(path):
        print(f"[step_14] Constraint file missing for {kind}: {path}")
        return []

    df = pd.read_csv(path)
    if {"cause", "effect"}.issubset(df.columns):
        source_col, target_col = "cause", "effect"
    elif {"source", "target"}.issubset(df.columns):
        source_col, target_col = "source", "target"
    else:
        raise ValueError(
            f"{path} is missing required columns: expected cause/effect "
            "or source/target"
        )

    required_columns = {source_col, target_col}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if "approved" in df.columns:
        approved = df["approved"].fillna("").astype(str).str.strip().str.lower()
        df = df[approved == "yes"].copy()

    pairs: list[Constraint] = []
    for _, row in df.iterrows():
        source = "" if pd.isna(row[source_col]) else str(row[source_col]).strip()
        target = "" if pd.isna(row[target_col]) else str(row[target_col]).strip()
        if not source or not target:
            continue
        pairs.append((source, target))

    return pairs


def _load_constraints_from_module(path: str) -> tuple[list[Constraint], list[Constraint]]:
    module_path = Path(path)
    if not module_path.is_absolute():
        module_path = Path(__file__).resolve().parent / module_path
    if not module_path.exists():
        raise FileNotFoundError(f"Constraint module not found: {module_path}")

    module_name = "constraint_module_" + module_path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load constraint module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forbidden = list(getattr(module, "FORBIDDEN_EDGES", []))
    required = list(getattr(module, "REQUIRED_EDGES", []))
    return forbidden, required


def _load_variable_names_from_csv(path: str) -> list[str]:
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Variable reference CSV not found: {csv_path}")
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def _add_background_edge(
    bk: Any,
    method_candidates: list[str],
    source_node: Any,
    target_node: Any,
) -> None:
    for method_name in method_candidates:
        method = getattr(bk, method_name, None)
        if method is not None:
            method(source_node, target_node)
            return
    raise AttributeError(
        "BackgroundKnowledge object does not expose any supported edge method: "
        f"{method_candidates}"
    )


def _write_skipped_log(skipped: list[dict[str, Any]], context: str) -> None:
    if not skipped:
        return

    rows = []
    for item in skipped:
        row = dict(item)
        row["context"] = context
        rows.append(row)

    os.makedirs(os.path.dirname(CONSTRAINT_ADAPTER_LOG_PATH), exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.exists(CONSTRAINT_ADAPTER_LOG_PATH):
        old_df = pd.read_csv(CONSTRAINT_ADAPTER_LOG_PATH)
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(CONSTRAINT_ADAPTER_LOG_PATH, index=False)
    print(f"[step_14] Skipped-constraint log -> {CONSTRAINT_ADAPTER_LOG_PATH}")


def _standalone_check(dataset: str) -> None:
    print(f"[step_14] Standalone constraint adapter sanity check ({dataset})")
    forbidden, required, variable_names = load_constraints_for_dataset(dataset)
    print(f"[step_14] {dataset} schema variables: {len(variable_names)}")

    try:
        from causallearn.graph.GraphNode import GraphNode

        nodes = [GraphNode(name) for name in variable_names]
        bk = build_causal_learn_bk(variable_names, forbidden, required, nodes)
        print(f"[step_14] causal-learn BK built: {type(bk).__name__}")
    except ImportError as exc:
        print(f"[step_14] causal-learn BK skipped: {exc}")

    prior = build_gcastle_prior_matrix(variable_names, forbidden, required)
    print(
        f"[step_14] gCastle sanity stats: "
        f"{int(np.sum(prior == -1))} forbidden, {int(np.sum(prior == 1))} required"
    )

    try:
        causica_matrix = build_causica_constraint_matrix(variable_names, forbidden, required)
        shape = tuple(causica_matrix.shape)
        print(f"[step_14] Causica sanity stats: shape={shape}")
        write_causica_constraint_matrix_npy(variable_names, forbidden, required)
    except ImportError as exc:
        print(f"[step_14] Causica skipped: {exc}")

    dummy_lingam = np.ones((len(variable_names), len(variable_names)), dtype=int)
    np.fill_diagonal(dummy_lingam, 0)
    post = apply_lingam_postprocess(dummy_lingam, variable_names, forbidden)
    print(
        f"[step_14] LiNGAM postprocess sanity stats: "
        f"{int(np.count_nonzero(dummy_lingam) - np.count_nonzero(post))} edges removed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check constraint adapter formats."
    )
    parser.add_argument("--dataset", choices=["real", "synthetic", "both"],
                        default="synthetic",
                        help="Dataset-specific finalized constraints to check")
    args = parser.parse_args()

    datasets = ["real", "synthetic"] if args.dataset == "both" else [args.dataset]
    for dataset_name in datasets:
        _standalone_check(dataset_name)
