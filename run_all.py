# run_all.py
# ============================================================
# Full experiment runner for constrained causal discovery.
#
# Runs PC, NOTEARS, LiNGAM, and guarded DECI/Causica-style runs on
# synthetic and real ESG datasets, with bootstrap resampling per seed.
#
# Outputs:
#   outputs/experiments/results.csv
#   outputs/experiments/results_summary.csv
#   outputs/experiments/failures.csv
#
# Usage:
#   python run_all.py
#   python run_all.py --skip-deci
#   python run_all.py --datasets synthetic_n2000,real
# ============================================================

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as project_config


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "outputs" / "experiments"
RESULTS_PATH = RESULTS_DIR / "results.csv"
SUMMARY_PATH = RESULTS_DIR / "results_summary.csv"
FAILURES_PATH = RESULTS_DIR / "failures.csv"
DECI_DIAGNOSTICS_PATH = RESULTS_DIR / "deci_diagnostics.csv"
DECI_THRESHOLD_SWEEP_PATH = RESULTS_DIR / "deci_threshold_sweep.csv"
DECI_STABLE_EDGES_PATH = RESULTS_DIR / "deci_stable_edges.csv"
DECI_WORK_DIR = RESULTS_DIR / "deci_work"
DECI_TIMEOUT_SECONDS = 300
DECI_GLOBAL_ABORT_SECONDS = 1800
NOTEARS_NOTE_PRINTED = False
CURRENT_DECI_THRESHOLD = float(getattr(project_config, "DECI_THRESHOLD", 0.275))
DECI_RUN_CACHE: dict[tuple[str, str, int], dict[str, Any]] = {}

DATASETS = {
    "synthetic_n2000": {
        "path": ROOT / "data" / "synthetic" / "synthetic_n2000.csv",
        "ground_truth": ROOT / "data" / "synthetic" / "ground_truth_adjacency.csv",
        "constraint_dataset": "synthetic",
        "has_ground_truth": True,
    },
    "real": {
        "path": ROOT / "data" / "processed" / "data_ready.csv",
        "ground_truth": None,
        "constraint_dataset": "real",
        "has_ground_truth": False,
    },
}

RESULT_COLUMNS = [
    "algorithm",
    "mode",
    "dataset",
    "seed",
    "runtime_seconds",
    "status",
    "edge_count_predicted",
    "edge_count_true",
    "shd",
    "f1_directed",
    "f1_skeleton",
    "precision",
    "recall",
    "literature_agreement_count",
    "literature_violation_count",
    "literature_alignment_score",
]
FAILURE_COLUMNS = ["algorithm", "mode", "dataset", "seed", "error", "traceback"]
NUMERIC_SUMMARY_COLUMNS = [
    "runtime_seconds",
    "edge_count_predicted",
    "edge_count_true",
    "shd",
    "f1_directed",
    "f1_skeleton",
    "precision",
    "recall",
    "literature_agreement_count",
    "literature_violation_count",
    "literature_alignment_score",
]

DECI_DIAGNOSTIC_COLUMNS = [
    "algorithm",
    "dataset",
    "mode",
    "seed",
    "n_samples",
    "n_variables",
    "training_status",
    "runtime_seconds",
    "device",
    "preset",
    "backend_used",
    "causica_compat_status",
    "causica_error",
    "threshold_mode",
    "threshold_used",
    "native_constraints_supported",
    "constraint_handling",
    "small_data_warning",
    "raw_adjacency_shape",
    "raw_min",
    "raw_max",
    "raw_mean",
    "raw_std",
    "abs_q50",
    "abs_q75",
    "abs_q90",
    "abs_q95",
    "abs_q99",
    "weighted_nonzero_edges",
    "weighted_near_nonzero_edges",
    "edges_after_threshold",
    "forbidden_edges_predicted_before_enforcement",
    "required_edges_missing_before_enforcement",
    "forbidden_edges_removed",
    "required_edges_added",
    "constraint_cells_changed",
    "edges_after_constraint_enforcement",
    "diagnostic_message",
    "shd",
    "f1_directed",
    "f1_skeleton",
    "precision",
    "recall",
    "literature_agreement_count",
    "literature_violation_count",
    "literature_alignment_score",
    "raw_weights_path",
    "final_adjacency_path",
]

DECI_THRESHOLD_SWEEP_COLUMNS = [
    "dataset",
    "mode",
    "seed",
    "threshold",
    "edge_count",
    "edge_count_true",
    "f1_directed",
    "precision",
    "recall",
    "shd",
    "violations",
    "constraint_cells_changed",
    "selected",
]

DECI_STABLE_EDGE_COLUMNS = [
    "dataset",
    "mode",
    "source",
    "target",
    "frequency",
    "mean_weight",
    "sign_if_available",
    "passes_60_percent",
    "passes_80_percent",
    "forbidden_edge",
    "required_edge",
]

DECI_PRESETS = {
    "fast_debug": {
        "max_epochs": 5,
        "learning_rate": 3e-3,
        "batch_size_cap": 64,
        "hidden_dim": 16,
        "l1_lambda": 0.1,
        "device": "cpu",
        "timeout_seconds": 90,
    },
    "small_data": {
        "max_epochs": 20,
        "learning_rate": 1e-3,
        "batch_size_cap": 64,
        "hidden_dim": 16,
        "l1_lambda": 0.05,
        "device": "cpu",
        "timeout_seconds": 180,
    },
    "default": {
        "max_epochs": 50,
        "learning_rate": 3e-3,
        "batch_size_cap": 128,
        "hidden_dim": 32,
        "l1_lambda": 0.1,
        "device": "cpu",
        "timeout_seconds": 300,
    },
}


def log(message: str) -> None:
    """Print a runner-prefixed log line."""
    print(f"[run_all] {message}", flush=True)


def parse_csv_arg(value: str) -> list[str]:
    """Parse a comma-separated CLI option."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_arg(value: str) -> list[int]:
    """Parse comma-separated seeds."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ensure_outputs() -> None:
    """
    Create output directory and reset output CSVs.

    Returns
    -------
    None
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writeheader()
    with FAILURES_PATH.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS).writeheader()
    with DECI_DIAGNOSTICS_PATH.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DECI_DIAGNOSTIC_COLUMNS).writeheader()
    with DECI_THRESHOLD_SWEEP_PATH.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DECI_THRESHOLD_SWEEP_COLUMNS).writeheader()
    with DECI_STABLE_EDGES_PATH.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DECI_STABLE_EDGE_COLUMNS).writeheader()


def append_result(row: dict[str, Any]) -> None:
    """
    Append one result row immediately.

    Parameters
    ----------
    row : dict[str, Any]
        Result values.

    Returns
    -------
    None
    """
    full_row = {column: row.get(column, "") for column in RESULT_COLUMNS}
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writerow(full_row)


def append_failure(
    algorithm: str,
    mode: str,
    dataset: str,
    seed: int,
    exc: BaseException,
) -> None:
    """
    Append a detailed failure row.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    mode : str
        Experiment mode.
    dataset : str
        Dataset label.
    seed : int
        Seed.
    exc : BaseException
        Exception caught.

    Returns
    -------
    None
    """
    with FAILURES_PATH.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS).writerow({
            "algorithm": algorithm,
            "mode": mode,
            "dataset": dataset,
            "seed": seed,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def append_csv_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    """
    Append a row to a CSV with a fixed schema.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV.
    columns : list[str]
        Ordered output columns.
    row : dict[str, Any]
        Row values.

    Returns
    -------
    None
    """
    full_row = {column: row.get(column, "") for column in columns}
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writerow(full_row)


def load_module_from_path(module_name: str, path: Path) -> Any:
    """
    Import a Python module from a path.

    Parameters
    ----------
    module_name : str
        Import name to assign.
    path : pathlib.Path
        Module file path.

    Returns
    -------
    Any
        Loaded module.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter() -> Any:
    """Load ``14_constraint_adapter.py``."""
    return load_module_from_path("constraint_adapter_step_14", ROOT / "14_constraint_adapter.py")


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """
    Load a configured dataset and optional synthetic ground truth.

    Parameters
    ----------
    dataset_name : str
        Dataset key.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray | None, list[str]]
        Complete numeric data, aligned ground-truth adjacency if available,
        and variable names.
    """
    config = DATASETS[dataset_name]
    df = pd.read_csv(config["path"])
    df = df.select_dtypes(include="number").dropna().copy()

    true_adj: np.ndarray | None = None
    if config["has_ground_truth"]:
        true_df = pd.read_csv(config["ground_truth"], index_col=0)
        common = [
            column for column in df.columns
            if column in true_df.index and column in true_df.columns
        ]
        df = df[common].copy()
        true_adj = (true_df.loc[common, common].to_numpy() == 1).astype(int)
        np.fill_diagonal(true_adj, 0)
    columns = df.columns.tolist()
    log(f"Loaded {dataset_name}: rows={len(df)}, columns={len(columns)}")
    return df, true_adj, columns


def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Z-score data columns.

    Parameters
    ----------
    data : np.ndarray
        Raw numeric data.

    Returns
    -------
    np.ndarray
        Standardized data.
    """
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std


def bootstrap_data(df: pd.DataFrame, seed: int) -> np.ndarray:
    """
    Bootstrap-sample 80 percent of rows with replacement.

    Parameters
    ----------
    df : pd.DataFrame
        Complete numeric dataset.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Standardized bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    sample_size = max(2, int(0.8 * n_rows))
    indices = rng.choice(n_rows, sample_size, replace=True)
    data = df.iloc[indices].to_numpy(dtype=float)
    return standardize_data(data)


def load_constraints(dataset_name: str, adapter: Any) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Load finalized constraints for one dataset target.

    Parameters
    ----------
    dataset_name : str
        Experiment dataset name.
    adapter : Any
        Constraint adapter module.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
        Forbidden and required constraints.
    """
    constraint_dataset = DATASETS[dataset_name]["constraint_dataset"]
    forbidden, required, _ = adapter.load_constraints_for_dataset(constraint_dataset)
    return forbidden, required


def get_deci_preset() -> dict[str, Any]:
    """
    Return the configured DECI hyperparameter preset.

    Returns
    -------
    dict[str, Any]
        Preset values.
    """
    preset_name = str(getattr(project_config, "DECI_PRESET", "small_data"))
    if preset_name not in DECI_PRESETS:
        raise ValueError(f"Unknown DECI_PRESET={preset_name!r}")
    preset = dict(DECI_PRESETS[preset_name])
    preset["name"] = preset_name
    return preset


def small_data_warning(n_samples: int, n_variables: int) -> str:
    """
    Build a DECI small-data warning message.

    Parameters
    ----------
    n_samples : int
        Number of rows.
    n_variables : int
        Number of variables.

    Returns
    -------
    str
        Warning message or empty string.
    """
    threshold = int(getattr(project_config, "MIN_SAMPLES_PER_VARIABLE_WARNING", 10))
    if n_variables <= 0:
        return ""
    ratio = n_samples / n_variables
    if ratio < 5:
        return (
            f"strong_warning: n_samples/n_variables={ratio:.2f} < 5; "
            "DECI is exploratory and unstable for this dataset"
        )
    if ratio < threshold:
        return (
            f"warning: n_samples/n_variables={ratio:.2f} < {threshold}; "
            "DECI is exploratory for this small dataset"
        )
    return ""


def threshold_weight_matrix(
    weights: np.ndarray,
    mode: str,
    fixed_threshold: float,
    percentile: float,
    topk: int | None,
) -> tuple[np.ndarray, float]:
    """
    Threshold a weighted DECI adjacency matrix.

    Parameters
    ----------
    weights : np.ndarray
        Weighted adjacency matrix.
    mode : str
        fixed, percentile, or topk.
    fixed_threshold : float
        Numeric fixed threshold.
    percentile : float
        Percentile threshold for diagnostic mode.
    topk : int or None
        Number of strongest edges to keep for top-k mode.

    Returns
    -------
    tuple[np.ndarray, float]
        Binary adjacency and effective numeric threshold.
    """
    values = np.abs(np.asarray(weights, dtype=float)).copy()
    np.fill_diagonal(values, 0.0)
    non_diag = values[~np.eye(values.shape[0], dtype=bool)]

    if mode == "fixed":
        threshold = float(fixed_threshold)
        adjacency = (values > threshold).astype(int)
    elif mode == "percentile":
        threshold = float(np.percentile(non_diag, percentile))
        adjacency = (values > threshold).astype(int)
    elif mode == "topk":
        k = int(topk or 0)
        threshold = float("inf")
        adjacency = np.zeros_like(values, dtype=int)
        if k > 0:
            flat = values.ravel()
            valid = np.where(~np.eye(values.shape[0], dtype=bool).ravel())[0]
            k = min(k, len(valid))
            chosen = valid[np.argsort(flat[valid])[-k:]]
            threshold = float(flat[chosen].min()) if len(chosen) else float("inf")
            adjacency.ravel()[chosen] = 1
    else:
        raise ValueError(f"Unsupported DECI_THRESHOLD_MODE={mode!r}")

    np.fill_diagonal(adjacency, 0)
    return adjacency, threshold


def count_pairs(adjacency: np.ndarray, pairs: list[tuple[str, str]], columns: list[str]) -> int:
    """
    Count how many directed pairs are present in an adjacency matrix.

    Parameters
    ----------
    adjacency : np.ndarray
        Binary adjacency.
    pairs : list[tuple[str, str]]
        Directed pairs.
    columns : list[str]
        Matrix column order.

    Returns
    -------
    int
        Count present.
    """
    index = {name: i for i, name in enumerate(columns)}
    count = 0
    for source, target in pairs:
        if source in index and target in index and adjacency[index[source], index[target]]:
            count += 1
    return count


def enforce_deci_constraints(
    adjacency: np.ndarray,
    columns: list[str],
    forbidden: list[tuple[str, str]],
    required: list[tuple[str, str]],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Apply post-processing constraints to a DECI binary adjacency.

    Parameters
    ----------
    adjacency : np.ndarray
        Thresholded adjacency before enforcement.
    columns : list[str]
        Variable order.
    forbidden : list[tuple[str, str]]
        Forbidden edges.
    required : list[tuple[str, str]]
        Required edges.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        Final adjacency and enforcement counters.
    """
    index = {name: i for i, name in enumerate(columns)}
    final = np.asarray(adjacency, dtype=int).copy()
    before = final.copy()

    forbidden_removed = 0
    required_added = 0
    for source, target in forbidden:
        if source in index and target in index:
            i, j = index[source], index[target]
            if final[i, j]:
                forbidden_removed += 1
            final[i, j] = 0
    for source, target in required:
        if source in index and target in index:
            i, j = index[source], index[target]
            if not final[i, j]:
                required_added += 1
            final[i, j] = 1
            final[j, i] = 0

    np.fill_diagonal(final, 0)
    return final, {
        "forbidden_removed": forbidden_removed,
        "required_added": required_added,
        "constraint_cells_changed": int(np.sum(final != before)),
        "edges_after_constraint_enforcement": int(final.sum()),
    }


def causallearn_to_directed_adj(cg: Any) -> np.ndarray:
    """
    Convert causal-learn PC output to binary directed adjacency.

    Parameters
    ----------
    cg : Any
        causal-learn CausalGraph.

    Returns
    -------
    np.ndarray
        Binary adjacency where A[i, j] means i -> j.
    """
    graph = np.asarray(cg.G.graph)
    n_vars = graph.shape[0]
    directed = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars):
        for j in range(n_vars):
            if graph[i, j] == -1 and graph[j, i] == 1:
                directed[i, j] = 1
    np.fill_diagonal(directed, 0)
    return directed


def run_pc(
    data: np.ndarray,
    columns: list[str],
    mode: str,
    dataset_name: str,
    adapter: Any,
) -> np.ndarray:
    """
    Run causal-learn PC.

    Parameters
    ----------
    data : np.ndarray
        Standardized bootstrap sample.
    columns : list[str]
        Variable names.
    mode : str
        unconstrained or constrained.
    dataset_name : str
        Dataset label.
    adapter : Any
        Constraint adapter module.

    Returns
    -------
    np.ndarray
        Directed binary adjacency.
    """
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc

    background_knowledge = None
    if mode == "constrained":
        forbidden, required = load_constraints(dataset_name, adapter)
        node_objects = [GraphNode(name) for name in columns]
        background_knowledge = adapter.build_causal_learn_bk(
            variable_names=columns,
            forbidden=forbidden,
            required=required,
            node_objects=node_objects,
        )

    cg = pc(
        data,
        alpha=0.05,
        indep_test="fisherz",
        stable=True,
        background_knowledge=background_knowledge,
        verbose=False,
        show_progress=False,
        node_names=columns,
    )
    return causallearn_to_directed_adj(cg)


def run_notears(
    data: np.ndarray,
    columns: list[str],
    mode: str,
    dataset_name: str,
    adapter: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Run gCastle NOTEARS.

    Parameters
    ----------
    data : np.ndarray
        Standardized bootstrap sample.
    columns : list[str]
        Variable names.
    mode : str
        unconstrained or constrained.
    dataset_name : str
        Dataset label.
    adapter : Any
        Constraint adapter module.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        Directed binary adjacency thresholded at 0.3 and metadata containing
        the number of post-processing cell changes.
    """
    from castle.algorithms import Notears

    global NOTEARS_NOTE_PRINTED
    if not NOTEARS_NOTE_PRINTED:
        log(
            "NOTE: NOTEARS uses post-processing for constraints because "
            "gCastle's Notears.learn() does not accept prior_knowledge. "
            "Constrained variant applies forbidden/required edges to the "
            "discovered weighted matrix after thresholding."
        )
        NOTEARS_NOTE_PRINTED = True

    model = Notears(lambda1=0.1, loss_type="l2", w_threshold=0.3)
    model.learn(data, columns=columns)

    raw_adj = getattr(model, "weight_causal_matrix", None)
    if raw_adj is None:
        raw_adj = model.causal_matrix
    raw_adj = np.asarray(raw_adj, dtype=float)
    directed = (np.abs(raw_adj) > 0.3).astype(int)
    np.fill_diagonal(directed, 0)

    post_processing_changed = 0
    if mode == "constrained":
        forbidden, required = load_constraints(dataset_name, adapter)
        prior = adapter.build_gcastle_prior_matrix(columns, forbidden, required)
        before = directed.copy()

        directed[prior == -1] = 0
        for source_idx, target_idx in np.argwhere(prior == 1):
            directed[source_idx, target_idx] = 1
            directed[target_idx, source_idx] = 0

        np.fill_diagonal(directed, 0)
        post_processing_changed = int(np.sum(directed != before))

    return directed, {"post_processing_changed": post_processing_changed}


def run_lingam(
    data: np.ndarray,
    columns: list[str],
    mode: str,
    dataset_name: str,
    adapter: Any,
) -> np.ndarray:
    """
    Run DirectLiNGAM, with constrained mode as post-processing.

    Parameters
    ----------
    data : np.ndarray
        Standardized bootstrap sample.
    columns : list[str]
        Variable names.
    mode : str
        unconstrained or constrained.
    dataset_name : str
        Dataset label.
    adapter : Any
        Constraint adapter module.

    Returns
    -------
    np.ndarray
        Directed binary adjacency.
    """
    import lingam

    model = lingam.DirectLiNGAM()
    model.fit(data)
    # lingam adjacency_matrix_[effect, cause] means cause -> effect.
    raw_adj = np.asarray(model.adjacency_matrix_, dtype=float).T
    directed = (np.abs(raw_adj) > 0.01).astype(int)
    np.fill_diagonal(directed, 0)

    if mode == "constrained":
        forbidden, _ = load_constraints(dataset_name, adapter)
        directed = adapter.apply_lingam_postprocess(directed, columns, forbidden)
        directed = (np.asarray(directed) != 0).astype(int)
        np.fill_diagonal(directed, 0)

    return directed


def build_deci_constraint_matrix(
    columns: list[str],
    forbidden: list[tuple[str, str]],
    required: list[tuple[str, str]],
) -> np.ndarray:
    """
    Build the -1/0/1 matrix expected by the existing DECI helper.

    Parameters
    ----------
    columns : list[str]
        Variable names.
    forbidden : list[tuple[str, str]]
        Forbidden edges.
    required : list[tuple[str, str]]
        Required edges.

    Returns
    -------
    np.ndarray
        Matrix where -1 = forbidden, 1 = required, 0 = unconstrained.
    """
    index = {name: i for i, name in enumerate(columns)}
    matrix = np.zeros((len(columns), len(columns)), dtype=np.float32)
    for source, target in forbidden:
        if source in index and target in index:
            matrix[index[source], index[target]] = -1.0
    for source, target in required:
        if source in index and target in index:
            matrix[index[source], index[target]] = 1.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def deci_subprocess_entry(input_path: str, output_path: str) -> int:
    """
    Run one DECI job from serialized files.

    Parameters
    ----------
    input_path : str
        ``.npz`` payload path.
    output_path : str
        JSON result path.

    Returns
    -------
    int
        Process exit code.
    """
    try:
        payload = np.load(input_path, allow_pickle=True)
        data = np.asarray(payload["data"], dtype=float)
        columns = [str(item) for item in payload["columns"].tolist()]
        has_constraint = bool(payload["has_constraint"][0])
        constraint_matrix = (
            np.asarray(payload["constraint_matrix"], dtype=np.float32)
            if has_constraint
            else None
        )
        run_name = str(payload["run_name"].item())
        adjacency_path = Path(str(payload["adjacency_path"].item()))
        log_path = str(payload["log_path"].item())
        max_epochs = int(payload["max_epochs"][0])
        learning_rate = float(payload["learning_rate"][0])
        batch_size_cap = int(payload["batch_size_cap"][0])
        device = str(payload["device"].item())
        hidden_dim = int(payload["hidden_dim"][0])
        l1_lambda = float(payload["l1_lambda"][0])
        seed = int(payload["seed"][0])
        backend = str(payload["backend"].item())
        allow_manual_fallback = bool(payload["allow_manual_fallback"][0])

        np.random.seed(seed)

        (ROOT / "outputs" / "graphs").mkdir(parents=True, exist_ok=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        deci_module = load_module_from_path("step_07_run_deci", ROOT / "07_run_deci.py")
        batch_size = max(8, min(batch_size_cap, data.shape[0] // 2))
        adj = deci_module.run_deci(
            data=data,
            columns=columns,
            constraint_matrix=constraint_matrix,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            edge_threshold=0.5,
            run_name=run_name,
            log_path=log_path,
            hidden_dim=hidden_dim,
            l1_lambda=l1_lambda,
            seed=seed,
            backend=backend,
            allow_manual_fallback=allow_manual_fallback,
        )
        np.save(adjacency_path, np.asarray(adj, dtype=int))
        result = {
            "status": "success",
            "adjacency_path": str(adjacency_path),
            "backend_metadata": getattr(deci_module.run_deci, "last_metadata", {}),
        }
    except BaseException as exc:
        result = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    Path(output_path).write_text(json.dumps(result), encoding="utf-8")
    return 0


def run_deci_guarded(
    data: np.ndarray,
    columns: list[str],
    mode: str,
    dataset_name: str,
    seed: int,
    adapter: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Run the existing DECI helper in a timed subprocess.

    Parameters
    ----------
    data : np.ndarray
        Standardized bootstrap sample.
    columns : list[str]
        Variable names.
    mode : str
        unconstrained or constrained.
    dataset_name : str
        Dataset label.
    seed : int
        Seed.
    adapter : Any
        Constraint adapter module.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Directed binary adjacency and diagnostic metadata.
    """
    cache_key = (dataset_name, mode, seed)
    if cache_key in DECI_RUN_CACHE:
        training = DECI_RUN_CACHE[cache_key]
    else:
        training = train_deci_guarded(data, columns, mode, dataset_name, seed, adapter)
        DECI_RUN_CACHE[cache_key] = training

    forbidden, required = load_constraints(dataset_name, adapter)
    threshold_mode = str(getattr(project_config, "DECI_THRESHOLD_MODE", "fixed"))
    percentile = float(getattr(project_config, "DECI_THRESHOLD_PERCENTILE", 95.0))
    topk = getattr(project_config, "DECI_TOPK_EDGES", None)
    threshold_value = float(CURRENT_DECI_THRESHOLD)

    raw_weights = np.asarray(training["raw_weights"], dtype=float)
    thresholded, effective_threshold = threshold_weight_matrix(
        raw_weights,
        threshold_mode,
        threshold_value,
        percentile,
        topk,
    )

    forbidden_before = count_pairs(thresholded, forbidden, columns)
    required_present_before = count_pairs(thresholded, required, columns)
    required_missing_before = max(0, len([
        pair for pair in required
        if pair[0] in columns and pair[1] in columns
    ]) - required_present_before)

    final = thresholded.copy()
    enforcement = {
        "forbidden_removed": 0,
        "required_added": 0,
        "constraint_cells_changed": 0,
        "edges_after_constraint_enforcement": int(final.sum()),
    }
    if mode == "constrained":
        final, enforcement = enforce_deci_constraints(thresholded, columns, forbidden, required)

    run_dir = Path(training["run_dir"])
    final_adjacency_path = run_dir / "final_adjacency_thresholded.npy"
    np.save(final_adjacency_path, final.astype(int))

    metadata = build_deci_metadata(
        training=training,
        raw_weights=raw_weights,
        thresholded=thresholded,
        final=final,
        effective_threshold=effective_threshold,
        forbidden_before=forbidden_before,
        required_missing_before=required_missing_before,
        enforcement=enforcement,
        final_adjacency_path=final_adjacency_path,
    )
    print_deci_run_diagnostic(metadata)
    return final, metadata


def train_deci_guarded(
    data: np.ndarray,
    columns: list[str],
    mode: str,
    dataset_name: str,
    seed: int,
    adapter: Any,
) -> dict[str, Any]:
    """
    Train DECI in a Windows-safe subprocess and return raw weights.

    Parameters
    ----------
    data : np.ndarray
        Standardized bootstrap sample.
    columns : list[str]
        Variable names.
    mode : str
        unconstrained or constrained.
    dataset_name : str
        Dataset label.
    seed : int
        Seed.
    adapter : Any
        Constraint adapter module.

    Returns
    -------
    dict[str, Any]
        Raw DECI artifacts and run metadata.
    """
    preset = get_deci_preset()
    constraint_matrix = None
    if mode == "constrained":
        forbidden, required = load_constraints(dataset_name, adapter)
        # Build the real adapter tensor for API verification/documentation,
        # then use the existing Step 07 helper's -1/0/1 convention.
        adapter.build_causica_constraint_matrix(columns, forbidden, required)
        constraint_matrix = build_deci_constraint_matrix(columns, forbidden, required)

    run_name = f"deci_{dataset_name}_{mode}_seed{seed}"

    safe_run_name = run_name.replace("/", "_").replace("\\", "_")
    run_dir = DECI_WORK_DIR / safe_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.npz"
    output_path = run_dir / "result.json"
    adjacency_path = run_dir / "helper_adjacency.npy"
    log_path = ROOT / "outputs" / "metrics" / "deci_logs" / f"{safe_run_name}_run_log.csv"

    np.savez_compressed(
        input_path,
        data=data,
        columns=np.asarray(columns, dtype=object),
        has_constraint=np.asarray([constraint_matrix is not None], dtype=bool),
        constraint_matrix=(
            constraint_matrix
            if constraint_matrix is not None
            else np.zeros((0, 0), dtype=np.float32)
        ),
        run_name=np.asarray(run_name),
        adjacency_path=np.asarray(str(adjacency_path)),
        log_path=np.asarray(str(log_path)),
        max_epochs=np.asarray([preset["max_epochs"]], dtype=int),
        learning_rate=np.asarray([preset["learning_rate"]], dtype=float),
        batch_size_cap=np.asarray([preset["batch_size_cap"]], dtype=int),
        device=np.asarray(preset["device"]),
        hidden_dim=np.asarray([preset["hidden_dim"]], dtype=int),
        l1_lambda=np.asarray([preset["l1_lambda"]], dtype=float),
        seed=np.asarray([seed], dtype=int),
        backend=np.asarray(str(getattr(project_config, "DECI_BACKEND", "causica"))),
        allow_manual_fallback=np.asarray([
            bool(getattr(project_config, "DECI_ALLOW_MANUAL_FALLBACK", True))
        ], dtype=bool),
    )

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--deci-worker",
        str(input_path),
        str(output_path),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "ESG_DECI_BACKEND": str(getattr(project_config, "DECI_BACKEND", "causica")),
                "ESG_DECI_ALLOW_MANUAL_FALLBACK": (
                    "1" if bool(getattr(project_config, "DECI_ALLOW_MANUAL_FALLBACK", True)) else "0"
                ),
            },
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=int(preset.get("timeout_seconds", DECI_TIMEOUT_SECONDS)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"DECI exceeded {preset.get('timeout_seconds', DECI_TIMEOUT_SECONDS)}s timeout"
        ) from exc
    elapsed = time.perf_counter() - started

    if not output_path.exists():
        stderr_tail = (completed.stderr or "").strip()[-1000:]
        stdout_tail = (completed.stdout or "").strip()[-1000:]
        detail = stderr_tail or stdout_tail or f"exit code {completed.returncode}"
        raise RuntimeError(f"DECI worker produced no result: {detail}")

    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result["status"] != "success":
        raise RuntimeError(result.get("error", "DECI failed"))
    backend_metadata = result.get("backend_metadata", {}) or {}

    raw_weights_path = ROOT / "outputs" / "graphs" / f"{run_name}_raw_edge_probabilities.csv"
    final_weights_path = ROOT / "outputs" / "graphs" / f"{run_name}_edge_probabilities.csv"
    if not raw_weights_path.exists():
        raise RuntimeError(f"DECI raw probability output missing: {raw_weights_path}")
    raw_weights = pd.read_csv(raw_weights_path, index_col=0).loc[columns, columns].to_numpy(dtype=float)
    helper_adjacency = (np.load(result["adjacency_path"]) != 0).astype(int)
    np.fill_diagonal(helper_adjacency, 0)

    return {
        "dataset": dataset_name,
        "mode": mode,
        "seed": seed,
        "n_samples": int(data.shape[0]),
        "n_variables": int(len(columns)),
        "device": preset["device"],
        "preset": preset["name"],
        "runtime_seconds": round(elapsed, 4),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "raw_weights": raw_weights,
        "raw_weights_path": str(raw_weights_path),
        "final_weights_path": str(final_weights_path),
        "helper_adjacency_path": str(result["adjacency_path"]),
        "helper_edge_count": int(helper_adjacency.sum()),
        "backend_used": backend_metadata.get("backend_used", "unknown"),
        "causica_compat_status": backend_metadata.get("causica_compat_status", ""),
        "causica_error": backend_metadata.get("causica_error", ""),
        "native_constraints_supported": bool(
            backend_metadata.get("native_constraints_supported", False)
        ),
        "constraint_handling": backend_metadata.get(
            "constraint_handling",
            "unknown DECI constraint handling"
        ),
        "small_data_warning": small_data_warning(int(data.shape[0]), int(len(columns))),
        "stdout_tail": (completed.stdout or "").strip()[-2000:],
        "stderr_tail": (completed.stderr or "").strip()[-2000:],
    }


def build_deci_metadata(
    training: dict[str, Any],
    raw_weights: np.ndarray,
    thresholded: np.ndarray,
    final: np.ndarray,
    effective_threshold: float,
    forbidden_before: int,
    required_missing_before: int,
    enforcement: dict[str, int],
    final_adjacency_path: Path,
) -> dict[str, Any]:
    """
    Build diagnostic metadata for one DECI run.

    Parameters
    ----------
    training : dict[str, Any]
        Training artifacts.
    raw_weights : np.ndarray
        Raw weighted adjacency before final thresholding.
    thresholded : np.ndarray
        Binary adjacency before constraint enforcement.
    final : np.ndarray
        Final binary adjacency.
    effective_threshold : float
        Threshold used.
    forbidden_before : int
        Forbidden edges present before enforcement.
    required_missing_before : int
        Required edges absent before enforcement.
    enforcement : dict[str, int]
        Constraint post-processing counters.
    final_adjacency_path : pathlib.Path
        Saved final adjacency path.

    Returns
    -------
    dict[str, Any]
        Diagnostic metadata.
    """
    weights = np.asarray(raw_weights, dtype=float)
    abs_weights = np.abs(weights.copy())
    np.fill_diagonal(abs_weights, 0.0)
    non_diag = abs_weights[~np.eye(abs_weights.shape[0], dtype=bool)]
    quantiles = np.quantile(non_diag, [0.50, 0.75, 0.90, 0.95, 0.99])
    max_abs = float(non_diag.max()) if non_diag.size else 0.0
    near_nonzero = int(np.sum(non_diag > 1e-8))
    weighted_nonzero = int(np.sum(non_diag != 0))
    edges_after_threshold = int(thresholded.sum())
    edges_after_constraints = int(final.sum())
    possible_edges = int(weights.shape[0] * max(0, weights.shape[0] - 1))

    messages: list[str] = []
    if max_abs < 1e-8 or near_nonzero == 0:
        messages.append("DECI learned near-zero adjacency before thresholding.")
    elif edges_after_threshold == 0:
        messages.append("DECI produced weights, but the chosen threshold removed all edges.")
    elif possible_edges and edges_after_threshold >= int(0.8 * possible_edges):
        messages.append(
            "DECI fixed threshold produced an extremely dense graph; threshold does not transfer cleanly to this dataset."
        )
    if (
        training["mode"] == "constrained"
        and edges_after_threshold > 0
        and edges_after_constraints <= max(1, int(0.2 * edges_after_threshold))
    ):
        messages.append("Constraint enforcement removed all or most DECI edges.")
    elif possible_edges and edges_after_constraints >= int(0.8 * possible_edges):
        messages.append("Final DECI graph is near-complete after constraint enforcement.")
    if training.get("small_data_warning"):
        messages.append(str(training["small_data_warning"]))
    if not messages:
        messages.append("DECI produced a non-empty diagnosable weighted adjacency.")

    return {
        "algorithm": (
            "deci_native"
            if training.get("backend_used") == "causica_native"
            else "deci_postproc"
        ),
        "dataset": training["dataset"],
        "mode": training["mode"],
        "seed": training["seed"],
        "n_samples": training["n_samples"],
        "n_variables": training["n_variables"],
        "training_status": "success",
        "runtime_seconds": training.get("runtime_seconds"),
        "device": training["device"],
        "preset": training["preset"],
        "backend_used": training.get("backend_used", ""),
        "causica_compat_status": training.get("causica_compat_status", ""),
        "causica_error": training.get("causica_error", ""),
        "threshold_mode": getattr(project_config, "DECI_THRESHOLD_MODE", "fixed"),
        "threshold_used": effective_threshold,
        "native_constraints_supported": training["native_constraints_supported"],
        "constraint_handling": training["constraint_handling"],
        "small_data_warning": training["small_data_warning"],
        "raw_adjacency_shape": f"{weights.shape[0]}x{weights.shape[1]}",
        "raw_min": float(non_diag.min()) if non_diag.size else 0.0,
        "raw_max": max_abs,
        "raw_mean": float(non_diag.mean()) if non_diag.size else 0.0,
        "raw_std": float(non_diag.std()) if non_diag.size else 0.0,
        "abs_q50": float(quantiles[0]),
        "abs_q75": float(quantiles[1]),
        "abs_q90": float(quantiles[2]),
        "abs_q95": float(quantiles[3]),
        "abs_q99": float(quantiles[4]),
        "weighted_nonzero_edges": weighted_nonzero,
        "weighted_near_nonzero_edges": near_nonzero,
        "edges_after_threshold": edges_after_threshold,
        "forbidden_edges_predicted_before_enforcement": forbidden_before,
        "required_edges_missing_before_enforcement": required_missing_before,
        "forbidden_edges_removed": enforcement["forbidden_removed"],
        "required_edges_added": enforcement["required_added"],
        "constraint_cells_changed": enforcement["constraint_cells_changed"],
        "edges_after_constraint_enforcement": edges_after_constraints,
        "diagnostic_message": " ".join(messages),
        "raw_weights_path": training["raw_weights_path"],
        "final_adjacency_path": str(final_adjacency_path),
    }


def print_deci_run_diagnostic(metadata: dict[str, Any]) -> None:
    """
    Print a compact DECI diagnostic line.

    Parameters
    ----------
    metadata : dict[str, Any]
        DECI metadata.

    Returns
    -------
    None
    """
    log(
        "DECI diag "
        f"{metadata['dataset']}/{metadata['mode']}/seed={metadata['seed']}: "
        f"raw_max={metadata['raw_max']:.4f}, q95={metadata['abs_q95']:.4f}, "
        f"threshold={metadata['threshold_used']:.4f}, "
        f"edges_pre={metadata['edges_after_threshold']}, "
        f"changed={metadata['constraint_cells_changed']}, "
        f"edges_final={metadata['edges_after_constraint_enforcement']} | "
        f"{metadata['diagnostic_message']}"
    )


def evaluate_deci_threshold(
    raw_weights: np.ndarray,
    threshold: float,
    mode: str,
    columns: list[str],
    forbidden: list[tuple[str, str]],
    required: list[tuple[str, str]],
    true_adj: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate one DECI threshold on synthetic ground truth.

    Parameters
    ----------
    raw_weights : np.ndarray
        Raw DECI weights.
    threshold : float
        Candidate threshold.
    mode : str
        unconstrained or constrained.
    columns : list[str]
        Variable names.
    forbidden : list[tuple[str, str]]
        Forbidden constraints.
    required : list[tuple[str, str]]
        Required constraints.
    true_adj : np.ndarray
        Synthetic ground truth.

    Returns
    -------
    dict[str, Any]
        Sweep metrics.
    """
    thresholded, _ = threshold_weight_matrix(
        raw_weights,
        "fixed",
        threshold,
        float(getattr(project_config, "DECI_THRESHOLD_PERCENTILE", 95.0)),
        getattr(project_config, "DECI_TOPK_EDGES", None),
    )
    final = thresholded
    changes = 0
    if mode == "constrained":
        final, enforcement = enforce_deci_constraints(thresholded, columns, forbidden, required)
        changes = enforcement["constraint_cells_changed"]

    metrics = compute_synthetic_metrics(final, true_adj)
    return {
        "edge_count": metrics["edge_count_predicted"],
        "edge_count_true": metrics["edge_count_true"],
        "f1_directed": metrics["f1_directed"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "shd": metrics["shd"],
        "violations": count_pairs(final, forbidden, columns),
        "constraint_cells_changed": changes,
    }


def append_deci_threshold_sweep(
    training: dict[str, Any],
    columns: list[str],
    true_adj: np.ndarray,
    adapter: Any,
) -> None:
    """
    Append synthetic threshold-sweep rows for one DECI run.

    Parameters
    ----------
    training : dict[str, Any]
        Raw DECI training artifacts.
    columns : list[str]
        Variable names.
    true_adj : np.ndarray
        Ground-truth adjacency.
    adapter : Any
        Constraint adapter.

    Returns
    -------
    None
    """
    forbidden, required = load_constraints(training["dataset"], adapter)
    candidates = list(getattr(project_config, "DECI_THRESHOLD_CANDIDATES", []))
    if not candidates:
        candidates = [float(getattr(project_config, "DECI_THRESHOLD", 0.275))]
    for threshold in candidates:
        metrics = evaluate_deci_threshold(
            raw_weights=training["raw_weights"],
            threshold=float(threshold),
            mode=training["mode"],
            columns=columns,
            forbidden=forbidden,
            required=required,
            true_adj=true_adj,
        )
        append_csv_row(DECI_THRESHOLD_SWEEP_PATH, DECI_THRESHOLD_SWEEP_COLUMNS, {
            "dataset": training["dataset"],
            "mode": training["mode"],
            "seed": training["seed"],
            "threshold": float(threshold),
            **metrics,
            "selected": "",
        })


def choose_deci_threshold_from_sweep() -> tuple[float, str]:
    """
    Select a fixed DECI threshold from synthetic sweep rows only.

    Returns
    -------
    tuple[float, str]
        Selected threshold and human-readable reason.
    """
    if not DECI_THRESHOLD_SWEEP_PATH.exists():
        fallback = float(getattr(project_config, "DECI_THRESHOLD", 0.275))
        return fallback, "no sweep rows available; using configured fallback"

    sweep = pd.read_csv(DECI_THRESHOLD_SWEEP_PATH)
    if sweep.empty:
        fallback = float(getattr(project_config, "DECI_THRESHOLD", 0.275))
        return fallback, "empty sweep; using configured fallback"

    selection_pool = sweep[
        (sweep["dataset"] == "synthetic_n2000")
        & (sweep["mode"] == "unconstrained")
    ].copy()
    if selection_pool.empty:
        selection_pool = sweep[sweep["dataset"] == "synthetic_n2000"].copy()
    if selection_pool.empty:
        fallback = float(getattr(project_config, "DECI_THRESHOLD", 0.275))
        return fallback, "no synthetic sweep rows; using configured fallback"

    grouped = selection_pool.groupby("threshold", as_index=False).agg({
        "f1_directed": "mean",
        "shd": "mean",
        "edge_count": "mean",
        "edge_count_true": "mean",
    })
    true_edges = float(grouped["edge_count_true"].iloc[0])
    max_edges = float(getattr(project_config, "DECI_MAX_DENSITY_MULTIPLE", 1.5)) * true_edges
    min_edges = float(getattr(project_config, "DECI_MIN_DENSITY_MULTIPLE", 0.25)) * true_edges
    feasible = grouped[
        (grouped["edge_count"] <= max_edges)
        & (grouped["edge_count"] >= min_edges)
    ].copy()

    if feasible.empty:
        grouped["density_gap"] = (grouped["edge_count"] - true_edges).abs()
        chosen = grouped.sort_values(
            ["density_gap", "f1_directed", "shd"],
            ascending=[True, False, True],
        ).iloc[0]
        reason = (
            "selected synthetic threshold with closest mean edge count to true "
            f"edge count ({true_edges:.0f}) because no candidate passed density bounds"
        )
    else:
        chosen = feasible.sort_values(
            ["f1_directed", "shd", "edge_count"],
            ascending=[False, True, True],
        ).iloc[0]
        reason = (
            "selected on synthetic unconstrained sweep: best mean F1 among "
            f"thresholds with edge count between {min_edges:.1f} and {max_edges:.1f}; "
            "real data was not used"
        )

    selected = float(chosen["threshold"])
    sweep["selected"] = np.where(sweep["threshold"].astype(float) == selected, "yes", "")
    sweep.to_csv(DECI_THRESHOLD_SWEEP_PATH, index=False)
    return selected, reason


def calibrate_deci_threshold(
    requested_algorithms: list[str],
    requested_datasets: list[str],
    seeds: list[int],
    dataset_cache: dict[str, tuple[pd.DataFrame, np.ndarray | None, list[str]]],
    adapter: Any,
) -> tuple[float, str]:
    """
    Run synthetic-only DECI threshold calibration.

    Parameters
    ----------
    requested_algorithms : list[str]
        Algorithms requested by the run.
    requested_datasets : list[str]
        Datasets requested by the run.
    seeds : list[int]
        Bootstrap seeds.
    dataset_cache : dict
        Loaded datasets.
    adapter : Any
        Constraint adapter.

    Returns
    -------
    tuple[float, str]
        Selected threshold and reason.
    """
    fallback = float(getattr(project_config, "DECI_THRESHOLD", 0.275))
    if "deci" not in requested_algorithms:
        return fallback, "DECI not requested"
    if not bool(getattr(project_config, "DECI_CALIBRATE_THRESHOLD_ON_SYNTHETIC", True)):
        return fallback, "synthetic calibration disabled; using configured threshold"
    if "synthetic_n2000" not in dataset_cache:
        return fallback, "synthetic_n2000 not loaded; using configured threshold"

    df, true_adj, columns = dataset_cache["synthetic_n2000"]
    if true_adj is None:
        return fallback, "synthetic ground truth unavailable; using configured threshold"

    log("Calibrating DECI threshold on synthetic_n2000 only; real data is not used.")
    for mode in ["unconstrained", "constrained"]:
        for seed in seeds:
            cache_key = ("synthetic_n2000", mode, seed)
            if cache_key not in DECI_RUN_CACHE:
                data_boot = bootstrap_data(df, seed)
                DECI_RUN_CACHE[cache_key] = train_deci_guarded(
                    data=data_boot,
                    columns=columns,
                    mode=mode,
                    dataset_name="synthetic_n2000",
                    seed=seed,
                    adapter=adapter,
                )
            append_deci_threshold_sweep(DECI_RUN_CACHE[cache_key], columns, true_adj, adapter)

    selected, reason = choose_deci_threshold_from_sweep()
    log(f"Selected DECI fixed threshold={selected:.4f}. Reason: {reason}")
    return selected, reason


def compute_synthetic_metrics(predicted: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    """
    Compute directed and skeleton metrics against synthetic ground truth.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted adjacency.
    true : np.ndarray
        True adjacency.

    Returns
    -------
    dict[str, float | int]
        Synthetic metrics.
    """
    pred = (np.asarray(predicted) != 0).astype(int)
    truth = (np.asarray(true) != 0).astype(int)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: predicted={pred.shape}, true={truth.shape}")
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(truth, 0)

    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_directed = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    pred_skel = ((pred + pred.T) > 0).astype(int)
    true_skel = ((truth + truth.T) > 0).astype(int)
    np.fill_diagonal(pred_skel, 0)
    np.fill_diagonal(true_skel, 0)
    upper = np.triu(np.ones_like(pred_skel, dtype=bool), k=1)
    ps = pred_skel[upper]
    ts = true_skel[upper]
    tp_s = int(((ps == 1) & (ts == 1)).sum())
    fp_s = int(((ps == 1) & (ts == 0)).sum())
    fn_s = int(((ps == 0) & (ts == 1)).sum())
    p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
    r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
    f1_skeleton = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) else 0.0

    return {
        "edge_count_predicted": int(pred.sum()),
        "edge_count_true": int(truth.sum()),
        "shd": int(np.sum(np.abs(pred - truth))),
        "f1_directed": f1_directed,
        "f1_skeleton": f1_skeleton,
        "precision": precision,
        "recall": recall,
    }


def load_literature_supported_pairs(columns: list[str]) -> set[tuple[str, str]]:
    """
    Load approved forward literature directions for real-data scoring.

    Parameters
    ----------
    columns : list[str]
        Dataset variables.

    Returns
    -------
    set[tuple[str, str]]
        Approved cause-effect pairs present in the dataset.
    """
    review_path = ROOT / "reports" / "constraints_for_review.csv"
    if not review_path.exists():
        return set()
    df = pd.read_csv(review_path)
    required = {"cause", "effect", "approved"}
    if not required.issubset(df.columns):
        return set()
    variable_set = set(columns)
    approved = df["approved"].fillna("").astype(str).str.strip().str.lower().eq("yes")
    pairs: set[tuple[str, str]] = set()
    for _, row in df[approved].iterrows():
        source = str(row["cause"]).strip()
        target = str(row["effect"]).strip()
        if source in variable_set and target in variable_set and source != target:
            pairs.add((source, target))
    return pairs


def compute_real_metrics(
    predicted: np.ndarray,
    columns: list[str],
    forbidden: list[tuple[str, str]],
    literature_supported: set[tuple[str, str]],
) -> dict[str, float | int]:
    """
    Compute literature-alignment metrics for real data.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted adjacency.
    columns : list[str]
        Variable names.
    forbidden : list[tuple[str, str]]
        Forbidden edge constraints.
    literature_supported : set[tuple[str, str]]
        Approved forward literature pairs.

    Returns
    -------
    dict[str, float | int]
        Real-data literature metrics.
    """
    pred = (np.asarray(predicted) != 0).astype(int)
    np.fill_diagonal(pred, 0)
    edge_pairs = {
        (columns[i], columns[j])
        for i, j in zip(*np.where(pred == 1))
    }
    forbidden_set = set(forbidden)
    agreement = len(edge_pairs & literature_supported)
    violations = len(edge_pairs & forbidden_set)
    alignment = agreement / (agreement + violations + 1e-9)
    return {
        "edge_count_predicted": int(pred.sum()),
        "literature_agreement_count": agreement,
        "literature_violation_count": violations,
        "literature_alignment_score": alignment,
    }


def run_algorithm(
    algorithm: str,
    mode: str,
    dataset_name: str,
    seed: int,
    data: np.ndarray,
    columns: list[str],
    adapter: Any,
    total_started: float,
) -> np.ndarray:
    """
    Dispatch one algorithm run.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    mode : str
        unconstrained or constrained.
    dataset_name : str
        Dataset label.
    seed : int
        Seed.
    data : np.ndarray
        Standardized bootstrap data.
    columns : list[str]
        Variables.
    adapter : Any
        Constraint adapter module.
    total_started : float
        Overall run start time.

    Returns
    -------
    np.ndarray
        Predicted adjacency.
    """
    if algorithm == "pc":
        return run_pc(data, columns, mode, dataset_name, adapter)
    if algorithm == "notears":
        return run_notears(data, columns, mode, dataset_name, adapter)
    if algorithm == "lingam":
        return run_lingam(data, columns, mode, dataset_name, adapter)
    if algorithm == "deci":
        cache_key = (dataset_name, mode, seed)
        if (
            cache_key not in DECI_RUN_CACHE
            and time.perf_counter() - total_started > DECI_GLOBAL_ABORT_SECONDS
        ):
            raise TimeoutError("DECI skipped because total runtime exceeded 30 minutes")
        return run_deci_guarded(data, columns, mode, dataset_name, seed, adapter)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_one(
    algorithm: str,
    mode: str,
    dataset_name: str,
    seed: int,
    df: pd.DataFrame,
    true_adj: np.ndarray | None,
    columns: list[str],
    adapter: Any,
    total_started: float,
) -> str:
    """
    Run one experiment cell and append the result.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    mode : str
        Mode.
    dataset_name : str
        Dataset label.
    seed : int
        Seed.
    df : pd.DataFrame
        Complete numeric data.
    true_adj : np.ndarray or None
        Synthetic ground truth.
    columns : list[str]
        Variable names.
    adapter : Any
        Constraint adapter.
    total_started : float
        Overall run start time.

    Returns
    -------
    str
        success or failed.
    """
    started = time.perf_counter()
    if algorithm == "notears":
        result_algorithm = "notears_postproc"
    elif algorithm == "deci":
        result_algorithm = (
            "deci_native"
            if str(getattr(project_config, "DECI_BACKEND", "causica")).lower() == "causica"
            else "deci_postproc"
        )
    else:
        result_algorithm = algorithm
    try:
        data_boot = bootstrap_data(df, seed)
        prediction = run_algorithm(
            algorithm,
            mode,
            dataset_name,
            seed,
            data_boot,
            columns,
            adapter,
            total_started,
        )
        metadata: dict[str, Any] = {}
        if isinstance(prediction, tuple):
            predicted = prediction[0]
            metadata = prediction[1]
        else:
            predicted = prediction
        if algorithm == "deci" and metadata.get("backend_used") == "causica_native":
            result_algorithm = "deci_native"
        elapsed = time.perf_counter() - started

        row: dict[str, Any] = {
            "algorithm": result_algorithm,
            "mode": mode,
            "dataset": dataset_name,
            "seed": seed,
            "runtime_seconds": round(elapsed, 4),
            "status": "success",
        }

        if true_adj is not None:
            row.update(compute_synthetic_metrics(predicted, true_adj))
            if algorithm == "deci":
                log(
                    f"{result_algorithm}/{mode}/{dataset_name}/seed={seed}: "
                    f"edges={row['edge_count_predicted']}, "
                    f"SHD={row['shd']}, F1={row['f1_directed']:.3f}, "
                    f"threshold={metadata.get('threshold_used', '')} "
                    f"({elapsed:.1f}s)"
                )
            elif algorithm == "notears":
                log(
                    f"{result_algorithm}/{mode}/{dataset_name}/seed={seed}: "
                    f"edges={row['edge_count_predicted']}, "
                    f"F1={row['f1_directed']:.3f}, "
                    f"post_processing_changed={metadata.get('post_processing_changed', 0)} cells "
                    f"({elapsed:.1f}s)"
                )
            else:
                log(
                    f"{result_algorithm}/{mode}/{dataset_name}/seed={seed}: "
                    f"SHD={row['shd']}, F1={row['f1_directed']:.3f} "
                    f"({elapsed:.1f}s)"
                )
        else:
            forbidden, _ = load_constraints(dataset_name, adapter)
            literature_supported = load_literature_supported_pairs(columns)
            row.update(compute_real_metrics(predicted, columns, forbidden, literature_supported))
            if algorithm == "deci":
                log(
                    f"{result_algorithm}/{mode}/{dataset_name}/seed={seed}: "
                    f"edges={row['edge_count_predicted']}, "
                    f"align={row['literature_alignment_score']:.3f}, "
                    f"violations={row['literature_violation_count']}, "
                    f"threshold={metadata.get('threshold_used', '')} "
                    f"({elapsed:.1f}s)"
                )
            elif algorithm == "notears":
                log(
                    f"{result_algorithm}/{mode}/{dataset_name}/seed={seed}: "
                    f"edges={row['edge_count_predicted']}, "
                    f"F1=nan, "
                    f"post_processing_changed={metadata.get('post_processing_changed', 0)} cells "
                    f"({elapsed:.1f}s)"
                )
            else:
                log(
                    f"{result_algorithm}/{mode}/{dataset_name}/seed={seed}: "
                    f"edges={row['edge_count_predicted']}, "
                    f"align={row['literature_alignment_score']:.3f} "
                    f"({elapsed:.1f}s)"
                )

        if algorithm == "deci":
            diagnostic = {**row, **metadata}
            diagnostic["runtime_seconds"] = metadata.get("runtime_seconds", round(elapsed, 4))
            append_csv_row(DECI_DIAGNOSTICS_PATH, DECI_DIAGNOSTIC_COLUMNS, diagnostic)

        append_result(row)
        return "success"
    except Exception as exc:
        elapsed = time.perf_counter() - started
        append_failure(result_algorithm, mode, dataset_name, seed, exc)
        append_result({
            "algorithm": result_algorithm,
            "mode": mode,
            "dataset": dataset_name,
            "seed": seed,
            "runtime_seconds": round(elapsed, 4),
            "status": f"failed: {exc}",
            "edge_count_true": int(true_adj.sum()) if true_adj is not None else "",
        })
        if algorithm == "deci":
            append_csv_row(DECI_DIAGNOSTICS_PATH, DECI_DIAGNOSTIC_COLUMNS, {
                "algorithm": result_algorithm,
                "dataset": dataset_name,
                "mode": mode,
                "seed": seed,
                "n_samples": max(2, int(0.8 * len(df))),
                "n_variables": len(columns),
                "training_status": f"failed: {exc}",
                "runtime_seconds": round(elapsed, 4),
                "device": get_deci_preset().get("device", "cpu"),
                "preset": get_deci_preset().get("name", ""),
                "backend_used": str(getattr(project_config, "DECI_BACKEND", "causica")),
                "diagnostic_message": f"DECI training failed; no zero matrix was substituted. Error: {exc}",
            })
        log(
            f"{result_algorithm}/{mode}/{dataset_name}/seed={seed}: "
            f"FAILED ({elapsed:.1f}s): {exc}"
        )
        return "failed"


def append_skipped_rows(
    algorithm: str,
    datasets: list[str],
    seeds: list[int],
    reason: str,
) -> None:
    """
    Append skipped rows for a disabled algorithm.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    datasets : list[str]
        Dataset labels.
    seeds : list[int]
        Seeds.
    reason : str
        Skip reason.

    Returns
    -------
    None
    """
    for dataset_name in datasets:
        for mode in ["unconstrained", "constrained"]:
            for seed in seeds:
                append_result({
                    "algorithm": algorithm,
                    "mode": mode,
                    "dataset": dataset_name,
                    "seed": seed,
                    "runtime_seconds": 0.0,
                    "status": f"skipped: {reason}",
                })
    log(f"{algorithm} skipped: {reason}")


def write_summary() -> pd.DataFrame:
    """
    Write grouped summary CSV.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    results = pd.read_csv(RESULTS_PATH)
    success = results[results["status"] == "success"].copy()
    if success.empty:
        summary = pd.DataFrame(columns=["algorithm", "mode", "dataset", "successful_runs"])
        summary.to_csv(SUMMARY_PATH, index=False)
        return summary

    for column in NUMERIC_SUMMARY_COLUMNS:
        success[column] = pd.to_numeric(success[column], errors="coerce")

    grouped = success.groupby(["algorithm", "mode", "dataset"], dropna=False)
    pieces = [grouped.size().rename("successful_runs")]
    for column in NUMERIC_SUMMARY_COLUMNS:
        pieces.append(grouped[column].mean().rename(f"{column}_mean"))
        pieces.append(grouped[column].std().fillna(0.0).rename(f"{column}_std"))
    summary = pd.concat(pieces, axis=1).reset_index()
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def print_final_summaries(summary: pd.DataFrame) -> None:
    """
    Print synthetic and real headline summary tables.

    Parameters
    ----------
    summary : pd.DataFrame
        Grouped results summary.

    Returns
    -------
    None
    """
    synthetic = summary[summary["dataset"].astype(str).str.startswith("synthetic")].copy()
    real = summary[summary["dataset"].eq("real")].copy()

    log("Synthetic results:")
    if synthetic.empty:
        log("  No successful synthetic runs.")
    else:
        for algorithm in sorted(synthetic["algorithm"].unique()):
            sub = synthetic[synthetic["algorithm"] == algorithm]
            uncon = sub[sub["mode"] == "unconstrained"]
            cons = sub[sub["mode"] == "constrained"]
            if uncon.empty or cons.empty:
                continue
            delta = float(cons["f1_directed_mean"].iloc[0] - uncon["f1_directed_mean"].iloc[0])
            log(
                f"  {algorithm}: "
                f"F1 uncon={uncon['f1_directed_mean'].iloc[0]:.3f}, "
                f"con={cons['f1_directed_mean'].iloc[0]:.3f}, "
                f"DeltaF1={delta:+.3f}; "
                f"SHD uncon={uncon['shd_mean'].iloc[0]:.2f}, "
                f"con={cons['shd_mean'].iloc[0]:.2f}"
            )

    log("Real results:")
    if real.empty:
        log("  No successful real runs.")
    else:
        for _, row in real.sort_values(["algorithm", "mode"]).iterrows():
            log(
                f"  {row['algorithm']}/{row['mode']}: "
                f"alignment={row['literature_alignment_score_mean']:.3f} "
                f"+/- {row['literature_alignment_score_std']:.3f}, "
                f"edges={row['edge_count_predicted_mean']:.2f}, "
                f"violations={row['literature_violation_count_mean']:.2f}"
            )


def write_deci_stable_edges(adapter: Any) -> pd.DataFrame:
    """
    Write DECI edge-frequency stability table across seeds.

    Parameters
    ----------
    adapter : Any
        Constraint adapter.

    Returns
    -------
    pd.DataFrame
        Stable-edge table.
    """
    if not DECI_DIAGNOSTICS_PATH.exists():
        stable = pd.DataFrame(columns=DECI_STABLE_EDGE_COLUMNS)
        stable.to_csv(DECI_STABLE_EDGES_PATH, index=False)
        return stable

    diagnostics = pd.read_csv(DECI_DIAGNOSTICS_PATH)
    diagnostics = diagnostics[diagnostics["training_status"] == "success"].copy()
    if diagnostics.empty:
        stable = pd.DataFrame(columns=DECI_STABLE_EDGE_COLUMNS)
        stable.to_csv(DECI_STABLE_EDGES_PATH, index=False)
        return stable

    rows: list[dict[str, Any]] = []
    for (dataset_name, mode), group in diagnostics.groupby(["dataset", "mode"]):
        config = DATASETS[str(dataset_name)]
        _, _, variables = load_dataset(str(dataset_name))
        forbidden, required = load_constraints(str(dataset_name), adapter)
        forbidden_set = set(forbidden)
        required_set = set(required)
        n_runs = len(group)
        edge_counts: dict[tuple[str, str], int] = {}
        weight_sums: dict[tuple[str, str], float] = {}

        for _, diag in group.iterrows():
            adj_path = Path(str(diag["final_adjacency_path"]))
            raw_path = Path(str(diag["raw_weights_path"]))
            if not adj_path.exists() or not raw_path.exists():
                continue
            adjacency = (np.load(adj_path) != 0).astype(int)
            raw_weights = pd.read_csv(raw_path, index_col=0).loc[variables, variables].to_numpy(dtype=float)
            for i, source in enumerate(variables):
                for j, target in enumerate(variables):
                    if i == j:
                        continue
                    pair = (source, target)
                    if adjacency[i, j]:
                        edge_counts[pair] = edge_counts.get(pair, 0) + 1
                    weight_sums[pair] = weight_sums.get(pair, 0.0) + float(raw_weights[i, j])

        for pair, count in sorted(edge_counts.items()):
            frequency = count / n_runs if n_runs else 0.0
            rows.append({
                "dataset": dataset_name,
                "mode": mode,
                "source": pair[0],
                "target": pair[1],
                "frequency": frequency,
                "mean_weight": weight_sums.get(pair, 0.0) / n_runs if n_runs else 0.0,
                "sign_if_available": "unsigned_probability",
                "passes_60_percent": frequency >= 0.60,
                "passes_80_percent": frequency >= 0.80,
                "forbidden_edge": pair in forbidden_set,
                "required_edge": pair in required_set,
            })

    stable = pd.DataFrame(rows, columns=DECI_STABLE_EDGE_COLUMNS)
    stable.to_csv(DECI_STABLE_EDGES_PATH, index=False)
    return stable


def print_deci_interpretation(summary: pd.DataFrame) -> None:
    """
    Print automated DECI diagnostic interpretation.

    Parameters
    ----------
    summary : pd.DataFrame
        Results summary.

    Returns
    -------
    None
    """
    if not DECI_DIAGNOSTICS_PATH.exists():
        return
    diagnostics = pd.read_csv(DECI_DIAGNOSTICS_PATH)
    deci_success = diagnostics[diagnostics["training_status"] == "success"].copy()
    if deci_success.empty:
        log("DECI diagnostic interpretation:")
        log("  No successful DECI runs; inspect deci_diagnostics.csv and failures.csv.")
        return

    log("DECI diagnostic interpretation:")
    if (deci_success["weighted_near_nonzero_edges"].fillna(0) == 0).any():
        log("  At least one DECI run learned near-zero adjacency before thresholding.")
    if (
        (deci_success["weighted_near_nonzero_edges"].fillna(0) > 0)
        & (deci_success["edges_after_threshold"].fillna(0) == 0)
    ).any():
        log("  DECI learned nonzero weights but the fixed threshold removed all edges in some runs.")
    constrained = deci_success[deci_success["mode"] == "constrained"]
    if not constrained.empty and (constrained["constraint_cells_changed"].fillna(0) > 0).any():
        log("  Constrained DECI is post-processed: constraints changed at least one cell.")
    if (deci_success["small_data_warning"].fillna("") != "").any():
        log("  Real-data DECI should be treated as exploratory due to small sample size.")
    real_runs = deci_success[deci_success["dataset"] == "real"]
    if not real_runs.empty:
        possible = real_runs["n_variables"].astype(float) * (real_runs["n_variables"].astype(float) - 1.0)
        dense = real_runs["edges_after_constraint_enforcement"].astype(float) >= 0.8 * possible
        if dense.any():
            log(
                "  Real-data DECI fixed-threshold graphs are near-complete, "
                "so their literature alignment is not reliable evidence of causal recovery."
            )

    synthetic = summary[
        (summary["algorithm"].astype(str).str.startswith("deci_"))
        & (summary["dataset"] == "synthetic_n2000")
    ]
    if not synthetic.empty:
        uncon = synthetic[synthetic["mode"] == "unconstrained"]
        cons = synthetic[synthetic["mode"] == "constrained"]
        if not uncon.empty and not cons.empty:
            delta_f1 = cons["f1_directed_mean"].iloc[0] - uncon["f1_directed_mean"].iloc[0]
            delta_shd = cons["shd_mean"].iloc[0] - uncon["shd_mean"].iloc[0]
            log(
                f"  Synthetic constrained-minus-unconstrained DECI: "
                f"DeltaF1={delta_f1:+.3f}, DeltaSHD={delta_shd:+.2f}."
            )


def main() -> int:
    """Run the full experiment matrix."""
    parser = argparse.ArgumentParser(description="Run constrained causal discovery experiments.")
    parser.add_argument("--skip-deci", action="store_true", help="Skip DECI runs.")
    parser.add_argument("--skip-notears", action="store_true", help="Skip NOTEARS runs.")
    parser.add_argument("--skip-lingam", action="store_true", help="Skip LiNGAM runs.")
    parser.add_argument("--datasets", default="synthetic_n2000,real",
                        help="Comma-separated subset of synthetic_n2000,real.")
    parser.add_argument("--algorithms", default="pc,notears,lingam,deci",
                        help="Comma-separated subset of pc,notears,lingam,deci.")
    parser.add_argument("--seeds", default="42,43,44,45,46",
                        help="Comma-separated bootstrap seeds.")
    args = parser.parse_args()

    requested_datasets = parse_csv_arg(args.datasets)
    requested_algorithms = parse_csv_arg(args.algorithms)
    seeds = parse_seed_arg(args.seeds)

    bad_datasets = sorted(set(requested_datasets) - set(DATASETS))
    if bad_datasets:
        parser.error(f"Unsupported datasets: {bad_datasets}")

    allowed_algorithms = {"pc", "notears", "lingam", "deci"}
    bad_algorithms = sorted(set(requested_algorithms) - allowed_algorithms)
    if bad_algorithms:
        parser.error(f"Unsupported algorithms: {bad_algorithms}")

    algorithms = []
    for algorithm in requested_algorithms:
        if algorithm == "notears" and args.skip_notears:
            continue
        if algorithm == "lingam" and args.skip_lingam:
            continue
        if algorithm == "deci" and args.skip_deci:
            continue
        algorithms.append(algorithm)

    ensure_outputs()
    total_started = time.perf_counter()
    adapter = load_adapter()

    if "notears" in requested_algorithms and args.skip_notears:
        append_skipped_rows("notears", requested_datasets, seeds, "--skip-notears")
    if "lingam" in requested_algorithms and args.skip_lingam:
        append_skipped_rows("lingam", requested_datasets, seeds, "--skip-lingam")
    if "deci" in requested_algorithms and args.skip_deci:
        append_skipped_rows("deci_postproc", requested_datasets, seeds, "--skip-deci")

    dataset_cache = {
        dataset_name: load_dataset(dataset_name)
        for dataset_name in requested_datasets
    }

    global CURRENT_DECI_THRESHOLD
    CURRENT_DECI_THRESHOLD, deci_threshold_reason = calibrate_deci_threshold(
        requested_algorithms=algorithms,
        requested_datasets=requested_datasets,
        seeds=seeds,
        dataset_cache=dataset_cache,
        adapter=adapter,
    )
    if "deci" in algorithms:
        log(
            f"DECI preset={get_deci_preset()['name']}; "
            f"threshold_mode={getattr(project_config, 'DECI_THRESHOLD_MODE', 'fixed')}; "
            f"threshold={CURRENT_DECI_THRESHOLD:.4f}; {deci_threshold_reason}"
        )

    for dataset_name in requested_datasets:
        df, true_adj, columns = dataset_cache[dataset_name]
        for algorithm in algorithms:
            for mode in ["unconstrained", "constrained"]:
                for seed in seeds:
                    run_one(
                        algorithm=algorithm,
                        mode=mode,
                        dataset_name=dataset_name,
                        seed=seed,
                        df=df,
                        true_adj=true_adj,
                        columns=columns,
                        adapter=adapter,
                        total_started=total_started,
                    )

    summary = write_summary()
    write_deci_stable_edges(adapter)
    print_final_summaries(summary)
    print_deci_interpretation(summary)
    log(f"Total runtime: {time.perf_counter() - total_started:.1f}s")
    log(f"Results -> {RESULTS_PATH.relative_to(ROOT)}")
    log(f"Summary -> {SUMMARY_PATH.relative_to(ROOT)}")
    log(f"DECI diagnostics -> {DECI_DIAGNOSTICS_PATH.relative_to(ROOT)}")
    log(f"DECI threshold sweep -> {DECI_THRESHOLD_SWEEP_PATH.relative_to(ROOT)}")
    log(f"DECI stable edges -> {DECI_STABLE_EDGES_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--deci-worker":
        raise SystemExit(deci_subprocess_entry(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
