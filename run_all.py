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


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "outputs" / "experiments"
RESULTS_PATH = RESULTS_DIR / "results.csv"
SUMMARY_PATH = RESULTS_DIR / "results_summary.csv"
FAILURES_PATH = RESULTS_DIR / "failures.csv"
DECI_WORK_DIR = RESULTS_DIR / "deci_work"
DECI_TIMEOUT_SECONDS = 300
DECI_GLOBAL_ABORT_SECONDS = 1800
NOTEARS_NOTE_PRINTED = False

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

        (ROOT / "outputs" / "graphs").mkdir(parents=True, exist_ok=True)
        metrics_dir = ROOT / "outputs" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        deci_module = load_module_from_path("step_07_run_deci", ROOT / "07_run_deci.py")
        batch_size = max(8, min(64, data.shape[0] // 2))
        adj = deci_module.run_deci(
            data=data,
            columns=columns,
            constraint_matrix=constraint_matrix,
            max_epochs=20,
            learning_rate=3e-3,
            batch_size=batch_size,
            device="cpu",
            edge_threshold=0.5,
            run_name=run_name,
            log_path=str(metrics_dir / "run_log_deci.csv"),
        )
        np.save(adjacency_path, np.asarray(adj, dtype=int))
        result = {"status": "success", "adjacency_path": str(adjacency_path)}
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
) -> np.ndarray:
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
    np.ndarray
        Directed binary adjacency.
    """
    constraint_matrix = None
    if mode == "constrained":
        forbidden, required = load_constraints(dataset_name, adapter)
        # Build the real adapter tensor for API verification/documentation,
        # then use the existing Step 07 helper's -1/0/1 convention.
        adapter.build_causica_constraint_matrix(columns, forbidden, required)
        constraint_matrix = build_deci_constraint_matrix(columns, forbidden, required)

    run_name = f"deci_{dataset_name}_{mode}_seed{seed}"

    DECI_WORK_DIR.mkdir(parents=True, exist_ok=True)
    safe_run_name = run_name.replace("/", "_").replace("\\", "_")
    input_path = DECI_WORK_DIR / f"{safe_run_name}_input.npz"
    output_path = DECI_WORK_DIR / f"{safe_run_name}_result.json"
    adjacency_path = DECI_WORK_DIR / f"{safe_run_name}_adjacency.npy"

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
    )

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--deci-worker",
        str(input_path),
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "ESG_FORCE_MANUAL_DECI": "1"},
            text=True,
            capture_output=True,
            timeout=DECI_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"DECI exceeded {DECI_TIMEOUT_SECONDS}s timeout") from exc

    if not output_path.exists():
        stderr_tail = (completed.stderr or "").strip()[-1000:]
        stdout_tail = (completed.stdout or "").strip()[-1000:]
        detail = stderr_tail or stdout_tail or f"exit code {completed.returncode}"
        raise RuntimeError(f"DECI worker produced no result: {detail}")

    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result["status"] != "success":
        raise RuntimeError(result.get("error", "DECI failed"))

    directed = (np.load(result["adjacency_path"]) != 0).astype(int)
    np.fill_diagonal(directed, 0)
    return directed


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
        if time.perf_counter() - total_started > DECI_GLOBAL_ABORT_SECONDS:
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
    result_algorithm = "notears_postproc" if algorithm == "notears" else algorithm
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
            if algorithm == "notears":
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
            if algorithm == "notears":
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
        append_skipped_rows("deci", requested_datasets, seeds, "--skip-deci")

    dataset_cache = {
        dataset_name: load_dataset(dataset_name)
        for dataset_name in requested_datasets
    }

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
    print_final_summaries(summary)
    log(f"Total runtime: {time.perf_counter() - total_started:.1f}s")
    log(f"Results -> {RESULTS_PATH.relative_to(ROOT)}")
    log(f"Summary -> {SUMMARY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--deci-worker":
        raise SystemExit(deci_subprocess_entry(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
