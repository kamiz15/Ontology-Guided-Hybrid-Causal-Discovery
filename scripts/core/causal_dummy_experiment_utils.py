"""Shared helpers for causal-dummy ESG causal-discovery experiments.

The helpers in this module keep the quantitative experiments aligned on:

- generated causal dummy v2 data;
- the fixed ground-truth DAG from ``generate_causal_dummy.py``;
- forbidden-only ontology constraints for constrained runs;
- per-row failure capture and runtime tracking.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "outputs" / "experiments"
FIG_DIR = ROOT / "outputs" / "figures"
GRAPH_DIR = EXP_DIR / "graphs"
COMMAND_LOG = EXP_DIR / "experiment_command_log.csv"

DEFAULT_N = 3000
DEFAULT_SNR = 0.6
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_SNR_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_SAMPLE_SIZES = [100, 250, 500, 1000, 2000, 3000]

ALGORITHMS_CORE = ["PC", "LiNGAM", "GES"]
ALGORITHMS_FINAL = ["PC", "LiNGAM", "NOTEARS", "GES"]
CONSTRAINT_MODES = ["unconstrained", "constrained"]

RESULT_COLUMNS = [
    "algorithm",
    "dataset",
    "constraint_mode",
    "seed",
    "n",
    "snr",
    "status",
    "runtime_seconds",
    "error_type",
    "error_message",
    "f1",
    "shd",
    "precision",
    "recall",
    "edge_count",
    "violations",
    "true_edge_count",
    "n_columns",
    "n_forbidden",
    "graph_path",
]

METRIC_COLUMNS = [
    "f1",
    "shd",
    "precision",
    "recall",
    "edge_count",
    "violations",
    "runtime_seconds",
]


def ensure_dirs() -> None:
    """Create output directories used by the experiment scripts."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_float_for_path(value: float) -> str:
    """Return a file-safe compact float label."""
    return f"{value:g}".replace(".", "p").replace("-", "m")


def script_command() -> str:
    """Return the command used to invoke the current Python script."""
    return " ".join([Path(sys.executable).as_posix(), *sys.argv])


def log_command(script_name: str, args: dict[str, Any]) -> None:
    """Append one command-level log row."""
    ensure_dirs()
    exists = COMMAND_LOG.exists()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": script_name,
        "command": script_command(),
        "args": repr(args),
    }
    with COMMAND_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_module_from_path(module_name: str, path: Path) -> Any:
    """Import a Python module from a local path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_generator() -> Any:
    """Load ``generate_causal_dummy.py``."""
    return load_module_from_path(
        "causal_dummy_generator",
        ROOT / "scripts" / "experiments" / "generate_causal_dummy.py",
    )


def causal_dummy_columns(generator: Any | None = None) -> list[str]:
    """Return structural variables in the fixed DAG order."""
    gen = generator if generator is not None else load_generator()
    return [str(column) for column in gen.TOPO_ORDER]


def build_ground_truth_adj(columns: list[str], generator: Any | None = None) -> np.ndarray:
    """Build the fixed ground-truth adjacency for a column order."""
    gen = generator if generator is not None else load_generator()
    index = {column: idx for idx, column in enumerate(columns)}
    adj = np.zeros((len(columns), len(columns)), dtype=int)
    for target, parents in gen.STRUCTURAL_EDGES.items():
        for source, _sign in parents:
            if source in index and target in index:
                adj[index[source], index[target]] = 1
    np.fill_diagonal(adj, 0)
    return adj


def generate_causal_dummy_numeric(n: int, seed: int, snr: float) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Generate numeric causal dummy v2 data without injected quality issues."""
    gen = load_generator()
    df_raw, _ = gen.generate(n=n, seed=seed, snr=snr, skip_quality_issues=True)
    columns = [column for column in causal_dummy_columns(gen) if column in df_raw.columns]
    df = df_raw[columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0).copy()
    columns = df.columns.tolist()
    truth = build_ground_truth_adj(columns, gen)
    return df, truth, columns


def standardize(data: np.ndarray) -> np.ndarray:
    """Z-score numeric data, guarding constant columns."""
    arr = np.asarray(data, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std == 0, 1.0, std)
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_forbidden_constraints(columns: list[str]) -> list[tuple[str, str]]:
    """Load forbidden ontology constraints filtered to the active variables."""
    candidates = [
        EXP_DIR / "advisor_dummy_constraints_forbidden.csv",
        EXP_DIR / "dummy_constraints_forbidden.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []

    df = pd.read_csv(path)
    source_col = "cause" if "cause" in df.columns else "source"
    target_col = "effect" if "effect" in df.columns else "target"
    if source_col not in df.columns or target_col not in df.columns:
        return []

    present = set(columns)
    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        source = str(row[source_col]).strip()
        target = str(row[target_col]).strip()
        if source in present and target in present and source != target:
            pairs.append((source, target))
    return sorted(set(pairs))


def apply_forbidden(adjacency: np.ndarray, columns: list[str], forbidden: Iterable[tuple[str, str]]) -> np.ndarray:
    """Remove forbidden directed edges from a binary adjacency matrix."""
    index = {column: idx for idx, column in enumerate(columns)}
    final = np.asarray(adjacency, dtype=int).copy()
    for source, target in forbidden:
        if source in index and target in index:
            final[index[source], index[target]] = 0
    np.fill_diagonal(final, 0)
    return final


def count_violations(adjacency: np.ndarray, columns: list[str], forbidden: Iterable[tuple[str, str]]) -> int:
    """Count predicted forbidden directed edges."""
    index = {column: idx for idx, column in enumerate(columns)}
    pred = (np.asarray(adjacency) != 0).astype(int)
    count = 0
    for source, target in forbidden:
        if source in index and target in index and pred[index[source], index[target]]:
            count += 1
    return count


def dag_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    """Compute directed causal-recovery metrics."""
    pred = (np.asarray(predicted) != 0).astype(int)
    true = (np.asarray(truth) != 0).astype(int)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: predicted={pred.shape}, truth={true.shape}")
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(true, 0)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "f1": f1,
        "shd": int(np.abs(pred - true).sum()),
        "precision": precision,
        "recall": recall,
        "edge_count": int(pred.sum()),
        "true_edge_count": int(true.sum()),
    }


def run_pc(data: np.ndarray, columns: list[str], constraint_mode: str, forbidden: list[tuple[str, str]]) -> np.ndarray:
    """Run causal-learn PC with optional forbidden background knowledge."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc as causallearn_pc

    background = None
    if constraint_mode == "constrained" and forbidden:
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

        node_objects = [GraphNode(column) for column in columns]
        node_map = dict(zip(columns, node_objects))
        background = BackgroundKnowledge()
        for source, target in forbidden:
            if source in node_map and target in node_map:
                background.add_forbidden_by_node(node_map[source], node_map[target])

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
    index = {column: idx for idx, column in enumerate(columns)}
    adj = np.zeros((len(columns), len(columns)), dtype=int)
    for edge in graph.G.get_graph_edges():
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        if n1 not in index or n2 not in index:
            continue
        ep1 = str(edge.get_endpoint1())
        ep2 = str(edge.get_endpoint2())
        i, j = index[n1], index[n2]
        if "TAIL" in ep1 and "ARROW" in ep2:
            adj[i, j] = 1
        elif "ARROW" in ep1 and "TAIL" in ep2:
            adj[j, i] = 1
    np.fill_diagonal(adj, 0)
    return adj


def run_lingam(data: np.ndarray, columns: list[str], constraint_mode: str, forbidden: list[tuple[str, str]]) -> np.ndarray:
    """Run DirectLiNGAM with optional forbidden-edge post-processing."""
    import lingam

    model = lingam.DirectLiNGAM()
    model.fit(data)
    raw = np.asarray(model.adjacency_matrix_, dtype=float).T
    adj = (np.abs(raw) > 0.01).astype(int)
    np.fill_diagonal(adj, 0)
    if constraint_mode == "constrained":
        adj = apply_forbidden(adj, columns, forbidden)
    return adj


def run_notears(data: np.ndarray, columns: list[str], constraint_mode: str, forbidden: list[tuple[str, str]]) -> np.ndarray:
    """Run gCastle NOTEARS with optional forbidden-edge post-processing."""
    from castle.algorithms import Notears

    model = Notears(lambda1=0.1, loss_type="l2", w_threshold=0.3)
    model.learn(data, columns=columns)
    raw = getattr(model, "weight_causal_matrix", None)
    if raw is None:
        raw = model.causal_matrix
    adj = (np.abs(np.asarray(raw, dtype=float)) > 0.3).astype(int)
    np.fill_diagonal(adj, 0)
    if constraint_mode == "constrained":
        adj = apply_forbidden(adj, columns, forbidden)
    return adj


def run_ges(data: np.ndarray, columns: list[str], constraint_mode: str, forbidden: list[tuple[str, str]], seed: int) -> np.ndarray:
    """Run GES via the project wrapper and keep only forbidden constraints."""
    module = load_module_from_path(
        "project_ges_runner",
        ROOT / "scripts" / "experiments" / "08_run_ges.py",
    )
    adj, _metadata = module.run_ges(
        data=data,
        columns=columns,
        mode=constraint_mode,
        dataset_name="causal_dummy",
        seed=seed,
        forbidden=forbidden if constraint_mode == "constrained" else [],
        required=[],
        constraint_mode="forbidden_only",
        output_dir=GRAPH_DIR,
    )
    return (np.asarray(adj) != 0).astype(int)


def run_algorithm(
    algorithm: str,
    data: np.ndarray,
    columns: list[str],
    constraint_mode: str,
    forbidden: list[tuple[str, str]],
    seed: int,
) -> np.ndarray:
    """Dispatch a supported algorithm."""
    name = algorithm.strip().lower()
    if name == "pc":
        return run_pc(data, columns, constraint_mode, forbidden)
    if name == "lingam":
        return run_lingam(data, columns, constraint_mode, forbidden)
    if name == "notears":
        return run_notears(data, columns, constraint_mode, forbidden)
    if name == "ges":
        return run_ges(data, columns, constraint_mode, forbidden, seed)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def graph_path_for(experiment: str, algorithm: str, constraint_mode: str, seed: int, n: int, snr: float) -> Path:
    """Return the canonical graph CSV path for one experiment row."""
    safe_alg = algorithm.replace(" ", "_")
    return GRAPH_DIR / (
        f"{experiment}_{safe_alg}_{constraint_mode}_seed{seed}"
        f"_n{n}_snr{format_float_for_path(snr)}.csv"
    )


def save_adjacency(adjacency: np.ndarray, columns: list[str], path: Path) -> None:
    """Save an adjacency matrix as a labeled CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(adjacency, index=columns, columns=columns).to_csv(path)


def empty_result_row(
    algorithm: str,
    dataset: str,
    constraint_mode: str,
    seed: int,
    n: int,
    snr: float,
) -> dict[str, Any]:
    """Return a result row prefilled with metadata and NaN metrics."""
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update({
        "algorithm": algorithm,
        "dataset": dataset,
        "constraint_mode": constraint_mode,
        "seed": seed,
        "n": n,
        "snr": snr,
        "status": "pending",
        "runtime_seconds": np.nan,
        "error_type": "",
        "error_message": "",
        "graph_path": "",
    })
    return row


def run_experiment_cell(
    algorithm: str,
    constraint_mode: str,
    seed: int,
    n: int,
    snr: float,
    experiment: str,
    dataset: str = "causal_dummy_v2",
) -> dict[str, Any]:
    """Run one algorithm/configuration and return a complete result row."""
    row = empty_result_row(algorithm, dataset, constraint_mode, seed, n, snr)
    started = time.perf_counter()
    try:
        df, truth, columns = generate_causal_dummy_numeric(n=n, seed=seed, snr=snr)
        forbidden = load_forbidden_constraints(columns)
        data = standardize(df.to_numpy(dtype=float))
        adj = run_algorithm(
            algorithm=algorithm,
            data=data,
            columns=columns,
            constraint_mode=constraint_mode,
            forbidden=forbidden,
            seed=seed,
        )
        graph_path = graph_path_for(experiment, algorithm, constraint_mode, seed, n, snr)
        save_adjacency(adj, columns, graph_path)
        metrics = dag_metrics(adj, truth)
        row.update(metrics)
        row.update({
            "status": "success",
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "violations": count_violations(adj, columns, forbidden),
            "n_columns": len(columns),
            "n_forbidden": len(forbidden),
            "graph_path": str(graph_path.relative_to(ROOT)),
        })
    except Exception as exc:
        row.update({
            "status": "failed",
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc).replace("\n", " ")[:1000],
        })
        trace_path = EXP_DIR / f"{experiment}_failures.log"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                f"{algorithm}/{constraint_mode}/seed={seed}/n={n}/snr={snr}\n"
            )
            handle.write(traceback.format_exc())
    return row


def run_experiment_pair(
    algorithm: str,
    seed: int,
    n: int,
    snr: float,
    experiment: str,
    dataset: str = "causal_dummy_v2",
) -> list[dict[str, Any]]:
    """Run unconstrained/constrained rows for one algorithm and data seed.

    PC uses native background knowledge, so the two modes are fit separately.
    LiNGAM, NOTEARS, and GES use forbidden-edge post-processing, so the
    constrained row is derived from the same learned unconstrained graph.
    """
    if algorithm.strip().lower() == "pc":
        return [
            run_experiment_cell(algorithm, "unconstrained", seed, n, snr, experiment, dataset),
            run_experiment_cell(algorithm, "constrained", seed, n, snr, experiment, dataset),
        ]

    rows = [
        empty_result_row(algorithm, dataset, "unconstrained", seed, n, snr),
        empty_result_row(algorithm, dataset, "constrained", seed, n, snr),
    ]
    started = time.perf_counter()
    try:
        df, truth, columns = generate_causal_dummy_numeric(n=n, seed=seed, snr=snr)
        forbidden = load_forbidden_constraints(columns)
        data = standardize(df.to_numpy(dtype=float))
        unconstrained = run_algorithm(
            algorithm=algorithm,
            data=data,
            columns=columns,
            constraint_mode="unconstrained",
            forbidden=forbidden,
            seed=seed,
        )
        constrained = apply_forbidden(unconstrained, columns, forbidden)
        elapsed = round(time.perf_counter() - started, 4)
        for row, constraint_mode, adjacency in zip(rows, CONSTRAINT_MODES, [unconstrained, constrained]):
            graph_path = graph_path_for(experiment, algorithm, constraint_mode, seed, n, snr)
            save_adjacency(adjacency, columns, graph_path)
            metrics = dag_metrics(adjacency, truth)
            row.update(metrics)
            row.update({
                "status": "success",
                "runtime_seconds": elapsed,
                "violations": count_violations(adjacency, columns, forbidden),
                "n_columns": len(columns),
                "n_forbidden": len(forbidden),
                "graph_path": str(graph_path.relative_to(ROOT)),
            })
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 4)
        for row in rows:
            row.update({
                "status": "failed",
                "runtime_seconds": elapsed,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc).replace("\n", " ")[:1000],
            })
        trace_path = EXP_DIR / f"{experiment}_failures.log"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                f"{algorithm}/paired/seed={seed}/n={n}/snr={snr}\n"
            )
            handle.write(traceback.format_exc())
    return rows


def summarize_results(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Create mean/std summaries for successful rows, with failure counts."""
    if results.empty:
        return pd.DataFrame()
    data = results.copy()
    ok = data[data["status"].astype(str).eq("success")].copy()
    grouped_all = data.groupby(group_cols, dropna=False)
    counts = grouped_all["status"].agg(
        total_runs="count",
        successful_runs=lambda s: int((s.astype(str) == "success").sum()),
        failed_runs=lambda s: int((s.astype(str) != "success").sum()),
    )
    if ok.empty:
        return counts.reset_index()
    summary = ok.groupby(group_cols, dropna=False)[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.join(counts, how="outer").reset_index()
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    return summary


def write_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Write a CSV with stable column order where requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        for column in columns:
            if column not in df.columns:
                df[column] = np.nan
        df = df[columns]
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a compact Markdown table."""
    if df.empty:
        return "No rows available."
    render = df.head(max_rows).copy()
    for column in render.columns:
        if pd.api.types.is_numeric_dtype(render[column]):
            render[column] = render[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
        else:
            render[column] = render[column].fillna("").astype(str)
    header = "| " + " | ".join(render.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(render.columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in render.columns) + " |"
        for _, row in render.iterrows()
    ]
    if len(df) > max_rows:
        body.append(f"| ... | {len(df) - max_rows} more rows |" + " |" * max(0, len(render.columns) - 2))
    return "\n".join([header, separator, *body])


def write_experiment_report(
    path: Path,
    title: str,
    command: str,
    design_lines: list[str],
    outputs: list[Path],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    headline_cols: list[str],
) -> None:
    """Write a concise experiment report."""
    failures = results[~results["status"].astype(str).eq("success")].copy() if not results.empty else pd.DataFrame()
    headline = summary[headline_cols].copy() if not summary.empty else pd.DataFrame()
    output_lines = "\n".join(f"- `{path.relative_to(ROOT)}`" for path in outputs)
    text = f"""# {title}

## Command

`{command}`

## Design

{chr(10).join(f"- {line}" for line in design_lines)}

## Outputs

{output_lines}

## Failures

{markdown_table(failures[["algorithm", "constraint_mode", "seed", "n", "snr", "error_type", "error_message"]]) if not failures.empty else "No failed rows."}

## Headline Metrics

{markdown_table(headline)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_line_metric(
    summary: pd.DataFrame,
    x_col: str,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    algorithms: list[str],
) -> None:
    """Plot a metric sweep over SNR or sample size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    colors = {
        "PC": "#2563EB",
        "LiNGAM": "#D97706",
        "GES": "#059669",
        "NOTEARS": "#7C3AED",
        "DECI": "#DC2626",
    }
    styles = {"unconstrained": "--", "constrained": "-"}
    labels = {"unconstrained": "unconstrained", "constrained": "constrained"}

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    x_vals = sorted(summary[x_col].dropna().unique())
    for algorithm in algorithms:
        for constraint_mode in CONSTRAINT_MODES:
            sub = summary[
                summary["algorithm"].astype(str).eq(algorithm)
                & summary["constraint_mode"].astype(str).eq(constraint_mode)
            ].copy()
            if sub.empty or mean_col not in sub.columns:
                continue
            sub = sub.set_index(x_col).reindex(x_vals)
            y = sub[mean_col].astype(float)
            yerr = sub[std_col].fillna(0.0).astype(float) if std_col in sub else pd.Series(0.0, index=y.index)
            label = f"{algorithm} {labels[constraint_mode]}"
            ax.plot(x_vals, y, marker="o", linestyle=styles[constraint_mode], color=colors.get(algorithm), label=label)
            ax.fill_between(x_vals, y - yerr, y + yerr, color=colors.get(algorithm), alpha=0.08)
    ax.set_xlabel("SNR" if x_col == "snr" else "Sample size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_grouped_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    algorithms: list[str],
) -> None:
    """Plot final comparison grouped bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    pivot = summary.pivot_table(index="algorithm", columns="constraint_mode", values=mean_col, aggfunc="first")
    err = summary.pivot_table(index="algorithm", columns="constraint_mode", values=std_col, aggfunc="first") if std_col in summary else None
    pivot = pivot.reindex([algorithm for algorithm in algorithms if algorithm in pivot.index])
    if pivot.empty:
        return
    if err is not None:
        err = err.reindex(pivot.index)
    x = np.arange(len(pivot.index))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.bar(
        x - width / 2,
        pivot.get("unconstrained", pd.Series(index=pivot.index, dtype=float)),
        width,
        label="unconstrained",
        yerr=None if err is None else err.get("unconstrained"),
        capsize=3,
        color="#60A5FA",
    )
    ax.bar(
        x + width / 2,
        pivot.get("constrained", pd.Series(index=pivot.index, dtype=float)),
        width,
        label="constrained",
        yerr=None if err is None else err.get("constrained"),
        capsize=3,
        color="#34D399",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
