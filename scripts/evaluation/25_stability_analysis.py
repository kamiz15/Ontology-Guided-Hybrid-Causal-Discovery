"""Compute edge-stability summaries for available experiment graphs."""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CORE_DIR = REPO_ROOT / "scripts" / "core"
for _path in (SCRIPT_DIR, CORE_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from causal_dummy_experiment_utils import (
    EXP_DIR,
    FIG_DIR,
    GRAPH_DIR,
    ROOT,
    build_ground_truth_adj,
    causal_dummy_columns,
    ensure_dirs,
    log_command,
    markdown_table,
)


OUTPUTS = {
    "causal_dummy": {
        "edges": EXP_DIR / "stability_edges_causal_dummy.csv",
        "summary": EXP_DIR / "stability_summary_causal_dummy.csv",
        "figure": FIG_DIR / "stability_causal_dummy.png",
    },
    "advisor_dummy": {
        "edges": EXP_DIR / "stability_edges_advisor_dummy.csv",
        "summary": EXP_DIR / "stability_summary_advisor_dummy.csv",
        "figure": FIG_DIR / "stability_advisor_dummy.png",
    },
    "real_ecb": {
        "edges": EXP_DIR / "stability_edges_real_ecb.csv",
        "summary": EXP_DIR / "stability_summary_real_ecb.csv",
        "figure": FIG_DIR / "stability_real_ecb.png",
    },
}


def read_csv_graph(path: Path) -> set[tuple[str, str]]:
    """Read a labeled adjacency CSV into an edge set."""
    adj = pd.read_csv(path, index_col=0)
    columns = [str(column) for column in adj.columns]
    matrix = (adj.to_numpy() != 0).astype(int)
    edges: set[tuple[str, str]] = set()
    for i, j in zip(*np.where(matrix == 1)):
        if i < len(columns) and j < len(columns) and i != j:
            edges.add((columns[i], columns[j]))
    return edges


def read_gml_graph(path: Path) -> set[tuple[str, str]]:
    """Read a directed GML graph into an edge set."""
    graph = nx.read_gml(path)
    return {(str(source), str(target)) for source, target in graph.edges()}


def safe_read_edges(path: Path) -> set[tuple[str, str]]:
    """Read supported graph formats."""
    if path.suffix.lower() == ".csv":
        return read_csv_graph(path)
    if path.suffix.lower() == ".gml":
        return read_gml_graph(path)
    return set()


def causal_dummy_records() -> list[dict[str, Any]]:
    """Load graph records from the final causal-dummy comparison."""
    raw_path = EXP_DIR / "causal_dummy_final_comparison_raw.csv"
    if not raw_path.exists():
        return []
    raw = pd.read_csv(raw_path)
    raw = raw[raw["status"].astype(str).eq("success")].copy()
    records: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        graph_rel = str(row.get("graph_path", ""))
        if not graph_rel:
            continue
        path = ROOT / graph_rel
        if not path.exists():
            continue
        records.append({
            "dataset": "causal_dummy",
            "algorithm": str(row["algorithm"]),
            "constraint_mode": str(row["constraint_mode"]),
            "seed": int(row["seed"]),
            "path": path,
        })
    return records


def ges_records(dataset: str) -> list[dict[str, Any]]:
    """Load available GES graph records for advisor dummy or real ECB."""
    pattern = re.compile(r"ges_(advisor_dummy|real)_(.+)_seed(\d+)\.csv$")
    records: list[dict[str, Any]] = []
    for path in (EXP_DIR / "graphs").glob("ges_*_seed*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        file_dataset, constraint_label, seed = match.groups()
        normalized = "real_ecb" if file_dataset == "real" else file_dataset
        if normalized != dataset:
            continue
        records.append({
            "dataset": dataset,
            "algorithm": "GES",
            "constraint_mode": constraint_label,
            "seed": int(seed),
            "path": path,
        })
    return records


def true_edges_for(dataset: str) -> set[tuple[str, str]]:
    """Return true edges where an actual ground-truth DAG exists."""
    if dataset != "causal_dummy":
        return set()
    columns = causal_dummy_columns()
    truth = build_ground_truth_adj(columns)
    return {
        (columns[i], columns[j])
        for i, j in zip(*np.where(truth == 1))
    }


def jaccard(edges_a: set[tuple[str, str]], edges_b: set[tuple[str, str]]) -> float:
    """Jaccard similarity between two edge sets."""
    union = edges_a | edges_b
    if not union:
        return 1.0
    return len(edges_a & edges_b) / len(union)


def analyze_dataset(dataset: str, records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute edge-frequency and summary stability tables."""
    edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    truth = true_edges_for(dataset)

    loaded: list[dict[str, Any]] = []
    for record in records:
        edges = safe_read_edges(Path(record["path"]))
        loaded.append({**record, "edges": edges, "edge_count": len(edges)})

    if not loaded:
        return pd.DataFrame(edge_rows), pd.DataFrame(summary_rows)

    for (algorithm, constraint_mode), group_df in pd.DataFrame(loaded).groupby(["algorithm", "constraint_mode"]):
        group = group_df.to_dict("records")
        n_graphs = len(group)
        counts: dict[tuple[str, str], int] = {}
        for record in group:
            for edge in record["edges"]:
                counts[edge] = counts.get(edge, 0) + 1
        for (source, target), present_count in sorted(counts.items()):
            frequency = present_count / n_graphs if n_graphs else 0.0
            row = {
                "dataset": dataset,
                "algorithm": algorithm,
                "constraint_mode": constraint_mode,
                "source": source,
                "target": target,
                "present_count": present_count,
                "n_graphs": n_graphs,
                "frequency": frequency,
                "stable_60": frequency >= 0.60,
            }
            if truth:
                row["true_positive"] = (source, target) in truth
                row["false_positive"] = (source, target) not in truth
            else:
                row["true_positive"] = np.nan
                row["false_positive"] = np.nan
            edge_rows.append(row)

        similarities = [
            jaccard(left["edges"], right["edges"])
            for left, right in combinations(group, 2)
        ]
        stable_edges = [edge for edge, count in counts.items() if count / n_graphs >= 0.60]
        summary = {
            "dataset": dataset,
            "algorithm": algorithm,
            "constraint_mode": constraint_mode,
            "n_graphs": n_graphs,
            "n_unique_edges": len(counts),
            "mean_edge_count": float(np.mean([record["edge_count"] for record in group])) if group else 0.0,
            "stable_edges_60": len(stable_edges),
            "jaccard_mean": float(np.mean(similarities)) if similarities else np.nan,
            "jaccard_std": float(np.std(similarities, ddof=1)) if len(similarities) > 1 else 0.0,
        }
        if truth:
            summary["stable_true_positive_edges"] = sum(1 for edge in stable_edges if edge in truth)
            summary["stable_false_positive_edges"] = sum(1 for edge in stable_edges if edge not in truth)
        else:
            summary["stable_true_positive_edges"] = np.nan
            summary["stable_false_positive_edges"] = np.nan
        summary_rows.append(summary)

    return pd.DataFrame(edge_rows), pd.DataFrame(summary_rows)


def plot_stability(summary: pd.DataFrame, path: Path, title: str) -> None:
    """Plot stable-edge counts and Jaccard means."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
        ax.text(0.5, 0.5, "No graph files available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    data = summary.copy()
    data["label"] = data["algorithm"].astype(str) + "\n" + data["constraint_mode"].astype(str)
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(max(8, len(data) * 0.8), 5), dpi=160)
    ax1.bar(x, data["stable_edges_60"].astype(float), color="#60A5FA", label="stable edges >=60%")
    ax1.set_ylabel("Stable edge count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(data["label"], rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, data["jaccard_mean"].astype(float), color="#059669", marker="o", label="mean Jaccard")
    ax2.set_ylabel("Mean Jaccard")
    ax2.set_ylim(0, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(all_summaries: dict[str, pd.DataFrame]) -> None:
    """Write a concise stability report for the command log."""
    path = EXP_DIR / "stability_analysis_report.md"
    sections = ["# Stability Analysis Report", "", f"`{' '.join([str(Path(__file__).name), *([])])}`", ""]
    for dataset, summary in all_summaries.items():
        sections.append(f"## {dataset}")
        sections.append(markdown_table(summary))
        sections.append("")
    path.write_text("\n".join(sections), encoding="utf-8")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute graph stability for available outputs.")
    args = parser.parse_args()
    ensure_dirs()
    log_command("25_stability_analysis.py", vars(args))

    datasets = {
        "causal_dummy": causal_dummy_records(),
        "advisor_dummy": ges_records("advisor_dummy"),
        "real_ecb": ges_records("real_ecb"),
    }

    summaries: dict[str, pd.DataFrame] = {}
    for dataset, records in datasets.items():
        edges, summary = analyze_dataset(dataset, records)
        outputs = OUTPUTS[dataset]
        edges.to_csv(outputs["edges"], index=False)
        summary.to_csv(outputs["summary"], index=False)
        plot_stability(summary, outputs["figure"], f"Stability: {dataset.replace('_', ' ')}")
        summaries[dataset] = summary
        print(
            f"[stability] {dataset}: records={len(records)} "
            f"summary -> {outputs['summary']}",
            flush=True,
        )

    write_report(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
