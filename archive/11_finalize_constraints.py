# 11_finalize_constraints.py
# ============================================================
# Step 11 - Finalize human-reviewed literature constraints.
#
# Reads the reviewed constraint CSV, filters approved rows separately for
# real and synthetic variable sets, writes dataset-specific constraint
# modules, and writes literature-derived adjacency matrices and coverage.
#
# Usage:
#   python 11_finalize_constraints.py
#   python 11_finalize_constraints.py --dry-run
#   python 11_finalize_constraints.py --dataset real
#   python 11_finalize_constraints.py --dataset synthetic
#
# Output:
#   04_forbidden_edges_real.py
#   04_forbidden_edges_synthetic.py
#   04_forbidden_edges.py
#   data/processed/ground_truth_adjacency_real.csv
#   data/processed/ground_truth_adjacency.csv
#   data/synthetic/ground_truth_constraints_synthetic.csv
#   reports/constraint_coverage_real.md
#   reports/constraint_coverage_synthetic.md
#   reports/constraints_skipped_real.csv
#   reports/constraints_skipped_synthetic.csv
# ============================================================

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from config import (
    CLAIM_AGGREGATION_PATH,
    CONSTRAINT_COVERAGE_REAL_PATH,
    CONSTRAINT_COVERAGE_SYNTHETIC_PATH,
    CONSTRAINTS_REVIEW_PATH,
    CONSTRAINTS_SKIPPED_REAL_PATH,
    CONSTRAINTS_SKIPPED_SYNTHETIC_PATH,
    FORBIDDEN_EDGES_MODULE_PATH,
    FORBIDDEN_EDGES_REAL_MODULE_PATH,
    FORBIDDEN_EDGES_SYNTHETIC_MODULE_PATH,
    GROUND_TRUTH_ADJACENCY_PATH,
    GROUND_TRUTH_ADJACENCY_REAL_PATH,
    READY_DATA_PATH,
    SYNTHETIC_CONSTRAINT_ADJACENCY_PATH,
    SYNTHETIC_DIR,
)


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_ACTIONS = {"required", "forbid_reverse"}
DATASET_CHOICES = {"real", "synthetic", "both"}


@dataclass(frozen=True)
class DatasetOutputs:
    """Output configuration for one dataset-specific constraint target."""

    name: str
    variable_path: str
    module_path: str
    adjacency_path: str
    coverage_path: str
    skipped_path: str
    compatibility_adjacency_path: str | None = None


def _project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _split_constraint_cell(value: Any) -> list[str]:
    """
    Split a semicolon-bundled constraint endpoint cell.

    Parameters
    ----------
    value : Any
        Raw cause/effect value.

    Returns
    -------
    list[str]
        Non-empty stripped endpoint names.
    """
    text = _clean_text(value)
    parts = [part.strip() for part in text.split(";") if part.strip()]
    return parts if parts else [text]


def _paper_count_int(value: Any) -> int:
    """
    Parse a paper-count value.

    Parameters
    ----------
    value : Any
        Raw `paper_count` cell.

    Returns
    -------
    int
        Integer paper count, or zero if unavailable.
    """
    try:
        if value is None or pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _is_truthy(value: Any) -> bool:
    """
    Interpret common truthy encodings from CSV values.

    Parameters
    ----------
    value : Any
        Raw boolean-like value.

    Returns
    -------
    bool
        True when the value represents a true/yes/1 flag.
    """
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _normalise_action(value: Any) -> str:
    return _clean_text(value).lower()


def _ensure_output_dir(path: str | Path) -> None:
    directory = Path(path).parent
    if str(directory) and str(directory) != ".":
        directory.mkdir(parents=True, exist_ok=True)


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return
    backup_path = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup_path)
    print(f"[step_11] Backup written -> {backup_path}")


def _dataset_configs() -> dict[str, DatasetOutputs]:
    return {
        "real": DatasetOutputs(
            name="real",
            variable_path=READY_DATA_PATH,
            module_path=FORBIDDEN_EDGES_REAL_MODULE_PATH,
            adjacency_path=GROUND_TRUTH_ADJACENCY_REAL_PATH,
            compatibility_adjacency_path=GROUND_TRUTH_ADJACENCY_PATH,
            coverage_path=CONSTRAINT_COVERAGE_REAL_PATH,
            skipped_path=CONSTRAINTS_SKIPPED_REAL_PATH,
        ),
        "synthetic": DatasetOutputs(
            name="synthetic",
            variable_path=f"{SYNTHETIC_DIR}/synthetic_n2000.csv",
            module_path=FORBIDDEN_EDGES_SYNTHETIC_MODULE_PATH,
            adjacency_path=SYNTHETIC_CONSTRAINT_ADJACENCY_PATH,
            coverage_path=CONSTRAINT_COVERAGE_SYNTHETIC_PATH,
            skipped_path=CONSTRAINTS_SKIPPED_SYNTHETIC_PATH,
        ),
    }


def _selected_datasets(choice: str) -> list[DatasetOutputs]:
    configs = _dataset_configs()
    if choice == "both":
        return [configs["real"], configs["synthetic"]]
    return [configs[choice]]


def _load_columns(path: str) -> list[str]:
    csv_path = _project_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Variable reference dataset not found: {csv_path}")
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def _parse_paper_ids(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return sorted({_clean_text(v) for v in parsed if _clean_text(v)})
    except json.JSONDecodeError:
        pass

    parts = text.split(";") if ";" in text else text.split(",")
    return sorted({_clean_text(part) for part in parts if _clean_text(part)})


def _load_contradiction_index(path: str = CLAIM_AGGREGATION_PATH) -> dict[tuple[str, str], bool]:
    """
    Load contradiction flags from the claim aggregation table.

    The aggregation table can itself contain semicolon-bundled cause/effect
    cells, so each bundled row is expanded into the same Cartesian-product
    edge keys used by the review CSV splitter.

    Parameters
    ----------
    path : str, optional
        Claim aggregation CSV path.

    Returns
    -------
    dict[tuple[str, str], bool]
        Mapping from `(cause, effect)` to whether any matching aggregated
        evidence was flagged as contradictory.
    """
    aggregation_path = _project_path(path)
    if not aggregation_path.exists():
        print(f"[step_11] Claim aggregation not found for contradiction lookup: {aggregation_path}")
        return {}

    df = pd.read_csv(aggregation_path)
    required_cols = {"has_contradiction"}
    if "cause" not in df.columns and "cause_mapped" not in df.columns:
        return {}
    if "effect" not in df.columns and "effect_mapped" not in df.columns:
        return {}
    if not required_cols.issubset(df.columns):
        return {}

    cause_col = "cause" if "cause" in df.columns else "cause_mapped"
    effect_col = "effect" if "effect" in df.columns else "effect_mapped"
    contradiction_index: dict[tuple[str, str], bool] = {}

    for _, row in df.iterrows():
        causes = _split_constraint_cell(row[cause_col])
        effects = _split_constraint_cell(row[effect_col])
        has_contradiction = _is_truthy(row["has_contradiction"])
        for cause in causes:
            for effect in effects:
                key = (cause, effect)
                contradiction_index[key] = contradiction_index.get(key, False) or has_contradiction

    print(f"[step_11] Loaded contradiction lookup for {len(contradiction_index)} aggregated edges.")
    return contradiction_index


def _row_has_contradiction(
    row: pd.Series,
    contradiction_index: dict[tuple[str, str], bool] | None,
) -> bool:
    """
    Check whether a reviewed row has contradictory evidence.

    Parameters
    ----------
    row : pd.Series
        Approved review row.
    contradiction_index : dict[tuple[str, str], bool] or None
        Lookup produced by :func:`_load_contradiction_index`.

    Returns
    -------
    bool
        True when the row or its aggregation lookup indicates contradiction.
    """
    if "has_contradiction" in row.index and _is_truthy(row["has_contradiction"]):
        return True
    if not contradiction_index:
        return False
    key = (_clean_text(row["cause"]), _clean_text(row["effect"]))
    return contradiction_index.get(key, False)


def _paper_comment(value: Any) -> str:
    papers = _parse_paper_ids(value)
    return "[" + ", ".join(papers) + "]"


def _py_string(value: str) -> str:
    return json.dumps(value)


def _extract_constraints(
    df: pd.DataFrame,
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    required = []
    forbidden = []
    promoted_required = 0

    for _, row in df.iterrows():
        action = _normalise_action(row["proposed_action"])
        cause = _clean_text(row["cause"])
        effect = _clean_text(row["effect"])
        paper_count = _paper_count_int(row.get("paper_count", 0))
        has_contradiction = _row_has_contradiction(row, contradiction_index)
        record = {
            "cause": cause,
            "effect": effect,
            "tier": int(row["tier"]),
            "paper_ids": _clean_text(row.get("paper_ids", "")),
            "paper_count": paper_count,
            "has_contradiction": has_contradiction,
        }

        if action not in VALID_ACTIONS:
            raise ValueError(
                "Approved rows must use proposed_action 'required' or "
                f"'forbid_reverse', got: {action}"
            )

        if action == "required":
            required.append({"source": cause, "target": effect, **record})
        else:
            if paper_count >= 2 and not has_contradiction:
                required.append({
                    "source": cause,
                    "target": effect,
                    "required_reason": "multi-paper forbid_reverse promotion",
                    **record,
                })
                promoted_required += 1
            forbidden.append({"source": effect, "target": cause, **record})

    return required, forbidden, promoted_required


def load_reviewed(path: str) -> pd.DataFrame:
    """
    Load approved rows from the human-reviewed constraint CSV.

    Parameters
    ----------
    path : str
        Path to the reviewed CSV. Rows are kept only when `approved` equals
        `yes`, case-insensitively.

    Returns
    -------
    pd.DataFrame
        Approved constraint rows.

    Raises
    ------
    FileNotFoundError
        If the reviewed CSV does not exist.
    ValueError
        If required review columns are missing.
    """
    review_path = _project_path(path)
    if not review_path.exists():
        raise FileNotFoundError(f"Reviewed CSV not found: {review_path}")

    df = pd.read_csv(review_path)
    required_cols = [
        "cause",
        "effect",
        "tier",
        "proposed_action",
        "paper_count",
        "top_quote",
        "paper_ids",
        "approved",
        "notes",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Reviewed CSV missing required columns: {missing}")

    approved = df[df["approved"].map(_normalise_action) == "yes"].copy()
    approved["proposed_action"] = approved["proposed_action"].map(_normalise_action)
    approved["cause"] = approved["cause"].map(_clean_text)
    approved["effect"] = approved["effect"].map(_clean_text)

    print(f"[step_11] Approved rows loaded: {len(approved)}")
    return approved


def filter_constraints_for_dataset(
    df: pd.DataFrame,
    columns: list[str],
    skipped_path: str,
    dataset_name: str,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, int]:
    """
    Keep only approved constraints whose endpoints exist in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Approved rows from :func:`load_reviewed`.
    columns : list[str]
        Dataset variable names.
    skipped_path : str
        CSV path for skipped rows.
    dataset_name : str
        Label used in logs and skipped output.
    dry_run : bool, optional
        If True, do not write the skipped-row CSV.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Filtered approved rows and number skipped for missing variables.
    """
    variable_set = set(columns)
    kept_rows = []
    skipped_rows = []

    for _, row in df.iterrows():
        cause = _clean_text(row["cause"])
        effect = _clean_text(row["effect"])
        missing = sorted({node for node in [cause, effect] if node not in variable_set})
        if missing:
            skipped_rows.append({
                "dataset": dataset_name,
                "cause": cause,
                "effect": effect,
                "tier": row["tier"],
                "proposed_action": row["proposed_action"],
                "paper_ids": row.get("paper_ids", ""),
                "missing_variables": "; ".join(missing),
            })
            continue
        kept_rows.append(row)

    filtered = pd.DataFrame(kept_rows, columns=df.columns)
    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=[
            "dataset",
            "cause",
            "effect",
            "tier",
            "proposed_action",
            "paper_ids",
            "missing_variables",
        ],
    )

    skipped_out = _project_path(skipped_path)
    if dry_run:
        print(f"[step_11] Dry run: would write skipped log -> {skipped_out}")
    else:
        _ensure_output_dir(skipped_out)
        skipped_df.to_csv(skipped_out, index=False)
        print(f"[step_11] Skipped log -> {skipped_out}")

    return filtered, len(skipped_rows)


def validate_constraints(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str = "dataset",
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> None:
    """
    Validate filtered constraints for one dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Approved and dataset-filtered constraint rows.
    columns : list[str]
        Dataset variable names.
    dataset_name : str, optional
        Dataset label for logs and error messages.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If required edges contain a cycle or an edge is both required and
        forbidden after dataset filtering.
    """
    required, forbidden, _ = _extract_constraints(df, contradiction_index)
    required_edges = {(item["source"], item["target"]) for item in required}
    forbidden_edges = {(item["source"], item["target"]) for item in forbidden}

    overlap = sorted(required_edges & forbidden_edges)
    if overlap:
        raise ValueError(
            f"{dataset_name}: edges present in both required and forbidden sets: {overlap}"
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    graph.add_edges_from(required_edges)
    try:
        cycle = nx.find_cycle(graph, orientation="original")
    except nx.NetworkXNoCycle:
        cycle = []
    if cycle:
        readable = [(src, tgt) for src, tgt, _ in cycle]
        raise ValueError(f"{dataset_name}: required edges contain a cycle: {readable}")

    print(f"[step_11] {dataset_name}: constraint validation passed.")


def _build_module_content(
    df: pd.DataFrame,
    dataset_name: str,
    alias_note: str | None = None,
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> str:
    required, forbidden, _ = _extract_constraints(df, contradiction_index)

    def tuple_line(item: dict[str, Any]) -> str:
        source = _py_string(item["source"])
        target = _py_string(item["target"])
        papers = _paper_comment(item.get("paper_ids", ""))
        return f"    ({source}, {target}),  # tier={item['tier']}; papers={papers}"

    forbidden_lines = "\n".join(tuple_line(item) for item in forbidden)
    required_lines = "\n".join(tuple_line(item) for item in required)
    module_label = "04_forbidden_edges.py" if alias_note else f"04_forbidden_edges_{dataset_name}.py"
    docstring = (
        '"""\n'
        f"{alias_note}\n"
        'Canonical files are 04_forbidden_edges_real.py and '
        '04_forbidden_edges_synthetic.py.\n'
        '"""\n\n'
        if alias_note else ""
    )

    return f'''{docstring}# {module_label}
# ============================================================
# Step 4 - Literature-reviewed causal constraints for {dataset_name} data.
# Defines forbidden and required edges from human-approved
# RAG/literature claims after dataset-specific variable filtering.
#
# This file is generated by 11_finalize_constraints.py.
#
# Usage (standalone check):
#   python {module_label}
# ============================================================

from __future__ import annotations


# Each tuple: (source, target) means "forbid a causal arrow
# FROM source TO target".
FORBIDDEN_EDGES: list[tuple[str, str]] = [
{forbidden_lines}
]


# Each tuple: (source, target) means "require a causal arrow
# FROM source TO target".
REQUIRED_EDGES: list[tuple[str, str]] = [
{required_lines}
]


def build_background_knowledge(available_columns: list[str]):
    """
    Returns a causal-learn BackgroundKnowledge object with all
    forbidden and required edges that apply to the given columns.

    Parameters
    ----------
    available_columns : list of str
        Column names in the dataset.

    Returns
    -------
    bk : BackgroundKnowledge
    applied_forbidden : int
    applied_required : int
    """
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode

    col_set = set(available_columns)
    node_map = {{col: GraphNode(col) for col in available_columns}}
    bk = BackgroundKnowledge()
    applied_forbidden = 0
    applied_required = 0
    skipped = []

    for src, tgt in FORBIDDEN_EDGES:
        if src in col_set and tgt in col_set:
            bk.add_forbidden_by_node(node_map[src], node_map[tgt])
            applied_forbidden += 1
        else:
            skipped.append(("forbidden", src, tgt))

    for src, tgt in REQUIRED_EDGES:
        if src in col_set and tgt in col_set:
            bk.add_required_by_node(node_map[src], node_map[tgt])
            applied_required += 1
        else:
            skipped.append(("required", src, tgt))

    print(f"[constraints] Forbidden edges applied : {{applied_forbidden}}")
    print(f"[constraints] Required edges applied  : {{applied_required}}")
    if skipped:
        print(f"[constraints] Skipped (cols not in data): {{len(skipped)}}")
        for kind, source, target in skipped:
            print(f"[constraints] {{kind}}: {{source}} -> {{target}}")

    return bk, applied_forbidden, applied_required


if __name__ == "__main__":
    print(f"[constraints] Forbidden edges defined : {{len(FORBIDDEN_EDGES)}}")
    print(f"[constraints] Required edges defined  : {{len(REQUIRED_EDGES)}}")
'''


def regenerate_forbidden_edges_module(
    df: pd.DataFrame,
    path: str,
    dataset_name: str = "dataset",
    alias_note: str | None = None,
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> None:
    """
    Rewrite a dataset-specific forbidden-edge module.

    Parameters
    ----------
    df : pd.DataFrame
        Approved and dataset-filtered rows.
    path : str
        Path to the module to overwrite.
    dataset_name : str, optional
        Dataset label used in generated comments.
    alias_note : str or None, optional
        Optional compatibility docstring for alias modules.

    Returns
    -------
    None
    """
    module_path = _project_path(path)
    _ensure_output_dir(module_path)
    _backup_existing(module_path)
    module_path.write_text(
        _build_module_content(
            df,
            dataset_name=dataset_name,
            alias_note=alias_note,
            contradiction_index=contradiction_index,
        ),
        encoding="utf-8",
    )
    print(f"[step_11] Module rewritten -> {module_path}")


def build_ground_truth_adjacency(
    df: pd.DataFrame,
    columns: list[str],
    path: str,
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> None:
    """
    Build a literature-derived constraint adjacency matrix.

    This artifact is derived from approved literature/RAG claims, not from
    the observed dataset. For synthetic experiments, it is distinct from the
    SCM-derived true DAG at `data/synthetic/ground_truth_adjacency.csv`.

    Parameters
    ----------
    df : pd.DataFrame
        Approved and dataset-filtered rows.
    columns : list[str]
        Dataset variable names.
    path : str
        Output CSV path.

    Returns
    -------
    None
    """
    required, forbidden, _ = _extract_constraints(df, contradiction_index)
    matrix = pd.DataFrame(0, index=columns, columns=columns, dtype=int)

    for item in required:
        matrix.loc[item["source"], item["target"]] = 1
    for item in forbidden:
        matrix.loc[item["source"], item["target"]] = -1

    out_path = _project_path(path)
    _ensure_output_dir(out_path)
    _backup_existing(out_path)
    matrix.to_csv(out_path)
    print(f"[step_11] Literature constraint adjacency -> {out_path}")


def coverage_report(
    df: pd.DataFrame,
    columns: list[str],
    path: str,
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> None:
    """
    Write a markdown coverage report for one dataset's constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Approved and dataset-filtered rows.
    columns : list[str]
        Dataset variable names.
    path : str
        Output markdown path.

    Returns
    -------
    None
    """
    required, forbidden, _ = _extract_constraints(df, contradiction_index)
    all_edges = required + forbidden

    touched = {item["source"] for item in all_edges} | {item["target"] for item in all_edges}
    isolated = [col for col in columns if col not in touched]
    coverage = (len(touched) / len(columns) * 100) if columns else 0.0

    counts = Counter()
    for item in all_edges:
        counts[item["source"]] += 1
        counts[item["target"]] += 1

    lines = [
        "# Constraint Coverage",
        "",
        f"- Dataset variables: {len(columns)}",
        f"- Approved required edges: {len(required)}",
        f"- Approved forbidden-reverse edges: {len(forbidden)}",
        f"- Variables touched: {len(touched)} ({coverage:.1f}%)",
        f"- Isolated variables: {len(isolated)}",
        "",
        "## Top 10 Most-Constrained Variables",
        "",
    ]

    if counts:
        for var, count in counts.most_common(10):
            lines.append(f"- `{var}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Isolated Variables", ""])
    if isolated:
        lines.extend(f"- `{var}`" for var in isolated)
    else:
        lines.append("- None")

    out_path = _project_path(path)
    _ensure_output_dir(out_path)
    _backup_existing(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[step_11] Coverage report -> {out_path}")


def _write_compatibility_adjacency(source_path: str, target_path: str) -> None:
    source = _project_path(source_path)
    target = _project_path(target_path)
    _ensure_output_dir(target)
    _backup_existing(target)
    shutil.copy2(source, target)
    print(f"[step_11] Backward-compatible adjacency -> {target}")


def finalize_for_dataset(
    reviewed: pd.DataFrame,
    config: DatasetOutputs,
    dry_run: bool = False,
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> dict[str, int]:
    """
    Finalize constraints for one dataset target.

    Parameters
    ----------
    reviewed : pd.DataFrame
        Approved rows from :func:`load_reviewed`.
    config : DatasetOutputs
        Dataset-specific paths and labels.
    dry_run : bool, optional
        If True, validate and print planned writes without changing files.

    Returns
    -------
    dict[str, int]
        Counts for approved input, skipped rows, written rows, required edges,
        and forbidden edges.
    """
    columns = _load_columns(config.variable_path)
    filtered, skipped = filter_constraints_for_dataset(
        reviewed,
        columns,
        config.skipped_path,
        dataset_name=config.name,
        dry_run=dry_run,
    )
    validate_constraints(
        filtered,
        columns,
        dataset_name=config.name,
        contradiction_index=contradiction_index,
    )
    required, forbidden, promoted_required = _extract_constraints(filtered, contradiction_index)

    print(f"[step_11] {config.name}: approved in={len(reviewed)}, skipped={skipped}, written={len(filtered)}")
    print(f"[step_11] {config.name}: Required edges written: {len(required)}")
    print(f"[step_11] {config.name}: Forbidden edges written: {len(forbidden)}")
    print(
        f"[step_11] {config.name}: Reasoning: Required edges generated from "
        f"{promoted_required} multi-paper-supported forbid_reverse claims "
        "with paper_count >= 2"
    )

    if dry_run:
        print(f"[step_11] Dry run only for {config.name}; no files written.")
        print(f"[step_11] Would rewrite module    : {_project_path(config.module_path)}")
        print(f"[step_11] Would write adjacency  : {_project_path(config.adjacency_path)}")
        print(f"[step_11] Would write coverage   : {_project_path(config.coverage_path)}")
        if config.compatibility_adjacency_path:
            print(f"[step_11] Would update compat adjacency: {_project_path(config.compatibility_adjacency_path)}")
    else:
        regenerate_forbidden_edges_module(
            filtered,
            config.module_path,
            dataset_name=config.name,
            contradiction_index=contradiction_index,
        )
        build_ground_truth_adjacency(
            filtered,
            columns,
            config.adjacency_path,
            contradiction_index=contradiction_index,
        )
        if config.compatibility_adjacency_path:
            _write_compatibility_adjacency(config.adjacency_path, config.compatibility_adjacency_path)
        coverage_report(
            filtered,
            columns,
            config.coverage_path,
            contradiction_index=contradiction_index,
        )

    return {
        "input": len(reviewed),
        "skipped": skipped,
        "written": len(filtered),
        "required": len(required),
        "forbidden": len(forbidden),
        "promoted_required": promoted_required,
    }


def _write_synthetic_alias(
    reviewed: pd.DataFrame,
    dry_run: bool,
    contradiction_index: dict[tuple[str, str], bool] | None = None,
) -> None:
    synthetic_config = _dataset_configs()["synthetic"]
    columns = _load_columns(synthetic_config.variable_path)
    synthetic_filtered, _ = filter_constraints_for_dataset(
        reviewed,
        columns,
        synthetic_config.skipped_path,
        dataset_name="synthetic",
        dry_run=True,
    )
    alias_note = (
        "Backward-compatible alias of the synthetic literature-constraint "
        "module. Use 04_forbidden_edges_real.py or "
        "04_forbidden_edges_synthetic.py for canonical dataset-specific files."
    )
    if dry_run:
        print(f"[step_11] Would rewrite backward-compatible alias: {_project_path(FORBIDDEN_EDGES_MODULE_PATH)}")
        return
    regenerate_forbidden_edges_module(
        synthetic_filtered,
        FORBIDDEN_EDGES_MODULE_PATH,
        dataset_name="synthetic",
        alias_note=alias_note,
        contradiction_index=contradiction_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize reviewed literature constraints for real and synthetic datasets."
    )
    parser.add_argument("--reviewed-csv", default=CONSTRAINTS_REVIEW_PATH,
                        help="Human-reviewed constraint CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print planned writes without changing files")
    parser.add_argument("--dataset", choices=sorted(DATASET_CHOICES), default="both",
                        help="Dataset output set to finalize")
    args = parser.parse_args()

    reviewed = load_reviewed(args.reviewed_csv)
    contradiction_index = _load_contradiction_index()
    selected = _selected_datasets(args.dataset)
    summaries = {}

    for config in selected:
        summaries[config.name] = finalize_for_dataset(
            reviewed,
            config,
            dry_run=args.dry_run,
            contradiction_index=contradiction_index,
        )

    if args.dataset in {"synthetic", "both"}:
        _write_synthetic_alias(
            reviewed,
            dry_run=args.dry_run,
            contradiction_index=contradiction_index,
        )

    print("\n[step_11] Summary")
    for name in [cfg.name for cfg in selected]:
        item = summaries[name]
        print(
            f"[step_11] {name.title():10s}: "
            f"{item['input']} constraints in, "
            f"{item['skipped']} skipped (var missing), "
            f"{item['written']} written "
            f"({item['required']} required, {item['forbidden']} forbidden, "
            f"{item['promoted_required']} promoted required)"
        )


if __name__ == "__main__":
    main()
