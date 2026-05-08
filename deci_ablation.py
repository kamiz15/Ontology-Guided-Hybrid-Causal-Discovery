"""DECI/Causica ablation and diagnostic runner.

This module is intentionally called from ``run_all.py`` rather than replacing
the main experiment loop. It keeps DECI calibration separate from the PC,
LiNGAM, and NOTEARS comparisons, and uses synthetic data only for model and
threshold selection.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as project_config
import run_all as runner


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "experiments"
ABLATION_PATH = OUT / "deci_ablation_synthetic.csv"
ABLATION_SUMMARY_PATH = OUT / "deci_ablation_synthetic_summary.csv"
SELECTED_CONFIG_PATH = OUT / "deci_selected_config.json"
REAL_SELECTED_PATH = OUT / "deci_real_selected_config.csv"
VARIABLE_REPORT_PATH = OUT / "deci_variable_set_report.csv"
REDUCED_VARIABLES_PATH = OUT / "deci_reduced_variables.txt"
CONSTRAINT_VALIDATION_PATH = OUT / "deci_constraint_matrix_validation.csv"
CONSTRAINT_TYPE_ABLATION_PATH = OUT / "deci_constraint_type_ablation.csv"
STABLE_DETAILED_PATH = OUT / "deci_stable_edges_detailed.csv"
DECI_FAILURES_PATH = OUT / "deci_failures.csv"
DECI_REPORT_PATH = OUT / "deci_report.md"

ENV_VARS = {
    "scope_1_emissions_tco2e",
    "scope_2_emissions_tco2e",
    "scope_3_emissions_tco2e",
    "emission_reduction_policy_score",
    "renewable_energy_share",
    "total_energy_consumption",
    "environmental_fines",
    "iso_14001_exists",
    "carbon_intensity",
    "reporting_quality_score",
    "water_withdrawal",
    "hazardous_waste_generated",
    "env_pillar_score",
    "green_financing_eur",
}
SOC_VARS = {
    "training_hours",
    "injury_frequency_rate",
    "turnover_rate",
    "diversity_representation",
    "community_investment_eur",
    "customer_satisfaction_score",
    "health_safety_score",
    "fair_wage_gap",
    "human_rights_violations",
    "collective_bargaining_coverage",
    "union_membership",
    "soc_pillar_score",
}
GOV_VARS = {
    "board_strategy_esg_oversight_score",
    "board_diversity",
    "ceo_chair_split",
    "auditor_independence_score",
    "corruption_cases",
    "ethical_breaches",
    "shareholder_rights_score",
    "governance_compliance_score",
    "anti_competitive_violations",
    "gov_pillar_score",
}
COMPOSITES = {"overall_esg_score", "env_pillar_score", "soc_pillar_score", "gov_pillar_score"}
FINANCIAL_VARS = {
    "total_asset",
    "total_revenue_eur",
    "roa_eat",
    "roe_eat",
    "debt_to_equity_ratio",
    "tobins_q",
    "pe_ratio",
    "asset_growth_pct",
    "npl_ratio",
    "solvency_ratio",
    "systemic_risk",
    "funding_cost",
    "liquidity_risk",
    "liquidity_creation",
    "capital_adequacy_ratio",
    "default_probability",
    "credit_rating",
    "wacc",
}

ABLATION_COLUMNS = [
    "config_id",
    "dataset",
    "variable_set",
    "mode",
    "constraint_mode",
    "seed",
    "epochs",
    "threshold",
    "sparsity_strength",
    "l1_lambda",
    "backend",
    "status",
    "error_message",
    "runtime_seconds",
    "edge_count",
    "edge_count_true",
    "graph_density",
    "f1_directed",
    "precision",
    "recall",
    "shd",
    "violations",
    "forbidden_constraints_passed",
    "required_constraints_passed",
    "constraint_cells_changed",
    "raw_weights_path",
    "final_adjacency_path",
]

SUMMARY_COLUMNS = [
    "config_id",
    "variable_set",
    "mode",
    "constraint_mode",
    "epochs",
    "threshold",
    "sparsity_strength",
    "l1_lambda",
    "mean_f1",
    "std_f1",
    "mean_precision",
    "mean_recall",
    "mean_shd",
    "std_shd",
    "mean_edge_count",
    "std_edge_count",
    "mean_violations",
    "mean_runtime",
    "successful_runs",
    "failed_runs",
    "timed_out_runs",
    "success_rate",
    "acceptable",
    "selection_rank",
]

REAL_COLUMNS = [
    "config_id",
    "dataset",
    "variable_set",
    "mode",
    "seed",
    "threshold",
    "epochs",
    "sparsity_strength",
    "status",
    "error_message",
    "runtime_seconds",
    "edge_count",
    "alignment",
    "violations",
    "stable_edges_60",
    "stable_edges_80",
]

VALIDATION_COLUMNS = [
    "dataset",
    "variable_set",
    "source",
    "target",
    "constraint_type",
    "source_index",
    "target_index",
    "source_exists_in_dataset",
    "target_exists_in_dataset",
    "direction_confirmed",
    "conflicts_with_synthetic_ground_truth",
    "passed_to_causica",
    "notes",
]

STABLE_COLUMNS = [
    "dataset",
    "variable_set",
    "mode",
    "config_id",
    "source",
    "target",
    "frequency",
    "mean_weight",
    "median_weight",
    "std_weight",
    "appears_in_40_percent",
    "appears_in_60_percent",
    "appears_in_80_percent",
    "forbidden_edge",
    "required_edge",
    "ontology_supported",
    "in_synthetic_ground_truth",
    "correct_direction",
]

FAILURE_COLUMNS = [
    "config_id",
    "dataset",
    "variable_set",
    "mode",
    "constraint_mode",
    "seed",
    "epochs",
    "backend",
    "status",
    "exception_type",
    "error_message",
    "full_error_message",
    "traceback_short",
    "failure_phase",
    "runtime_seconds",
]


def log(message: str) -> None:
    """Print a DECI-ablation-prefixed log line."""
    print(f"[deci_ablation] {message}", flush=True)


def init_csv(path: Path, columns: list[str]) -> None:
    """Create or reset a CSV with a fixed header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()


def append_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    """Append a row with missing values filled by blanks."""
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writerow(
            {column: row.get(column, "") for column in columns}
        )


def dataset_key(cli_dataset: str) -> str:
    """Map CLI dataset aliases to the internal dataset key."""
    if cli_dataset == "synthetic":
        return "synthetic_n2000"
    return cli_dataset


def variable_domain(variable: str) -> str:
    """Return a coarse domain label for one variable."""
    if variable in ENV_VARS:
        return "E"
    if variable in SOC_VARS:
        return "S"
    if variable in GOV_VARS:
        return "G"
    if variable in FINANCIAL_VARS:
        return "financial"
    return "other"


def ontology_role(variable: str) -> str:
    """Return the ontology role used in the variable-set report."""
    if variable in COMPOSITES:
        return "composite_esg"
    if variable in ENV_VARS | SOC_VARS | GOV_VARS:
        return "ontology_component"
    if variable in FINANCIAL_VARS:
        return "financial_outcome_or_control"
    return "unmapped_numeric"


def compute_variable_stats(variable: str, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute missingness, variance, and correlation stats for a variable."""
    frame = frames["real"] if variable in frames["real"].columns else frames["synthetic_n2000"]
    series = pd.to_numeric(frame[variable], errors="coerce")
    missing_rate = float(series.isna().mean())
    variance = float(series.var(skipna=True)) if series.notna().sum() > 1 else 0.0
    numeric = frame.select_dtypes(include="number")
    max_corr = 0.0
    if variable in numeric.columns and len(numeric.columns) > 1:
        corr = numeric.corr(numeric_only=True)[variable].drop(labels=[variable], errors="ignore")
        corr = corr.replace([np.inf, -np.inf], np.nan).dropna().abs()
        max_corr = float(corr.max()) if not corr.empty else 0.0
    return {
        "missing_rate": missing_rate,
        "variance": variance,
        "max_abs_correlation_with_other_variable": max_corr,
    }


def constrained_variables() -> set[str]:
    """Return variables directly referenced by finalized constraints."""
    variables: set[str] = set()
    for module_path in ["04_forbidden_edges_synthetic.py", "04_forbidden_edges_real.py"]:
        path = ROOT / module_path
        if not path.exists():
            continue
        module = runner.load_module_from_path(f"deci_constraint_vars_{path.stem}", path)
        for source, target in list(getattr(module, "FORBIDDEN_EDGES", [])) + list(
            getattr(module, "REQUIRED_EDGES", [])
        ):
            variables.add(str(source))
            variables.add(str(target))
    return variables


def should_include_reduced(
    variable: str,
    stats: dict[str, Any],
    essential_variables: set[str],
) -> tuple[bool, str]:
    """Apply deterministic thesis-facing rules for reduced DECI variables."""
    if stats["missing_rate"] > 0.30:
        return False, "excluded: missing_rate > 0.30"
    if stats["variance"] <= 1e-12:
        return False, "excluded: near-constant"
    if variable in essential_variables:
        return True, "included: referenced by ontology constraint"
    if variable in COMPOSITES:
        return False, "excluded: composite ESG score; components retained"
    if stats["max_abs_correlation_with_other_variable"] >= 0.995:
        return False, "excluded: nearly duplicate correlation >= 0.995"
    if variable in ENV_VARS | SOC_VARS | GOV_VARS | FINANCIAL_VARS:
        return True, "included: ontology-connected or financial variable"
    return False, "excluded: weak ontology connection"


def write_variable_set_report() -> list[str]:
    """Write the reduced-variable report and return the reduced list."""
    frames = {
        "synthetic_n2000": pd.read_csv(runner.DATASETS["synthetic_n2000"]["path"]),
        "real": pd.read_csv(runner.DATASETS["real"]["path"]),
    }
    variables = sorted(set(frames["synthetic_n2000"].select_dtypes(include="number").columns) |
                       set(frames["real"].select_dtypes(include="number").columns))
    essential_variables = constrained_variables()
    rows: list[dict[str, Any]] = []
    reduced: list[str] = []
    for variable in variables:
        stats = compute_variable_stats(variable, frames)
        included, reason = should_include_reduced(variable, stats, essential_variables)
        if included:
            reduced.append(variable)
        rows.append({
            "variable": variable,
            "included_full": True,
            "included_reduced": included,
            "reason": reason,
            **stats,
            "ontology_role": ontology_role(variable),
            "domain": variable_domain(variable),
            "is_composite": variable in COMPOSITES,
            "is_component": variable in (ENV_VARS | SOC_VARS | GOV_VARS) - COMPOSITES,
        })
    pd.DataFrame(rows).to_csv(VARIABLE_REPORT_PATH, index=False)
    REDUCED_VARIABLES_PATH.write_text("\n".join(reduced) + "\n", encoding="utf-8")
    log(f"Variable set report -> {VARIABLE_REPORT_PATH.relative_to(ROOT)}")
    log(f"Reduced variable list ({len(reduced)} vars) -> {REDUCED_VARIABLES_PATH.relative_to(ROOT)}")
    return reduced


def load_reduced_variables() -> list[str]:
    """Load or create the reduced variable list."""
    if not REDUCED_VARIABLES_PATH.exists():
        return write_variable_set_report()
    return [
        line.strip()
        for line in REDUCED_VARIABLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dataset(dataset_name: str, variable_set: str) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Load a dataset and optionally apply the reduced DECI variable set."""
    df, true_adj, columns = runner.load_dataset(dataset_name)
    if variable_set == "full":
        return df, true_adj, columns

    keep = [column for column in load_reduced_variables() if column in columns]
    if len(keep) < 3:
        raise ValueError(f"Reduced variable set for {dataset_name} has fewer than 3 variables")
    index = [columns.index(column) for column in keep]
    df = df[keep].copy()
    true_reduced = None
    if true_adj is not None:
        true_reduced = true_adj[np.ix_(index, index)]
    log(f"Applied reduced variable set to {dataset_name}: {len(columns)} -> {len(keep)}")
    return df, true_reduced, keep


def sparsity_value(strength: str, base_l1: float) -> float:
    """Map ablation sparsity labels onto the valid Causica sparsity parameter."""
    if strength == "current":
        return float(base_l1)
    if strength == "medium":
        return max(float(base_l1), 0.10)
    if strength == "strong":
        return max(float(base_l1), 0.20)
    raise ValueError(f"Unsupported sparsity_strength={strength!r}")


def config_id(epochs: int, sparsity_strength: str, variable_set: str) -> str:
    """Build a stable base configuration identifier."""
    return f"ep{epochs}_sp{sparsity_strength}_vars{variable_set}"


def build_options(
    *,
    config_id_value: str,
    epochs: int,
    sparsity_strength: str,
    variable_set: str,
    backend: str,
    run_suffix: str,
) -> dict[str, Any]:
    """Build options for the existing guarded DECI trainer."""
    preset = runner.get_deci_preset()
    l1_lambda = sparsity_value(sparsity_strength, float(preset["l1_lambda"]))
    return {
        "config_id": config_id_value,
        "preset_name": f"ablation_{config_id_value}",
        "max_epochs": int(epochs),
        "l1_lambda": float(l1_lambda),
        "backend": "manual" if backend == "fallback" else backend,
        "allow_manual_fallback": backend == "fallback" or bool(
            getattr(project_config, "DECI_ALLOW_MANUAL_FALLBACK", True)
        ),
        "timeout_seconds": int(getattr(project_config, "DECI_TIMEOUT_SECONDS", 1800)),
        "variable_set": variable_set,
        "sparsity_strength": sparsity_strength,
        "run_suffix": run_suffix,
    }


def constraints_for_mode(
    dataset_name: str,
    columns: list[str],
    adapter: Any,
    constraint_mode: str,
) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], bool]:
    """Return training mode and constraints for a DECI constraint ablation mode."""
    forbidden, required = runner.load_constraints(dataset_name, adapter)
    variable_set = set(columns)
    forbidden = [pair for pair in forbidden if pair[0] in variable_set and pair[1] in variable_set]
    required = [pair for pair in required if pair[0] in variable_set and pair[1] in variable_set]
    if constraint_mode == "unconstrained":
        return "unconstrained", [], [], False
    if constraint_mode == "native_constrained":
        return "constrained", forbidden, required, False
    if constraint_mode == "native_forbidden_only":
        return "constrained", forbidden, [], False
    if constraint_mode == "native_required_only":
        return "constrained", [], required, False
    if constraint_mode == "native_forbidden_postprocessed_required":
        return "constrained", forbidden, [], True
    raise ValueError(f"Unsupported constraint_mode={constraint_mode!r}")


def evaluate_from_weights(
    *,
    raw_weights: np.ndarray,
    threshold: float,
    columns: list[str],
    dataset_name: str,
    true_adj: np.ndarray | None,
    all_forbidden: list[tuple[str, str]],
    required_for_postprocess: list[tuple[str, str]],
    postprocess_required: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Threshold raw DECI weights and compute metrics."""
    thresholded, _ = runner.threshold_weight_matrix(raw_weights, "fixed", threshold, 95.0, None)
    final = thresholded.copy()
    enforcement = {
        "constraint_cells_changed": 0,
        "required_added": 0,
        "forbidden_removed": 0,
    }
    if postprocess_required:
        final, enforcement = runner.enforce_deci_constraints(
            thresholded,
            columns,
            [],
            required_for_postprocess,
        )

    metrics: dict[str, Any]
    if true_adj is not None:
        metrics = runner.compute_synthetic_metrics(final, true_adj)
        metrics["violations"] = runner.count_pairs(final, all_forbidden, columns)
        metrics["graph_density"] = (
            metrics["edge_count_predicted"] / (len(columns) * (len(columns) - 1))
            if len(columns) > 1 else 0.0
        )
    else:
        forbidden, _ = runner.load_constraints(dataset_name, runner.load_adapter())
        literature = runner.load_literature_supported_pairs(columns)
        real_metrics = runner.compute_real_metrics(final, columns, forbidden, literature)
        metrics = {
            "edge_count_predicted": real_metrics["edge_count_predicted"],
            "literature_alignment_score": real_metrics["literature_alignment_score"],
            "literature_violation_count": real_metrics["literature_violation_count"],
            "graph_density": (
                real_metrics["edge_count_predicted"] / (len(columns) * (len(columns) - 1))
                if len(columns) > 1 else 0.0
            ),
            "violations": real_metrics["literature_violation_count"],
        }
    metrics["constraint_cells_changed"] = enforcement["constraint_cells_changed"]
    return final, metrics


def append_failure(
    dataset_name: str,
    mode: str,
    constraint_mode: str,
    variable_set: str,
    config_id_value: str,
    seed: int,
    epochs: int,
    backend: str,
    status: str,
    exc: BaseException,
    runtime_seconds: float,
    failure_phase: str,
) -> None:
    """Write a DECI-specific failure row."""
    exception_type = type(exc).__name__
    full_message = str(exc).strip() or repr(exc)
    append_row(DECI_FAILURES_PATH, FAILURE_COLUMNS, {
        "config_id": config_id_value,
        "dataset": dataset_name,
        "variable_set": variable_set,
        "mode": mode,
        "constraint_mode": constraint_mode,
        "seed": seed,
        "epochs": epochs,
        "backend": backend,
        "status": status,
        "exception_type": exception_type,
        "error_message": full_message,
        "full_error_message": f"{exception_type}: {full_message}",
        "traceback_short": traceback.format_exc()[-1200:],
        "failure_phase": failure_phase,
        "runtime_seconds": round(runtime_seconds, 4),
    })


def infer_failure_phase(exc: BaseException, default_phase: str) -> str:
    """Infer the most specific DECI failure phase available from an exception."""
    text = f"{type(exc).__name__}: {str(exc)}".lower()
    if "data preparation" in text or "nan" in text or "infinite" in text:
        return "data_preparation"
    if "constraint" in text and ("matrix" in text or "cycle" in text or "path" in text):
        return "constraint_matrix_loading"
    if "model initialization" in text or "decimodule" in text or "initialization" in text:
        return "causica_model_initialization"
    if "raw probability output missing" in text or "adjacency" in text or "edge probabilities" in text:
        return "adjacency_extraction"
    if "threshold" in text:
        return "threshold_evaluation"
    if "trainer.fit" in text or "training" in text or "timeout" in text:
        return "training"
    return default_phase


def run_training_once(
    *,
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
    seed: int,
    training_mode: str,
    forbidden_passed: list[tuple[str, str]],
    required_passed: list[tuple[str, str]],
    adapter: Any,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Train one DECI model with explicit constraint overrides."""
    data_boot = runner.bootstrap_data(df, seed)
    options = dict(options)
    options["forbidden_override"] = forbidden_passed
    options["required_override"] = required_passed
    return runner.train_deci_guarded(
        data=data_boot,
        columns=columns,
        mode=training_mode,
        dataset_name=dataset_name,
        seed=seed,
        adapter=adapter,
        deci_options=options,
    )


def write_constraint_validation(
    *,
    datasets: list[str],
    variable_sets: list[str],
    adapter: Any,
) -> pd.DataFrame:
    """Validate and write the Causica constraint matrix contents."""
    rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for variable_set in variable_sets:
            df, true_adj, columns = load_dataset(dataset_name, variable_set)
            forbidden, required = runner.load_constraints(dataset_name, adapter)
            index = {name: i for i, name in enumerate(columns)}
            for kind, pairs in [("forbidden", forbidden), ("required", required)]:
                for source, target in pairs:
                    src_exists = source in index
                    tgt_exists = target in index
                    conflict: str | bool = ""
                    if true_adj is not None and src_exists and tgt_exists:
                        present = bool(true_adj[index[source], index[target]])
                        conflict = present if kind == "forbidden" else not present
                    rows.append({
                        "dataset": dataset_name,
                        "variable_set": variable_set,
                        "source": source,
                        "target": target,
                        "constraint_type": kind,
                        "source_index": index.get(source, ""),
                        "target_index": index.get(target, ""),
                        "source_exists_in_dataset": src_exists,
                        "target_exists_in_dataset": tgt_exists,
                        "direction_confirmed": src_exists and tgt_exists and source != target,
                        "conflicts_with_synthetic_ground_truth": conflict,
                        "passed_to_causica": src_exists and tgt_exists,
                        "notes": (
                            "Causica native constraint_matrix_path supports 0=forbidden, "
                            "1=required, NaN=unconstrained in this codebase."
                        ),
                    })
    validation = pd.DataFrame(rows, columns=VALIDATION_COLUMNS)
    validation.to_csv(CONSTRAINT_VALIDATION_PATH, index=False)
    passed = validation[validation["passed_to_causica"] == True]
    dropped = validation[validation["passed_to_causica"] != True]
    conflicts = validation[validation["conflicts_with_synthetic_ground_truth"] == True]
    log(f"Constraint validation -> {CONSTRAINT_VALIDATION_PATH.relative_to(ROOT)}")
    log(f"Forbidden constraints passed: {(passed['constraint_type'] == 'forbidden').sum()}")
    log(f"Required constraints passed: {(passed['constraint_type'] == 'required').sum()}")
    log(f"Constraints dropped due to missing variables: {len(dropped)}")
    log(f"Constraints conflicting with synthetic ground truth: {len(conflicts)}")
    log("Causica native path supports forbidden and required constraints via constraint_matrix_path.")
    return validation


def run_synthetic_ablation(args: Any) -> pd.DataFrame:
    """Run synthetic-only DECI calibration over the configured grid."""
    if not bool(getattr(project_config, "DECI_ABLATION_ENABLED", True)):
        raise RuntimeError("DECI_ABLATION_ENABLED=False")

    init_csv(ABLATION_PATH, ABLATION_COLUMNS)
    init_csv(ABLATION_SUMMARY_PATH, SUMMARY_COLUMNS)
    init_csv(DECI_FAILURES_PATH, FAILURE_COLUMNS)
    init_csv(STABLE_DETAILED_PATH, STABLE_COLUMNS)
    write_variable_set_report()

    adapter = runner.load_adapter()
    dataset_name = "synthetic_n2000"
    grid = dict(getattr(project_config, "DECI_ABLATION_GRID"))
    smoke_mode = os.environ.get("DECI_ABLATION_SMOKE") == "1"
    if smoke_mode:
        grid = {
            "epochs": [20],
            "thresholds": [0.20, 0.25, 0.30],
            "sparsity_strength": ["current"],
            "variable_sets": ["reduced"],
            "constraint_modes": ["unconstrained", "native_constrained"],
        }
        log("Using DECI_ABLATION_SMOKE=1 tiny Windows grid.")
    thresholds = [float(value) for value in grid["thresholds"]]
    seeds = runner.parse_seed_arg(args.seeds)
    backend = args.backend or str(getattr(project_config, "DECI_BACKEND", "causica"))
    variable_sets = [args.variable_set] if args.variable_set != "all" else list(grid["variable_sets"])
    constraint_modes = list(grid["constraint_modes"])

    write_constraint_validation(
        datasets=[dataset_name],
        variable_sets=variable_sets,
        adapter=adapter,
    )

    log("Running synthetic-only DECI ablation. Real data is not used for selection.")
    log(
        "Sparsity ablation uses Causica DECIModule prior_sparsity_lambda; "
        "thresholds are swept from each trained raw matrix without retraining."
    )
    all_records: list[dict[str, Any]] = []
    stable_inputs: list[dict[str, Any]] = []

    for epochs, sparsity_strength, variable_set, constraint_mode in itertools.product(
        grid["epochs"],
        grid["sparsity_strength"],
        variable_sets,
        constraint_modes,
    ):
        base_id = config_id(int(epochs), str(sparsity_strength), str(variable_set))
        phase = "data_preparation"
        try:
            df, true_adj, columns = load_dataset(dataset_name, str(variable_set))
            if true_adj is None:
                raise RuntimeError("Synthetic ground truth is required for DECI ablation")
            phase = "constraint_matrix_loading"
            all_forbidden, all_required = runner.load_constraints(dataset_name, adapter)
            training_mode, forbidden_passed, required_passed, post_required = constraints_for_mode(
                dataset_name,
                columns,
                adapter,
                str(constraint_mode),
            )
            options = build_options(
                config_id_value=base_id,
                epochs=int(epochs),
                sparsity_strength=str(sparsity_strength),
                variable_set=str(variable_set),
                backend=backend,
                run_suffix=f"{base_id}_{constraint_mode}",
            )
        except BaseException as exc:
            runtime_seconds = 0.0
            status = "timeout" if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() else "failed"
            failure_phase = infer_failure_phase(exc, phase)
            full_message = str(exc).strip() or repr(exc)
            for seed in seeds:
                append_failure(
                    dataset_name=dataset_name,
                    mode="constrained" if constraint_mode != "unconstrained" else "unconstrained",
                    constraint_mode=str(constraint_mode),
                    variable_set=str(variable_set),
                    config_id_value=base_id,
                    seed=seed,
                    epochs=int(epochs),
                    backend=backend,
                    status=status,
                    exc=exc,
                    runtime_seconds=runtime_seconds,
                    failure_phase=failure_phase,
                )
                for threshold in thresholds:
                    row = {
                        "config_id": base_id,
                        "dataset": dataset_name,
                        "variable_set": variable_set,
                        "mode": "constrained" if constraint_mode != "unconstrained" else "unconstrained",
                        "constraint_mode": constraint_mode,
                        "seed": seed,
                        "epochs": epochs,
                        "threshold": threshold,
                        "sparsity_strength": sparsity_strength,
                        "backend": backend,
                        "status": status,
                        "error_message": f"{type(exc).__name__}: {full_message}",
                        "runtime_seconds": runtime_seconds,
                    }
                    append_row(ABLATION_PATH, ABLATION_COLUMNS, row)
                    all_records.append(row)
            log(f"{base_id}/{constraint_mode} {status} during {failure_phase}: {full_message}")
            continue
        for seed in seeds:
            started = time.perf_counter()
            phase = "training"
            try:
                training = run_training_once(
                    df=df,
                    columns=columns,
                    dataset_name=dataset_name,
                    seed=seed,
                    training_mode=training_mode,
                    forbidden_passed=forbidden_passed,
                    required_passed=required_passed,
                    adapter=adapter,
                    options=options,
                )
                runtime_seconds = float(training["runtime_seconds"])
                phase = "adjacency_extraction"
                raw_weights = np.asarray(training["raw_weights"], dtype=float)
                phase = "threshold_evaluation"
                for threshold in thresholds:
                    final, metrics = evaluate_from_weights(
                        raw_weights=raw_weights,
                        threshold=threshold,
                        columns=columns,
                        dataset_name=dataset_name,
                        true_adj=true_adj,
                        all_forbidden=all_forbidden,
                        required_for_postprocess=all_required if post_required else [],
                        postprocess_required=post_required,
                    )
                    final_path = Path(training["run_dir"]) / f"ablation_final_thr{threshold:.3f}.npy"
                    np.save(final_path, final.astype(int))
                    row = {
                        "config_id": base_id,
                        "dataset": dataset_name,
                        "variable_set": variable_set,
                        "mode": "constrained" if constraint_mode != "unconstrained" else "unconstrained",
                        "constraint_mode": constraint_mode,
                        "seed": seed,
                        "epochs": epochs,
                        "threshold": threshold,
                        "sparsity_strength": sparsity_strength,
                        "l1_lambda": options["l1_lambda"],
                        "backend": training.get("backend_used", backend),
                        "status": "success",
                        "runtime_seconds": runtime_seconds,
                        "edge_count": metrics["edge_count_predicted"],
                        "edge_count_true": metrics["edge_count_true"],
                        "graph_density": metrics["graph_density"],
                        "f1_directed": metrics["f1_directed"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "shd": metrics["shd"],
                        "violations": metrics["violations"],
                        "forbidden_constraints_passed": len(forbidden_passed),
                        "required_constraints_passed": len(required_passed),
                        "constraint_cells_changed": metrics["constraint_cells_changed"],
                        "raw_weights_path": training["raw_weights_path"],
                        "final_adjacency_path": str(final_path),
                    }
                    append_row(ABLATION_PATH, ABLATION_COLUMNS, row)
                    all_records.append(row)
                    stable_inputs.append({**row, "columns": columns, "true_adj": true_adj})
            except BaseException as exc:
                runtime_seconds = time.perf_counter() - started
                status = "timeout" if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() else "failed"
                failure_phase = infer_failure_phase(exc, phase)
                full_message = str(exc).strip() or repr(exc)
                append_failure(
                    dataset_name=dataset_name,
                    mode="constrained" if constraint_mode != "unconstrained" else "unconstrained",
                    constraint_mode=str(constraint_mode),
                    variable_set=str(variable_set),
                    config_id_value=base_id,
                    seed=seed,
                    epochs=int(epochs),
                    backend=backend,
                    status=status,
                    exc=exc,
                    runtime_seconds=runtime_seconds,
                    failure_phase=failure_phase,
                )
                for threshold in thresholds:
                    row = {
                        "config_id": base_id,
                        "dataset": dataset_name,
                        "variable_set": variable_set,
                        "mode": "constrained" if constraint_mode != "unconstrained" else "unconstrained",
                        "constraint_mode": constraint_mode,
                        "seed": seed,
                        "epochs": epochs,
                        "threshold": threshold,
                        "sparsity_strength": sparsity_strength,
                        "l1_lambda": options["l1_lambda"],
                        "backend": backend,
                        "status": status,
                        "error_message": f"{type(exc).__name__}: {full_message}",
                        "runtime_seconds": round(runtime_seconds, 4),
                    }
                    append_row(ABLATION_PATH, ABLATION_COLUMNS, row)
                    all_records.append(row)
                log(f"{base_id}/{constraint_mode}/seed={seed} {status} during {failure_phase}: {full_message}")

    summary = summarize_ablation(pd.DataFrame(all_records))
    write_stable_edges_detailed(pd.DataFrame(stable_inputs), adapter, dataset_name)
    selected = select_best_config(summary)
    SELECTED_CONFIG_PATH.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    log("Selected DECI configuration based on synthetic data only.")
    log(f"Selected config -> {SELECTED_CONFIG_PATH.relative_to(ROOT)}")
    if smoke_mode:
        pd.DataFrame(columns=ABLATION_COLUMNS).to_csv(CONSTRAINT_TYPE_ABLATION_PATH, index=False)
        log("Smoke mode: skipped extended constraint-type ablation.")
    else:
        run_constraint_type_ablation(selected, args, adapter)
    write_deci_report()
    return summary


def summarize_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate synthetic ablation rows."""
    if ablation.empty:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
        summary.to_csv(ABLATION_SUMMARY_PATH, index=False)
        return summary

    group_cols = [
        "config_id",
        "variable_set",
        "mode",
        "constraint_mode",
        "epochs",
        "threshold",
        "sparsity_strength",
        "l1_lambda",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in ablation.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))
        success = group[group["status"] == "success"].copy()
        row = {**key}
        for source, target in [
            ("f1_directed", "f1"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("shd", "shd"),
            ("edge_count", "edge_count"),
            ("violations", "violations"),
            ("runtime_seconds", "runtime"),
        ]:
            values = pd.to_numeric(success.get(source, pd.Series(dtype=float)), errors="coerce")
            if target in {"f1", "shd", "edge_count"}:
                row[f"mean_{target}"] = float(values.mean()) if not values.empty else math.nan
                row[f"std_{target}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            else:
                row[f"mean_{target}"] = float(values.mean()) if not values.empty else math.nan
        row["successful_runs"] = int((group["status"] == "success").sum())
        row["failed_runs"] = int((group["status"] == "failed").sum())
        row["timed_out_runs"] = int((group["status"] == "timeout").sum())
        row["success_rate"] = row["successful_runs"] / len(group) if len(group) else 0.0
        row["acceptable"] = False
        row["selection_rank"] = ""
        rows.append(row)

    summary = pd.DataFrame(rows)
    true_edges = float(pd.to_numeric(ablation["edge_count_true"], errors="coerce").dropna().max())
    baseline_shd = current_deci_baseline_shd()
    if not math.isfinite(baseline_shd):
        baseline_shd = float("inf")
    lower_edges = 0.5 * true_edges
    upper_edges = 1.25 * true_edges
    summary["acceptable"] = (
        (summary["successful_runs"] >= 1)
        & (summary["success_rate"] >= 0.80)
        & (summary["mean_edge_count"].between(lower_edges, upper_edges))
        & (summary["mean_precision"] >= 0.10)
        & (summary["mean_shd"] <= baseline_shd * 1.10)
    )
    summary.to_csv(ABLATION_SUMMARY_PATH, index=False)
    log(f"Synthetic ablation summary -> {ABLATION_SUMMARY_PATH.relative_to(ROOT)}")
    return summary


def current_deci_baseline_shd() -> float:
    """Read the latest native-DECI constrained SHD baseline if available."""
    path = runner.SUMMARY_PATH
    if not path.exists():
        return 99.0
    summary = pd.read_csv(path)
    mask = (
        summary["algorithm"].astype(str).isin(["deci_native", "deci_native_constrained"])
        & summary["mode"].astype(str).eq("constrained")
        & summary["dataset"].astype(str).eq("synthetic_n2000")
    )
    if not mask.any():
        return 99.0
    return float(pd.to_numeric(summary.loc[mask, "shd_mean"], errors="coerce").dropna().iloc[0])


def select_best_config(summary: pd.DataFrame) -> dict[str, Any]:
    """Select the DECI configuration using synthetic data only."""
    if summary.empty:
        raise RuntimeError("Cannot select a DECI config from an empty ablation summary")

    constrained = summary[summary["constraint_mode"] == "native_constrained"].copy()
    unconstrained = summary[summary["constraint_mode"] == "unconstrained"].copy()
    merged = constrained.merge(
        unconstrained[["config_id", "variable_set", "epochs", "threshold", "sparsity_strength",
                       "mean_f1", "mean_shd", "mean_edge_count", "mean_runtime"]],
        on=["config_id", "variable_set", "epochs", "threshold", "sparsity_strength"],
        how="left",
        suffixes=("", "_unconstrained"),
    )
    merged["f1_preserved_or_improved"] = merged["mean_f1"] >= (
        merged["mean_f1_unconstrained"].fillna(-np.inf) - 0.005
    )
    merged["shd_reduced"] = merged["mean_shd"] <= merged["mean_shd_unconstrained"].fillna(np.inf)
    true_edges = 65.0
    if ABLATION_PATH.exists():
        ablation = pd.read_csv(ABLATION_PATH)
        truth = pd.to_numeric(ablation["edge_count_true"], errors="coerce").dropna()
        if not truth.empty:
            true_edges = float(truth.max())
    merged["edge_count_deviation"] = (merged["mean_edge_count"] - true_edges).abs()
    pool = merged[merged["acceptable"] == True].copy()
    if pool.empty:
        pool = merged.copy()
        decision_note = "No configuration met all acceptability rules; selected best available synthetic-only compromise."
    else:
        decision_note = "Selected among acceptable configurations using synthetic data only."
    pool = pool.sort_values(
        ["mean_f1", "mean_shd", "edge_count_deviation", "mean_runtime"],
        ascending=[False, True, True, True],
    )
    selected = pool.iloc[0].to_dict()
    selected["decision_note"] = decision_note
    selected["true_synthetic_edge_count"] = true_edges
    selected["native_required_supported"] = True
    selected["backend"] = str(getattr(project_config, "DECI_BACKEND", "causica"))
    return json_ready(selected)


def json_ready(value: Any) -> Any:
    """Convert numpy/pandas scalar values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def run_selected_real(args: Any) -> pd.DataFrame:
    """Run the synthetic-selected DECI configuration on real data."""
    if not SELECTED_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{SELECTED_CONFIG_PATH} is missing. Run --deci-ablation --dataset synthetic first."
        )
    init_csv(REAL_SELECTED_PATH, REAL_COLUMNS)
    init_csv(DECI_FAILURES_PATH, FAILURE_COLUMNS)
    selected = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    adapter = runner.load_adapter()
    dataset_name = "real"
    variable_set = args.variable_set if args.variable_set != "all" else str(selected["variable_set"])
    df, true_adj, columns = load_dataset(dataset_name, variable_set)
    seeds = runner.parse_seed_arg(args.seeds)
    threshold = float(selected["threshold"])
    backend = args.backend or str(selected.get("backend", getattr(project_config, "DECI_BACKEND", "causica")))
    rows: list[dict[str, Any]] = []
    stable_inputs: list[dict[str, Any]] = []
    write_constraint_validation(datasets=[dataset_name], variable_sets=[variable_set], adapter=adapter)

    for constraint_mode in ["unconstrained", "native_constrained"]:
        training_mode, forbidden_passed, required_passed, post_required = constraints_for_mode(
            dataset_name,
            columns,
            adapter,
            constraint_mode,
        )
        options = build_options(
            config_id_value=str(selected["config_id"]),
            epochs=int(selected["epochs"]),
            sparsity_strength=str(selected["sparsity_strength"]),
            variable_set=variable_set,
            backend=backend,
            run_suffix=f"selected_{selected['config_id']}_{constraint_mode}",
        )
        for seed in seeds:
            started = time.perf_counter()
            phase = "training"
            try:
                training = run_training_once(
                    df=df,
                    columns=columns,
                    dataset_name=dataset_name,
                    seed=seed,
                    training_mode=training_mode,
                    forbidden_passed=forbidden_passed,
                    required_passed=required_passed,
                    adapter=adapter,
                    options=options,
                )
                phase = "constraint_matrix_loading"
                all_forbidden, all_required = runner.load_constraints(dataset_name, adapter)
                phase = "adjacency_extraction"
                raw_weights = np.asarray(training["raw_weights"], dtype=float)
                phase = "threshold_evaluation"
                final, metrics = evaluate_from_weights(
                    raw_weights=raw_weights,
                    threshold=threshold,
                    columns=columns,
                    dataset_name=dataset_name,
                    true_adj=None,
                    all_forbidden=all_forbidden,
                    required_for_postprocess=all_required if post_required else [],
                    postprocess_required=post_required,
                )
                final_path = Path(training["run_dir"]) / f"real_selected_final_thr{threshold:.3f}.npy"
                np.save(final_path, final.astype(int))
                row = {
                    "config_id": selected["config_id"],
                    "dataset": dataset_name,
                    "variable_set": variable_set,
                    "mode": "constrained" if constraint_mode != "unconstrained" else "unconstrained",
                    "seed": seed,
                    "threshold": threshold,
                    "epochs": selected["epochs"],
                    "sparsity_strength": selected["sparsity_strength"],
                    "status": "success",
                    "runtime_seconds": training["runtime_seconds"],
                    "edge_count": metrics["edge_count_predicted"],
                    "alignment": metrics["literature_alignment_score"],
                    "violations": metrics["literature_violation_count"],
                    "stable_edges_60": "",
                    "stable_edges_80": "",
                }
                append_row(REAL_SELECTED_PATH, REAL_COLUMNS, row)
                rows.append(row)
                stable_inputs.append({
                    **row,
                    "constraint_mode": constraint_mode,
                    "raw_weights_path": training["raw_weights_path"],
                    "final_adjacency_path": str(final_path),
                    "columns": columns,
                    "true_adj": None,
                })
            except BaseException as exc:
                runtime_seconds = time.perf_counter() - started
                status = "timeout" if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() else "failed"
                failure_phase = infer_failure_phase(exc, phase)
                full_message = str(exc).strip() or repr(exc)
                append_failure(
                    dataset_name=dataset_name,
                    mode="constrained" if constraint_mode != "unconstrained" else "unconstrained",
                    constraint_mode=str(constraint_mode),
                    variable_set=variable_set,
                    config_id_value=str(selected["config_id"]),
                    seed=seed,
                    epochs=int(selected["epochs"]),
                    backend=backend,
                    status=status,
                    exc=exc,
                    runtime_seconds=runtime_seconds,
                    failure_phase=failure_phase,
                )
                row = {
                    "config_id": selected["config_id"],
                    "dataset": dataset_name,
                    "variable_set": variable_set,
                    "mode": "constrained" if constraint_mode != "unconstrained" else "unconstrained",
                    "seed": seed,
                    "threshold": threshold,
                    "epochs": selected["epochs"],
                    "sparsity_strength": selected["sparsity_strength"],
                    "status": status,
                    "error_message": f"{type(exc).__name__}: {full_message}",
                    "runtime_seconds": round(runtime_seconds, 4),
                }
                append_row(REAL_SELECTED_PATH, REAL_COLUMNS, row)
                rows.append(row)

    stable = write_stable_edges_detailed(pd.DataFrame(stable_inputs), adapter, dataset_name)
    real = pd.DataFrame(rows)
    for mode in ["unconstrained", "constrained"]:
        mode_stable = stable[(stable["dataset"] == "real") & (stable["mode"] == mode)]
        mask = real["mode"] == mode
        real.loc[mask, "stable_edges_60"] = int(mode_stable["appears_in_60_percent"].sum())
        real.loc[mask, "stable_edges_80"] = int(mode_stable["appears_in_80_percent"].sum())
    real.to_csv(REAL_SELECTED_PATH, index=False)
    write_deci_report()
    log(f"Real selected-config results -> {REAL_SELECTED_PATH.relative_to(ROOT)}")
    return real


def write_stable_edges_detailed(records: pd.DataFrame, adapter: Any, dataset_name: str) -> pd.DataFrame:
    """Write detailed DECI stability rows from run artifacts."""
    if records.empty:
        stable = pd.DataFrame(columns=STABLE_COLUMNS)
        stable.to_csv(STABLE_DETAILED_PATH, index=False)
        return stable
    records = records.copy()
    if "threshold" in records.columns:
        records["_stable_config_id"] = records.apply(
            lambda row: f"{row['config_id']}_thr{float(row['threshold']):.3f}",
            axis=1,
        )
    else:
        records["_stable_config_id"] = records["config_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for (variable_set, mode, config_id_value), group in records.groupby(
        ["variable_set", "mode", "_stable_config_id"], dropna=False
    ):
        columns = list(group.iloc[0]["columns"])
        true_adj = group.iloc[0].get("true_adj")
        forbidden, required = runner.load_constraints(dataset_name, adapter)
        forbidden_set = set(forbidden)
        required_set = set(required)
        n_runs = len(group)
        edge_counts: dict[tuple[str, str], int] = {}
        weights: dict[tuple[str, str], list[float]] = {}
        for _, item in group.iterrows():
            adj_path = Path(str(item["final_adjacency_path"]))
            raw_path = Path(str(item["raw_weights_path"]))
            if not adj_path.exists() or not raw_path.exists():
                continue
            adjacency = (np.load(adj_path) != 0).astype(int)
            raw = pd.read_csv(raw_path, index_col=0).loc[columns, columns].to_numpy(dtype=float)
            for i, source in enumerate(columns):
                for j, target in enumerate(columns):
                    if i == j:
                        continue
                    pair = (source, target)
                    weights.setdefault(pair, []).append(float(raw[i, j]))
                    if adjacency[i, j]:
                        edge_counts[pair] = edge_counts.get(pair, 0) + 1
        for pair, count in sorted(edge_counts.items()):
            frequency = count / n_runs if n_runs else 0.0
            observed_weights = weights.get(pair, [])
            in_truth: str | bool = ""
            correct_direction: str | bool = ""
            if isinstance(true_adj, np.ndarray):
                idx = {name: i for i, name in enumerate(columns)}
                in_truth = bool(true_adj[idx[pair[0]], idx[pair[1]]])
                correct_direction = in_truth
            rows.append({
                "dataset": dataset_name,
                "variable_set": variable_set,
                "mode": mode,
                "config_id": config_id_value,
                "source": pair[0],
                "target": pair[1],
                "frequency": frequency,
                "mean_weight": float(np.mean(observed_weights)) if observed_weights else 0.0,
                "median_weight": float(np.median(observed_weights)) if observed_weights else 0.0,
                "std_weight": float(np.std(observed_weights)) if observed_weights else 0.0,
                "appears_in_40_percent": frequency >= 0.40,
                "appears_in_60_percent": frequency >= 0.60,
                "appears_in_80_percent": frequency >= 0.80,
                "forbidden_edge": pair in forbidden_set,
                "required_edge": pair in required_set,
                "ontology_supported": pair in required_set or pair in forbidden_set,
                "in_synthetic_ground_truth": in_truth,
                "correct_direction": correct_direction,
            })
    stable = pd.DataFrame(rows, columns=STABLE_COLUMNS)
    if STABLE_DETAILED_PATH.exists():
        old = pd.read_csv(STABLE_DETAILED_PATH)
        stable = pd.concat([old, stable], ignore_index=True)
        stable = stable.drop_duplicates(
            subset=["dataset", "variable_set", "mode", "config_id", "source", "target"],
            keep="last",
        )
    stable.to_csv(STABLE_DETAILED_PATH, index=False)
    log(f"Detailed stable edges -> {STABLE_DETAILED_PATH.relative_to(ROOT)}")
    return stable


def run_constraint_type_ablation(selected: dict[str, Any], args: Any, adapter: Any) -> pd.DataFrame:
    """Run selected synthetic constraint-type ablation."""
    init_csv(CONSTRAINT_TYPE_ABLATION_PATH, ABLATION_COLUMNS)
    dataset_name = "synthetic_n2000"
    variable_set = str(selected["variable_set"])
    df, true_adj, columns = load_dataset(dataset_name, variable_set)
    if true_adj is None:
        raise RuntimeError("Synthetic ground truth is required")
    all_forbidden, all_required = runner.load_constraints(dataset_name, adapter)
    modes = [
        "unconstrained",
        "native_forbidden_only",
        "native_required_only",
        "native_constrained",
        "native_forbidden_postprocessed_required",
    ]
    seeds = runner.parse_seed_arg(args.seeds)
    backend = args.backend or str(selected.get("backend", getattr(project_config, "DECI_BACKEND", "causica")))
    rows: list[dict[str, Any]] = []
    for constraint_mode in modes:
        training_mode, forbidden_passed, required_passed, post_required = constraints_for_mode(
            dataset_name,
            columns,
            adapter,
            constraint_mode,
        )
        options = build_options(
            config_id_value=str(selected["config_id"]),
            epochs=int(selected["epochs"]),
            sparsity_strength=str(selected["sparsity_strength"]),
            variable_set=variable_set,
            backend=backend,
            run_suffix=f"constraint_type_{selected['config_id']}_{constraint_mode}",
        )
        for seed in seeds:
            started = time.perf_counter()
            try:
                training = run_training_once(
                    df=df,
                    columns=columns,
                    dataset_name=dataset_name,
                    seed=seed,
                    training_mode=training_mode,
                    forbidden_passed=forbidden_passed,
                    required_passed=required_passed,
                    adapter=adapter,
                    options=options,
                )
                final, metrics = evaluate_from_weights(
                    raw_weights=np.asarray(training["raw_weights"], dtype=float),
                    threshold=float(selected["threshold"]),
                    columns=columns,
                    dataset_name=dataset_name,
                    true_adj=true_adj,
                    all_forbidden=all_forbidden,
                    required_for_postprocess=all_required if post_required else [],
                    postprocess_required=post_required,
                )
                final_path = Path(training["run_dir"]) / "constraint_type_final.npy"
                np.save(final_path, final.astype(int))
                row = {
                    "config_id": selected["config_id"],
                    "dataset": dataset_name,
                    "variable_set": variable_set,
                    "mode": "constrained" if constraint_mode != "unconstrained" else "unconstrained",
                    "constraint_mode": constraint_mode,
                    "seed": seed,
                    "epochs": selected["epochs"],
                    "threshold": selected["threshold"],
                    "sparsity_strength": selected["sparsity_strength"],
                    "l1_lambda": selected["l1_lambda"],
                    "backend": training.get("backend_used", backend),
                    "status": "success",
                    "runtime_seconds": training["runtime_seconds"],
                    "edge_count": metrics["edge_count_predicted"],
                    "edge_count_true": metrics["edge_count_true"],
                    "graph_density": metrics["graph_density"],
                    "f1_directed": metrics["f1_directed"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "shd": metrics["shd"],
                    "violations": metrics["violations"],
                    "forbidden_constraints_passed": len(forbidden_passed),
                    "required_constraints_passed": len(required_passed),
                    "constraint_cells_changed": metrics["constraint_cells_changed"],
                    "raw_weights_path": training["raw_weights_path"],
                    "final_adjacency_path": str(final_path),
                }
            except BaseException as exc:
                row = {
                    "config_id": selected["config_id"],
                    "dataset": dataset_name,
                    "variable_set": variable_set,
                    "mode": "constrained" if constraint_mode != "unconstrained" else "unconstrained",
                    "constraint_mode": constraint_mode,
                    "seed": seed,
                    "epochs": selected["epochs"],
                    "threshold": selected["threshold"],
                    "sparsity_strength": selected["sparsity_strength"],
                    "l1_lambda": selected["l1_lambda"],
                    "backend": backend,
                    "status": "timeout" if "timeout" in str(exc).lower() else "failed",
                    "error_message": str(exc),
                    "runtime_seconds": round(time.perf_counter() - started, 4),
                }
            append_row(CONSTRAINT_TYPE_ABLATION_PATH, ABLATION_COLUMNS, row)
            rows.append(row)
    result = pd.DataFrame(rows)
    log(f"Constraint-type ablation -> {CONSTRAINT_TYPE_ABLATION_PATH.relative_to(ROOT)}")
    return result


def write_deci_report() -> None:
    """Write a markdown report summarizing the DECI calibration state."""
    selected = {}
    if SELECTED_CONFIG_PATH.exists():
        selected = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    summary = pd.read_csv(ABLATION_SUMMARY_PATH) if ABLATION_SUMMARY_PATH.exists() else pd.DataFrame()
    real = pd.read_csv(REAL_SELECTED_PATH) if REAL_SELECTED_PATH.exists() else pd.DataFrame()
    validation = pd.read_csv(CONSTRAINT_VALIDATION_PATH) if CONSTRAINT_VALIDATION_PATH.exists() else pd.DataFrame()
    stable = pd.read_csv(STABLE_DETAILED_PATH) if STABLE_DETAILED_PATH.exists() else pd.DataFrame()
    failures = pd.read_csv(DECI_FAILURES_PATH) if DECI_FAILURES_PATH.exists() else pd.DataFrame()

    recommendation = "DECI is suitable as exploratory only"
    reason = "Synthetic recovery remains mixed or the real sample is too small for a main-result claim."
    if not summary.empty and selected:
        chosen = summary[
            (summary["config_id"] == selected.get("config_id"))
            & (summary["constraint_mode"] == "native_constrained")
            & (summary["threshold"].astype(float) == float(selected.get("threshold", -1)))
        ]
        uncon = summary[
            (summary["config_id"] == selected.get("config_id"))
            & (summary["constraint_mode"] == "unconstrained")
            & (summary["threshold"].astype(float) == float(selected.get("threshold", -1)))
        ]
        if not chosen.empty and not uncon.empty:
            improves_f1 = chosen["mean_f1"].iloc[0] > uncon["mean_f1"].iloc[0]
            reduces_shd = chosen["mean_shd"].iloc[0] < uncon["mean_shd"].iloc[0]
            stable_runtime = failures.empty or len(failures) <= 1
            if improves_f1 and reduces_shd and stable_runtime:
                recommendation = "DECI is suitable as a main result"
                reason = "Constrained DECI improves F1, reduces SHD, and runs stably under the selected configuration."
            elif chosen["mean_f1"].iloc[0] < 0.10 or chosen["successful_runs"].iloc[0] == 0:
                recommendation = "DECI should be excluded from main comparisons"
                reason = "Synthetic F1 is very low or runs fail too often."

    lines = [
        "# DECI Diagnostic Report",
        "",
        f"- Current backend: `{getattr(project_config, 'DECI_BACKEND', 'causica')}`",
        "- Native Causica constraints used: yes, when backend is `causica` and mode is constrained",
        "- Required constraints natively supported: yes in this codebase via `constraint_matrix_path` with `1.0` entries",
        f"- Selected threshold: `{selected.get('threshold', getattr(project_config, 'DECI_THRESHOLD', ''))}`",
        f"- Selected epochs: `{selected.get('epochs', '')}`",
        f"- Selected sparsity setting: `{selected.get('sparsity_strength', '')}`",
        f"- Selected variable set: `{selected.get('variable_set', '')}`",
        "",
        "## Synthetic Best Configuration",
        "",
        selected_table(summary, selected),
        "",
        "## Real Selected Configuration",
        "",
        dataframe_preview(real),
        "",
        "## Constraint Matrix Validation Summary",
        "",
        validation_summary(validation),
        "",
        "## Stability Summary",
        "",
        stability_summary(stable),
        "",
        "## Runtime And Failure Summary",
        "",
        failure_summary(failures),
        "",
        "## Final Interpretation",
        "",
        f"**{recommendation}.** {reason}",
    ]
    DECI_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"DECI report -> {DECI_REPORT_PATH.relative_to(ROOT)}")


def dataframe_preview(df: pd.DataFrame, max_rows: int = 12) -> str:
    """Return a compact markdown table preview."""
    if df.empty:
        return "_No rows yet._"
    return simple_markdown_table(df.head(max_rows))


def selected_table(summary: pd.DataFrame, selected: dict[str, Any]) -> str:
    """Return the selected synthetic summary row as markdown."""
    if summary.empty or not selected:
        return "_No selected synthetic summary yet._"
    mask = (
        (summary["config_id"] == selected.get("config_id"))
        & (summary["constraint_mode"] == "native_constrained")
        & (summary["threshold"].astype(float) == float(selected.get("threshold", -1)))
    )
    cols = [
        "config_id",
        "variable_set",
        "constraint_mode",
        "epochs",
        "threshold",
        "sparsity_strength",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "mean_shd",
        "mean_edge_count",
        "successful_runs",
    ]
    return simple_markdown_table(summary.loc[mask, cols]) if mask.any() else "_Selected row not present._"


def validation_summary(validation: pd.DataFrame) -> str:
    """Summarize constraint validation rows."""
    if validation.empty:
        return "_No validation rows yet._"
    passed = validation[validation["passed_to_causica"] == True]
    conflicts = validation[validation["conflicts_with_synthetic_ground_truth"] == True]
    return "\n".join([
        f"- Forbidden constraints passed: `{(passed['constraint_type'] == 'forbidden').sum()}`",
        f"- Required constraints passed: `{(passed['constraint_type'] == 'required').sum()}`",
        f"- Constraints dropped due to missing variables: `{len(validation) - len(passed)}`",
        f"- Synthetic ground-truth conflicts: `{len(conflicts)}`",
    ])


def stability_summary(stable: pd.DataFrame) -> str:
    """Summarize detailed stable edges."""
    if stable.empty:
        return "_No stable edge rows yet._"
    grouped = stable.groupby(["dataset", "mode"], dropna=False).agg(
        edges=("source", "count"),
        stable_60=("appears_in_60_percent", "sum"),
        stable_80=("appears_in_80_percent", "sum"),
    ).reset_index()
    return simple_markdown_table(grouped)


def failure_summary(failures: pd.DataFrame) -> str:
    """Summarize DECI failures."""
    if failures.empty:
        return "No DECI ablation failures recorded."
    grouped = failures.groupby(["dataset", "mode", "status"], dropna=False).size().reset_index(name="count")
    return simple_markdown_table(grouped)


def simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as GitHub-flavored markdown without tabulate."""
    if df.empty:
        return "_No rows yet._"
    text = df.copy()
    for column in text.columns:
        text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in text.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in text.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(args: Any) -> int:
    """Entry point called by ``run_all.py``."""
    OUT.mkdir(parents=True, exist_ok=True)
    if args.deci_ablation:
        run_synthetic_ablation(args)
    if args.deci_selected_only:
        target = dataset_key(args.dataset or "real")
        if target == "synthetic_n2000":
            raise ValueError("--deci-selected-only is intended for --dataset real")
        run_selected_real(args)
    return 0
