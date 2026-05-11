# 05_build_reference_dag_from_dummy.py
# ============================================================
# Build an ontology-derived reference DAG for the advisor-provided
# ESG-Finance dummy dataset.
#
# The final advisor_dummy path must use the provided CSV and XLSX files.
# The text generator is development-only fallback, enabled explicitly with
# --allow-dummy-regeneration.
# ============================================================

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ADVISOR_DIR = ROOT / "data" / "advisor_dummy"
OUT = ROOT / "outputs" / "experiments"
DATA_PROCESSED = ROOT / "data" / "processed"

ADVISOR_CSV = ADVISOR_DIR / "ESG-Finance_dummy_data.csv"
ADVISOR_XLSX = ADVISOR_DIR / "ESG-Finance_Metadata.xlsx"
ADVISOR_SPEC = ADVISOR_DIR / "Dummy_dataset_ESG.txt"

EXPLICIT_RULE_FILENAMES = [
    "generation_rules.csv",
    "advisor_rules.csv",
    "ground_truth_dag.csv",
    "reference_dag_edges.csv",
    "dag_edges.csv",
    "causal_rules.csv",
]

DATA_AUDIT_PATH = OUT / "advisor_dummy_data_audit.csv"
METADATA_REGISTRY_PATH = OUT / "advisor_dummy_metadata_registry.csv"
VARIABLE_REGISTRY_PATH = OUT / "dummy_variable_registry.csv"
CLEANED_FULL_PATH = OUT / "advisor_dummy_cleaned_full.csv"
CLEANED_MODEL_PATH = OUT / "advisor_dummy_cleaned.csv"
PROCESSED_DUMMY_PATH = DATA_PROCESSED / "advisor_dummy_ready.csv"
REFERENCE_EDGES_PATH = OUT / "advisor_dummy_reference_dag_edges.csv"
REFERENCE_ADJACENCY_PATH = OUT / "advisor_dummy_reference_dag_adjacency.csv"
VALIDATION_PATH = OUT / "advisor_dummy_reference_dag_validation.md"
FORBIDDEN_PATH = OUT / "advisor_dummy_constraints_forbidden.csv"
REQUIRED_LIGHT_PATH = OUT / "advisor_dummy_constraints_required_light.csv"
FULL_REQUIRED_PATH = OUT / "advisor_dummy_constraints_full_reference_required.csv"

# Legacy paths retained for compatibility with earlier drafts.
LEGACY_REFERENCE_EDGES_PATH = OUT / "dummy_reference_dag_edges.csv"
LEGACY_ADJACENCY_PATH = OUT / "dummy_reference_dag_adjacency.csv"
LEGACY_FORBIDDEN_PATH = OUT / "dummy_constraints_forbidden.csv"
LEGACY_REQUIRED_PATH = OUT / "dummy_constraints_required.csv"
LEGACY_VALIDATION_PATH = OUT / "dummy_reference_dag_validation.md"


@dataclass(frozen=True)
class VariableSpec:
    """Schema record for one advisor dummy variable."""

    name: str
    domain: str
    ontology_type: str
    causal_role: str
    unit: str
    data_type: str
    source_sheet: str
    description: str = ""
    is_identifier: bool = False
    is_static: bool = False
    is_composite: bool = False
    is_component: bool = False


@dataclass(frozen=True)
class EdgeRule:
    """Reference-DAG edge with provenance and constraint status."""

    source: str
    target: str
    rule_name: str
    justification: str
    confidence_level: str
    required_light: bool = False


def log(message: str) -> None:
    """Print a builder-prefixed log line."""
    print(f"[dummy_ref] {message}", flush=True)


def normalize_name(value: Any) -> str:
    """Normalize advisor CSV/metadata names to snake_case."""
    text = str(value).strip().lower()
    text = text.replace("%", " pct ")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text and text[0].isdigit():
        text = f"var_{text}"
    return text


def unique_columns(columns: list[str]) -> list[str]:
    """Make normalized column names unique while preserving order."""
    counts: dict[str, int] = {}
    unique: list[str] = []
    for column in columns:
        base = normalize_name(column) or "unnamed"
        counts[base] = counts.get(base, 0) + 1
        unique.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return unique


def parse_variable_specs_from_text(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the ``variables = {...}`` dictionary from the advisor text spec."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"variables\s*=\s*(\{.*?\})\s*N\s*=", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not locate variables dictionary in {path}")
    tree = ast.parse("variables = " + match.group(1))
    assignment = tree.body[0]
    if not isinstance(assignment, ast.Assign):
        raise ValueError(f"Unexpected variables assignment in {path}")
    return ast.literal_eval(assignment.value)


def domain_from_name(name: str) -> str:
    """Infer broad ESG/financial domain from a variable name."""
    environmental = [
        "carbon", "co2", "ch4", "n2o", "emission", "energy", "renewable",
        "water", "waste", "spill", "biodiversity", "iso_14001", "fsc",
        "climate", "resilience", "environmental", "green", "resource",
        "air", "land", "gmo", "ods", "emf",
    ]
    social = [
        "training", "turnover", "injury", "diversity", "wage", "human",
        "union", "collective", "community", "customer", "health", "csr",
        "labor", "indigenous", "animal", "marketing", "supplier",
        "services", "employee", "product_safety", "hiv",
    ]
    governance = [
        "board", "governance", "oversight", "ceo", "auditor", "ethical",
        "anti_competitive", "corruption", "privacy", "shareholder", "tax",
        "lobby", "compact", "inclusion", "systemic", "assurance",
        "incentive", "closure", "reporting", "compliance",
    ]
    financial = [
        "market", "asset", "debt", "equity", "profit", "margin", "roa",
        "roe", "tobin", "solvency", "ratio", "sales", "revenue", "eps",
        "earnings", "cash", "dividend", "inventory", "liabilities",
        "capital", "pe", "pbv", "wacc", "financing",
    ]
    lower = name.lower()
    parts = set(re.split(r"[^a-z0-9]+", lower))

    def has_token(patterns: list[str]) -> bool:
        for pattern in patterns:
            if len(pattern) <= 3:
                if pattern in parts:
                    return True
            elif pattern in lower:
                return True
        return False

    if has_token(environmental):
        return "environmental"
    if has_token(governance):
        return "governance"
    if has_token(social):
        return "social"
    if has_token(financial):
        return "financial"
    return "company_static"


def ontology_type_from_name(name: str, dtype: str) -> str:
    """Infer ontology type from name and declared data type."""
    lower = name.lower()
    if any(token in lower for token in ["lei", "name", "nace", "currency", "country", "status"]):
        return "identifier_or_static"
    if any(token in lower for token in ["score", "index", "quality", "resilience", "ratio", "margin"]):
        return "score_or_ratio"
    if dtype == "bool" or any(token in lower for token in ["policy", "commitment", "exists", "done", "membership", "compliance", "split"]):
        return "policy_or_structure"
    if any(token in lower for token in ["fines", "violations", "breaches", "cases", "spills"]):
        return "risk_or_event"
    if any(token in lower for token in ["revenue", "financing", "spending", "investment", "asset", "debt", "equity", "profit", "earnings"]):
        return "financial_kpi"
    if any(token in lower for token in ["consumption", "emissions", "waste", "withdrawal", "hours", "rate", "share", "gap"]):
        return "operational_indicator"
    return "indicator"


def causal_role_from_type(ontology_type: str, domain: str) -> str:
    """Map ontology type/domain to a conservative causal role."""
    if ontology_type == "identifier_or_static":
        return "identifier_or_static"
    if domain == "financial":
        return "financial_outcome"
    return {
        "policy_or_structure": "driver",
        "risk_or_event": "event_driver",
        "operational_indicator": "operational_outcome",
        "score_or_ratio": "composite_or_outcome",
        "financial_kpi": "financial_outcome",
        "indicator": "indicator",
    }.get(ontology_type, "indicator")


def is_identifier_name(name: str) -> bool:
    """Return whether a variable is an identifier/text descriptor."""
    tokens = ["lei", "name", "currency", "nace", "shareholder_info", "country", "status"]
    return any(token in name for token in tokens)


def specs_from_text(path: Path) -> dict[str, VariableSpec]:
    """Build variable specs from the advisor dummy text specification."""
    raw = parse_variable_specs_from_text(path)
    specs: dict[str, VariableSpec] = {}
    for original_name, payload in raw.items():
        name = normalize_name(original_name)
        dtype = str(payload.get("dtype", "unknown")).lower()
        domain = domain_from_name(name)
        ontology_type = ontology_type_from_name(name, dtype)
        specs[name] = VariableSpec(
            name=name,
            domain=domain,
            ontology_type=ontology_type,
            causal_role=causal_role_from_type(ontology_type, domain),
            unit=str(payload.get("unit", "")),
            data_type=dtype,
            source_sheet=path.name,
            description="Advisor text-spec variable definition.",
            is_identifier=is_identifier_name(name),
            is_static=is_identifier_name(name),
            is_composite=ontology_type == "score_or_ratio",
            is_component=ontology_type in {"operational_indicator", "indicator"},
        )
    return specs


def load_metadata_specs(path: Path) -> dict[str, VariableSpec]:
    """Load variable hints from the advisor metadata workbook."""
    sheets = pd.read_excel(path, sheet_name=None)
    specs: dict[str, VariableSpec] = {}

    if "Financial KPIs" in sheets:
        financial = sheets["Financial KPIs"]
        for _, row in financial.iterrows():
            original = str(row.get("Variables", "")).strip()
            if not original or original.lower() == "nan":
                continue
            name = normalize_name(original)
            data_type = str(row.get("Data Type", "")).strip().lower() or "unknown"
            domain = "financial"
            ontology_type = ontology_type_from_name(name, data_type)
            specs[name] = VariableSpec(
                name=name,
                domain=domain,
                ontology_type=ontology_type,
                causal_role=causal_role_from_type(ontology_type, domain),
                unit=str(row.get("Unit", "") if str(row.get("Unit", "")).lower() != "nan" else ""),
                data_type=data_type,
                source_sheet="Financial KPIs",
                description=str(row.get("Description", "") if str(row.get("Description", "")).lower() != "nan" else ""),
                is_identifier=is_identifier_name(name) or "text" in data_type,
                is_static=any(token in name for token in ["incorporation", "ipo", "company", "nace"]),
                is_composite=ontology_type == "score_or_ratio",
                is_component=False,
            )

    if "ESG Ontology" in sheets:
        esg = sheets["ESG Ontology"]
        for _, row in esg.iterrows():
            original = str(row.get("Unnamed: 7", "")).strip()
            if not original or original.lower() in {"nan", "variable measured"}:
                continue
            name = normalize_name(original)
            data_type = str(row.get("Unnamed: 9", "")).strip().lower()
            unit = str(row.get("Unnamed: 8", "")).strip()
            description = str(row.get("Unnamed: 10", "")).strip()
            category = str(row.get("Unnamed: 6", "")).strip()
            domain = domain_from_name(name)
            ontology_type = ontology_type_from_name(name, data_type)
            specs.setdefault(
                name,
                VariableSpec(
                    name=name,
                    domain=domain,
                    ontology_type=ontology_type,
                    causal_role=causal_role_from_type(ontology_type, domain),
                    unit="" if unit.lower() == "nan" else unit,
                    data_type=data_type or "unknown",
                    source_sheet="ESG Ontology",
                    description="" if description.lower() == "nan" else description,
                    is_identifier=False,
                    is_static=False,
                    is_composite=ontology_type == "score_or_ratio",
                    is_component=ontology_type in {"operational_indicator", "indicator"},
                ),
            )
            if category and category.lower() != "nan":
                current = specs[name]
                specs[name] = VariableSpec(**{**current.__dict__, "description": current.description or category})

    if "ECB List of corporates" in sheets:
        ecb = sheets["ECB List of corporates"]
        for column in ecb.columns:
            name = normalize_name(column)
            if not name:
                continue
            specs.setdefault(
                name,
                VariableSpec(
                    name=name,
                    domain="company_static",
                    ontology_type="identifier_or_static",
                    causal_role="identifier_or_static",
                    unit="",
                    data_type="text",
                    source_sheet="ECB List of corporates",
                    description="ECB company/static metadata field.",
                    is_identifier=True,
                    is_static=True,
                    is_composite=False,
                    is_component=False,
                ),
            )
    return specs


def infer_spec_for_column(name: str, series: pd.Series) -> VariableSpec:
    """Infer a variable spec for a CSV column missing from metadata."""
    data_type = "numeric" if pd.to_numeric(series, errors="coerce").notna().mean() > 0.6 else "text"
    domain = domain_from_name(name)
    ontology_type = ontology_type_from_name(name, data_type)
    return VariableSpec(
        name=name,
        domain=domain,
        ontology_type=ontology_type,
        causal_role=causal_role_from_type(ontology_type, domain),
        unit="",
        data_type=data_type,
        source_sheet="csv_inferred",
        description="Inferred from advisor dummy CSV column name.",
        is_identifier=is_identifier_name(name) or data_type == "text",
        is_static=is_identifier_name(name),
        is_composite=ontology_type == "score_or_ratio",
        is_component=ontology_type in {"operational_indicator", "indicator"},
    )


def require_advisor_files(allow_dummy_regeneration: bool) -> None:
    """Require project-local advisor CSV/XLSX unless development fallback is enabled."""
    if ADVISOR_CSV.exists() and ADVISOR_XLSX.exists():
        return
    if allow_dummy_regeneration and ADVISOR_SPEC.exists():
        return
    raise FileNotFoundError(
        "Advisor dummy CSV/XLSX files not found. Place them in data/advisor_dummy/. "
        "The final advisor_dummy path must use the provided files."
    )


def generate_dummy_from_specs(specs: dict[str, VariableSpec], n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Development-only deterministic fallback matching the advisor text generator."""
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {}
    for name, spec in specs.items():
        dtype = spec.data_type.lower()
        if dtype == "int":
            data[name] = rng.integers(0, 50, n_rows).astype(float)
        elif dtype in {"float", "numeric"}:
            data[name] = np.round(rng.uniform(0, 100, n_rows), 2)
        elif dtype == "bool":
            data[name] = rng.choice([True, False], n_rows)
        elif dtype == "str":
            data[name] = rng.choice(["Low", "Medium", "High"], n_rows)
        else:
            data[name] = np.round(rng.uniform(0, 100, n_rows), 2)
    df = pd.DataFrame(data).astype(object)
    for column in df.columns:
        missing_count = int(rng.integers(int(0.05 * n_rows), int(0.2 * n_rows) + 1))
        idx = rng.choice(df.index, missing_count, replace=False)
        df.loc[idx, column] = None
    numeric_cols = [name for name, spec in specs.items() if spec.data_type.lower() in {"float", "int", "numeric"}]
    for column in numeric_cols[:10]:
        clean_numeric = pd.to_numeric(df[column], errors="coerce")
        idx = rng.choice(df.index, 5, replace=False)
        df.loc[idx, column] = clean_numeric.max() * float(rng.choice([10, 50, 100]))
    for column in numeric_cols[:8]:
        idx = rng.choice(df.index, 5, replace=False)
        df.loc[idx, column] = "error"
    if "systemic_risk_level" in df.columns:
        idx = rng.choice(df.index, 6, replace=False)
        df.loc[idx, "systemic_risk_level"] = "Very High"
    return df


def load_raw_advisor_data(specs: dict[str, VariableSpec], allow_dummy_regeneration: bool) -> tuple[pd.DataFrame, str, tuple[int, int]]:
    """Load actual advisor CSV or explicit development fallback."""
    if ADVISOR_CSV.exists():
        raw = pd.read_csv(ADVISOR_CSV)
        original_shape = raw.shape
        raw.columns = unique_columns([str(column) for column in raw.columns])
        return raw, str(ADVISOR_CSV.relative_to(ROOT)), original_shape
    if allow_dummy_regeneration:
        raw = generate_dummy_from_specs(specs)
        return raw, "development fallback generated from Dummy_dataset_ESG.txt", raw.shape
    raise FileNotFoundError(
        "Advisor dummy CSV/XLSX files not found. Place them in data/advisor_dummy/. "
        "The final advisor_dummy path must use the provided files."
    )


def preferred_variables() -> list[str]:
    """Manageable advisor-dummy subset for reference-DAG evaluation."""
    return [
        "emission_reduction_policy",
        "carbon_neutral_commitment",
        "climate_risk_assessment_done",
        "iso_14001_exists",
        "total_energy_consumption",
        "renewable_energy_share",
        "carbon_intensity",
        "co2_ch4_n2o_scope_1_3",
        "hazardous_waste_generated",
        "waste_recycled_share",
        "water_withdrawal",
        "toxic_spills",
        "environmental_fines",
        "reporting_quality_score",
        "resilience_score",
        "training_hours",
        "turnover_rate",
        "injury_frequency_rate",
        "diversity_representation",
        "collective_bargaining_coverage",
        "fair_wage_gap",
        "customer_satisfaction_score",
        "human_rights_violations",
        "union_membership",
        "esg_oversight_policy",
        "board_diversity",
        "governance_compliance_score",
        "esg_incentive_bonus",
        "assurance_score",
        "ceo_chair_split",
        "auditor_independence_score",
        "ethical_breaches",
        "anti_competitive_violations",
        "corruption_cases",
        "market_value_equity",
        "asset_growth_pct",
        "debt_to_equity_ratio",
        "gross_profit_margin",
        "net_profit_margin",
        "roa_eat",
        "roe_eat",
        "tobins_q",
        "solvency_ratio",
        "pe_ratio",
    ]


def invalid_examples(series: pd.Series, invalid_mask: pd.Series) -> str:
    """Return a compact semicolon-separated list of invalid examples."""
    examples = series[invalid_mask].dropna().astype(str).unique().tolist()
    return "; ".join(examples[:5])


def clean_advisor_dummy_dataset(
    raw: pd.DataFrame,
    specs: dict[str, VariableSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean advisor dummy data and write a column-level audit."""
    cleaned_columns: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []
    ordinal = {"low": 0.0, "medium": 1.0, "high": 2.0}
    bool_map = {
        True: 1.0,
        False: 0.0,
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "1": 1.0,
        "0": 0.0,
        1: 1.0,
        0: 0.0,
    }

    for name in raw.columns:
        spec = specs.get(name, infer_spec_for_column(name, raw[name]))
        series = raw[name]
        original_missing = series.isna()
        dtype = spec.data_type.lower()
        inferred_dtype = dtype
        invalid_mask = pd.Series(False, index=series.index)

        if spec.is_identifier or dtype in {"text", "object"} and name not in {"systemic_risk_level"}:
            numeric = pd.Series(np.nan, index=series.index, dtype=float)
            inferred_dtype = "identifier_or_text"
            invalid_mask = series.notna()
        elif dtype == "bool" or spec.ontology_type == "policy_or_structure":
            lowered = series.astype(str).str.strip().str.lower()
            numeric = lowered.map(bool_map)
            invalid_mask = series.notna() & numeric.isna()
            inferred_dtype = "boolean"
        elif dtype in {"str", "categorical"} or name == "systemic_risk_level":
            lowered = series.astype(str).str.strip().str.lower()
            numeric = lowered.map(ordinal)
            invalid_mask = series.notna() & numeric.isna()
            inferred_dtype = "categorical_ordinal"
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid_mask = series.notna() & numeric.isna()
            inferred_dtype = "numeric"

        missing_count = int(original_missing.sum())
        invalid_count = int(invalid_mask.sum())
        cleaned_dtype = ""
        if numeric.notna().sum() > 0:
            lower = numeric.quantile(0.01)
            upper = numeric.quantile(0.99)
            numeric = numeric.clip(lower=lower, upper=upper)
            numeric = numeric.fillna(numeric.median())
            cleaned_columns[name] = numeric.astype(float)
            cleaned_dtype = "float64"
        else:
            cleaned_dtype = "excluded"

        included = False
        exclusion_reason = ""
        if name not in preferred_variables():
            exclusion_reason = "excluded: outside conservative manageable reference subset"
        elif spec.is_identifier:
            exclusion_reason = "excluded: identifier/static text field"
        elif cleaned_dtype == "excluded":
            exclusion_reason = "excluded: not convertible to model numeric form"
        elif (missing_count + invalid_count) / max(1, len(series)) > 0.35:
            exclusion_reason = "excluded: more than 35% missing/invalid before imputation"
        elif name in cleaned_columns and float(cleaned_columns[name].var(ddof=0)) <= 1e-12:
            exclusion_reason = "excluded: near-constant after cleaning"
        else:
            included = True
            exclusion_reason = "included: advisor CSV variable, cleaned numeric/boolean, ontology-covered"

        audit_rows.append({
            "column": name,
            "original_dtype": str(series.dtype),
            "inferred_dtype": inferred_dtype,
            "missing_count": missing_count,
            "missing_rate": missing_count / max(1, len(series)),
            "invalid_count": invalid_count,
            "invalid_examples": invalid_examples(series, invalid_mask),
            "cleaned_dtype": cleaned_dtype,
            "included_in_model": included,
            "exclusion_reason": exclusion_reason,
        })

    cleaned_full = pd.DataFrame(cleaned_columns, index=raw.index)
    audit = pd.DataFrame(audit_rows)
    included_columns = audit[audit["included_in_model"] == True]["column"].tolist()
    included_columns = [column for column in included_columns if column in cleaned_full.columns]
    cleaned_model = cleaned_full[included_columns].copy()
    return cleaned_full, cleaned_model, audit


def build_metadata_registry(
    specs: dict[str, VariableSpec],
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Build metadata registry in the required thesis format."""
    included = set(audit[audit["included_in_model"] == True]["column"])
    rows: list[dict[str, Any]] = []
    for _, audit_row in audit.iterrows():
        name = str(audit_row["column"])
        spec = specs.get(name, infer_spec_for_column(name, pd.Series(dtype=float)))
        rows.append({
            "variable_name": name,
            "source_sheet": spec.source_sheet,
            "domain": spec.domain,
            "ontology_type": spec.ontology_type,
            "causal_role": spec.causal_role,
            "unit": spec.unit,
            "description": spec.description,
            "data_type": spec.data_type,
            "is_identifier": spec.is_identifier,
            "is_static": spec.is_static,
            "is_composite": spec.is_composite,
            "is_component": spec.is_component,
            "included_in_reference_dag": name in included,
            "notes": audit_row["exclusion_reason"],
        })
    registry = pd.DataFrame(rows)
    registry.to_csv(METADATA_REGISTRY_PATH, index=False)
    registry.rename(columns={"notes": "reason"}).to_csv(VARIABLE_REGISTRY_PATH, index=False)
    return registry


def find_explicit_rules_file() -> Path | None:
    """Find an explicit advisor causal rule/DAG file if one exists."""
    for filename in EXPLICIT_RULE_FILENAMES:
        path = ADVISOR_DIR / filename
        if path.exists():
            return path
    return None


def load_explicit_edges(path: Path, included: set[str], specs: dict[str, VariableSpec]) -> pd.DataFrame:
    """Load an explicit advisor-provided edge list."""
    raw = pd.read_csv(path)
    normalized = {normalize_name(column): column for column in raw.columns}
    source_col = normalized.get("source") or normalized.get("cause") or normalized.get("from")
    target_col = normalized.get("target") or normalized.get("effect") or normalized.get("to")
    if source_col is None or target_col is None:
        raise ValueError(f"Explicit rule file {path} must contain source/target or cause/effect columns")

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        source = normalize_name(row[source_col])
        target = normalize_name(row[target_col])
        if source not in included or target not in included or source == target:
            continue
        source_spec = specs[source]
        target_spec = specs[target]
        rows.append({
            "source": source,
            "target": target,
            "rule_name": str(row.get("rule_name", "explicit_advisor_rule")),
            "justification": str(row.get("justification", f"Explicit advisor-provided causal rule from {path.name}.")),
            "source_domain": source_spec.domain,
            "target_domain": target_spec.domain,
            "source_type": source_spec.ontology_type,
            "target_type": target_spec.ontology_type,
            "confidence_level": str(row.get("confidence_level", "high")),
            "origin": "explicit_advisor_rule",
            "required_light": bool(row.get("required_light", False)),
        })
    return pd.DataFrame(rows)


def conservative_edges(included: set[str]) -> list[EdgeRule]:
    """Construct conservative metadata/ontology-derived reference edges."""
    candidates = [
        EdgeRule("esg_oversight_policy", "governance_compliance_score", "governance_structure_to_compliance", "Board-level ESG oversight plausibly strengthens governance compliance processes.", "high"),
        EdgeRule("esg_oversight_policy", "reporting_quality_score", "governance_structure_to_reporting", "Formal ESG oversight supports more complete and reliable ESG reporting.", "high"),
        EdgeRule("esg_oversight_policy", "assurance_score", "governance_structure_to_reporting", "ESG oversight increases the likelihood and quality of assurance processes.", "medium"),
        EdgeRule("esg_oversight_policy", "emission_reduction_policy", "governance_to_environmental_policy", "Governance oversight can drive adoption of environmental reduction policies.", "high"),
        EdgeRule("board_diversity", "governance_compliance_score", "governance_structure_to_compliance", "Board composition is a structural governance input to compliance quality.", "medium"),
        EdgeRule("board_diversity", "reporting_quality_score", "governance_structure_to_reporting", "More diverse boards are associated with broader transparency and disclosure oversight.", "medium"),
        EdgeRule("ceo_chair_split", "governance_compliance_score", "governance_structure_to_compliance", "CEO-chair separation is a governance structure affecting oversight and compliance.", "medium"),
        EdgeRule("auditor_independence_score", "assurance_score", "governance_structure_to_reporting", "Auditor independence directly supports assurance quality.", "high", True),
        EdgeRule("auditor_independence_score", "governance_compliance_score", "governance_structure_to_compliance", "Independent audit oversight supports governance compliance.", "medium"),
        EdgeRule("esg_incentive_bonus", "emission_reduction_policy", "governance_to_environmental_policy", "ESG-linked incentives can encourage environmental policy adoption.", "medium"),
        EdgeRule("emission_reduction_policy", "carbon_intensity", "environmental_policy_to_operational_outcome", "Emission-reduction policies are intended to reduce carbon intensity.", "high"),
        EdgeRule("emission_reduction_policy", "co2_ch4_n2o_scope_1_3", "environmental_policy_to_operational_outcome", "Emission-reduction policies target greenhouse gas emissions.", "high"),
        EdgeRule("emission_reduction_policy", "renewable_energy_share", "environmental_policy_to_operational_outcome", "Emission policies often increase renewable energy adoption.", "medium"),
        EdgeRule("emission_reduction_policy", "total_energy_consumption", "environmental_policy_to_operational_outcome", "Operational energy efficiency is a common mechanism of emission-reduction policy.", "medium"),
        EdgeRule("carbon_neutral_commitment", "emission_reduction_policy", "environmental_commitment_to_policy", "A carbon-neutral commitment should precede concrete emission-reduction policies.", "high", True),
        EdgeRule("carbon_neutral_commitment", "renewable_energy_share", "environmental_commitment_to_operational_outcome", "Carbon-neutral commitments often require greater renewable energy use.", "medium"),
        EdgeRule("climate_risk_assessment_done", "resilience_score", "risk_assessment_to_resilience", "Climate-risk assessment is a precursor to resilience planning and scoring.", "high"),
        EdgeRule("iso_14001_exists", "reporting_quality_score", "environmental_management_to_reporting", "ISO 14001 systems support structured environmental reporting.", "medium"),
        EdgeRule("iso_14001_exists", "environmental_fines", "environmental_management_to_compliance_outcome", "Environmental management certification can reduce regulatory compliance failures.", "medium"),
        EdgeRule("renewable_energy_share", "carbon_intensity", "operational_indicator_to_operational_outcome", "Higher renewable energy share mechanically lowers carbon intensity in many settings.", "high", True),
        EdgeRule("total_energy_consumption", "co2_ch4_n2o_scope_1_3", "operational_indicator_to_operational_outcome", "Energy consumption is an operational driver of emissions.", "high", True),
        EdgeRule("total_energy_consumption", "carbon_intensity", "operational_indicator_to_operational_outcome", "Energy consumption affects emissions intensity.", "medium"),
        EdgeRule("co2_ch4_n2o_scope_1_3", "environmental_fines", "esg_risk_event_to_fines", "Higher greenhouse gas emissions can increase exposure to environmental penalties.", "medium"),
        EdgeRule("carbon_intensity", "environmental_fines", "esg_risk_event_to_fines", "High carbon intensity can increase environmental compliance and penalty risk.", "medium"),
        EdgeRule("hazardous_waste_generated", "environmental_fines", "esg_risk_event_to_fines", "Hazardous waste generation increases regulatory penalty exposure.", "high"),
        EdgeRule("toxic_spills", "environmental_fines", "esg_risk_event_to_fines", "Toxic spills are direct environmental incidents that can trigger fines.", "high", True),
        EdgeRule("waste_recycled_share", "hazardous_waste_generated", "operational_indicator_to_operational_outcome", "Recycling practices can reduce waste burden.", "medium"),
        EdgeRule("water_withdrawal", "environmental_fines", "esg_risk_event_to_fines", "High water withdrawal can increase environmental regulatory exposure in constrained contexts.", "low"),
        EdgeRule("training_hours", "injury_frequency_rate", "social_action_to_operational_outcome", "Employee training can reduce workplace injury frequency.", "high"),
        EdgeRule("training_hours", "turnover_rate", "social_action_to_operational_outcome", "Training investment can influence retention and turnover.", "medium"),
        EdgeRule("training_hours", "gross_profit_margin", "social_action_to_financial_outcome", "Training can improve workforce productivity and gross margin.", "medium"),
        EdgeRule("collective_bargaining_coverage", "fair_wage_gap", "social_structure_to_workforce_outcome", "Collective bargaining coverage can affect pay equity outcomes.", "medium"),
        EdgeRule("union_membership", "turnover_rate", "social_structure_to_workforce_outcome", "Union membership can affect employee separation and retention.", "medium"),
        EdgeRule("diversity_representation", "human_rights_violations", "social_indicator_to_rights_outcome", "Workforce diversity is linked to lower employee-rights violation risk in ESG literature.", "medium"),
        EdgeRule("diversity_representation", "tobins_q", "social_indicator_to_market_value", "Diversity representation is linked to innovation and market valuation in ESG literature.", "medium"),
        EdgeRule("customer_satisfaction_score", "turnover_rate", "social_outcome_interdependency", "Customer satisfaction and employee turnover are linked operational social outcomes; direction is treated conservatively from satisfaction to retention pressure.", "low"),
        EdgeRule("customer_satisfaction_score", "net_profit_margin", "social_outcome_to_financial_outcome", "Customer satisfaction can improve retention, pricing power, and profitability.", "medium"),
        EdgeRule("injury_frequency_rate", "net_profit_margin", "social_risk_to_financial_outcome", "Workplace injuries create direct and indirect costs that can reduce margins.", "medium"),
        EdgeRule("ethical_breaches", "governance_compliance_score", "governance_event_to_compliance", "Ethical breaches are governance events that affect compliance standing.", "high"),
        EdgeRule("corruption_cases", "governance_compliance_score", "governance_event_to_compliance", "Corruption cases directly degrade governance compliance.", "high"),
        EdgeRule("anti_competitive_violations", "governance_compliance_score", "governance_event_to_compliance", "Anti-competitive violations are governance/legal compliance events.", "high"),
        EdgeRule("corruption_cases", "ethical_breaches", "governance_risk_event_relationship", "Corruption cases are a specific source of broader ethical-breach exposure.", "medium"),
        EdgeRule("governance_compliance_score", "solvency_ratio", "governance_to_financial_stability", "Governance quality supports prudent risk management and financial stability.", "medium"),
        EdgeRule("governance_compliance_score", "debt_to_equity_ratio", "governance_to_capital_structure", "Governance quality influences capital structure and creditor trust.", "medium"),
        EdgeRule("governance_compliance_score", "tobins_q", "governance_to_market_value", "Governance quality can affect market valuation.", "medium"),
        EdgeRule("environmental_fines", "market_value_equity", "environmental_event_to_market_value", "Environmental fines can reduce market value through legal and reputational channels.", "high"),
        EdgeRule("environmental_fines", "debt_to_equity_ratio", "environmental_event_to_financing", "Environmental penalties can affect financing conditions and leverage.", "medium"),
        EdgeRule("carbon_intensity", "market_value_equity", "environmental_indicator_to_market_value", "Carbon intensity can affect market valuation through transition-risk channels.", "medium"),
        EdgeRule("carbon_intensity", "tobins_q", "environmental_indicator_to_market_value", "Carbon intensity can affect Tobin's Q through transition-risk channels.", "medium"),
        EdgeRule("emission_reduction_policy", "roa_eat", "environmental_policy_to_financial_outcome", "Emission-reduction policies can improve operating efficiency over time.", "medium"),
        EdgeRule("emission_reduction_policy", "roe_eat", "environmental_policy_to_financial_outcome", "Emission-reduction policies can affect shareholder returns through efficiency and financing channels.", "medium"),
        EdgeRule("iso_14001_exists", "net_profit_margin", "environmental_management_to_financial_outcome", "Environmental management systems can improve operational discipline and margins.", "medium"),
    ]
    return [edge for edge in candidates if edge.source in included and edge.target in included]


def edges_to_frame(edges: list[EdgeRule], specs: dict[str, VariableSpec]) -> pd.DataFrame:
    """Convert edge records to a dataframe."""
    rows = []
    for edge in edges:
        source = specs[edge.source]
        target = specs[edge.target]
        rows.append({
            "source": edge.source,
            "target": edge.target,
            "rule_name": edge.rule_name,
            "justification": edge.justification,
            "source_domain": source.domain,
            "target_domain": target.domain,
            "source_type": source.ontology_type,
            "target_type": target.ontology_type,
            "confidence_level": edge.confidence_level,
            "origin": "metadata_ontology_rule",
            "required_light": edge.required_light,
        })
    return pd.DataFrame(rows)


def build_reference_edges(
    included: list[str],
    specs: dict[str, VariableSpec],
) -> tuple[pd.DataFrame, str, str, Path | None]:
    """Build explicit-rule or metadata-reference edge dataframe."""
    explicit_path = find_explicit_rules_file()
    included_set = set(included)
    if explicit_path is not None:
        edges = load_explicit_edges(explicit_path, included_set, specs)
        graph_source_mode = "explicit_rules"
        evaluation_target = "advisor_rule_ground_truth_dag"
        return edges.drop_duplicates(["source", "target"]), graph_source_mode, evaluation_target, explicit_path

    edges = edges_to_frame(conservative_edges(included_set), specs)
    graph_source_mode = "metadata_reference"
    evaluation_target = "ontology_derived_reference_dag"
    return edges.drop_duplicates(["source", "target"]), graph_source_mode, evaluation_target, None


def build_forbidden(edges: pd.DataFrame, variables: list[str], specs: dict[str, VariableSpec]) -> pd.DataFrame:
    """Create forbidden-only constraints from reverse edges and ontology impossibilities."""
    required_pairs = set(zip(edges["source"], edges["target"]))
    rows: list[dict[str, str]] = []
    for source, target in sorted(required_pairs):
        if (target, source) not in required_pairs:
            rows.append({
                "cause": target,
                "effect": source,
                "rule_name": "forbid_reverse_of_reference_edge",
                "justification": f"Reverse of reference edge {source} -> {target} is not allowed in the main forbidden-only constraint set.",
            })

    financial = [name for name in variables if specs[name].causal_role == "financial_outcome"]
    esg_prior = [
        name for name in variables
        if specs[name].domain in {"environmental", "social", "governance"}
        and specs[name].causal_role in {"driver", "event_driver", "operational_outcome", "indicator"}
    ]
    for fin in financial:
        for esg in esg_prior:
            if fin != esg and (fin, esg) not in required_pairs:
                rows.append({
                    "cause": fin,
                    "effect": esg,
                    "rule_name": "forbid_financial_outcome_to_prior_esg_indicator",
                    "justification": "Financial outcomes should not cause prior ESG operational, risk, or policy indicators without explicit temporal justification.",
                })
    forbidden = pd.DataFrame(rows).drop_duplicates(["cause", "effect"])
    return forbidden


def save_constraint_sets(edges: pd.DataFrame, variables: list[str], specs: dict[str, VariableSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save forbidden-only, required-light, and full-reference sanity constraints."""
    forbidden = build_forbidden(edges, variables, specs)
    required_light = edges[edges["required_light"] == True].rename(columns={"source": "cause", "target": "effect"})[
        ["cause", "effect", "rule_name", "justification", "confidence_level", "origin"]
    ].copy()
    full_required = edges.rename(columns={"source": "cause", "target": "effect"})[
        ["cause", "effect", "rule_name", "justification", "confidence_level", "origin"]
    ].copy()

    forbidden.to_csv(FORBIDDEN_PATH, index=False)
    required_light.to_csv(REQUIRED_LIGHT_PATH, index=False)
    full_required.to_csv(FULL_REQUIRED_PATH, index=False)

    # Legacy compatibility copies.
    forbidden.to_csv(LEGACY_FORBIDDEN_PATH, index=False)
    full_required.to_csv(LEGACY_REQUIRED_PATH, index=False)
    return forbidden, required_light, full_required


def validate_reference(
    edges: pd.DataFrame,
    forbidden: pd.DataFrame,
    variables: list[str],
    cleaned: pd.DataFrame,
    data_source: str,
    metadata_source: str,
    graph_source_mode: str,
    evaluation_target: str,
    explicit_rules_path: Path | None,
    original_shape: tuple[int, int],
) -> tuple[bool, str]:
    """Validate reference graph and write markdown validation report."""
    edge_pairs = list(zip(edges["source"], edges["target"]))
    duplicate_count = int(pd.DataFrame(edge_pairs).duplicated().sum()) if edge_pairs else 0
    self_loops = [(source, target) for source, target in edge_pairs if source == target]
    missing_nodes = sorted({node for pair in edge_pairs for node in pair if node not in cleaned.columns})
    forbidden_pairs = set(zip(forbidden["cause"], forbidden["effect"])) if not forbidden.empty else set()
    contradictions = sorted(set(edge_pairs) & forbidden_pairs)

    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    graph.add_edges_from(edge_pairs)
    acyclic = nx.is_directed_acyclic_graph(graph)
    cycles: list[Any] = []
    if not acyclic:
        try:
            cycles = list(nx.find_cycle(graph))
        except Exception:
            cycles = []

    adjacency = pd.DataFrame(0, index=variables, columns=variables, dtype=int)
    for source, target in edge_pairs:
        adjacency.loc[source, target] = 1
    adjacency.to_csv(REFERENCE_ADJACENCY_PATH)
    adjacency.to_csv(LEGACY_ADJACENCY_PATH)

    ok = acyclic and not missing_nodes and duplicate_count == 0 and not self_loops and not contradictions
    warning = ""
    if graph_source_mode == "metadata_reference":
        warning = (
            "\nNo explicit advisor-provided causal generation DAG/rule file was found. "
            "The current graph is an ontology-derived reference DAG reconstructed from "
            "metadata and domain rules. It should not be described as a true data-generating DAG.\n"
        )
    report = f"""# Advisor Dummy Reference DAG Validation

The advisor-provided dummy CSV and metadata workbook are used as the official
dummy-data source for this evaluation.
{warning}
- Data source: `{data_source}`
- Metadata source: `{metadata_source}`
- Explicit rule file: `{explicit_rules_path if explicit_rules_path else "not found"}`
- Graph source mode: `{graph_source_mode}`
- Evaluation target: `{evaluation_target}`
- Original CSV shape: {original_shape[0]} rows x {original_shape[1]} columns
- Cleaned model dataset: `{PROCESSED_DUMMY_PATH.as_posix()}`
- Cleaned model shape: {cleaned.shape[0]} rows x {cleaned.shape[1]} columns
- Nodes included: {len(variables)}
- Reference edges: {len(edge_pairs)}
- Forbidden constraints: {len(forbidden)}
- Acyclic: {acyclic}
- Duplicate edges: {duplicate_count}
- Self-loops: {len(self_loops)}
- Missing edge variables from cleaned data: {len(missing_nodes)}
- Edges contradicting forbidden constraints: {len(contradictions)}
- Validation status: {"PASS" if ok else "FAIL"}

## Notes

F1 and SHD for `advisor_dummy` are computed against the evaluation target
listed above. In metadata-reference mode, these metrics indicate alignment
with an ontology-derived reference DAG, not recovery of an experimentally
known causal mechanism.
"""
    if cycles:
        report += "\n## Cycles\n\n" + "\n".join(f"- {cycle}" for cycle in cycles) + "\n"
    if missing_nodes:
        report += "\n## Missing Variables\n\n" + "\n".join(f"- {node}" for node in missing_nodes) + "\n"
    if contradictions:
        report += "\n## Constraint Contradictions\n\n" + "\n".join(f"- {s} -> {t}" for s, t in contradictions) + "\n"
    VALIDATION_PATH.write_text(report, encoding="utf-8")
    LEGACY_VALIDATION_PATH.write_text(report, encoding="utf-8")
    return ok, report


def build_reference(args: argparse.Namespace | SimpleNamespace) -> int:
    """Build all advisor dummy reference-DAG artifacts."""
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    allow_regeneration = bool(getattr(args, "allow_dummy_regeneration", False))
    require_advisor_files(allow_regeneration)

    specs: dict[str, VariableSpec] = {}
    if ADVISOR_SPEC.exists():
        specs.update(specs_from_text(ADVISOR_SPEC))
    if ADVISOR_XLSX.exists():
        specs.update(load_metadata_specs(ADVISOR_XLSX))
    if not specs:
        raise ValueError("No variable specs could be parsed from advisor metadata/text source.")

    raw, data_source, original_shape = load_raw_advisor_data(specs, allow_regeneration)
    for column in raw.columns:
        specs.setdefault(column, infer_spec_for_column(column, raw[column]))

    cleaned_full, cleaned_model, audit = clean_advisor_dummy_dataset(raw, specs)
    if cleaned_model.empty:
        raise ValueError("No advisor dummy variables passed the model inclusion filters.")

    cleaned_full.to_csv(CLEANED_FULL_PATH, index=False)
    cleaned_model.to_csv(CLEANED_MODEL_PATH, index=False)
    cleaned_model.to_csv(PROCESSED_DUMMY_PATH, index=False)
    audit.to_csv(DATA_AUDIT_PATH, index=False)

    registry = build_metadata_registry(specs, audit)
    included = registry[registry["included_in_reference_dag"] == True]["variable_name"].tolist()
    edges, graph_source_mode, evaluation_target, explicit_rules_path = build_reference_edges(included, specs)
    edges.to_csv(REFERENCE_EDGES_PATH, index=False)
    edges.to_csv(LEGACY_REFERENCE_EDGES_PATH, index=False)
    forbidden, required_light, full_required = save_constraint_sets(edges, included, specs)

    metadata_source = str(ADVISOR_XLSX.relative_to(ROOT)) if ADVISOR_XLSX.exists() else (
        str(ADVISOR_SPEC.relative_to(ROOT)) if ADVISOR_SPEC.exists() else "not found"
    )
    ok, _ = validate_reference(
        edges=edges,
        forbidden=forbidden,
        variables=included,
        cleaned=cleaned_model,
        data_source=data_source,
        metadata_source=metadata_source,
        graph_source_mode=graph_source_mode,
        evaluation_target=evaluation_target,
        explicit_rules_path=explicit_rules_path,
        original_shape=original_shape,
    )

    log(f"Data audit -> {DATA_AUDIT_PATH.relative_to(ROOT)}")
    log(f"Metadata registry -> {METADATA_REGISTRY_PATH.relative_to(ROOT)}")
    log(f"Cleaned advisor dummy -> {CLEANED_MODEL_PATH.relative_to(ROOT)}")
    log(f"Reference DAG edges -> {REFERENCE_EDGES_PATH.relative_to(ROOT)}")
    log(f"Forbidden constraints -> {FORBIDDEN_PATH.relative_to(ROOT)}")
    log(f"Required-light constraints -> {REQUIRED_LIGHT_PATH.relative_to(ROOT)}")
    log(f"Full-reference required constraints -> {FULL_REQUIRED_PATH.relative_to(ROOT)}")
    log(f"Validation -> {VALIDATION_PATH.relative_to(ROOT)}")
    log(
        f"Validation status: {'success' if ok else 'failed'}; "
        f"mode={graph_source_mode}; nodes={len(included)}, edges={len(edges)}; "
        f"required_light={len(required_light)}"
    )
    return 0 if ok else 1


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build advisor dummy ontology-derived reference DAG.")
    parser.add_argument("--allow-dummy-regeneration", action="store_true",
                        help="Development-only fallback if the advisor CSV is absent.")
    args = parser.parse_args()
    return build_reference(args)


if __name__ == "__main__":
    raise SystemExit(main())
