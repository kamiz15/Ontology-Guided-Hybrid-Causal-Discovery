# 10_constraints_from_claims.py
# ============================================================
# Step 10 - Aggregate RAG literature claims into draft causal
# constraints for human review.
#
# Usage:
#   python 10_constraints_from_claims.py
#   python 10_constraints_from_claims.py --claims-dir data/raw/rag_claims
#
# Output:
#   reports/claim_aggregation.csv
#   reports/constraints_for_review.csv
#   data/processed/forbidden_edges_draft.csv
#   data/processed/required_edges_draft.csv
# ============================================================

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CLAIM_AGGREGATION_PATH,
    CONSTRAINTS_REVIEW_PATH,
    FORBIDDEN_EDGES_DRAFT_PATH,
    RAG_CLAIMS_DIR,
    REQUIRED_EDGES_DRAFT_PATH,
)


REQUIRED_SCHEMA = [
    "paper_id",
    "page_or_section",
    "quote",
    "claim_type",
    "cause_raw",
    "effect_raw",
    "cause_mapped",
    "effect_mapped",
    "direction",
    "evidence_type",
    "sample_or_scope",
    "effect_sign",
    "lag",
    "forbidden_reverse",
    "caveats",
    "confidence",
]

ALLOWED_VALUES = {
    "claim_type": {
        "causal",
        "structural",
        "temporal",
        "impossibility",
        "method_note",
    },
    "direction": {"cause_to_effect", "bidirectional", "ambiguous"},
    "evidence_type": {
        "empirical_quantitative",
        "empirical_qualitative",
        "theoretical",
        "definitional",
        "physical_law",
        "regulatory",
    },
    "effect_sign": {"positive", "negative", "mixed", "unspecified"},
    "forbidden_reverse": {"yes", "no", "contested", "not_addressed"},
    "confidence": {"high", "medium", "low"},
}

AGGREGATE_TOKEN_MAP = {
    "ESG": "overall_esg_score",
    "E": "env_pillar_score",
    "S": "soc_pillar_score",
    "G": "gov_pillar_score",
}

FORBIDDEN_REVERSE_CLAIM_TYPES = {"structural", "impossibility", "temporal"}

REVIEW_SAMPLE_PER_TIER = 10


def _empty_claims_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_SCHEMA + ["source_file"])


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_enum(value: Any) -> str:
    return _clean_text(value).lower()


def _contains_any(value: str, needles: list[str]) -> bool:
    return any(needle.lower() in value for needle in needles)


def _normalise_claim_type(value: Any) -> str:
    text = _normalise_enum(value)
    if text in ALLOWED_VALUES["claim_type"]:
        return text
    if _contains_any(text, ["causal", "effect", "empirical_finding", "relationship", "empirical"]):
        return "causal"
    if _contains_any(text, ["temporal", "lag", "delayed"]):
        return "temporal"
    if _contains_any(text, ["structural", "composite", "definitional", "decomposition"]):
        return "structural"
    if "impossib" in text:
        return "impossibility"
    if _contains_any(text, ["method", "review", "limitation"]):
        return "method_note"
    print(f"[step_10] WARNING: unknown claim_type={value!r}; defaulting to causal")
    return "causal"


def _normalise_direction(value: Any) -> str:
    text = _normalise_enum(value)
    if text in ALLOWED_VALUES["direction"]:
        return text
    if _contains_any(text, ["cause", "forward", "predictor"]):
        return "cause_to_effect"
    if _contains_any(text, ["bidirec", "both", "mutual"]):
        return "bidirectional"
    if _contains_any(text, ["ambig", "unclear", "mixed"]):
        return "ambiguous"
    return "cause_to_effect"


def _normalise_evidence_type(value: Any) -> str:
    text = _normalise_enum(value)
    if text in ALLOWED_VALUES["evidence_type"]:
        return text
    if _contains_any(text, ["quantitative", "regression", "panel", "gmm", "iv"]):
        return "empirical_quantitative"
    if _contains_any(text, ["qualitative", "case study", "interview"]):
        return "empirical_qualitative"
    if _contains_any(text, ["theoretical", "theory"]):
        return "theoretical"
    if _contains_any(text, ["definition", "definitional", "tautological"]):
        return "definitional"
    if _contains_any(text, ["physical", "law"]):
        return "physical_law"
    if _contains_any(text, ["regulatory", "regulation", "policy"]):
        return "regulatory"
    return "empirical_quantitative"


def _normalise_effect_sign(value: Any) -> str:
    text = _normalise_enum(value)
    if text in ALLOWED_VALUES["effect_sign"]:
        return text
    if _contains_any(text, ["positive", "+"]):
        return "positive"
    if _contains_any(text, ["negative", "-", "inverse"]):
        return "negative"
    if _contains_any(text, ["mixed", "context-dependent"]):
        return "mixed"
    if _contains_any(text, ["null", "none", "no effect", "unspecified"]):
        return "unspecified"
    return "unspecified"


def _normalise_forbidden_reverse(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    text = _normalise_enum(value)
    if text in {"yes", "true"}:
        return "yes"
    if text in {"no", "false"}:
        return "no"
    if "contest" in text:
        return "contested"
    return "not_addressed"


def _normalise_confidence(value: Any) -> str:
    text = _normalise_enum(value)
    if text in ALLOWED_VALUES["confidence"]:
        return text
    if _contains_any(text, ["high", "strong"]):
        return "high"
    if _contains_any(text, ["medium", "moderate"]):
        return "medium"
    if _contains_any(text, ["low", "weak"]):
        return "low"
    return "medium"


def normalize_enums(claims_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map variant RAG enum spellings to canonical values.

    Parameters
    ----------
    claims_df : pd.DataFrame
        Raw claim table after required schema validation.

    Returns
    -------
    pd.DataFrame
        Claim table with canonical enum values.
    """
    out = claims_df.copy()
    normalizers = {
        "claim_type": _normalise_claim_type,
        "direction": _normalise_direction,
        "evidence_type": _normalise_evidence_type,
        "effect_sign": _normalise_effect_sign,
        "forbidden_reverse": _normalise_forbidden_reverse,
        "confidence": _normalise_confidence,
    }
    changed_counts: dict[str, int] = {}

    print(f"[step_10] Normalizing enum values across {len(out)} claims...")
    for col, normalizer in normalizers.items():
        original = out[col].copy()
        normalized = original.map(normalizer)
        original_clean = original.map(_normalise_enum)
        changed = int((original_clean != normalized).sum())
        changed_counts[col] = changed
        out[col] = normalized

    print(
        "[step_10] Normalized: "
        f"claim_type={changed_counts['claim_type']}, "
        f"direction={changed_counts['direction']}, "
        f"evidence_type={changed_counts['evidence_type']}, "
        f"effect_sign={changed_counts['effect_sign']}, "
        f"forbidden_reverse={changed_counts['forbidden_reverse']}, "
        f"confidence={changed_counts['confidence']}"
    )
    return out


def _jsonish(value: Any) -> str:
    if isinstance(value, set):
        value = sorted(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _counter(series: pd.Series) -> dict[str, int]:
    values = [_normalise_enum(v) for v in series if _clean_text(v)]
    return dict(sorted(Counter(values).items()))


def _sorted_unique(series: pd.Series) -> list[str]:
    values = {_clean_text(v) for v in series if _clean_text(v)}
    return sorted(values)


def _choose_top_claim(group: pd.DataFrame) -> pd.Series:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    evidence_rank = {
        "empirical_quantitative": 0,
        "empirical_qualitative": 1,
        "regulatory": 2,
        "physical_law": 3,
        "theoretical": 4,
        "definitional": 5,
    }

    ranked = group.copy()
    ranked["_confidence_rank"] = ranked["confidence"].map(
        lambda v: confidence_rank.get(_normalise_enum(v), 9)
    )
    ranked["_evidence_rank"] = ranked["evidence_type"].map(
        lambda v: evidence_rank.get(_normalise_enum(v), 9)
    )
    ranked["_quote_len"] = ranked["quote"].map(lambda v: len(_clean_text(v)))
    ranked = ranked.sort_values(
        ["_confidence_rank", "_evidence_rank", "_quote_len"],
        ascending=[True, True, False],
    )
    return ranked.iloc[0]


def _has_forbidden_reverse_support(group: pd.DataFrame) -> bool:
    claim_type_match = group["claim_type"].map(_normalise_enum).isin(
        FORBIDDEN_REVERSE_CLAIM_TYPES
    )
    reverse_vote_match = group["forbidden_reverse"].map(_normalise_enum) == "yes"
    return bool((claim_type_match & reverse_vote_match).any())


def _has_high_quantitative(group: pd.DataFrame) -> bool:
    high = group["confidence"].map(_normalise_enum) == "high"
    quantitative = group["evidence_type"].map(_normalise_enum) == "empirical_quantitative"
    return bool((high & quantitative).any())


def _contradiction_reasons(group: pd.DataFrame) -> list[str]:
    reasons = []

    signs = {
        _normalise_enum(v)
        for v in group["effect_sign"]
        if _normalise_enum(v) not in {"", "mixed", "unspecified"}
    }
    if {"positive", "negative"}.issubset(signs):
        reasons.append("mixed positive/negative effect signs")

    reverse_votes = {
        _normalise_enum(v)
        for v in group["forbidden_reverse"]
        if _normalise_enum(v) not in {"", "not_addressed"}
    }
    if "yes" in reverse_votes and ({"no", "contested"} & reverse_votes):
        reasons.append("conflicting forbidden_reverse votes")

    directions = {
        _normalise_enum(v)
        for v in group["direction"]
        if _normalise_enum(v) not in {"", "cause_to_effect"}
    }
    if "ambiguous" in directions and "bidirectional" not in directions:
        reasons.append("ambiguous direction")

    return reasons


def _ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _serialise_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (dict, list, set))).any():
            out[col] = out[col].map(_jsonish)
    return out


def load_claims(claims_dir: str) -> pd.DataFrame:
    """
    Load and validate raw RAG claim JSON files.

    Parameters
    ----------
    claims_dir : str
        Directory containing one or more JSON files. Each file must contain a
        JSON array of claim objects.

    Returns
    -------
    pd.DataFrame
        Concatenated claim table with the required schema columns.

    Raises
    ------
    ValueError
        If any file is not a JSON array or any row violates the schema. Missing
        or empty input directories return an empty DataFrame and log a warning.
    """
    claims_path = Path(claims_dir)
    if not claims_path.exists():
        print(f"[step_10] WARNING: claims directory not found: {claims_dir}")
        return _empty_claims_frame()

    files = sorted(claims_path.glob("*.json"))
    if not files:
        print(f"[step_10] WARNING: no JSON claim files found in {claims_dir}")
        return _empty_claims_frame()

    rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array")
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"{path} row {i} is not a JSON object")
            missing = [col for col in REQUIRED_SCHEMA if col not in item]
            if missing:
                raise ValueError(f"{path} row {i} missing required keys: {missing}")
            row = {col: item.get(col) for col in REQUIRED_SCHEMA}
            row["source_file"] = path.name
            rows.append(row)

    if not rows:
        print(f"[step_10] WARNING: JSON files contained no claim rows: {claims_dir}")
        return _empty_claims_frame()

    df = pd.DataFrame(rows)
    df = normalize_enums(df)

    for col, allowed in ALLOWED_VALUES.items():
        values = df[col].map(_normalise_enum)
        bad = sorted(set(values) - allowed)
        if bad:
            bad_rows = df.loc[values.isin(bad), [col, "source_file"]].head(10)
            bad_examples = bad_rows.to_dict(orient="records")
            raise ValueError(
                f"Invalid values in {col}: {bad}; "
                f"normalized examples: {bad_examples}"
            )
        df[col] = values

    for col in ["paper_id", "page_or_section", "quote", "cause_raw", "effect_raw",
                "cause_mapped", "effect_mapped", "sample_or_scope", "lag", "source_file"]:
        df[col] = df[col].map(_clean_text)

    df["caveats"] = df["caveats"].map(lambda v: None if v is None else _clean_text(v))
    df["cause_mapped"] = df["cause_mapped"].replace("", "UNMAPPED")
    df["effect_mapped"] = df["effect_mapped"].replace("", "UNMAPPED")

    print(f"[step_10] Loaded {len(df)} claims from {len(files)} files.")
    return df


def normalize_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map aggregate ESG shorthand tokens when the mapped field is unmapped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw claim table from :func:`load_claims`.

    Returns
    -------
    pd.DataFrame
        Claim table with `cause_mapped` and `effect_mapped` updated for raw
        aggregate tokens `ESG`, `E`, `S`, and `G`.
    """
    out = df.copy()
    replacements = 0

    for side in ["cause", "effect"]:
        raw_col = f"{side}_raw"
        mapped_col = f"{side}_mapped"
        for idx, row in out.iterrows():
            if _clean_text(row[mapped_col]).upper() != "UNMAPPED":
                continue
            raw_token = _clean_text(row[raw_col]).upper()
            if raw_token in AGGREGATE_TOKEN_MAP:
                out.at[idx, mapped_col] = AGGREGATE_TOKEN_MAP[raw_token]
                replacements += 1

    print(f"[step_10] Aggregate shorthand replacements: {replacements}")
    return out


def aggregate_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate normalized claims into one row per directed edge.

    Parameters
    ----------
    df : pd.DataFrame
        Claim table after aggregate normalization.

    Returns
    -------
    pd.DataFrame
        Aggregated edge audit table. Object-valued columns such as `papers`,
        `confidence_dist`, and `claim_types` are serialized by `write_outputs`.
    """
    working = df.copy()
    working["cause_mapped"] = working["cause_mapped"].map(_clean_text)
    working["effect_mapped"] = working["effect_mapped"].map(_clean_text)

    self_loop_mask = working["cause_mapped"] == working["effect_mapped"]
    unmapped_mask = (
        working["cause_mapped"].str.upper().eq("UNMAPPED")
        | working["effect_mapped"].str.upper().eq("UNMAPPED")
    )

    self_loop_count = int(self_loop_mask.sum())
    unmapped_count = int((~self_loop_mask & unmapped_mask).sum())
    print(f"[step_10] Dropping self-loop claims: {self_loop_count}")
    print(f"[step_10] Dropping claims with UNMAPPED endpoint: {unmapped_count}")

    aggregate_edges.last_self_loop_drop_count = self_loop_count
    aggregate_edges.last_unmapped_drop_count = unmapped_count

    working = working.loc[~self_loop_mask & ~unmapped_mask].copy()
    if working.empty:
        return pd.DataFrame(columns=[
            "cause_mapped",
            "effect_mapped",
            "cause",
            "effect",
            "paper_count",
            "papers",
            "confidence_dist",
            "evidence_dist",
            "claim_types",
            "forbidden_reverse_votes",
            "has_contradiction",
            "contradiction_reasons",
            "has_high_quantitative",
            "has_forbidden_reverse_support",
            "top_quote",
            "top_quote_paper",
            "top_quote_section",
            "effect_sign_dist",
            "direction_dist",
            "source_files",
        ])

    rows = []
    grouped = working.groupby(["cause_mapped", "effect_mapped"], dropna=False)
    for (cause, effect), group in grouped:
        top = _choose_top_claim(group)
        papers = _sorted_unique(group["paper_id"])
        reasons = _contradiction_reasons(group)

        rows.append({
            "cause_mapped": cause,
            "effect_mapped": effect,
            "cause": cause,
            "effect": effect,
            "paper_count": len(papers),
            "papers": papers,
            "confidence_dist": _counter(group["confidence"]),
            "evidence_dist": _counter(group["evidence_type"]),
            "claim_types": set(_sorted_unique(group["claim_type"])),
            "forbidden_reverse_votes": _counter(group["forbidden_reverse"]),
            "has_contradiction": bool(reasons),
            "contradiction_reasons": reasons,
            "has_high_quantitative": _has_high_quantitative(group),
            "has_forbidden_reverse_support": _has_forbidden_reverse_support(group),
            "top_quote": _clean_text(top["quote"]),
            "top_quote_paper": _clean_text(top["paper_id"]),
            "top_quote_section": _clean_text(top["page_or_section"]),
            "effect_sign_dist": _counter(group["effect_sign"]),
            "direction_dist": _counter(group["direction"]),
            "source_files": _sorted_unique(group["source_file"]),
        })

    edges = pd.DataFrame(rows)

    edge_pairs = set(zip(edges["cause_mapped"], edges["effect_mapped"]))
    for idx, row in edges.iterrows():
        reverse_pair = (row["effect_mapped"], row["cause_mapped"])
        if reverse_pair not in edge_pairs:
            continue
        directions = set(row["direction_dist"].keys())
        reverse_dirs = set(
            edges.loc[
                (edges["cause_mapped"] == row["effect_mapped"])
                & (edges["effect_mapped"] == row["cause_mapped"]),
                "direction_dist",
            ].iloc[0].keys()
        )
        if "bidirectional" not in directions and "bidirectional" not in reverse_dirs:
            reasons = list(edges.at[idx, "contradiction_reasons"])
            if "reciprocal directed claims" not in reasons:
                reasons.append("reciprocal directed claims")
            edges.at[idx, "contradiction_reasons"] = reasons
            edges.at[idx, "has_contradiction"] = True

    edges = edges.sort_values(
        ["paper_count", "cause_mapped", "effect_mapped"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    print(f"[step_10] Aggregated {len(edges)} directed edges.")
    return edges


def apply_tiers(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign constraint tiers and proposed actions.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Aggregated edge audit table from :func:`aggregate_edges`.

    Returns
    -------
    pd.DataFrame
        Edge table with `tier` and `proposed_action` columns.
    """
    out = edges_df.copy()
    tiers = []
    actions = []

    for _, row in out.iterrows():
        tier1 = (
            int(row["paper_count"]) >= 3
            and not bool(row["has_contradiction"])
            and bool(row["has_high_quantitative"])
        )
        tier2 = bool(row["has_forbidden_reverse_support"])

        if tier1:
            tiers.append(1)
            actions.append("required")
        elif tier2:
            tiers.append(2)
            actions.append("forbid_reverse")
        else:
            tiers.append(3)
            actions.append("review")

    out["tier"] = tiers
    out["proposed_action"] = actions
    return out


def write_outputs(edges_df: pd.DataFrame, paths: dict[str, str]) -> None:
    """
    Write aggregation, review, and draft constraint CSV outputs.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Aggregated and tiered edge table.
    paths : dict[str, str]
        Output paths with keys `aggregation`, `review`, `forbidden_draft`,
        and `required_draft`.

    Returns
    -------
    None
    """
    for path in paths.values():
        _ensure_output_dir(path)

    aggregation = _serialise_for_csv(edges_df)
    aggregation.to_csv(paths["aggregation"], index=False)
    print(f"[step_10] Saved aggregation -> {paths['aggregation']}")

    tier3 = edges_df[edges_df["tier"] == 3].copy()
    tier_samples = []
    for tier in [1, 2]:
        sample = (
            edges_df[edges_df["tier"] == tier]
            .sort_values(["paper_count", "cause", "effect"], ascending=[False, True, True])
            .head(REVIEW_SAMPLE_PER_TIER)
        )
        tier_samples.append(sample)

    review = pd.concat([tier3] + tier_samples, ignore_index=True)
    review = review.sort_values(["tier", "paper_count", "cause", "effect"],
                                ascending=[True, False, True, True])
    review_out = pd.DataFrame({
        "cause": review["cause"],
        "effect": review["effect"],
        "tier": review["tier"],
        "proposed_action": review["proposed_action"],
        "paper_count": review["paper_count"],
        "top_quote": review["top_quote"],
        "paper_ids": review["papers"].map(lambda v: "; ".join(v)),
        "approved": "",
        "notes": "",
    })
    review_out.to_csv(paths["review"], index=False)
    print(f"[step_10] Saved review file -> {paths['review']}")

    required = edges_df[edges_df["proposed_action"] == "required"].copy()
    required_out = pd.DataFrame({
        "source": required["cause"],
        "target": required["effect"],
        "tier": required["tier"],
        "paper_count": required["paper_count"],
        "paper_ids": required["papers"].map(lambda v: "; ".join(v)),
        "top_quote": required["top_quote"],
    })
    required_out.to_csv(paths["required_draft"], index=False)
    print(f"[step_10] Saved required draft -> {paths['required_draft']}")

    forbidden = edges_df[edges_df["proposed_action"] == "forbid_reverse"].copy()
    forbidden_out = pd.DataFrame({
        "source": forbidden["effect"],
        "target": forbidden["cause"],
        "tier": forbidden["tier"],
        "paper_count": forbidden["paper_count"],
        "paper_ids": forbidden["papers"].map(lambda v: "; ".join(v)),
        "top_quote": forbidden["top_quote"],
    })
    forbidden_out.to_csv(paths["forbidden_draft"], index=False)
    print(f"[step_10] Saved forbidden draft -> {paths['forbidden_draft']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RAG literature claims into draft constraints."
    )
    parser.add_argument("--claims-dir", default=RAG_CLAIMS_DIR,
                        help="Directory containing RAG claim JSON arrays")
    args = parser.parse_args()

    paths = {
        "aggregation": CLAIM_AGGREGATION_PATH,
        "review": CONSTRAINTS_REVIEW_PATH,
        "forbidden_draft": FORBIDDEN_EDGES_DRAFT_PATH,
        "required_draft": REQUIRED_EDGES_DRAFT_PATH,
    }

    claims = load_claims(args.claims_dir)
    if claims.empty:
        print("[step_10] No claims loaded; nothing to aggregate.")
        print("\n[step_10] Summary")
        print("[step_10] Claims loaded        : 0")
        print("[step_10] Edges aggregated     : 0")
        print("[step_10] Tier counts          : {}")
        print("[step_10] Unmapped drops       : 0")
        print("[step_10] Self-loop drops      : 0")
        print("[step_10] Contradictions found : 0")
        return

    normalised = normalize_aggregates(claims)
    edges = aggregate_edges(normalised)
    tiered = apply_tiers(edges)
    write_outputs(tiered, paths)

    tier_counts = tiered["tier"].value_counts().sort_index().to_dict()
    contradictions = int(tiered["has_contradiction"].sum()) if not tiered.empty else 0
    unmapped_drop_count = getattr(aggregate_edges, "last_unmapped_drop_count", 0)
    self_loop_drop_count = getattr(aggregate_edges, "last_self_loop_drop_count", 0)

    print("\n[step_10] Summary")
    print(f"[step_10] Claims loaded        : {len(claims)}")
    print(f"[step_10] Edges aggregated     : {len(tiered)}")
    print(f"[step_10] Tier counts          : {tier_counts}")
    print(f"[step_10] Unmapped drops       : {unmapped_drop_count}")
    print(f"[step_10] Self-loop drops      : {self_loop_drop_count}")
    print(f"[step_10] Contradictions found : {contradictions}")


if __name__ == "__main__":
    main()
