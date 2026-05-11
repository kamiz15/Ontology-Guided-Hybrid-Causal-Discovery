"""Diagnose enum violations in raw RAG claim JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CLAIMS_DIR = Path("data/raw/rag_claims")
OUTPUT_PATH = Path("reports/claims_enum_violations.csv")

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def clean_enum(value: Any) -> str:
    """Normalize a raw enum value for validation.

    Parameters
    ----------
    value:
        Raw JSON value.

    Returns
    -------
    str
        Lowercase stripped string representation.
    """

    if value is None:
        return ""
    return str(value).strip().lower()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load one claim JSON array.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    list[dict[str, Any]]
        Loaded claims.

    Raises
    ------
    ValueError
        If the file is not a JSON array of objects.
    """

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path} row {idx} is not a JSON object")
    return payload


def diagnose_claims(claims_dir: Path = CLAIMS_DIR) -> pd.DataFrame:
    """Diagnose enum violations across claim files.

    Parameters
    ----------
    claims_dir:
        Directory containing raw RAG claim JSON arrays.

    Returns
    -------
    pd.DataFrame
        Violation table with one row per invalid enum value.
    """

    violations: list[dict[str, object]] = []
    files = sorted(claims_dir.glob("*.json"))
    print(f"[diagnose] Claims directory: {claims_dir}")
    print(f"[diagnose] JSON files found: {len(files)}")

    for path in files:
        claims = load_json_array(path)
        file_violations: list[tuple[int, str, str]] = []

        for row_index, claim in enumerate(claims):
            for field, allowed in ALLOWED_VALUES.items():
                normalized = clean_enum(claim.get(field))
                if normalized in allowed:
                    continue
                raw_value = claim.get(field)
                bad_value = "" if raw_value is None else str(raw_value)
                file_violations.append((row_index, field, bad_value))
                violations.append(
                    {
                        "source_file": path.name,
                        "row_index": row_index,
                        "field": field,
                        "bad_value": bad_value,
                        "normalized_value": normalized,
                    }
                )

        print(f"\n[diagnose] {path.name}")
        print(f"[diagnose] Total claims: {len(claims)}")
        for field in ALLOWED_VALUES:
            count = sum(1 for _, bad_field, _ in file_violations if bad_field == field)
            print(f"[diagnose] Invalid {field}: {count}")
        if file_violations:
            print("[diagnose] Violations:")
            for violation in file_violations:
                print(f"[diagnose]   {violation}")
        else:
            print("[diagnose] Violations: none")

    df = pd.DataFrame(
        violations,
        columns=["source_file", "row_index", "field", "bad_value", "normalized_value"],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[diagnose] Saved violations -> {OUTPUT_PATH}")
    print(f"[diagnose] Total violations: {len(df)}")
    return df


if __name__ == "__main__":
    diagnose_claims()
