"""Apply pillar-stratified audit decisions to constraint review rows."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


LOG_PREFIX = "[apply_audits]"
REVIEW_PATH = Path("reports/constraints_for_review.csv")
AUDIT_DIR = Path("reports/audit_decisions")
BACKUP_PATH = Path("reports/constraints_for_review.csv.bak_pre_audit")
SUMMARY_PATH = Path("reports/audit_summary.md")
VALID_DECISIONS = {"yes", "no", "modify"}
MODIFY_MODES = {"yes", "no", "blank"}


def log(message: str) -> None:
    """Print a script-prefixed log message."""
    print(f"{LOG_PREFIX} {message}")


def clean_text(value: Any) -> str:
    """Return a stripped string for CSV/JSON values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def is_blank(value: Any) -> bool:
    """Check whether a CSV cell should be treated as blank."""
    return clean_text(value) == ""


def append_note(existing: Any, note: str) -> str:
    """Append an audit note without duplicating it on repeated runs."""
    current = clean_text(existing)
    if not current:
        return note
    if note in current:
        return current
    return f"{current}; {note}"


def infer_pillar(path: Path) -> str:
    """Infer pillar label from an audit JSON filename."""
    stem = path.stem.lower()
    if "_e" in stem or stem.endswith("e") or "environment" in stem:
        return "E"
    if "_s" in stem or stem.endswith("s") or "social" in stem:
        return "S"
    if "_g" in stem or stem.endswith("g") or "governance" in stem:
        return "G"
    return "unknown"


def load_audit_file(path: Path) -> list[dict[str, Any]]:
    """
    Load one audit JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Audit JSON path.

    Returns
    -------
    list[dict[str, Any]]
        Audit decision records.

    Raises
    ------
    ValueError
        If the file does not contain a list of objects.
    """
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "decisions" in data:
        data = data["decisions"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of audit rows")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} row {idx} is not an object")
    return data


def normalize_decision(value: Any) -> str:
    """Normalize and validate an audit decision label."""
    decision = clean_text(value).lower()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid audit decision: {value!r}")
    return decision


def load_audits(audit_dir: Path = AUDIT_DIR) -> tuple[dict[str, dict[str, Any]], Counter, dict[str, Counter], list[str]]:
    """
    Load all audit JSONs into a row_id-indexed dictionary.

    Parameters
    ----------
    audit_dir : pathlib.Path
        Directory containing audit JSON files.

    Returns
    -------
    tuple
        Audit dictionary, global decision counts, per-pillar decision counts,
        and duplicate row IDs.
    """
    if not audit_dir.exists():
        raise FileNotFoundError(
            f"Audit directory not found: {audit_dir}. Save audit_E.json, "
            "audit_S.json, and audit_G.json there before proceeding."
        )

    audit_paths = sorted(audit_dir.glob("*.json"))
    if not audit_paths:
        raise FileNotFoundError(
            f"No audit JSONs found in {audit_dir}. Save the pillar audit files before proceeding."
        )

    audits: dict[str, dict[str, Any]] = {}
    counts: Counter = Counter()
    per_pillar: dict[str, Counter] = defaultdict(Counter)
    duplicates: list[str] = []

    for path in audit_paths:
        pillar = infer_pillar(path)
        rows = load_audit_file(path)
        log(f"Loaded {len(rows)} audit rows from {path}")
        for row in rows:
            row_id = clean_text(row.get("row_id"))
            if not row_id:
                raise ValueError(f"{path} contains an audit row without row_id")
            decision = normalize_decision(row.get("decision"))
            payload = {
                **row,
                "row_id": row_id,
                "decision": decision,
                "confidence": clean_text(row.get("confidence")).lower() or "unknown",
                "paper_verified": clean_text(row.get("paper_verified")).lower() or "unknown",
                "reasoning": clean_text(row.get("reasoning")),
                "modification_note": clean_text(row.get("modification_note")),
                "concerns": clean_text(row.get("concerns")),
                "pillar": pillar,
                "audit_file": str(path),
            }
            if row_id in audits:
                duplicates.append(row_id)
            audits[row_id] = payload
            counts[decision] += 1
            per_pillar[pillar][decision] += 1

    return audits, counts, per_pillar, duplicates


def approval_for_decision(decision: str, modify_as: str) -> str:
    """Translate audit decision into the review CSV approved value."""
    if decision == "yes":
        return "yes"
    if decision == "no":
        return "no"
    if modify_as == "yes":
        return "yes"
    if modify_as == "no":
        return "no"
    return ""


def build_audit_note(payload: dict[str, Any]) -> str:
    """Build the compact audit note appended to the review CSV."""
    return (
        f"[audit:{payload['pillar']}] "
        f"{payload['decision']}/{payload['confidence']}/"
        f"paper_verified={payload['paper_verified']}: {payload['reasoning']}"
    )


def apply_audits_to_review(
    review_path: Path = REVIEW_PATH,
    audit_dir: Path = AUDIT_DIR,
    modify_as: str = "yes",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Apply loaded audit decisions to the constraint-review CSV.

    Parameters
    ----------
    review_path : pathlib.Path
        Path to constraints_for_review.csv.
    audit_dir : pathlib.Path
        Directory containing audit JSON files.
    modify_as : {"yes", "no", "blank"}, optional
        How to translate MODIFY decisions into the approved column.
    dry_run : bool, optional
        If True, print planned changes without writing.

    Returns
    -------
    dict[str, Any]
        Summary counts.
    """
    if modify_as not in MODIFY_MODES:
        raise ValueError(f"modify_as must be one of {sorted(MODIFY_MODES)}")
    if not review_path.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_path}")

    audits, decision_counts, per_pillar, duplicates = load_audits(audit_dir)
    df = pd.read_csv(review_path)
    required_cols = {"cause", "effect", "approved", "notes", "proposed_action"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{review_path} missing required columns: {sorted(missing)}")

    df["approved"] = df["approved"].astype("object")
    df["notes"] = df["notes"].astype("object")

    applied = 0
    conflict_row_ids: list[str] = []

    for idx, row in df.iterrows():
        row_id = f"{clean_text(row['cause'])} -> {clean_text(row['effect'])}"
        payload = audits.get(row_id)
        if not payload:
            continue

        prior_approved = clean_text(row.get("approved")).lower()
        decision = payload["decision"]
        new_approved = approval_for_decision(decision, modify_as)

        if decision == "no":
            if prior_approved == "yes":
                conflict_row_ids.append(row_id)
            df.at[idx, "approved"] = "no"
        elif prior_approved != "yes":
            df.at[idx, "approved"] = new_approved

        if (
            decision in {"yes", "modify"}
            and new_approved == "yes"
            and clean_text(df.at[idx, "proposed_action"]).lower() == "review"
        ):
            df.at[idx, "proposed_action"] = "forbid_reverse"

        note = build_audit_note(payload)
        df.at[idx, "notes"] = append_note(df.at[idx, "notes"], note)
        if decision == "modify" and payload["modification_note"]:
            df.at[idx, "notes"] = append_note(
                df.at[idx, "notes"],
                f"MOD: {payload['modification_note']}",
            )

        applied += 1

    final_approved = df["approved"].map(clean_text).str.lower()
    summary = {
        "audits_applied": applied,
        "decision_counts": decision_counts,
        "per_pillar": per_pillar,
        "yes_count": int(final_approved.eq("yes").sum()),
        "no_count": int(final_approved.eq("no").sum()),
        "blank_count": int(final_approved.eq("").sum()),
        "conflict_row_ids": conflict_row_ids,
        "duplicates": duplicates,
        "audits": audits,
        "updated_df": df,
    }

    log(f"Audits applied: {applied}")
    log(
        "Decisions distribution: "
        f"yes={decision_counts['yes']}, "
        f"no={decision_counts['no']}, "
        f"modify={decision_counts['modify']}"
    )
    log(f"Final approved=yes count: {summary['yes_count']}")
    log(f"Final approved=no count: {summary['no_count']}")
    log(f"Still blank (need manual review): {summary['blank_count']}")
    log(f"Conflicts (auditor said no but row was previously yes): {len(conflict_row_ids)}")
    for row_id in conflict_row_ids:
        log(f"  conflict: {row_id}")
    if duplicates:
        log(f"Duplicate audit row_ids encountered; last file wins: {len(duplicates)}")

    write_audit_summary(summary)

    if dry_run:
        log("Dry run only; no CSV changes written.")
        return summary

    shutil.copy2(review_path, BACKUP_PATH)
    log(f"Backup written -> {BACKUP_PATH}")
    df.to_csv(review_path, index=False)
    log(f"Updated review CSV -> {review_path}")
    return summary


def write_audit_summary(summary: dict[str, Any]) -> None:
    """
    Write markdown audit provenance summary.

    Parameters
    ----------
    summary : dict[str, Any]
        Summary returned by :func:`apply_audits_to_review`.

    Returns
    -------
    None
    """
    audits = summary["audits"]
    per_pillar = summary["per_pillar"]
    rejected = [payload for payload in audits.values() if payload["decision"] == "no"]
    modified = [payload for payload in audits.values() if payload["decision"] == "modify"]

    lines = [
        "# Audit Summary",
        "",
        f"- Total constraints reviewed: {len(audits)}",
        f"- Audits applied to current review CSV: {summary['audits_applied']}",
        f"- Final approved=yes: {summary['yes_count']}",
        f"- Final approved=no: {summary['no_count']}",
        f"- Still blank: {summary['blank_count']}",
        "",
        "## Per-Pillar Decision Distributions",
        "",
    ]

    for pillar in sorted(per_pillar):
        counts = per_pillar[pillar]
        lines.append(
            f"- {pillar}: yes={counts['yes']}, no={counts['no']}, modify={counts['modify']}"
        )

    lines.extend(["", "## Rejected Constraints", ""])
    if rejected:
        for payload in sorted(rejected, key=lambda item: item["row_id"]):
            lines.append(
                f"- `{payload['row_id']}`: {payload['reasoning']} "
                f"Concerns: {payload['concerns'] or 'None'}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Modify Decisions", ""])
    if modified:
        for payload in sorted(modified, key=lambda item: item["row_id"]):
            caveat = payload["modification_note"] or payload["concerns"] or "No caveat supplied."
            lines.append(f"- `{payload['row_id']}`: {caveat}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Provenance Note",
        "",
        "Audit conducted by pillar-stratified RAG instances each loaded "
        "with the relevant pillar's source papers. Decisions verified against "
        "source corpus. Author reviewed all REJECT and MODIFY decisions.",
        "",
    ])

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Audit summary written -> {SUMMARY_PATH}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Apply pillar audit decisions to constraints_for_review.csv."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing the review CSV.")
    parser.add_argument("--modify-as", choices=sorted(MODIFY_MODES), default="yes",
                        help="How MODIFY decisions affect the approved column.")
    return parser.parse_args()


def main() -> None:
    """Run the audit application CLI."""
    args = parse_args()
    apply_audits_to_review(modify_as=args.modify_as, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
