"""Organize paper PDFs into classification folders.

This script reads ``paper_inverntory.md`` and copies or moves source PDFs into
classification-specific subfolders. It is intentionally conservative: default
mode copies files, dry-run mode writes nothing, and duplicate download copies
such as ``paper (1).pdf`` are ignored in favor of the base filename.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


LOG_PREFIX = "[organize]"

SOURCE = Path(r"C:\Users\User\Desktop\Outline and Materials needed")
PROJECT_ROOT = Path(__file__).resolve().parent

INVENTORY_CANDIDATES = [
    Path(r"C:\Users\User\Desktop\go\paper_inverntory.md"),
    SOURCE / "paper_inverntory.md",
    PROJECT_ROOT / "paper_inverntory.md",
]

GROUP_DIRS = {
    "E": "group_E_environmental",
    "S": "group_S_social",
    "G": "group_G_governance",
    "X": "group_X_cross_pillar",
    "SKIP": "group_SKIP_unused",
    "": "group_UNCLASSIFIED",
}
VALID_CLASSIFICATIONS = {"E", "S", "G", "X", "SKIP"}

# Classifications supplied by the user for newly added papers. The inventory is
# not edited; these are used only when the classification cell is blank.
SUPPLEMENTAL_CLASSIFICATIONS = {
    # Environmental
    "1-s2.0-S014098832400032X-main.pdf": "E",
    "1-s2.0-S0144818818302291-main.pdf": "E",
    "16+Aigienohuwa,+O.O.+&+Aigienohuwa,+U.S..pdf": "E",
    "500652 PDF.pdf": "E",
    "63cb168482b2d840d314f38a852a6117a840.pdf": "E",
    "679-Article Text-4016-1-10-20231104.pdf": "E",
    "6a4a8c9e802a9f5c602a300906b3b05c3e82.pdf": "E",
    "admbar,+199_Articles_S1807-76922012000500006_scielo.pdf": "E",
    "BA_Ferner.pdf": "E",
    "energies-14-06029-v2.pdf": "E",
    "Environmental Violations Legal Penalties and Reputation Costs.pdf": "E",
    "file151689.pdf": "E",
    "ijerph-19-11272-v2.pdf": "E",
    "JCP ISO14001.pdf": "E",
    "Price_Corporate_Carbon_Footprints_Nov_2024.pdf": "E",
    "risks-12-00197-v2.pdf": "E",
    "s10551-021-04881-6.pdf": "E",
    "s41598-025-16455-x.pdf": "E",
    "s43546-025-00944-2.pdf": "E",
    "ssrn-3915486.pdf": "E",
    "ssrn-4056529.pdf": "E",
    "ssrn-5021896.pdf": "E",
    "ssrn-5030796.pdf": "E",
    "State_of_transition_in_the_banking_sector_report_2024_December.pdf": "E",
    "sustainability-14-00989.pdf": "E",
    "water-16-02560.pdf": "E",
    "water-17-01881.pdf": "E",
    # Social
    "3429-10990-1-SM.pdf": "S",
    "40001_2025_Article_2698.pdf": "S",
    "An Institutional Approach to Gender Diversity and Firm Performance_4c0479f3-9d13-4af8-82da-7f1713af940d.pdf": "S",
    "Bryson_ueaa048.pdf": "S",
    "cesifo1_wp10873.pdf": "S",
    "dbbafcc4-en.pdf": "S",
    "Dodini_JMP_2023_2024.pdf": "S",
    "dp11111.pdf": "S",
    "From debt breaches to employee safety_ The hidden power of banking interventions - ScienceDirect.pdf": "S",
    "Garcia-ManglanoPerez-alfonso - Womens Representation and Financial Performance A Qualitative Sect....pdf": "S",
    "journal.pone.0292889.pdf": "S",
    "Park and Shaw Turnover rates and organizational performance_ A meta-analysis 2013.pdf": "S",
    "rest_a_00460-esupp.pdf": "S",
    "s11002-023-09671-w.pdf": "S",
    "s12651-025-00391-4.pdf": "S",
    "ssrn-1535969.pdf": "S",
    "ssrn-3505626.pdf": "S",
    "ssrn-4565288.pdf": "S",
    "ssrn-4788788.pdf": "S",
    "ssrn-4841977.pdf": "S",
    "Stock-Returns-on-Customer-Satisfaction-Do-Beat-the-Market.pdf": "S",
    "SvarstadC32C+collective+agreements+and+productivity.pdf": "S",
    "training_draft11ReStatv_final.pdf": "S",
    # Governance
    "1-s2.0-S105752192300279X-main.pdf": "G",
    "AFC_2025_1_Faiteh.pdf": "G",
    "Bank_credit_loss.pdf": "G",
    "BBS_2018_02_Braendle.pdf": "G",
    "ecb.wp3115~7444235074.en.pdf": "G",
    "ijefr7(1)14-20.pdf": "G",
    "pone.0276637.pdf": "G",
    "ssrn-246674.pdf": "G",
    "ssrn-4081494.pdf": "G",
    "sustainability-12-08386-v2.pdf": "G",
    "The Rights and Wrongs of Shareholder Rights.pdf": "G",
    # Cross-pillar / banking ESG
    "1-s2.0-S0275531925004702-main.pdf": "X",
    "10.3934_GF.2025011.pdf": "X",
    "830_Korzeb_et_al.pdf": "X",
    "9ea0a12f-8d10-4353-842a-ed21f547d194.pdf": "X",
    "Banking_tool_-_2025_consultation_report.pdf": "X",
    "banks-sector-guidance-apr-2024.pdf": "X",
    "ecb.wp2550~24c25d5791.en.pdf": "X",
    "ECTI_IDA(2025)773711_EN.pdf": "X",
    "energies-15-01292-v2.pdf": "X",
    "ijfs-13-00234.pdf": "X",
    "ijfs-14-00087.pdf": "X",
    "s41599-024-03876-8.pdf": "X",
    "s43621-025-01279-6.pdf": "X",
    "Sustainability and Stability_ ESG_Credit Risk Dynamics in EU Bank.pdf": "X",
    "Uwasa_2024_Porta_Michela.pdf": "X",
    "View of ESG performance and bank financial stability_ Global evidence _ Oeconomia Copernicana.pdf": "X",
    # Skip
    "1-s2.0-S0304405X16301969-main.pdf": "SKIP",
    "Estimating-Marginal-Q_v14.pdf": "SKIP",
    "FinacialSystemBenchmark_Methodology2026-V2025_1_1.pdf": "SKIP",
    "Final Guidelines on the management of ESG risks.pdf": "SKIP",
    "GABV-2021-Real-Economy-Real-Returns.pdf": "SKIP",
    "Morgan_Stanley_2025_Sustainable_Issuance_Report.pdf": "SKIP",
    "ssm.ECB_Report_on_climate_and_environmental_disclosures_202203~4ae33f2a70.en_.pdf": "SKIP",
    "ssrn-2405231.pdf": "SKIP",
    "ssrn-4902308.pdf": "SKIP",
    "The Misuse of Tobin_s q.pdf": "SKIP",
    "w14845.pdf": "SKIP",
    "Wharton-Peters_taylor_Intangible capital & Q.pdf": "SKIP",
}


@dataclass
class InventoryRow:
    """One parsed row from the inventory table."""

    paper_id: int
    paper_name: str
    classification_raw: str
    classification: str


@dataclass
class ManifestRow:
    """One output row for the organization manifest."""

    paper_id: int
    original_filename: str
    classification: str
    destination_subfolder: str
    action: str
    timestamp: str


def log(message: str) -> None:
    """Print an organizer-prefixed log message."""

    print(f"{LOG_PREFIX} {message}")


def canonical_name(name: str) -> str:
    """Normalize a filename for duplicate-copy comparison.

    Parameters
    ----------
    name:
        Filename to normalize.

    Returns
    -------
    str
        Case-folded filename with Windows duplicate suffixes such as
        ``" (1)"`` removed before ``.pdf``.
    """

    lowered = name.casefold().strip()
    for suffix in (" (1)", " (2)", " (3)", " (4)", " (5)"):
        lowered = lowered.replace(f"{suffix}.pdf", ".pdf")
    return lowered


def locate_inventory(override: str | None = None) -> Path:
    """Find the inventory file.

    Parameters
    ----------
    override:
        Optional explicit inventory path from the CLI.

    Returns
    -------
    pathlib.Path
        Existing inventory path.

    Raises
    ------
    FileNotFoundError
        If no candidate exists and no valid path is entered interactively.
    """

    if override:
        path = Path(override).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Inventory override not found: {path}")

    for path in INVENTORY_CANDIDATES:
        if path.exists():
            return path

    entered = input(
        f"{LOG_PREFIX} Inventory file not found. Enter full path to paper_inverntory.md: "
    ).strip()
    if entered:
        path = Path(entered).expanduser()
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in INVENTORY_CANDIDATES)
    raise FileNotFoundError(f"Could not locate paper_inverntory.md. Searched: {searched}")


def normalize_classification(raw_value: str) -> str:
    """Normalize a raw classification cell.

    Parameters
    ----------
    raw_value:
        Raw classification value from the inventory.

    Returns
    -------
    str
        One of ``E``, ``S``, ``G``, ``X``, ``SKIP``, or ``""``.
    """

    value = raw_value.strip().upper()
    if value in VALID_CLASSIFICATIONS:
        return value
    return ""


def load_classification_overrides(path: Path | None) -> dict[int, str]:
    """Load explicit classification overrides.

    Parameters
    ----------
    path:
        Optional CSV path with ``paper_id`` and ``classification`` columns.

    Returns
    -------
    dict[int, str]
        Mapping from inventory paper ID to normalized classification.
    """

    if path is None:
        return {}

    overrides: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"paper_id", "classification"}
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Override CSV must contain columns {sorted(expected)}; got {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            raw_id = (row.get("paper_id") or "").strip()
            raw_classification = (row.get("classification") or "").strip()
            if not raw_id:
                continue
            try:
                paper_id = int(raw_id)
            except ValueError:
                log(f"Skipping override line {line_number}: invalid paper_id={raw_id!r}")
                continue

            classification = normalize_classification(raw_classification)
            if not classification:
                log(
                    "Skipping override line "
                    f"{line_number}: invalid classification={raw_classification!r}"
                )
                continue
            overrides[paper_id] = classification

    log(f"Loaded {len(overrides)} classification overrides from {path}")
    return overrides


def supplemental_classification(paper_name: str) -> str:
    """Look up a user-supplied classification for blank inventory cells.

    Parameters
    ----------
    paper_name:
        Inventory filename.

    Returns
    -------
    str
        Supplemental classification, or ``""`` if none is known.
    """

    by_key = {canonical_name(name): value for name, value in SUPPLEMENTAL_CLASSIFICATIONS.items()}
    key = canonical_name(paper_name)
    if key in by_key:
        return by_key[key]

    prefix = key[:30]
    matches = [value for name, value in by_key.items() if name.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return ""


def split_inventory_records(text: str) -> list[str]:
    """Split a markdown table into row records.

    This parser tolerates long abstracts that spill onto physical continuation
    lines. A new record begins only at lines matching ``| <integer> |``.

    Parameters
    ----------
    text:
        Inventory file content.

    Returns
    -------
    list[str]
        Raw table row records.
    """

    records: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        is_row_start = stripped.startswith("|") and len(stripped.split("|")) > 2
        if is_row_start:
            first_cell = stripped.split("|")[1].strip()
            if first_cell.isdigit():
                if current:
                    records.append("\n".join(current))
                current = [line]
                continue
        if current:
            current.append(line)

    if current:
        records.append("\n".join(current))
    return records


def parse_inventory(
    path: Path,
    classification_overrides: dict[int, str] | None = None,
) -> list[InventoryRow]:
    """Parse inventory rows.

    Parameters
    ----------
    path:
        Path to ``paper_inverntory.md``.
    classification_overrides:
        Optional mapping from paper ID to explicit classification. Overrides
        take precedence over the inventory column.

    Returns
    -------
    list[InventoryRow]
        Parsed inventory rows with normalized classifications.
    """

    rows: list[InventoryRow] = []
    classification_overrides = classification_overrides or {}
    for record in split_inventory_records(path.read_text(encoding="utf-8")):
        first_line = record.splitlines()[0]
        parts = first_line.split("|")
        if len(parts) < 4:
            log(f"Skipping malformed inventory row: {first_line[:120]}")
            continue

        try:
            paper_id = int(parts[1].strip())
        except ValueError:
            log(f"Skipping row with non-integer id: {first_line[:120]}")
            continue

        paper_name = parts[2].strip()
        raw_classification = ""
        tail_parts = record.rstrip().rsplit("|", 2)
        if len(tail_parts) >= 2:
            raw_classification = tail_parts[-2].strip()

        if paper_id in classification_overrides:
            normalized = classification_overrides[paper_id]
        else:
            normalized = normalize_classification(raw_classification)
        if not normalized:
            normalized = supplemental_classification(paper_name)

        rows.append(
            InventoryRow(
                paper_id=paper_id,
                paper_name=paper_name,
                classification_raw=raw_classification.strip(),
                classification=normalized,
            )
        )

    return rows


def choose_preferred_file(paths: list[Path]) -> Path:
    """Choose the best file among fuzzy or duplicate-copy matches.

    Parameters
    ----------
    paths:
        Candidate file paths.

    Returns
    -------
    pathlib.Path
        Preferred candidate.
    """

    no_copy_suffix = [
        path
        for path in paths
        if " (1).pdf" not in path.name.casefold()
        and " (2).pdf" not in path.name.casefold()
        and " (3).pdf" not in path.name.casefold()
    ]
    return sorted(no_copy_suffix or paths, key=lambda path: (len(path.name), path.name.casefold()))[0]


def build_source_index(source: Path) -> tuple[dict[str, Path], dict[str, list[Path]], list[list[Path]]]:
    """Index top-level files in the source folder.

    Parameters
    ----------
    source:
        Folder containing the loose PDFs.

    Returns
    -------
    tuple
        Exact case-fold index, canonical duplicate index, and duplicate groups.
    """

    files = [path for path in source.iterdir() if path.is_file()]
    exact_index = {path.name.casefold(): path for path in files}

    canonical_index: dict[str, list[Path]] = {}
    for path in files:
        canonical_index.setdefault(canonical_name(path.name), []).append(path)

    duplicate_groups = [paths for paths in canonical_index.values() if len(paths) > 1]
    return exact_index, canonical_index, duplicate_groups


def find_source_file(
    paper_name: str,
    source: Path,
    exact_index: dict[str, Path],
    canonical_index: dict[str, list[Path]],
) -> Path | None:
    """Resolve an inventory filename to a source file.

    Matching order is exact, case-insensitive, duplicate-copy canonical match,
    then first-30-character fuzzy match.

    Parameters
    ----------
    paper_name:
        Filename from the inventory.
    source:
        Source folder.
    exact_index:
        Case-folded exact filename index.
    canonical_index:
        Duplicate-copy canonical filename index.

    Returns
    -------
    pathlib.Path or None
        Matched source file, or ``None`` if no match is found.
    """

    exact_path = source / paper_name
    if exact_path.exists() and exact_path.is_file():
        return exact_path

    case_match = exact_index.get(paper_name.casefold())
    if case_match:
        return case_match

    canonical_matches = canonical_index.get(canonical_name(paper_name), [])
    if canonical_matches:
        return choose_preferred_file(canonical_matches)

    prefix = paper_name.casefold()[:30]
    if prefix:
        fuzzy_matches = [
            path
            for paths in canonical_index.values()
            for path in paths
            if path.name.casefold().startswith(prefix)
        ]
        if fuzzy_matches:
            return choose_preferred_file(fuzzy_matches)

    return None


def destination_for(classification: str, source: Path) -> Path:
    """Resolve destination folder for a classification.

    Parameters
    ----------
    classification:
        Normalized classification.
    source:
        Source root folder.

    Returns
    -------
    pathlib.Path
        Destination subfolder.
    """

    folder_name = GROUP_DIRS.get(classification, GROUP_DIRS[""])
    return source / folder_name


def write_manifest(source: Path, rows: list[ManifestRow]) -> Path:
    """Write the organization manifest.

    Parameters
    ----------
    source:
        Source root folder.
    rows:
        Manifest rows to write.

    Returns
    -------
    pathlib.Path
        Written manifest path.
    """

    manifest_path = source / "organization_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "original_filename",
                "classification",
                "destination_subfolder",
                "action",
                "timestamp",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.paper_id,
                    "original_filename": row.original_filename,
                    "classification": row.classification,
                    "destination_subfolder": row.destination_subfolder,
                    "action": row.action,
                    "timestamp": row.timestamp,
                }
            )
    return manifest_path


def organize_papers(
    inventory_path: Path,
    source: Path,
    dry_run: bool = False,
    move_instead: bool = False,
    classification_overrides: dict[int, str] | None = None,
) -> None:
    """Copy or move papers into classification folders.

    Parameters
    ----------
    inventory_path:
        Path to the paper inventory.
    source:
        Folder containing the source PDFs.
    dry_run:
        If ``True``, print planned actions without writing.
    move_instead:
        If ``True``, move files instead of copying them.
    classification_overrides:
        Optional paper-ID classification overrides.
    """

    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    rows = parse_inventory(inventory_path, classification_overrides)
    exact_index, canonical_index, duplicate_groups = build_source_index(source)
    timestamp = datetime.now().isoformat(timespec="seconds")

    if duplicate_groups:
        log(f"Detected {len(duplicate_groups)} duplicate-copy groups in source; preferred base files will be used.")
        for group in duplicate_groups:
            names = ", ".join(path.name for path in sorted(group, key=lambda path: path.name.casefold()))
            log(f"Duplicate-copy group ignored as one paper: {names}")

    if not dry_run:
        for folder_name in GROUP_DIRS.values():
            (source / folder_name).mkdir(parents=True, exist_ok=True)

    routed_per_group = {folder_name: 0 for folder_name in GROUP_DIRS.values()}
    copied_per_group = {folder_name: 0 for folder_name in GROUP_DIRS.values()}
    not_found: list[InventoryRow] = []
    raw_empty_classification = 0
    final_unclassified = 0
    manifest_rows: list[ManifestRow] = []

    action_word = "moved" if move_instead else "copied"
    for row in rows:
        if not row.classification_raw:
            raw_empty_classification += 1
        if not row.classification:
            final_unclassified += 1

        source_file = find_source_file(row.paper_name, source, exact_index, canonical_index)
        final_classification = row.classification if row.classification in VALID_CLASSIFICATIONS else ""
        destination_folder = destination_for(final_classification, source)
        destination_file = destination_folder / (source_file.name if source_file else row.paper_name)

        if source_file is None:
            not_found.append(row)
            manifest_rows.append(
                ManifestRow(
                    paper_id=row.paper_id,
                    original_filename=row.paper_name,
                    classification=final_classification or "UNCLASSIFIED",
                    destination_subfolder=destination_folder.name,
                    action="not_found",
                    timestamp=timestamp,
                )
            )
            log(f"NOT FOUND id={row.paper_id}: {row.paper_name}")
            continue

        routed_per_group[destination_folder.name] += 1
        if dry_run:
            log(f"DRY RUN id={row.paper_id}: {source_file.name} -> {destination_folder.name}")
            action = "skipped"
        elif destination_file.exists():
            log(f"SKIP exists id={row.paper_id}: {destination_file}")
            action = "skipped"
        elif move_instead:
            shutil.move(str(source_file), str(destination_file))
            log(f"Moved id={row.paper_id}: {source_file.name} -> {destination_folder.name}")
            copied_per_group[destination_folder.name] += 1
            action = action_word
        else:
            shutil.copy2(source_file, destination_file)
            log(f"Copied id={row.paper_id}: {source_file.name} -> {destination_folder.name}")
            copied_per_group[destination_folder.name] += 1
            action = action_word

        manifest_rows.append(
            ManifestRow(
                paper_id=row.paper_id,
                original_filename=row.paper_name,
                classification=final_classification or "UNCLASSIFIED",
                destination_subfolder=destination_folder.name,
                action=action,
                timestamp=timestamp,
            )
        )

    if dry_run:
        log("Dry run complete; no files or manifest were written.")
    else:
        manifest_path = write_manifest(source, manifest_rows)
        log(f"Manifest written: {manifest_path}")

    log("Summary")
    log("Routing counts by group:")
    for folder_name, count in routed_per_group.items():
        log(f"{folder_name}: {count} files routed")
    log("Copy/move counts by group:")
    for folder_name, count in copied_per_group.items():
        log(f"{folder_name}: {count} files {action_word}")
    log(f"Inventory rows processed: {len(rows)}")
    log(f"Inventory rows skipped because file was not found: {len(not_found)}")
    log(f"Inventory rows with empty classification cell: {raw_empty_classification}")
    log(f"Rows still routed to UNCLASSIFIED after supplemental rules: {final_unclassified}")

    if not_found:
        log("File-not-found cases:")
        for row in not_found:
            log(f"  id={row.paper_id}; file={row.paper_name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Copy or move papers into classification subfolders."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without copying or writing a manifest.",
    )
    parser.add_argument(
        "--move-instead",
        action="store_true",
        help="Move files instead of copying them. Use only after backing up the source folder.",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default=None,
        help="Override inventory file path.",
    )
    parser.add_argument(
        "--override-classifications",
        type=str,
        default=None,
        help="CSV with paper_id,classification columns; overrides inventory classifications.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the organizer CLI."""

    args = parse_args()
    inventory_path = locate_inventory(args.inventory)
    override_path = Path(args.override_classifications) if args.override_classifications else None
    classification_overrides = load_classification_overrides(override_path)
    log(f"Inventory: {inventory_path}")
    log(f"Source: {SOURCE}")
    if args.move_instead:
        log("WARNING: --move-instead selected. Files will be moved, not copied.")
    organize_papers(
        inventory_path=inventory_path,
        source=SOURCE,
        dry_run=args.dry_run,
        move_instead=args.move_instead,
        classification_overrides=classification_overrides,
    )


if __name__ == "__main__":
    main()
