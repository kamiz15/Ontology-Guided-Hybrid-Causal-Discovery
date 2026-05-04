# 02a_parse_real_dataset.py
# ============================================================
# Step 02a - Parse the real ECB ESG workbook into a numeric
# analysis table with thesis-auditable parsing diagnostics.
#
# Usage:
#   python 02a_parse_real_dataset.py
#   python 02a_parse_real_dataset.py --input data/raw/df_asst_bnk_ecb.xlsx
#
# Output:
#   data/processed/data_real_parsed.csv
#   reports/parsing_decisions.md
#   reports/ordinal_parsing_unmapped.csv
#   reports/currency_parsing_multivalue.csv
#   reports/parsing_summary.txt
# ============================================================

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CURRENCY_PARSING_MULTIVALUE_PATH,
    CURRENCY_PARSING_FAILURES_PATH,
    CURRENCY_PARSING_WARNINGS_PATH,
    FX_CONVERSIONS_LOG_PATH,
    FX_RATES_CSV_PATH,
    FX_UNSUPPORTED_LOG_PATH,
    ORDINAL_PARSING_UNMAPPED_PATH,
    PARSING_DECISIONS_PATH,
    PARSING_SUMMARY_PATH,
    RAW_DATA_PATH,
    REAL_PARSED_DATA_PATH,
)


RAW_TO_TARGET = {
    "No": "no",
    "LEI, MFI code for branches": "lei_mfi_code_for_branches",
    "Type": "type",
    "Banks": "banks",
    "Ground for significance": "ground_for_significance",
    "Scope 1 GHG emissions": "scope_1_emissions_tco2e",
    "Scope 2 GHG emissions": "scope_2_emissions_tco2e",
    "Scope 3 GHG emissions": "scope_3_emissions_tco2e",
    "Emission reduction policy": "emission_reduction_policy_score",
    "Renewable energy share": "renewable_energy_share",
    "Community investment": "community_investment_eur",
    "Diversity / Women representation": "diversity_representation",
    "Health & Safety": "health_safety_score",
    "Board strategy / ESG oversight": "board_strategy_esg_oversight_score",
    "Sustainable finance / Green financing": "green_financing_eur",
    "Total revenue": "total_revenue_eur",
    "Reporting quality": "reporting_quality_score",
}

METADATA_COLUMNS = {
    "no",
    "lei_mfi_code_for_branches",
    "type",
    "banks",
    "ground_for_significance",
}

GHG_COLUMNS = {
    "scope_1_emissions_tco2e",
    "scope_2_emissions_tco2e",
    "scope_3_emissions_tco2e",
}

ORDINAL_COLUMNS = {
    "emission_reduction_policy_score",
    "health_safety_score",
    "board_strategy_esg_oversight_score",
    "reporting_quality_score",
}

CURRENCY_COLUMNS = {
    "community_investment_eur",
    "green_financing_eur",
    "total_revenue_eur",
}

MANUAL_CURRENCY_OVERRIDES = {
    (63, "green_financing_eur"): 2_944_000_000.0,
    (82, "green_financing_eur"): 698_000_000_000.0,
    (102, "green_financing_eur"): 650_000_000_000.0,
}

PERCENT_COLUMNS = {
    "renewable_energy_share",
    "diversity_representation",
}

ORDINAL_RULES = [
    ("excellent", 5),
    ("very good", 4),
    ("good", 3),
    ("moderate", 2),
    ("adequate", 2),
    ("limited", 1),
    ("weak", 1),
]


@dataclass
class ParseContext:
    """Accumulates diagnostics while parsing the raw workbook."""

    ordinal_unmapped: list[dict[str, Any]] = field(default_factory=list)
    currency_multivalue: list[dict[str, Any]] = field(default_factory=list)
    currency_warnings: list[dict[str, Any]] = field(default_factory=list)
    currency_failures: list[dict[str, Any]] = field(default_factory=list)
    fx_conversions: list[dict[str, Any]] = field(default_factory=list)
    fx_unsupported: list[dict[str, Any]] = field(default_factory=list)
    currency_default_eur: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    target_only_renewable: list[dict[str, Any]] = field(default_factory=list)


def _ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _normalise_for_matching(value: Any) -> str:
    text = _clean_text(value).lower()
    return (
        text.replace("co2", "co2")
        .replace("co₂", "co2")
        .replace("co\u2082", "co2")
        .replace("\u00a0", " ")
    )


def _normalise_number_token(token: str) -> float:
    token = token.strip().replace(" ", "").replace("+", "")
    if not token:
        raise ValueError("empty numeric token")

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        parts = token.split(",")
        if len(parts[-1]) <= 2 and len(parts) == 2:
            token = token.replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "." in token and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        token = token.replace(".", "")

    return float(token)


def _record_warning(
    context: ParseContext,
    row_index: int,
    column: str,
    raw_value: Any,
    warning: str,
) -> None:
    context.warnings.append({
        "row_index": row_index,
        "column": column,
        "raw_value": _clean_text(raw_value),
        "warning": warning,
    })


def load_raw_dataset(input_path: str) -> pd.DataFrame:
    """
    Load the raw workbook or delimited text file.

    Parameters
    ----------
    input_path : str
        Path to the raw `.xlsx`, `.xls`, `.csv`, `.txt`, or `.tsv` file.

    Returns
    -------
    pd.DataFrame
        Raw table with original headers preserved.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".txt", ".tsv"}:
        return pd.read_csv(path, sep=None, engine="python")

    raise ValueError(f"Unsupported input type: {path}")


def apply_required_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename required raw headers to the parsed snake_case schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw workbook table with original headers.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the required mapped columns.

    Raises
    ------
    ValueError
        If any required raw header is missing.
    """
    missing = [raw for raw in RAW_TO_TARGET if raw not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    ordered_raw = list(RAW_TO_TARGET.keys())
    out = df[ordered_raw].rename(columns=RAW_TO_TARGET).copy()
    print(f"[step_02a] Applied required column mapping: {len(ordered_raw)} columns.")
    return out


def parse_ghg_emissions(
    value: Any,
    row_index: int,
    column: str,
    context: ParseContext,
) -> float:
    """
    Parse GHG emissions and convert all values to tonnes CO2e.

    Parameters
    ----------
    value : Any
        Raw cell value.
    row_index : int
        Source row index for diagnostics.
    column : str
        Parsed target column name.
    context : ParseContext
        Warning accumulator.

    Returns
    -------
    float
        Tonnes CO2e, or NaN when no numeric value is available.
    """
    if _is_missing(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    raw = _clean_text(value)
    if not raw:
        return np.nan

    segments = [seg.strip() for seg in re.split(r"\s*/\s*", raw) if seg.strip()]
    chosen_segment = segments[0] if segments else raw
    for segment in segments:
        if "location-based" in _normalise_for_matching(segment):
            chosen_segment = segment
            break

    number_pattern = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,\.]*(?:\.\d+)?")
    matches = list(number_pattern.finditer(chosen_segment))
    if not matches:
        return np.nan

    if len(matches) > 1 and not re.search(
        r"(?i)\b(?:t|ton|tonne|kton|kt|mton|mio|million)\b|tco2", chosen_segment
    ):
        _record_warning(
            context,
            row_index,
            column,
            value,
            "multiple numeric values with no explicit unit; first value used",
        )

    match = matches[0]
    amount = _normalise_number_token(match.group())
    text = _normalise_for_matching(chosen_segment)
    local_text = _normalise_for_matching(chosen_segment[match.start(): match.end() + 40])

    multiplier = 1.0
    if re.search(r"\b(kton|kt|kiloton|thousand\s+t)", local_text):
        multiplier = 1_000.0
    elif re.search(r"\b(mton|mio|million\s+t|million\s+ton)", local_text):
        multiplier = 1_000_000.0
    elif re.search(r"\bkg\b", local_text):
        multiplier = 0.001
    elif len(segments) > 1 and chosen_segment == segments[0]:
        _record_warning(
            context,
            row_index,
            column,
            value,
            "multiple GHG values found; first value used",
        )

    if "location-based" in text:
        return amount * multiplier

    return amount * multiplier


def parse_ordinal_score(
    value: Any,
    row_index: int,
    column: str,
    context: ParseContext,
) -> float:
    """
    Parse a qualitative ESG rating to an ordinal 1-5 score.

    Parameters
    ----------
    value : Any
        Raw qualitative cell.
    row_index : int
        Source row index for diagnostics.
    column : str
        Parsed target column name.
    context : ParseContext
        Unmapped-value accumulator.

    Returns
    -------
    float
        Ordinal score from 1 to 5, or NaN for unmapped values.
    """
    if _is_missing(value):
        return np.nan

    text = _normalise_for_matching(value)
    if not text:
        return np.nan

    for token, score in ORDINAL_RULES:
        if token in text:
            return float(score)

    context.ordinal_unmapped.append({
        "row_index": row_index,
        "column": column,
        "raw_value": _clean_text(value),
        "reason": "no rating keyword matched",
    })
    _record_warning(context, row_index, column, value, "ordinal value unmapped")
    return np.nan


def _strip_parenthetical_qualifiers(value: str) -> str:
    return re.sub(r"\([^)]*\)", "", value)


@dataclass
class CurrencyParseResult:
    """Container for a parsed currency cell and its diagnostics."""

    value: float | None
    amounts: list[float] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    currency_code: str = "EUR"
    source_amount: float | None = None
    fx_rate_used: float = 1.0
    fx_rate_date: str = ""
    defaulted_to_eur: bool = False
    unsupported_currency: str | None = None


_FX_RATE_CACHE: dict[str, dict[str, float]] = {}
_FX_RATE_DATE_CACHE: dict[str, str] = {}

CURRENCY_MARKERS = [
    ("EUR", ["€", "â‚¬", "Ã¢â€šÂ¬", "EUR", "EURO", "Euro"]),
    ("USD", ["US$", "USD", "$"]),
    ("GBP", ["£", "Â£", "GBP"]),
    ("CHF", ["CHF"]),
    ("SEK", ["SEK"]),
    ("NOK", ["NOK"]),
    ("DKK", ["DKK"]),
    ("CZK", ["CZK", "Kč", "Kc"]),
    ("PLN", ["PLN", "zł", "zl"]),
    ("HUF", ["HUF", "Ft"]),
    ("RON", ["RON", "lei"]),
    ("BGN", ["BGN", "лв"]),
    ("HRK", ["HRK", "kn"]),
    ("COP", ["COP"]),
    ("BRL", ["BRL", "R$"]),
    ("MXN", ["MXN", "Mex$", "Ps."]),
    ("ARS", ["ARS", "AR$"]),
    ("ZAR", ["ZAR", "R "]),
    ("TRY", ["TRY", "₺"]),
    ("JPY", ["JPY", "¥"]),
    ("CNY", ["CNY", "RMB", "元"]),
]


def load_fx_rates(path: str = FX_RATES_CSV_PATH) -> dict[str, float]:
    """
    Load ECB FX rates and convert them to foreign-to-EUR multipliers.

    The project CSV is assumed to have columns `date`, `currency`, and
    `rate`, where `rate` follows the ECB convention: 1 EUR equals `rate`
    units of the foreign currency. This function returns the inverse form:
    1 unit of foreign currency equals X EUR. If several dates exist, the most
    recent date is used for all conversions.

    Parameters
    ----------
    path : str, optional
        Path to the extracted ECB rates CSV.

    Returns
    -------
    dict[str, float]
        Mapping from ISO currency code to EUR multiplier.

    Raises
    ------
    SystemExit
        If the FX file is missing or malformed.
    """
    if path in _FX_RATE_CACHE:
        return _FX_RATE_CACHE[path]

    if not os.path.exists(path):
        print(f"[step_02a] ERROR: FX rates file missing: {path}")
        print("[step_02a] Expected extracted rates in exchange_rates/.")
        raise SystemExit(1)

    rates = pd.read_csv(path)
    required = {"date", "currency", "rate"}
    missing = required.difference(rates.columns)
    if missing:
        print(f"[step_02a] ERROR: FX rates file missing columns: {sorted(missing)}")
        raise SystemExit(1)

    rates["date"] = pd.to_datetime(rates["date"], errors="coerce")
    rates["currency"] = rates["currency"].astype(str).str.upper().str.strip()
    rates["rate"] = pd.to_numeric(rates["rate"], errors="coerce")
    rates = rates.dropna(subset=["date", "currency", "rate"])
    if rates.empty:
        print(f"[step_02a] ERROR: No usable FX rows found in {path}")
        raise SystemExit(1)

    latest_date = rates["date"].max()
    latest = rates[rates["date"] == latest_date].copy()
    fx = {
        row["currency"]: 1.0 / float(row["rate"])
        for _, row in latest.iterrows()
        if float(row["rate"]) != 0.0
    }
    fx["EUR"] = 1.0

    date_text = latest_date.date().isoformat()
    _FX_RATE_CACHE[path] = fx
    _FX_RATE_DATE_CACHE[path] = date_text
    print(f"[step_02a] FX rates loaded from {path}; using date {date_text}")
    return fx


def get_fx_rate_date(path: str = FX_RATES_CSV_PATH) -> str:
    """
    Return the cached FX-rate date, loading rates if necessary.

    Parameters
    ----------
    path : str, optional
        Path to the extracted ECB rates CSV.

    Returns
    -------
    str
        ISO date used for FX conversion.
    """
    if path not in _FX_RATE_DATE_CACHE:
        load_fx_rates(path)
    return _FX_RATE_DATE_CACHE[path]


def detect_currency_code(raw_value: str) -> tuple[str, bool]:
    """
    Detect the currency code in a raw currency cell.

    Parameters
    ----------
    raw_value : str
        Raw currency cell text.

    Returns
    -------
    tuple[str, bool]
        Detected currency code and whether the parser defaulted to EUR because
        no supported currency token was found.
    """
    matches: list[tuple[int, int, str]] = []
    for currency, markers in CURRENCY_MARKERS:
        for marker in markers:
            pattern = _currency_marker_pattern(marker)
            match = re.search(pattern, raw_value, flags=re.IGNORECASE)
            if match:
                matches.append((match.start(), -(match.end() - match.start()), currency))

    if matches:
        matches.sort()
        return matches[0][2], False

    unknown_code = re.search(r"^\s*([A-Z]{3})\b", raw_value.strip())
    if unknown_code:
        return unknown_code.group(1).upper(), False

    return "EUR", True


def _currency_marker_pattern(marker: str) -> str:
    if marker == "R ":
        return r"(?<![A-Za-z])R\s+(?=\d)"
    escaped = re.escape(marker)
    if re.search(r"[A-Za-z]", marker):
        return rf"(?<![A-Za-z]){escaped}(?![A-Za-z])"
    return escaped


def _currency_multiplier(suffix: str) -> float:
    """
    Return the numeric multiplier implied by a currency suffix.

    Parameters
    ----------
    suffix : str
        Unit suffix such as million, bn, mio, or mrd.

    Returns
    -------
    float
        Multiplier to convert the numeric value to EUR.
    """
    unit = suffix.lower().strip().strip(".")
    if unit in {"thousand", "k", "tsd"}:
        return 1_000.0
    if unit in {"million", "m", "m$", "mio"}:
        return 1_000_000.0
    if unit in {"billion", "bn", "mrd", "mld"}:
        return 1_000_000_000.0
    if unit in {"trillion", "tn"}:
        return 1_000_000_000_000.0
    return 1.0


def _has_large_suffix(raw_value: str) -> bool:
    text = raw_value.lower()
    return bool(re.search(r"\b(billion|bn|mrd\.?|mld\.?)\b", text))


def _has_trillion_suffix(raw_value: str) -> bool:
    text = raw_value.lower()
    return bool(re.search(r"\b(trillion|tn)\b", text))


def _strip_currency_tokens(value: str) -> str:
    cleaned = _strip_parenthetical_qualifiers(value)
    cleaned = cleaned.replace("+", "")
    for _, markers in CURRENCY_MARKERS:
        for marker in markers:
            cleaned = re.sub(_currency_marker_pattern(marker), " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\u00a0", " ")
    return cleaned.strip()


def _currency_bounds(column: str) -> tuple[float, float]:
    if column == "total_revenue_eur":
        return 5e7, 1e12
    if column in {"community_investment_eur", "green_financing_eur"}:
        return 1e4, 1e11
    return 0.0, 1e12


def _choose_ambiguous_number(
    decimal_candidate: float,
    thousands_candidate: float,
    suffix: str,
    column: str,
    separator: str,
) -> float:
    """
    Resolve a single-separator currency number using magnitude reasonableness.

    Parameters
    ----------
    decimal_candidate : float
        Value if the separator is treated as a decimal separator.
    thousands_candidate : float
        Value if the separator is treated as a thousands separator.
    suffix : str
        Detected currency suffix.
    column : str
        Target column name.
    separator : str
        Ambiguous separator, either comma or dot.

    Returns
    -------
    float
        Chosen numeric value before suffix multiplier.
    """
    multiplier = _currency_multiplier(suffix)
    suffix_norm = suffix.lower().strip().strip(".")
    decimal_total = decimal_candidate * multiplier
    thousands_total = thousands_candidate * multiplier
    lower, upper = _currency_bounds(column)

    if lower <= decimal_total <= upper and not (lower <= thousands_total <= upper):
        return decimal_candidate
    if lower <= thousands_total <= upper and not (lower <= decimal_total <= upper):
        return thousands_candidate

    if separator == "." and suffix_norm in {
        "million", "m", "m$", "mio", "billion", "bn", "mrd", "mld", "trillion", "tn"
    }:
        return decimal_candidate
    if separator == "," and suffix_norm in {"billion", "bn", "mrd", "mld", "trillion", "tn"} and thousands_total > upper:
        return decimal_candidate
    return thousands_candidate


def _parse_locale_number(number_text: str, suffix: str = "", column: str = "") -> float:
    """
    Parse a locale-aware numeric token.

    Parameters
    ----------
    number_text : str
        Numeric substring containing digits and optional comma/dot separators.
    suffix : str, optional
        Currency suffix used to resolve ambiguous cases.
    column : str, optional
        Target column name used for magnitude-aware ambiguity resolution.

    Returns
    -------
    float
        Parsed numeric value before suffix multiplier.

    Raises
    ------
    ValueError
        If the token cannot be parsed as a number.
    """
    token = number_text.strip().replace(" ", "").replace("+", "")
    if not token:
        raise ValueError("empty numeric token")

    if "," in token and "." in token:
        if token.rfind(".") > token.rfind(","):
            return float(token.replace(",", ""))
        return float(token.replace(".", "").replace(",", "."))

    if "," in token:
        parts = token.split(",")
        if len(parts) > 2:
            if all(len(part) == 3 for part in parts[1:]):
                return float(token.replace(",", ""))
            return float(token.replace(",", "."))

        integer_part, fractional_part = parts
        if len(fractional_part) == 3 and 1 <= len(integer_part) <= 3:
            decimal_candidate = float(token.replace(",", "."))
            thousands_candidate = float(token.replace(",", ""))
            return _choose_ambiguous_number(
                decimal_candidate=decimal_candidate,
                thousands_candidate=thousands_candidate,
                suffix=suffix,
                column=column,
                separator=",",
            )
        if len(fractional_part) != 3:
            return float(token.replace(",", "."))
        return float(token.replace(",", ""))

    if "." in token:
        parts = token.split(".")
        if len(parts) > 2:
            if all(len(part) == 3 for part in parts[1:]):
                return float(token.replace(".", ""))
            raise ValueError(f"ambiguous dotted numeric token: {number_text}")

        integer_part, fractional_part = parts
        if len(fractional_part) == 3 and 1 <= len(integer_part) <= 3:
            decimal_candidate = float(token)
            thousands_candidate = float(token.replace(".", ""))
            return _choose_ambiguous_number(
                decimal_candidate=decimal_candidate,
                thousands_candidate=thousands_candidate,
                suffix=suffix,
                column=column,
                separator=".",
            )
        if len(fractional_part) != 3:
            return float(token)
        return float(token)

    return float(token)


def _currency_warning(
    column: str,
    raw_value: str,
    parsed_value: float,
) -> dict[str, Any] | None:
    lower, upper = _currency_bounds(column)
    text = raw_value.lower()

    if parsed_value < lower and _has_large_suffix(text):
        return {
            "parsed_value": parsed_value,
            "suspected_correct_value": parsed_value * 1000.0,
            "warning": "suspected under-parse for large-unit currency value",
        }
    if parsed_value > upper and not _has_trillion_suffix(text):
        return {
            "parsed_value": parsed_value,
            "suspected_correct_value": parsed_value / 1000.0,
            "warning": "suspected over-parse for currency value",
        }
    return None


def _extract_currency_amounts(raw_value: str, column: str = "") -> CurrencyParseResult:
    if not raw_value or not re.search(r"\d", raw_value):
        return CurrencyParseResult(value=None, failure_reason="no recognizable numeric content")

    fx_rates = load_fx_rates()
    detection_text = _strip_parenthetical_qualifiers(raw_value).strip()
    currency_code, defaulted_to_eur = detect_currency_code(detection_text)
    fx_rate_date = get_fx_rate_date()

    if currency_code not in fx_rates:
        return CurrencyParseResult(
            value=None,
            failure_reason=f"unsupported currency: {currency_code}",
            currency_code=currency_code,
            defaulted_to_eur=defaulted_to_eur,
            unsupported_currency=currency_code,
        )

    cleaned = _strip_currency_tokens(raw_value)
    if not cleaned or not re.search(r"\d", cleaned):
        return CurrencyParseResult(
            value=None,
            failure_reason="no recognizable numeric content",
            currency_code=currency_code,
            fx_rate_used=fx_rates[currency_code],
            fx_rate_date=fx_rate_date,
            defaulted_to_eur=defaulted_to_eur,
        )

    amount_pattern = re.compile(
        r"(?P<number>\d[\d,\.]*)\s*"
        r"(?P<suffix>trillion|thousand|billion|million|mrd\.?|mld\.?|mio\.?|tsd\.?|bn|tn|k|m\$?|M\$?)?",
        re.IGNORECASE,
    )

    amounts: list[float] = []
    for match in amount_pattern.finditer(cleaned):
        number_text = match.group("number")
        suffix = match.group("suffix") or ""
        next_char = cleaned[match.end(): match.end() + 1]
        if next_char == "%":
            continue
        try:
            number = _parse_locale_number(number_text, suffix=suffix, column=column)
        except ValueError:
            continue
        amounts.append(number * _currency_multiplier(suffix))

    if not amounts:
        return CurrencyParseResult(
            value=None,
            failure_reason="numeric content could not be parsed",
            currency_code=currency_code,
            fx_rate_used=fx_rates[currency_code],
            fx_rate_date=fx_rate_date,
            defaulted_to_eur=defaulted_to_eur,
        )

    source_total = float(sum(amounts))
    fx_rate = fx_rates[currency_code]
    total = source_total * fx_rate
    warning = _currency_warning(column, raw_value, total)
    warnings = [warning] if warning else []
    return CurrencyParseResult(
        value=total,
        amounts=amounts,
        warnings=warnings,
        currency_code=currency_code,
        source_amount=source_total,
        fx_rate_used=fx_rate,
        fx_rate_date=fx_rate_date,
        defaulted_to_eur=defaulted_to_eur,
    )


def parse_currency(value: Any) -> float | None:
    """
    Parse a currency value to EUR without writing parser diagnostics.

    Parameters
    ----------
    value : Any
        Raw currency cell.

    Returns
    -------
    float | None
        Parsed EUR amount, or None when no amount can be parsed.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    raw = _clean_text(value)
    if not raw:
        return None
    return _extract_currency_amounts(raw).value


def parse_currency_eur(
    value: Any,
    row_index: int,
    column: str,
    context: ParseContext,
) -> float:
    """
    Parse mixed currency cells to numeric EUR amounts.

    Parameters
    ----------
    value : Any
        Raw currency cell.
    row_index : int
        Source row index for diagnostics.
    column : str
        Parsed target column name.
    context : ParseContext
        Multi-value and warning accumulator.

    Returns
    -------
    float
        EUR amount, or NaN when no amount is found.
    """
    if _is_missing(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    raw = _clean_text(value)
    result = _extract_currency_amounts(raw, column=column)
    if result.value is None:
        if result.unsupported_currency:
            context.fx_unsupported.append({
                "row_index": row_index,
                "column": column,
                "raw_value": raw,
                "currency_detected": result.unsupported_currency,
            })
        context.currency_failures.append({
            "row_index": row_index,
            "column": column,
            "raw_value": raw,
            "reason": result.failure_reason or "currency parsing failed",
        })
        _record_warning(context, row_index, column, value, "currency value failed to parse")
        return np.nan

    if len(result.amounts) > 1:
        context.currency_multivalue.append({
            "row_index": row_index,
            "column": column,
            "raw_value": raw,
            "parsed_amounts": "; ".join(f"{amount:.2f}" for amount in result.amounts),
            "parsed_total": result.value,
        })
        _record_warning(context, row_index, column, value, "multiple currency amounts summed")

    if result.currency_code != "EUR":
        context.fx_conversions.append({
            "row_index": row_index,
            "column": column,
            "raw_value": raw,
            "currency_detected": result.currency_code,
            "source_amount": result.source_amount,
            "fx_rate_used": result.fx_rate_used,
            "fx_rate_date": result.fx_rate_date,
            "eur_amount": result.value,
        })

    if result.defaulted_to_eur:
        context.currency_default_eur.append({
            "row_index": row_index,
            "column": column,
            "raw_value": raw,
            "parsed_value": result.value,
        })

    for warning in result.warnings:
        context.currency_warnings.append({
            "row_index": row_index,
            "column": column,
            "raw_value": raw,
            "parsed_value": warning["parsed_value"],
            "suspected_correct_value": warning["suspected_correct_value"],
            "warning": warning["warning"],
        })
        _record_warning(context, row_index, column, value, warning["warning"])

    return result.value
    if not raw or not re.search(r"\d", raw):
        return np.nan

    cleaned = _strip_parenthetical_qualifiers(raw)
    cleaned = cleaned.replace("+", "")
    amount_pattern = re.compile(
        r"(?P<prefix>€|eur)?\s*"
        r"(?P<number>\d[\d,\.]*)\s*"
        r"(?P<unit>billion|bn|million|m|thousand|k)?",
        re.IGNORECASE,
    )

    amounts: list[float] = []
    for match in amount_pattern.finditer(cleaned):
        number_text = match.group("number")
        unit = match.group("unit") or ""
        prefix = match.group("prefix") or ""
        next_char = cleaned[match.end(): match.end() + 1]

        if next_char == "%" and not prefix and not unit:
            continue

        has_currency_signal = bool(prefix) or bool(unit)
        plain_numeric_cell = bool(re.fullmatch(r"\s*\d[\d,\.]*\s*", cleaned))
        if not has_currency_signal and not plain_numeric_cell:
            continue

        amount = _normalise_number_token(number_text) * _currency_multiplier(unit)
        amounts.append(amount)

    if not amounts:
        return np.nan

    if len(amounts) > 1:
        context.currency_multivalue.append({
            "row_index": row_index,
            "column": column,
            "raw_value": raw,
            "parsed_amounts": "; ".join(f"{amount:.2f}" for amount in amounts),
            "parsed_total": float(sum(amounts)),
        })
        _record_warning(context, row_index, column, value, "multiple currency amounts summed")

    return float(sum(amounts))


def _extract_percent_candidates(value: str) -> list[dict[str, Any]]:
    candidates = []
    segments = [seg.strip() for seg in re.split(r"\s*[;/]\s*", value) if seg.strip()]
    if not segments:
        segments = [value]

    for segment_order, segment in enumerate(segments):
        for match in re.finditer(r"\d[\d,\.]*\s*%?", segment):
            number = _normalise_number_token(match.group().replace("%", ""))
            has_percent = "%" in match.group()
            candidates.append({
                "number": number,
                "has_percent": has_percent,
                "segment": segment,
                "segment_order": segment_order,
                "start": match.start(),
            })
    return candidates


def parse_renewable_share(
    value: Any,
    row_index: int,
    column: str,
    context: ParseContext,
) -> float:
    """
    Parse renewable energy share as a fraction in [0, 1].

    Parameters
    ----------
    value : Any
        Raw renewable-energy cell.
    row_index : int
        Source row index for diagnostics.
    column : str
        Parsed target column name.
    context : ParseContext
        Target-only and warning accumulator.

    Returns
    -------
    float
        Fraction from 0 to 1, or NaN when no actual value is available.
    """
    if _is_missing(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number / 100.0 if number > 1 else number

    raw = _clean_text(value)
    if not raw or raw == "%":
        return np.nan

    candidates = _extract_percent_candidates(raw)
    if not candidates:
        return np.nan

    actual_candidates = [
        item for item in candidates
        if "target" not in _normalise_for_matching(item["segment"])
    ]
    if not actual_candidates:
        context.target_only_renewable.append({
            "row_index": row_index,
            "column": column,
            "raw_value": raw,
            "reason": "target-only renewable energy value",
        })
        _record_warning(context, row_index, column, value, "target-only renewable value ignored")
        return np.nan

    first = sorted(actual_candidates, key=lambda item: (item["segment_order"], item["start"]))[0]
    number = first["number"]
    if first["has_percent"] or number > 1:
        number = number / 100.0

    if number < 0 or number > 1:
        _record_warning(context, row_index, column, value, "renewable share outside [0, 1]")
        return np.nan

    return float(number)


def _diversity_priority(segment: str) -> int:
    text = _normalise_for_matching(segment)
    if re.search(r"\b(board|bod|supervisory board|directors?)\b", text):
        return 1
    if re.search(r"\b(exco|executive committee|g(ec|ec)|management board)\b", text):
        return 2
    if re.search(r"\b(senior|leadership|top senior|key positions|executive and middle)\b", text):
        return 3
    if re.search(r"\b(total staff|total workforce|workforce|employees?)\b", text):
        return 4
    return 5


def parse_diversity_representation(
    value: Any,
    row_index: int,
    column: str,
    context: ParseContext,
) -> float:
    """
    Parse diversity representation with board-level priority.

    Parameters
    ----------
    value : Any
        Raw diversity representation cell.
    row_index : int
        Source row index for diagnostics.
    column : str
        Parsed target column name.
    context : ParseContext
        Warning accumulator.

    Returns
    -------
    float
        Fraction from 0 to 1, or NaN when no percentage is found.
    """
    if _is_missing(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number / 100.0 if number > 1 else number

    raw = _clean_text(value)
    if not raw:
        return np.nan

    candidates = _extract_percent_candidates(raw)
    if not candidates:
        return np.nan

    ranked = []
    for item in candidates:
        segment = item["segment"]
        number = item["number"]
        if item["has_percent"] or number > 1:
            number = number / 100.0
        ranked.append({
            "priority": _diversity_priority(segment),
            "segment_order": item["segment_order"],
            "start": item["start"],
            "number": number,
            "segment": segment,
        })

    ranked = sorted(ranked, key=lambda item: (item["priority"], item["segment_order"], item["start"]))
    chosen = ranked[0]
    if chosen["number"] < 0 or chosen["number"] > 1:
        _record_warning(context, row_index, column, value, "diversity share outside [0, 1]")
        return np.nan

    return float(chosen["number"])


def parse_column(
    value: Any,
    row_index: int,
    column: str,
    context: ParseContext,
) -> Any:
    """
    Dispatch one raw cell to the correct parser.

    Parameters
    ----------
    value : Any
        Raw cell value.
    row_index : int
        Source row index for diagnostics.
    column : str
        Parsed target column name.
    context : ParseContext
        Diagnostic accumulator.

    Returns
    -------
    Any
        Parsed scalar value.
    """
    if column in METADATA_COLUMNS:
        return value
    if column in GHG_COLUMNS:
        return parse_ghg_emissions(value, row_index, column, context)
    if column in ORDINAL_COLUMNS:
        return parse_ordinal_score(value, row_index, column, context)
    if column in CURRENCY_COLUMNS:
        return parse_currency_eur(value, row_index, column, context)
    if column == "renewable_energy_share":
        return parse_renewable_share(value, row_index, column, context)
    if column == "diversity_representation":
        return parse_diversity_representation(value, row_index, column, context)
    return value


def parse_dataset(df: pd.DataFrame, context: ParseContext) -> pd.DataFrame:
    """
    Parse all mapped columns in the raw real dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw table with required target columns already mapped.
    context : ParseContext
        Diagnostic accumulator.

    Returns
    -------
    pd.DataFrame
        Parsed table with metadata columns retained and all analytical columns
        converted to numeric values.
    """
    parsed = pd.DataFrame(index=df.index)
    for column in df.columns:
        parsed[column] = [
            parse_column(value, int(row_index), column, context)
            for row_index, value in df[column].items()
        ]
        if column not in METADATA_COLUMNS:
            parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed


def apply_manual_currency_overrides(
    parsed_df: pd.DataFrame,
    context: ParseContext,
) -> pd.DataFrame:
    """
    Apply manually reviewed currency corrections.

    Parameters
    ----------
    parsed_df : pd.DataFrame
        Parsed table before manual corrections.
    context : ParseContext
        Diagnostic accumulator to update after corrections.

    Returns
    -------
    pd.DataFrame
        Parsed table with manual currency overrides applied.
    """
    corrected = parsed_df.copy()
    override_keys = set(MANUAL_CURRENCY_OVERRIDES)

    for (row_index, column), value in MANUAL_CURRENCY_OVERRIDES.items():
        if column in corrected.columns and row_index in corrected.index:
            corrected.loc[row_index, column] = value

    context.currency_warnings = [
        item for item in context.currency_warnings
        if (int(item["row_index"]), str(item["column"])) not in override_keys
    ]
    context.warnings = [
        item for item in context.warnings
        if (int(item["row_index"]), str(item["column"])) not in override_keys
    ]
    return corrected


def build_column_summary(parsed_df: pd.DataFrame, context: ParseContext) -> pd.DataFrame:
    """
    Build per-column parsing summary counts.

    Parameters
    ----------
    parsed_df : pd.DataFrame
        Parsed output table.
    context : ParseContext
        Diagnostic accumulator.

    Returns
    -------
    pd.DataFrame
        Summary with n_parsed, n_nan, and n_warnings per column.
    """
    warning_counts = pd.Series(
        [item["column"] for item in context.warnings],
        dtype="object",
    ).value_counts()

    rows = []
    for column in parsed_df.columns:
        if column in METADATA_COLUMNS:
            n_nan = int(parsed_df[column].isna().sum())
            n_parsed = int(parsed_df[column].notna().sum())
        else:
            n_nan = int(parsed_df[column].isna().sum())
            n_parsed = int(parsed_df[column].notna().sum())
        rows.append({
            "column": column,
            "n_parsed": n_parsed,
            "n_nan": n_nan,
            "n_warnings": int(warning_counts.get(column, 0)),
        })
    return pd.DataFrame(rows)


def write_parsing_decisions(path: str) -> None:
    """
    Write the thesis appendix describing all parser decisions.

    Parameters
    ----------
    path : str
        Output markdown path.

    Returns
    -------
    None
    """
    lines = [
        "# Parsing Decisions",
        "",
        "This appendix documents the deterministic parsing rules used by `02a_parse_real_dataset.py`.",
        "",
    ]
    fx_date = get_fx_rate_date()

    sections = {
        "Metadata Columns": [
            "`No`, `LEI, MFI code for branches`, `Type`, `Banks`, and `Ground for significance` are retained as metadata.",
            "They are renamed to snake_case and are not numerically transformed.",
        ],
        "GHG Emissions": [
            "Scope 1, Scope 2, and Scope 3 emissions are converted to tonnes CO2e.",
            "Comma and dot thousands separators are normalized before numeric conversion.",
            "`kton`, `kt`, and `thousand t` are multiplied by 1,000.",
            "`mton`, `mio`, and `million t` are multiplied by 1,000,000.",
            "When location-based and market-based values are both present, the location-based value is used.",
            "If multiple values are present and no location-based label is available, the first value is used and a warning is counted.",
            "Examples: `1,004.36 t CO2e` -> `1004.36`; `30 kton CO2e` -> `30000`; `4,909 t CO2e (Location-based) / 480 t CO2e (Market-based)` -> `4909`.",
        ],
        "Ordinal Qualitative Scores": [
            "Ordinal ESG ratings are parsed with case-insensitive substring matching in priority order.",
            "`excellent` -> 5, `very good` -> 4, `good` -> 3, `moderate` or `adequate` -> 2, `limited` or `weak` -> 1.",
            "`Qualitative` without a rating word is mapped to NaN.",
            "Purely descriptive cells with no rating word are mapped to NaN and written to `reports/ordinal_parsing_unmapped.csv`.",
        ],
        "Currency Amounts": [
            "Community investment, green financing, and total revenue are parsed as EUR amounts.",
            "Raw numeric cells are assumed to already be EUR.",
            "`EUR`, `euro`, and euro-symbol markers are treated as EUR signals.",
            "Locale-aware number parsing was added on 2026-05-04 for EU and US/UK notation.",
            "When both comma and dot appear, the rightmost separator is treated as the decimal separator: `10,552.00` -> US/UK, `10.552,00` -> EU.",
            "When only one separator appears, decimal vs thousands use separator-specific rules and magnitude reasonableness for ambiguous cases.",
            "`thousand`/`k`/`tsd` multiplies by 1,000; `million`/`m`/`mio` multiplies by 1,000,000; `billion`/`bn`/`mrd`/`mld` multiplies by 1,000,000,000; `trillion`/`tn` multiplies by 1,000,000,000,000.",
            "Parenthetical qualifiers and plus signs are ignored for numeric conversion.",
            "Multiple listed amounts are summed and written to `reports/currency_parsing_multivalue.csv`.",
            "Suspicious magnitudes are logged to `reports/currency_parsing_warnings.csv`; unparseable text cells are logged to `reports/currency_parsing_failures.csv`.",
            "Discriminating tests: `EUR 8.78 billion` -> `8780000000`; `EUR 10.552 billion` -> `10552000000`; `EUR 10.552 Mrd.` -> `10552000000`.",
            "Examples: `EUR 246 million` -> `246000000`; `EUR 118,000` -> `118000`; `EUR 118.000` -> `118000`; `EUR 118,000 (Grants); EUR 700,000+ (Scholarships)` -> `818000`.",
            "Currency-code detection runs before numeric extraction. The priority list is EUR (`EUR`, `EURO`, `€`), USD (`US$`, `USD`, `$`), GBP (`GBP`, `£`), CHF, SEK, NOK, DKK, CZK (`CZK`, `Kč`), PLN (`PLN`, `zł`, `zl`), HUF (`HUF`, `Ft`), RON (`RON`, `lei`), BGN (`BGN`, `лв`), HRK (`HRK`, `kn`), COP, and BRL (`BRL`, `R$`).",
            f"Non-EUR values are converted using `exchange_rates/ecb_rates_2025.csv`; the applied FX date is `{fx_date}`. The parser converts ECB rates from `1 EUR = foreign rate` to `1 foreign unit = X EUR`.",
            "Fallback rules: if no currency token is detected, the cell defaults to EUR; if a detected currency is missing from the FX table, the value is mapped to NaN and written to `reports/fx_unsupported_currencies.csv`.",
            "Every non-EUR conversion is written to `reports/fx_conversions_applied.csv` with detected currency, source amount, FX rate/date, and EUR amount.",
        ],
        "Renewable Energy Share": [
            "Renewable energy share is converted to a fraction in [0, 1].",
            "Percent strings are divided by 100; numeric values already in [0, 1] are retained.",
            "When several actual percentages appear, the first/group-level value is used.",
            "Target-only values such as `90% (Target)` are mapped to NaN and listed separately in `reports/parsing_summary.txt`.",
            "Examples: `100 percent` -> `1.0`; `28% (Group level); 100% (Vienna headquarters)` -> `0.28`; `%` -> NaN.",
        ],
        "Diversity Representation": [
            "Diversity representation is converted to a fraction in [0, 1].",
            "When several percentages appear, the priority is Board > ExCo > Senior Management > Total Staff > generic percentage.",
            "Examples: `51.3 % (Total Staff); 33.3 % (Board)` -> `0.333`; `40% (Board) / 42% (ExCo)` -> `0.40`; `56.6% (Executive and middle management)` -> `0.566`.",
        ],
    }

    for title, bullets in sections.items():
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {bullet}" for bullet in bullets)
        lines.append("")

    _ensure_output_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_diagnostics(
    context: ParseContext,
    summary_df: pd.DataFrame,
    ordinal_path: str,
    currency_path: str,
    currency_warnings_path: str,
    currency_failures_path: str,
    fx_conversions_path: str,
    fx_unsupported_path: str,
    summary_path: str,
) -> None:
    """
    Write parser diagnostic files.

    Parameters
    ----------
    context : ParseContext
        Diagnostic accumulator.
    summary_df : pd.DataFrame
        Per-column summary counts.
    ordinal_path : str
        Output path for ordinal unmapped rows.
    currency_path : str
        Output path for multi-value currency rows.
    currency_warnings_path : str
        Output path for suspicious currency magnitudes.
    currency_failures_path : str
        Output path for currency cells that could not be parsed.
    fx_conversions_path : str
        Output path for non-EUR FX conversions applied.
    fx_unsupported_path : str
        Output path for detected currencies missing from the FX table.
    summary_path : str
        Output path for text summary.

    Returns
    -------
    None
    """
    _ensure_output_dir(ordinal_path)
    _ensure_output_dir(currency_path)
    _ensure_output_dir(currency_warnings_path)
    _ensure_output_dir(currency_failures_path)
    _ensure_output_dir(fx_conversions_path)
    _ensure_output_dir(fx_unsupported_path)
    _ensure_output_dir(summary_path)

    pd.DataFrame(
        context.ordinal_unmapped,
        columns=["row_index", "column", "raw_value", "reason"],
    ).to_csv(ordinal_path, index=False)

    pd.DataFrame(
        context.currency_multivalue,
        columns=["row_index", "column", "raw_value", "parsed_amounts", "parsed_total"],
    ).to_csv(currency_path, index=False)

    pd.DataFrame(
        context.currency_warnings,
        columns=[
            "row_index",
            "column",
            "raw_value",
            "parsed_value",
            "suspected_correct_value",
            "warning",
        ],
    ).to_csv(currency_warnings_path, index=False)

    pd.DataFrame(
        context.currency_failures,
        columns=["row_index", "column", "raw_value", "reason"],
    ).to_csv(currency_failures_path, index=False)

    pd.DataFrame(
        context.fx_conversions,
        columns=[
            "row_index",
            "column",
            "raw_value",
            "currency_detected",
            "source_amount",
            "fx_rate_used",
            "fx_rate_date",
            "eur_amount",
        ],
    ).to_csv(fx_conversions_path, index=False)

    pd.DataFrame(
        context.fx_unsupported,
        columns=["row_index", "column", "raw_value", "currency_detected"],
    ).to_csv(fx_unsupported_path, index=False)

    lines = ["Parsing summary", "===============", ""]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['column']}: n_parsed={row['n_parsed']}, "
            f"n_nan={row['n_nan']}, n_warnings={row['n_warnings']}"
        )

    if context.target_only_renewable:
        lines.extend(["", "Renewable target-only cells", "---------------------------"])
        for item in context.target_only_renewable:
            lines.append(
                f"row_index={item['row_index']}, column={item['column']}, "
                f"raw_value={item['raw_value']}"
            )

    lines.extend([
        "",
        "Currency FX diagnostics",
        "-----------------------",
        f"non_eur_conversions={len(context.fx_conversions)}",
        f"eur_default_no_token={len(context.currency_default_eur)}",
        f"unsupported_currencies={len(context.fx_unsupported)}",
    ])

    if context.warnings:
        lines.extend(["", "Warnings", "--------"])
        for item in context.warnings:
            lines.append(
                f"row_index={item['row_index']}, column={item['column']}, "
                f"warning={item['warning']}, raw_value={item['raw_value']}"
            )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_real_dataset(
    input_path: str,
    output_path: str,
    decisions_path: str,
    ordinal_unmapped_path: str,
    currency_multivalue_path: str,
    currency_warnings_path: str,
    currency_failures_path: str,
    fx_conversions_path: str,
    fx_unsupported_path: str,
    summary_path: str,
) -> pd.DataFrame:
    """
    Parse the real ESG banking dataset and write all diagnostics.

    Parameters
    ----------
    input_path : str
        Raw workbook path.
    output_path : str
        Parsed CSV output path.
    decisions_path : str
        Markdown parsing-decisions output path.
    ordinal_unmapped_path : str
        CSV path for unmapped ordinal values.
    currency_multivalue_path : str
        CSV path for multi-value currency cells.
    currency_warnings_path : str
        CSV path for suspicious currency magnitudes.
    currency_failures_path : str
        CSV path for currency cells that could not be parsed.
    fx_conversions_path : str
        CSV path for non-EUR FX conversions applied.
    fx_unsupported_path : str
        CSV path for detected currencies missing from the FX table.
    summary_path : str
        Text parsing summary output path.

    Returns
    -------
    pd.DataFrame
        Parsed output table.
    """
    raw = load_raw_dataset(input_path)
    print(f"[step_02a] Input shape: {raw.shape}")

    mapped = apply_required_column_mapping(raw)
    context = ParseContext()
    parsed = parse_dataset(mapped, context)
    parsed = apply_manual_currency_overrides(parsed, context)

    _ensure_output_dir(output_path)
    parsed.to_csv(output_path, index=False)
    print(f"[step_02a] Parsed output -> {output_path}")

    summary_df = build_column_summary(parsed, context)
    write_parsing_decisions(decisions_path)
    write_diagnostics(
        context,
        summary_df,
        ordinal_unmapped_path,
        currency_multivalue_path,
        currency_warnings_path,
        currency_failures_path,
        fx_conversions_path,
        fx_unsupported_path,
        summary_path,
    )

    print(f"[step_02a] Parsing decisions -> {decisions_path}")
    print(f"[step_02a] Ordinal unmapped -> {ordinal_unmapped_path}")
    print(f"[step_02a] Currency multi-value log -> {currency_multivalue_path}")
    print(f"[step_02a] Currency warnings -> {currency_warnings_path}")
    print(f"[step_02a] Currency failures -> {currency_failures_path}")
    print(f"[step_02a] FX conversions -> {fx_conversions_path}")
    print(f"[step_02a] FX unsupported -> {fx_unsupported_path}")
    print(f"[step_02a] Currency defaulted to EUR (no token): {len(context.currency_default_eur)}")
    print(f"[step_02a] Parsing summary -> {summary_path}")
    print(f"[step_02a] Final shape: {parsed.shape}")
    return parsed


def run_currency_parser_tests() -> None:
    """
    Run hardcoded unit tests for locale-aware currency parsing.

    Raises
    ------
    AssertionError
        If any parser test fails.
    """
    test_cases: list[tuple[Any, float | None]] = [
        ("€1,572 million", 1_572_000_000.0),
        ("€10.552 billion", 10_552_000_000.0),
        ("€8.78 billion", 8_780_000_000.0),
        ("EUR 246 million", 246_000_000.0),
        ("EUR 10.552 Mrd.", 10_552_000_000.0),
        ("€118,000", 118_000.0),
        ("€118.000", 118_000.0),
        ("0.89", 0.89),
        ("0,89", 0.89),
        (422192000, 422_192_000.0),
        ("€118,000 (Grants); €700,000+ (Scholarships)", 818_000.0),
        ("Volunteering programs", None),
        ("", None),
        (None, None),
    ]

    for raw_value, expected in test_cases:
        parsed = parse_currency(raw_value)
        if expected is None:
            assert parsed is None, f"{raw_value!r}: expected None, got {parsed!r}"
        else:
            assert parsed is not None, f"{raw_value!r}: expected {expected}, got None"
            assert abs(parsed - expected) < 1e-6, (
                f"{raw_value!r}: expected {expected}, got {parsed}"
            )

    # FX-dependent ranges use the ECB rates in exchange_rates/ecb_rates_2025.csv.
    fx_cases: list[tuple[Any, float, float]] = [
        ("$1,000 million", 8e8, 1.2e9),             # USD around 0.85 EUR in the current file.
        ("HUF 2,224,584 million", 5e9, 7e9),       # HUF around 0.0026 EUR in the current file.
        ("CHF 500 million", 4.5e8, 6e8),           # CHF close to EUR parity.
        ("Ps. 222,143 million", 1.0e10, 1.5e10),   # MXN around 0.047 EUR in the current file.
        ("£250 million", 2.5e8, 3.5e8),            # GBP around 1.15 EUR in the current file.
        ("2,224,584 million", 2e12, 2.5e12),       # No token defaults to EUR.
    ]
    for raw_value, lower, upper in fx_cases:
        parsed = parse_currency(raw_value)
        assert parsed is not None, f"{raw_value!r}: expected FX-converted value, got None"
        assert lower <= parsed <= upper, (
            f"{raw_value!r}: expected value in [{lower}, {upper}], got {parsed}"
        )

    cop_value = parse_currency("COP 3,187.36 Billion (MMM)")
    if "COP" in load_fx_rates():
        assert cop_value is not None and 5e8 <= cop_value <= 1e9, (
            f"COP conversion outside plausible range: {cop_value}"
        )
    else:
        assert cop_value is None, "COP is absent from the FX file and should parse as unsupported"

    assert parse_currency("XYZ 100 million") is None, "Unsupported XYZ should return None"

    print(f"[step_02a] Currency parser tests passed: {len(test_cases) + len(fx_cases) + 2} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the real ECB ESG workbook into a numeric CSV."
    )
    parser.add_argument("--test-currency-parser", action="store_true",
                        help="Run locale-aware currency parser tests and exit")
    parser.add_argument("--input", default=RAW_DATA_PATH, help="Path to raw workbook")
    parser.add_argument("--output", default=REAL_PARSED_DATA_PATH, help="Parsed CSV output path")
    parser.add_argument("--decisions", default=PARSING_DECISIONS_PATH,
                        help="Markdown parsing decisions output path")
    parser.add_argument("--ordinal-unmapped", default=ORDINAL_PARSING_UNMAPPED_PATH,
                        help="CSV of unmapped ordinal values")
    parser.add_argument("--currency-multivalue", default=CURRENCY_PARSING_MULTIVALUE_PATH,
                        help="CSV of multi-value currency cells")
    parser.add_argument("--currency-warnings", default=CURRENCY_PARSING_WARNINGS_PATH,
                        help="CSV of suspicious currency magnitudes")
    parser.add_argument("--currency-failures", default=CURRENCY_PARSING_FAILURES_PATH,
                        help="CSV of currency cells that could not be parsed")
    parser.add_argument("--fx-conversions", default=FX_CONVERSIONS_LOG_PATH,
                        help="CSV of non-EUR FX conversions applied")
    parser.add_argument("--fx-unsupported", default=FX_UNSUPPORTED_LOG_PATH,
                        help="CSV of detected currencies missing from FX table")
    parser.add_argument("--summary", default=PARSING_SUMMARY_PATH,
                        help="Text parsing summary output path")
    args = parser.parse_args()

    if args.test_currency_parser:
        run_currency_parser_tests()
        raise SystemExit(0)

    parse_real_dataset(
        args.input,
        args.output,
        args.decisions,
        args.ordinal_unmapped,
        args.currency_multivalue,
        args.currency_warnings,
        args.currency_failures,
        args.fx_conversions,
        args.fx_unsupported,
        args.summary,
    )
