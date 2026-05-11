import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FX_PROCESSED_PATH, FX_RATES_CSV_PATH, RAW_DATA_PATH


MONETARY_COLUMNS = [
    "Community investment",
    "Sustainable finance / Green financing",
    "Total revenue",
]

CURRENCY_ALIASES = {
    "EUR": "EUR",
    "EURO": "EUR",
    "\u20ac": "EUR",
    "USD": "USD",
    "$": "USD",
    "SEK": "SEK",
    "COP": "COP",
    "GBP": "GBP",
    "CHF": "CHF",
    "NOK": "NOK",
    "DKK": "DKK",
    "JPY": "JPY",
    "PLN": "PLN",
    "CZK": "CZK",
    "HUF": "HUF",
    "RON": "RON",
    "BGN": "BGN",
    "TRY": "TRY",
    "PS": "PS",
    "PS.": "PS",
}

UNIT_MULTIPLIERS = {
    "trillion": 1_000_000_000_000,
    "tn": 1_000_000_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "m": 1_000_000,
    "thousand": 1_000,
    "k": 1_000,
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def normalize_currency(raw_text: str) -> str | None:
    text = raw_text.upper()
    text = text.replace("\\U20AC", "\u20ac").replace("\\u20ac", "\u20ac")
    for token, code in CURRENCY_ALIASES.items():
        if token in text:
            return code
    return None


def parse_unit_multiplier(snippet: str) -> float:
    snippet = snippet.lower()
    for token, multiplier in UNIT_MULTIPLIERS.items():
        if re.search(rf"\b{re.escape(token)}\b", snippet):
            return multiplier
    return 1.0


def parse_amount_and_currency(value) -> tuple[float, str | None, str]:
    if pd.isna(value):
        return np.nan, None, "missing"

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), None, "numeric value with no explicit currency"

    text = str(value).strip()
    if not text:
        return np.nan, None, "blank string"

    currency = normalize_currency(text)
    matches = list(re.finditer(r"\d[\d,]*(?:\.\d+)?", text))
    if not matches:
        return np.nan, currency, "no numeric token found"

    candidates = []
    for match in matches:
        raw_number = match.group().replace(",", "")
        try:
            amount = float(raw_number)
        except ValueError:
            continue

        tail = text[match.end(): match.end() + 24]
        multiplier = parse_unit_multiplier(tail)
        candidates.append(amount * multiplier)

    if not candidates:
        return np.nan, currency, "numeric parsing failed"

    return max(candidates), currency, "parsed from text"


def load_rates(rates_path: str) -> pd.DataFrame:
    path = Path(rates_path)
    if not path.exists():
        print(f"[fx] Rates file not found at {rates_path}. Non-EUR values will remain unresolved.")
        return pd.DataFrame(columns=["date", "currency", "rate"])

    rates_df = pd.read_csv(path)
    expected = {"date", "currency", "rate"}
    missing = expected.difference(rates_df.columns)
    if missing:
        raise ValueError(f"Rates CSV is missing required columns: {sorted(missing)}")

    rates_df = rates_df.copy()
    rates_df["date"] = pd.to_datetime(rates_df["date"], errors="coerce")
    rates_df["currency"] = rates_df["currency"].astype(str).str.upper().str.strip()
    rates_df["rate"] = pd.to_numeric(rates_df["rate"], errors="coerce")
    rates_df = rates_df.dropna(subset=["date", "currency", "rate"]).sort_values(["currency", "date"])

    print(f"[fx] Loaded {len(rates_df)} exchange-rate rows from {rates_path}")
    return rates_df


def get_fx_rate(rates_df: pd.DataFrame, currency: str | None, year: int | None) -> tuple[float | None, str]:
    if currency in (None, "", "EUR"):
        return 1.0, "EUR or unspecified"

    subset = rates_df[rates_df["currency"] == currency]
    if subset.empty:
        return None, f"no rate found for {currency}"

    if year is not None:
        cutoff = pd.Timestamp(f"{year}-12-31")
        subset = subset[subset["date"] <= cutoff]
        if subset.empty:
            return None, f"no rate found for {currency} on or before {cutoff.date()}"

    row = subset.iloc[-1]
    return float(row["rate"]), row["date"].date().isoformat()


def convert_to_eur(amount: float, currency: str | None, rate: float | None, rate_kind: str) -> float:
    if pd.isna(amount):
        return np.nan
    if currency in (None, "", "EUR"):
        return float(amount)
    if rate is None or rate == 0:
        return np.nan
    if rate_kind == "foreign_per_eur":
        return float(amount) / float(rate)
    return float(amount) * float(rate)


def process_workbook(input_path: str, rates_path: str, output_path: str, year: int | None, rate_kind: str) -> None:
    df = pd.read_excel(input_path)
    rates_df = load_rates(rates_path)

    print(f"[fx] Loaded workbook {input_path}: {df.shape}")

    for column in MONETARY_COLUMNS:
        if column not in df.columns:
            print(f"[fx] Skipping missing column: {column}")
            continue

        slug = slugify(column)
        parsed = df[column].apply(parse_amount_and_currency)
        parsed_df = pd.DataFrame(parsed.tolist(), columns=["amount", "currency", "parse_note"], index=df.index)

        rate_lookup = parsed_df["currency"].apply(lambda cur: get_fx_rate(rates_df, cur, year))
        rate_df = pd.DataFrame(rate_lookup.tolist(), columns=["fx_rate_used", "fx_rate_source"], index=df.index)

        eur_values = [
            convert_to_eur(amount, currency, rate, rate_kind)
            for amount, currency, rate in zip(parsed_df["amount"], parsed_df["currency"], rate_df["fx_rate_used"])
        ]

        df[f"{slug}_parsed_amount"] = parsed_df["amount"]
        df[f"{slug}_currency"] = parsed_df["currency"]
        df[f"{slug}_fx_rate_used"] = rate_df["fx_rate_used"]
        df[f"{slug}_fx_rate_source"] = rate_df["fx_rate_source"]
        df[f"{slug}_eur"] = eur_values
        df[f"{slug}_fx_converted"] = (
            parsed_df["currency"].notna()
            & parsed_df["currency"].ne("EUR")
            & rate_df["fx_rate_used"].notna()
        )
        df[f"{slug}_parse_note"] = parsed_df["parse_note"]

        unresolved_mask = (
            parsed_df["currency"].notna()
            & parsed_df["currency"].ne("EUR")
            & rate_df["fx_rate_used"].isna()
        )
        converted_count = int(df[f"{slug}_fx_converted"].sum())
        unresolved_count = int(unresolved_mask.sum())
        print(
            f"[fx] {column}: parsed {parsed_df['amount'].notna().sum()} rows, "
            f"converted {converted_count}, unresolved non-EUR rows {unresolved_count}"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"[fx] Saved processed workbook -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse mixed-currency workbook values and convert them to EUR.")
    parser.add_argument("--input", default=RAW_DATA_PATH, help="Path to the source workbook.")
    parser.add_argument("--rates", default=FX_RATES_CSV_PATH, help="CSV with columns date,currency,rate.")
    parser.add_argument("--output", default=FX_PROCESSED_PATH, help="Path to the processed workbook.")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional reporting year. If set, uses the latest rate on or before Dec 31 of that year.",
    )
    parser.add_argument(
        "--rate-kind",
        choices=["foreign_per_eur", "eur_per_foreign"],
        default="foreign_per_eur",
        help="Interpret ECB rates as foreign-per-EUR (default) or EUR-per-foreign.",
    )
    args = parser.parse_args()

    process_workbook(args.input, args.rates, args.output, args.year, args.rate_kind)
