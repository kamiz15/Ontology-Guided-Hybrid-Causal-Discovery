import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FX_RATES_CSV_PATH, FX_RATES_PDF_PATH


DATE_HEADERS = {"date", "as of", "reference date", "obs_value_date"}
CURRENCY_HEADERS = {"currency", "curr", "currency code", "iso code"}
RATE_HEADERS = {"rate", "fx rate", "exchange rate", "value", "obs_value"}


def normalize_header(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def find_column_index(headers: list[str], candidates: set[str]) -> int | None:
    for idx, header in enumerate(headers):
        if header in candidates:
            return idx
    return None


def extract_rows_from_table(table: list[list[str]], page_number: int) -> list[dict]:
    if not table or len(table) < 2:
        return []

    headers = [normalize_header(cell) for cell in table[0]]
    date_idx = find_column_index(headers, DATE_HEADERS)
    currency_idx = find_column_index(headers, CURRENCY_HEADERS)
    rate_idx = find_column_index(headers, RATE_HEADERS)

    if None in (date_idx, currency_idx, rate_idx):
        return []

    rows = []
    for raw_row in table[1:]:
        if not raw_row or max(date_idx, currency_idx, rate_idx) >= len(raw_row):
            continue
        rows.append(
            {
                "date": raw_row[date_idx],
                "currency": raw_row[currency_idx],
                "rate": raw_row[rate_idx],
                "source_page": page_number,
            }
        )
    return rows


def extract_rows_from_text(page_text: str, page_number: int) -> list[dict]:
    if not page_text:
        return []

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if len(lines) < 3:
        return []

    page_date = pd.to_datetime(lines[1], errors="coerce")
    if pd.isna(page_date):
        return []

    rows = []
    for line in lines[3:]:
        if line.lower().startswith("source:"):
            continue
        match = re.match(r"^([A-Z]{3})\s+.+?\s+([-+]?\d[\d.,]*)$", line)
        if not match:
            continue
        rows.append(
            {
                "date": page_date,
                "currency": match.group(1),
                "rate": match.group(2),
                "source_page": page_number,
            }
        )
    return rows


def extract_rates(pdf_path: str, output_path: str) -> pd.DataFrame:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    rows = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_rows = []
            for table in page.extract_tables() or []:
                page_rows.extend(extract_rows_from_table(table, page_number))
            if not page_rows:
                page_rows.extend(extract_rows_from_text(page.extract_text() or "", page_number))
            rows.extend(page_rows)

    rates_df = pd.DataFrame(rows)
    if rates_df.empty:
        raise RuntimeError(
            "No rate rows were extracted. Inspect the PDF structure and adjust the header mapping in extract_ecb_rates.py."
        )

    rates_df["date"] = pd.to_datetime(rates_df["date"], errors="coerce")
    rates_df["currency"] = rates_df["currency"].astype(str).str.upper().str.strip()
    rates_df["rate"] = (
        rates_df["rate"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )
    rates_df["rate"] = pd.to_numeric(rates_df["rate"], errors="coerce")
    rates_df = rates_df.dropna(subset=["date", "currency", "rate"]).sort_values(["currency", "date"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rates_df.to_csv(output_path, index=False)

    print(f"[rates] Extracted {len(rates_df)} rows from {pdf_path}")
    print(f"[rates] Saved -> {output_path}")
    return rates_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ECB exchange-rate tables from a PDF into CSV.")
    parser.add_argument("--input", default=FX_RATES_PDF_PATH, help="Path to the ECB PDF.")
    parser.add_argument("--output", default=FX_RATES_CSV_PATH, help="Path to the output CSV.")
    args = parser.parse_args()

    extract_rates(args.input, args.output)
