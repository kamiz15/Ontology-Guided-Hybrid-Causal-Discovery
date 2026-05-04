# Parsing Decisions

This appendix documents the deterministic parsing rules used by `02a_parse_real_dataset.py`.

## Metadata Columns

- `No`, `LEI, MFI code for branches`, `Type`, `Banks`, and `Ground for significance` are retained as metadata.
- They are renamed to snake_case and are not numerically transformed.

## GHG Emissions

- Scope 1, Scope 2, and Scope 3 emissions are converted to tonnes CO2e.
- Comma and dot thousands separators are normalized before numeric conversion.
- `kton`, `kt`, and `thousand t` are multiplied by 1,000.
- `mton`, `mio`, and `million t` are multiplied by 1,000,000.
- When location-based and market-based values are both present, the location-based value is used.
- If multiple values are present and no location-based label is available, the first value is used and a warning is counted.
- Examples: `1,004.36 t CO2e` -> `1004.36`; `30 kton CO2e` -> `30000`; `4,909 t CO2e (Location-based) / 480 t CO2e (Market-based)` -> `4909`.

## Ordinal Qualitative Scores

- Ordinal ESG ratings are parsed with case-insensitive substring matching in priority order.
- `excellent` -> 5, `very good` -> 4, `good` -> 3, `moderate` or `adequate` -> 2, `limited` or `weak` -> 1.
- `Qualitative` without a rating word is mapped to NaN.
- Purely descriptive cells with no rating word are mapped to NaN and written to `reports/ordinal_parsing_unmapped.csv`.

## Currency Amounts

- Community investment, green financing, and total revenue are parsed as EUR amounts.
- Raw numeric cells are assumed to already be EUR.
- `EUR`, `euro`, and euro-symbol markers are treated as EUR signals.
- Locale-aware number parsing was added on 2026-05-04 for EU and US/UK notation.
- When both comma and dot appear, the rightmost separator is treated as the decimal separator: `10,552.00` -> US/UK, `10.552,00` -> EU.
- When only one separator appears, decimal vs thousands use separator-specific rules and magnitude reasonableness for ambiguous cases.
- `thousand`/`k`/`tsd` multiplies by 1,000; `million`/`m`/`mio` multiplies by 1,000,000; `billion`/`bn`/`mrd`/`mld` multiplies by 1,000,000,000; `trillion`/`tn` multiplies by 1,000,000,000,000.
- Parenthetical qualifiers and plus signs are ignored for numeric conversion.
- Multiple listed amounts are summed and written to `reports/currency_parsing_multivalue.csv`.
- Suspicious magnitudes are logged to `reports/currency_parsing_warnings.csv`; unparseable text cells are logged to `reports/currency_parsing_failures.csv`.
- Discriminating tests: `EUR 8.78 billion` -> `8780000000`; `EUR 10.552 billion` -> `10552000000`; `EUR 10.552 Mrd.` -> `10552000000`.
- Examples: `EUR 246 million` -> `246000000`; `EUR 118,000` -> `118000`; `EUR 118.000` -> `118000`; `EUR 118,000 (Grants); EUR 700,000+ (Scholarships)` -> `818000`.
- Currency-code detection runs before numeric extraction. The priority list is EUR (`EUR`, `EURO`, `€`), USD (`US$`, `USD`, `$`), GBP (`GBP`, `£`), CHF, SEK, NOK, DKK, CZK (`CZK`, `Kč`), PLN (`PLN`, `zł`, `zl`), HUF (`HUF`, `Ft`), RON (`RON`, `lei`), BGN (`BGN`, `лв`), HRK (`HRK`, `kn`), COP, and BRL (`BRL`, `R$`).
- Non-EUR values are converted using `exchange_rates/ecb_rates_2025.csv`; the applied FX date is `2025-12-31`. The parser converts ECB rates from `1 EUR = foreign rate` to `1 foreign unit = X EUR`.
- Fallback rules: if no currency token is detected, the cell defaults to EUR; if a detected currency is missing from the FX table, the value is mapped to NaN and written to `reports/fx_unsupported_currencies.csv`.
- Every non-EUR conversion is written to `reports/fx_conversions_applied.csv` with detected currency, source amount, FX rate/date, and EUR amount.

## Renewable Energy Share

- Renewable energy share is converted to a fraction in [0, 1].
- Percent strings are divided by 100; numeric values already in [0, 1] are retained.
- When several actual percentages appear, the first/group-level value is used.
- Target-only values such as `90% (Target)` are mapped to NaN and listed separately in `reports/parsing_summary.txt`.
- Examples: `100 percent` -> `1.0`; `28% (Group level); 100% (Vienna headquarters)` -> `0.28`; `%` -> NaN.

## Diversity Representation

- Diversity representation is converted to a fraction in [0, 1].
- When several percentages appear, the priority is Board > ExCo > Senior Management > Total Staff > generic percentage.
- Examples: `51.3 % (Total Staff); 33.3 % (Board)` -> `0.333`; `40% (Board) / 42% (ExCo)` -> `0.40`; `56.6% (Executive and middle management)` -> `0.566`.


## LLM-Assisted Ordinal Coding

Date: 2026-05-04
Model: gemma-3-27b-it
Source: reports/ordinal_parsing_unmapped.csv
Output log: reports/llm_scoring_log.csv

Gemma via Google AI Studio was used to score ordinal ESG disclosure cells that
were not covered by deterministic keyword parsing. Low-confidence scores are
flagged for manual review in LIMITATIONS_TO_REVIEW.md.
