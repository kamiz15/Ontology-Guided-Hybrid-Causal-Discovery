# config.py
# ============================================================
# Central config for the ESG Causal Discovery Pipeline.
# All scripts import from here — change a path once, it
# updates everywhere.
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw/esg_raw.csv"          # supervisor data goes here
CLEAN_DATA_PATH     = "data/processed/data_clean.csv"
READY_DATA_PATH     = "data/processed/data_ready.csv"
COLUMN_MAPPING_PATH = "data/processed/column_mapping.csv"
AUDIT_REPORT_PATH   = "reports/audit_report.txt"
HIGH_CORR_PATH      = "reports/high_correlation_pairs.csv"

# ── Thresholds (tune these without touching any other file) ──
MISSING_DROP_THRESHOLD   = 1.0   # drop column if 100% missing
MISSING_FLAG_THRESHOLD   = 0.30  # flag column if >30% missing
IMPUTE_THRESHOLD         = 0.30  # impute with median if ≤ 30% missing
HIGH_CORR_THRESHOLD      = 0.97  # flag pairs above this correlation
NEAR_CONSTANT_STD_RATIO  = 0.01  # flag if std < 1% of mean

# ── Columns excluded from causal discovery ───────────────────
# Add any column that is an identifier, timestamp, or
# administrative field that should never be a causal variable.
METADATA_COLS = [
    "incorporation_year",
    "ipo_year",
    "reporting_year",
    "nace",
    "num_business_segments",
    "num_employee",
]

# Z-score sub-components: highly collinear with financial cols.
# Keep only the composite z-score if needed, drop ingredients.
ZSCORE_COMPONENTS = ["z1", "z2", "z3", "z4", "z5"]

EXCLUDE_FROM_CAUSAL = METADATA_COLS + ZSCORE_COMPONENTS

# ── Ensure output directories exist ─────────────────────────
for folder in ["data/raw", "data/processed", "reports",
               "outputs/graphs", "outputs/metrics"]:
    os.makedirs(folder, exist_ok=True)
