# config.py
# ============================================================
# Central config for the ESG Causal Discovery Pipeline.
# All scripts import from here — change a path once, it
# updates everywhere.
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw/df_asst_bnk_ecb.xlsx"  # canonical workbook for all runs
RAG_CLAIMS_DIR      = "data/raw/rag_claims"
CLEAN_DATA_PATH     = "data/processed/data_clean.csv"
READY_DATA_PATH     = "data/processed/data_ready.csv"
REAL_PARSED_DATA_PATH = "data/processed/data_real_parsed.csv"
COLUMN_MAPPING_PATH = "data/processed/column_mapping.csv"
AUDIT_REPORT_PATH   = "reports/audit_report.txt"
HIGH_CORR_PATH      = "reports/high_correlation_pairs.csv"
CLEANING_SUMMARY_PATH = "reports/data_cleaning_summary.md"
PARSING_DECISIONS_PATH = "reports/parsing_decisions.md"
ORDINAL_PARSING_UNMAPPED_PATH = "reports/ordinal_parsing_unmapped.csv"
CURRENCY_PARSING_MULTIVALUE_PATH = "reports/currency_parsing_multivalue.csv"
CURRENCY_PARSING_WARNINGS_PATH = "reports/currency_parsing_warnings.csv"
CURRENCY_PARSING_FAILURES_PATH = "reports/currency_parsing_failures.csv"
REVENUE_PARSING_COMPARISON_PATH = "reports/revenue_parsing_comparison.csv"
FX_CONVERSIONS_LOG_PATH = "reports/fx_conversions_applied.csv"
FX_UNSUPPORTED_LOG_PATH = "reports/fx_unsupported_currencies.csv"
PARSING_SUMMARY_PATH = "reports/parsing_summary.txt"
LLM_SCORING_LOG_PATH = "reports/llm_scoring_log.csv"
LLM_SCORING_FAILURES_PATH = "reports/llm_scoring_failures.csv"
LIMITATIONS_PATH = "LIMITATIONS_TO_REVIEW.md"
CLAIM_AGGREGATION_PATH = "reports/claim_aggregation.csv"
CONSTRAINTS_REVIEW_PATH = "reports/constraints_for_review.csv"
FORBIDDEN_EDGES_DRAFT_PATH = "data/processed/forbidden_edges_draft.csv"
REQUIRED_EDGES_DRAFT_PATH = "data/processed/required_edges_draft.csv"
CONSTRAINT_ADAPTER_LOG_PATH = "reports/constraint_adapter_log.csv"
CAUSICA_CONSTRAINT_MATRIX_PATH = "data/processed/causica_constraint_matrix.npy"
FORBIDDEN_EDGES_MODULE_PATH = "04_forbidden_edges.py"
FORBIDDEN_EDGES_BACKUP_PATH = "04_forbidden_edges.py.bak"
FORBIDDEN_EDGES_REAL_MODULE_PATH = "04_forbidden_edges_real.py"
FORBIDDEN_EDGES_SYNTHETIC_MODULE_PATH = "04_forbidden_edges_synthetic.py"
GROUND_TRUTH_ADJACENCY_PATH = "data/processed/ground_truth_adjacency.csv"
GROUND_TRUTH_ADJACENCY_REAL_PATH = "data/processed/ground_truth_adjacency_real.csv"
CONSTRAINT_COVERAGE_PATH = "reports/constraint_coverage.md"
CONSTRAINT_COVERAGE_REAL_PATH = "reports/constraint_coverage_real.md"
CONSTRAINT_COVERAGE_SYNTHETIC_PATH = "reports/constraint_coverage_synthetic.md"
CONSTRAINTS_SKIPPED_REAL_PATH = "reports/constraints_skipped_real.csv"
CONSTRAINTS_SKIPPED_SYNTHETIC_PATH = "reports/constraints_skipped_synthetic.csv"
SYNTHETIC_DIR = "data/synthetic"
SYNTHETIC_GROUND_TRUTH_PATH = "data/synthetic/ground_truth_adjacency.csv"
SYNTHETIC_CONSTRAINT_ADJACENCY_PATH = "data/synthetic/ground_truth_constraints_synthetic.csv"
SYNTHETIC_EDGES_PATH = "data/synthetic/ground_truth_edges.csv"
SYNTHETIC_GENERATION_SUMMARY = "reports/synthetic_generation_summary.md"
FX_RATES_PDF_PATH   = "exchange_rates/exchange_rate_2025.pdf"
FX_RATES_CSV_PATH   = "exchange_rates/ecb_rates_2025.csv"
FX_PROCESSED_PATH   = "data/processed/df_asst_bnk_ecb_processed.xlsx"
LOCAL_VENV_PATH     = ".venv"

# ── Thresholds (tune these without touching any other file) ──
MISSING_DROP_THRESHOLD   = 1.0   # drop column if 100% missing
MISSING_FLAG_THRESHOLD   = 0.30  # flag column if >30% missing
IMPUTE_THRESHOLD         = 0.30  # impute with median if ≤ 30% missing
HIGH_CORR_THRESHOLD      = 0.97  # flag pairs above this correlation
NEAR_CONSTANT_STD_RATIO  = 0.01  # flag if std < 1% of mean

# ── DECI experiment controls ─────────────────────────────────────────────
# Main thesis runs use a fixed threshold. Adaptive thresholding is available
# only as a diagnostic/exploratory mode and should be labelled as such.
DECI_THRESHOLD = 0.250
DECI_THRESHOLD_MODE = "fixed"  # fixed | percentile | topk
DECI_THRESHOLD_PERCENTILE = 95.0
DECI_TOPK_EDGES = None
DECI_BACKEND = "causica"  # causica | manual
DECI_ALLOW_MANUAL_FALLBACK = True
DECI_PRESET = "small_data"  # default | small_data | fast_debug
MIN_SAMPLES_PER_VARIABLE_WARNING = 10

DECI_THRESHOLD_CANDIDATES = [
    0.001, 0.005, 0.01, 0.02, 0.03,
    0.05, 0.075, 0.1, 0.15, 0.2,
    0.25, 0.26, 0.27, 0.275, 0.28,
    0.285, 0.3, 0.302, 0.304, 0.306,
    0.308, 0.31, 0.312, 0.315, 0.32,
    0.325, 0.33, 0.335, 0.34, 0.35,
    0.4, 0.5,
]
DECI_CALIBRATE_THRESHOLD_ON_SYNTHETIC = True
DECI_MAX_DENSITY_MULTIPLE = 1.5
DECI_MIN_DENSITY_MULTIPLE = 0.25

# ── Columns excluded from causal discovery ───────────────────
# Add any column that is an identifier, timestamp, or
# administrative field that should never be a causal variable.
METADATA_COLS = [
    "no",
    "lei_mfi_code_for_branches",
    "type",
    "banks",
    "ground_for_significance",
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
for folder in [
    "data/raw",
    "data/processed",
    "reports",
    "outputs/graphs",
    "outputs/metrics",
    "outputs/figures",
    "scripts",
    "docs",
    "exchange_rates",
]:
    os.makedirs(folder, exist_ok=True)
