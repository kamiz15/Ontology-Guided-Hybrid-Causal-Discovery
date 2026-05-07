# Data Cleaning Summary

## Analysis Orientation

The cleaning step uses horizontal analysis: each row is one confidential bank/entity observation, and each column is an ESG or financial variable. Direct identifiers and administrative fields are excluded from causal discovery, so bank names, legal identifiers, and row IDs are not used as model variables.

## Cleaning Decisions

- Raw input shape: 110 rows x 17 columns
- After dropping fully empty columns: 110 rows x 17 columns
- Final causal-ready shape: 110 rows x 12 columns
- Metadata/admin columns excluded: 5
- Remaining non-numeric columns dropped after encoding: 0
- Near-constant numeric columns flagged: 0

## Correlation Analysis

The correlation check is a redundancy diagnostic, not causal evidence. The cleaner selects numeric variables, computes the Pearson correlation matrix with `pandas.DataFrame.corr()`, converts correlations to absolute values, and reports pairs above the configured threshold. Only exact duplicate pairs with correlation equal to 1.0 are automatically dropped.

- High-correlation threshold: 0.97
- High-correlation pairs reported: 0
- Exact-duplicate columns dropped: 0

Excluded metadata/admin columns:

- `no`
- `lei_mfi_code_for_branches`
- `type`
- `banks`
- `ground_for_significance`
