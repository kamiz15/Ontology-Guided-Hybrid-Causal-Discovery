# Ontology-Guided Hybrid Causal Discovery

This repository contains an ESG-focused causal discovery workflow built around one canonical banking workbook, ontology-style edge constraints, multiple causal discovery backends, optional Gemma-based reasoning, and a small exchange-rate helper pipeline for mixed-currency fields.

The current default raw input is:

- `data/raw/df_asst_bnk_ecb.xlsx`

The project is centered on the scripts in the repo root, with shared paths defined in [config.py](./config.py).

## What Is In The Repo Now

- A standardized 01-09 causal discovery pipeline for the ECB workbook
- An optional Gemma proposal step in [08_gemma_causal_proposals.py](./08_gemma_causal_proposals.py)
- A supplementary edge-evaluation step in [10_gemma_evaluate.py](./10_gemma_evaluate.py)
- A one-command orchestrator in [run_all.py](./run_all.py)
- A PowerShell environment bootstrap in [setup_venv.ps1](./setup_venv.ps1)
- An exchange-rate extraction and currency-normalization helper workflow under [exchange_rates](./exchange_rates) and [check_del_after.py](./check_del_after.py)
- Tracked reports, graphs, and figures under [reports](./reports) and [outputs](./outputs)

## Project Layout

```text
go/
|-- 01_audit.py
|-- 02_clean.py
|-- 03_build_column_mapping.py
|-- 04_forbidden_edges.py
|-- 05_run_baselines.py
|-- 06_run_notears.py
|-- 07_run_deci.py
|-- 08_gemma_causal_proposals.py
|-- 09_visualize_graphs.py
|-- 10_gemma_evaluate.py
|-- check_del_after.py
|-- config.py
|-- io_utils.py
|-- run_all.py
|-- setup_venv.ps1
|-- requirements.txt
|-- data/
|   |-- raw/
|   |   `-- df_asst_bnk_ecb.xlsx
|   `-- processed/
|       |-- data_clean.csv
|       |-- data_ready.csv
|       |-- column_mapping.csv
|       `-- df_asst_bnk_ecb_processed.xlsx
|-- docs/
|   `-- WEEK2_README.md
|-- exchange_rates/
|   |-- exchange_rate_2025.pdf
|   |-- ecb_rates_2025.csv
|   `-- extract_ecb_rates.py
|-- outputs/
|   |-- figures/
|   |-- gemma_eval/
|   |-- graphs/
|   `-- metrics/
|-- reports/
|   |-- audit_report.txt
|   |-- data_cleaning_summary.md
|   |-- gemma_causal_reasoning.txt
|   `-- high_correlation_pairs.csv
`-- scripts/
    |-- check_del_after.py
    `-- run_full_pipeline.ps1
```

## Quick Start

Create and populate the project-local virtual environment:

```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

The checked-in [requirements.txt](./requirements.txt) covers the core pipeline dependencies, including:

- `pandas`, `numpy`, `scipy`, `openpyxl`
- `causal-learn`, `lingam`, `gcastle`, `cdt`
- `networkx`, `matplotlib`, `pyvis`
- `torch`, `transformers`
- `pdfplumber`
- `rdflib`

Optional extras:

- Google Gemma evaluation currently needs `google-genai`
- Some `07_run_deci.py` paths may benefit from `causica`, but the script still has a fallback path when it is unavailable

Install the Google client only if you plan to use the Google backend:

```powershell
.\.venv\Scripts\python.exe -m pip install google-genai
```

## End-To-End Runs

Run the main pipeline from the repo root:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

Useful variants:

```powershell
.\.venv\Scripts\python.exe run_all.py --with-fx
.\.venv\Scripts\python.exe run_all.py --with-gemma --gemma-backend ollama
.\.venv\Scripts\python.exe run_all.py --with-gemma --gemma-backend google --gemma-model gemma-4-26b-a4b-it --gemma-api-key YOUR_KEY
.\.venv\Scripts\python.exe run_all.py --epochs 200 --mode both
```

Notes:

- [run_all.py](./run_all.py) automatically relaunches itself inside `.venv\Scripts\python.exe` when that interpreter exists
- `--with-fx` runs the exchange-rate helper flow before the main modeling pipeline
- `--with-gemma` adds the Gemma-backed steps
- The Hugging Face backend is supported by [08_gemma_causal_proposals.py](./08_gemma_causal_proposals.py), but [10_gemma_evaluate.py](./10_gemma_evaluate.py) currently supports only `ollama` and `google`, so `run_all.py` skips step 10 for `huggingface`

If you only want the non-LLM pipeline, the repo also includes:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1
```

## Pipeline Steps

- [01_audit.py](./01_audit.py): profiles the raw workbook and writes an audit report
- [02_clean.py](./02_clean.py): cleans, scores, coerces, and imputes the raw ESG data into modeling-ready numeric outputs, with a horizontal/anonymized cleaning summary
- [03_build_column_mapping.py](./03_build_column_mapping.py): creates the variable mapping used downstream
- [04_forbidden_edges.py](./04_forbidden_edges.py): builds ontology-inspired edge constraints
- [05_run_baselines.py](./05_run_baselines.py): runs PC and LiNGAM baselines
- [06_run_notears.py](./06_run_notears.py): runs NOTEARS
- [07_run_deci.py](./07_run_deci.py): runs DECI in constrained and unconstrained modes
- [08_gemma_causal_proposals.py](./08_gemma_causal_proposals.py): asks Gemma to propose plausible direct ESG causal edges
- [09_visualize_graphs.py](./09_visualize_graphs.py): produces high-resolution comparison figures, graph visuals, scatter diagnostics, and a data-correlation heatmap
- [10_gemma_evaluate.py](./10_gemma_evaluate.py): asks Gemma to qualitatively score discovered edges across the graph outputs

## Data And Processed Outputs

The repo is standardized around:

- raw input: [data/raw/df_asst_bnk_ecb.xlsx](./data/raw/df_asst_bnk_ecb.xlsx)

Main processed artifacts:

- [data/processed/data_clean.csv](./data/processed/data_clean.csv)
- [data/processed/data_ready.csv](./data/processed/data_ready.csv)
- [data/processed/column_mapping.csv](./data/processed/column_mapping.csv)
- [data/processed/df_asst_bnk_ecb_processed.xlsx](./data/processed/df_asst_bnk_ecb_processed.xlsx)

The current cleaner keeps a compact causal-ready feature set and converts mixed raw workbook fields into consistently numeric modeling inputs for the downstream graph algorithms.

### Cleaning Orientation

The data cleaning step is described as horizontal analysis: each row is one confidential bank/entity observation, and each column is an ESG or financial variable. Direct identifiers and administrative fields such as bank names, LEI/MFI codes, row IDs, and other metadata are excluded before causal discovery. The modeling dataset therefore uses anonymized row-level observations rather than identifiable bank names as causal variables.

The cleaner also writes [reports/data_cleaning_summary.md](./reports/data_cleaning_summary.md), which records the horizontal-analysis decision, excluded metadata columns, final data shape, and correlation-check method.

### Correlation Check

The correlation analysis is a redundancy diagnostic, not causal evidence. [02_clean.py](./02_clean.py) selects numeric columns, computes the Pearson correlation matrix, takes absolute correlations, and writes variable pairs above `HIGH_CORR_THRESHOLD` to [reports/high_correlation_pairs.csv](./reports/high_correlation_pairs.csv). Only exact duplicate columns with correlation equal to `1.0` are automatically dropped; high but non-perfect pairs are kept and reported for interpretation.

## Exchange-Rate Helper Workflow

The mixed-currency helper flow is kept separate from the main modeling pipeline so it can be rerun independently.

Step 1: extract ECB exchange rates from the PDF.

```powershell
.\.venv\Scripts\python.exe exchange_rates/extract_ecb_rates.py
```

Step 2: parse workbook money fields and write EUR-normalized helper columns.

```powershell
.\.venv\Scripts\python.exe check_del_after.py
```

Equivalent helper entry point:

```powershell
.\.venv\Scripts\python.exe scripts/check_del_after.py
```

Main exchange-rate artifacts:

- [exchange_rates/exchange_rate_2025.pdf](./exchange_rates/exchange_rate_2025.pdf)
- [exchange_rates/ecb_rates_2025.csv](./exchange_rates/ecb_rates_2025.csv)
- [data/processed/df_asst_bnk_ecb_processed.xlsx](./data/processed/df_asst_bnk_ecb_processed.xlsx)

## Gemma Workflows

### 08. Causal Edge Proposals

[08_gemma_causal_proposals.py](./08_gemma_causal_proposals.py) proposes direct ESG edges from variable descriptions and compares them with the algorithmic graphs.

Examples:

```powershell
.\.venv\Scripts\python.exe 08_gemma_causal_proposals.py --backend ollama
.\.venv\Scripts\python.exe 08_gemma_causal_proposals.py --backend google --api-key YOUR_KEY
```

Tracked outputs include:

- [outputs/graphs/gemma_proposed_adjacency.csv](./outputs/graphs/gemma_proposed_adjacency.csv)
- [outputs/graphs/gemma_proposed_edges.csv](./outputs/graphs/gemma_proposed_edges.csv)
- [outputs/graphs/gemma_proposed_graph.gml](./outputs/graphs/gemma_proposed_graph.gml)
- [reports/gemma_causal_reasoning.txt](./reports/gemma_causal_reasoning.txt)

### 10. Qualitative Edge Evaluation

[10_gemma_evaluate.py](./10_gemma_evaluate.py) evaluates discovered edges by asking Gemma for:

- whether any causal link is plausible
- the most plausible relationship direction
- a short mechanism
- likely confounders
- confidence

Examples:

```powershell
.\.venv\Scripts\python.exe 10_gemma_evaluate.py --backend ollama
.\.venv\Scripts\python.exe 10_gemma_evaluate.py --backend google --model gemma-4-26b-a4b-it --api-key YOUR_KEY
.\.venv\Scripts\python.exe 10_gemma_evaluate.py --backend google --model gemma-4-26b-a4b-it --no-cache --delay 4.5 --max-edges 10
```

If you prefer an environment variable for Google:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
.\.venv\Scripts\python.exe 10_gemma_evaluate.py --backend google --model gemma-4-26b-a4b-it
```

Tracked evaluation outputs:

- [outputs/gemma_eval/edge_scores.csv](./outputs/gemma_eval/edge_scores.csv)
- [outputs/gemma_eval/model_summary.csv](./outputs/gemma_eval/model_summary.csv)
- [outputs/gemma_eval/raw_responses.jsonl](./outputs/gemma_eval/raw_responses.jsonl)
- [outputs/gemma_eval/cache.json](./outputs/gemma_eval/cache.json)
- [outputs/gemma_eval/comparison.png](./outputs/gemma_eval/comparison.png)
- [outputs/gemma_eval/distribution.png](./outputs/gemma_eval/distribution.png)

This step is intended as a supplementary plausibility check, not as ground truth.

## Graphs, Figures, And Reports

Regenerate report-quality figures at the default 300 DPI:

```powershell
.\.venv\Scripts\python.exe 09_visualize_graphs.py
```

Useful variants:

```powershell
.\.venv\Scripts\python.exe 09_visualize_graphs.py --dpi 600
.\.venv\Scripts\python.exe 09_visualize_graphs.py --top-n 25 --subgraphs
.\.venv\Scripts\python.exe 09_visualize_graphs.py --skip-eda
```

Current tracked graph artifacts include:

- [outputs/graphs/unconstrained_pc_graph.gml](./outputs/graphs/unconstrained_pc_graph.gml)
- [outputs/graphs/constrained_pc_graph.gml](./outputs/graphs/constrained_pc_graph.gml)
- [outputs/graphs/unconstrained_lingam_graph.gml](./outputs/graphs/unconstrained_lingam_graph.gml)
- [outputs/graphs/notears_graph.gml](./outputs/graphs/notears_graph.gml)
- [outputs/graphs/deci_unconstrained_graph.gml](./outputs/graphs/deci_unconstrained_graph.gml)
- [outputs/graphs/deci_constrained_graph.gml](./outputs/graphs/deci_constrained_graph.gml)
- [outputs/graphs/gemma_proposed_graph.gml](./outputs/graphs/gemma_proposed_graph.gml)

Current tracked figures include:

- [outputs/figures/comparison_grid.png](./outputs/figures/comparison_grid.png)
- [outputs/figures/correlation_heatmap.png](./outputs/figures/correlation_heatmap.png)
- [outputs/figures/scatter_key_relationships.png](./outputs/figures/scatter_key_relationships.png)
- [outputs/figures/jaccard_heatmap.png](./outputs/figures/jaccard_heatmap.png)
- [outputs/figures/edge_count_comparison.png](./outputs/figures/edge_count_comparison.png)
- [outputs/figures/constraint_impact.png](./outputs/figures/constraint_impact.png)
- [outputs/figures/network_unconstrained_pc.png](./outputs/figures/network_unconstrained_pc.png)
- [outputs/figures/network_constrained_pc.png](./outputs/figures/network_constrained_pc.png)
- [outputs/figures/network_unconstrained_lingam.png](./outputs/figures/network_unconstrained_lingam.png)
- [outputs/figures/network_notears.png](./outputs/figures/network_notears.png)
- [outputs/figures/network_deci_unconstrained.png](./outputs/figures/network_deci_unconstrained.png)
- [outputs/figures/network_deci_constrained.png](./outputs/figures/network_deci_constrained.png)
- [outputs/figures/network_gemma_proposed.png](./outputs/figures/network_gemma_proposed.png)

Reports and diagnostics:

- [reports/audit_report.txt](./reports/audit_report.txt)
- [reports/data_cleaning_summary.md](./reports/data_cleaning_summary.md)
- [reports/high_correlation_pairs.csv](./reports/high_correlation_pairs.csv)
- [reports/gemma_causal_reasoning.txt](./reports/gemma_causal_reasoning.txt)
- [outputs/metrics/run_log.csv](./outputs/metrics/run_log.csv)

## Supporting Files

- [config.py](./config.py) is the single source of truth for default paths and output folders
- [docs/WEEK2_README.md](./docs/WEEK2_README.md) keeps the archived week-2 notes inside `docs/`
- [scripts/run_full_pipeline.ps1](./scripts/run_full_pipeline.ps1) is the quickest non-LLM rerun path
- [setup_venv.ps1](./setup_venv.ps1) bootstraps the local environment

## Notes And Current Limitations

- The Google Gemma backend requires `google-genai` and a valid API key
- The Hugging Face backend is available for proposal generation in step 08, but not for step 10 evaluation
- The FX-normalized workbook is currently a helper artifact; the main causal pipeline still defaults to the canonical raw workbook path from [config.py](./config.py)
- LLM-generated proposal and evaluation outputs should be treated as supporting evidence, not definitive causal truth
