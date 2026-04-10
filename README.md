# Ontology-Guided Hybrid Causal Discovery

This project builds an ontology-guided causal discovery pipeline for ESG-style tabular data.
It combines:

- dataset auditing and cleaning
- ontology-inspired forbidden and required edges
- multiple causal discovery backends
- graph export and visualization
- optional LLM-based causal edge proposals

The current default dataset is:

- `data/raw/df_asst_bnk_ecb.xlsx`

The current cleaner converts that workbook into a fully numeric causal-ready dataset with:

- `110` rows
- `12` modeled variables

## Current Pipeline

The project currently uses these scripts:

1. `01_audit.py`
2. `02_clean.py`
3. `03_build_column_mapping.py`
4. `04_forbidden_edges.py`
5. `05_run_baselines.py`
6. `06_run_notears.py`
7. `07_run_deci.py`
8. `08_gemma_causal_proposals.py`
9. `09_visualize_graphs.py`

## What Each Step Does

`01_audit.py`
- reads the raw dataset
- reports shape, dtypes, missingness, and categorical values
- writes `reports/audit_report.txt`

`02_clean.py`
- supports `.csv`, `.xlsx`, and `.xls`
- normalizes column names through `io_utils.py`
- removes metadata/admin columns
- coerces boolean-like and numeric-like object columns
- encodes qualitative ESG fields to ordinal numeric scores
- extracts numeric values from emissions, percentages, and financial-magnitude text
- imputes remaining numeric missing values with the median
- writes:
  - `data/processed/data_clean.csv`
  - `data/processed/data_ready.csv`
  - `reports/high_correlation_pairs.csv`

`03_build_column_mapping.py`
- creates or refreshes `data/processed/column_mapping.csv`
- fills known ontology/domain mappings
- flags unmapped columns for manual review

`04_forbidden_edges.py`
- defines ontology-guided forbidden and required edges
- exposes background knowledge for downstream constrained models

`05_run_baselines.py`
- runs:
  - unconstrained PC
  - unconstrained DirectLiNGAM
  - constrained PC
- writes adjacency CSVs, GML graphs, and updates `outputs/metrics/run_log.csv`

`06_run_notears.py`
- runs NOTEARS through gCastle
- writes adjacency, weights, and graph outputs

`07_run_deci.py`
- runs DECI in unconstrained and constrained modes
- currently falls back to a manual PyTorch implementation if `causica` is not installed
- writes adjacency, edge probabilities, and graph outputs

`08_gemma_causal_proposals.py`
- queries Gemma to propose causal edges from variable descriptions
- compares LLM-proposed edges to data-driven graphs
- optional and backend-dependent

`09_visualize_graphs.py`
- reads adjacency CSVs
- generates comparison figures in `outputs/figures`

## Current Modeled Variables

The current cleaned dataset keeps these 12 variables for causal discovery:

- `scope_1_ghg_emissions`
- `scope_2_ghg_emissions`
- `scope_3_ghg_emissions`
- `emission_reduction_policy`
- `renewable_energy_share`
- `community_investment`
- `diversity_women_representation`
- `health_safety`
- `board_strategy_esg_oversight`
- `sustainable_finance_green_financing`
- `total_revenue`
- `reporting_quality`

This means the current graphs now include Environmental, Social, Governance, and Financial variables rather than only the environmental subset.

## Repository Structure

Key files:

- [README.md](/c:/Users/User/Desktop/go/README.md)
- [config.py](/c:/Users/User/Desktop/go/config.py)
- [io_utils.py](/c:/Users/User/Desktop/go/io_utils.py)
- [forbidden_edges.py](/c:/Users/User/Desktop/go/forbidden_edges.py)
- [WEEK2_README.md](/c:/Users/User/Desktop/go/WEEK2_README.md)

Key output folders:

- `data/processed/`
- `reports/`
- `outputs/graphs/`
- `outputs/metrics/`
- `outputs/figures/`

## Setup

Base packages already listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Additional packages used by the current scripts but not listed in `requirements.txt` should also be installed:

```powershell
python -m pip install causal-learn lingam networkx matplotlib scikit-learn torch
```

Optional packages:

```powershell
python -m pip install pyvis
python -m pip install causica==0.4.5 --no-deps
```

Notes:

- `openpyxl` is required for `.xlsx` input
- `causica` is optional; without it, `07_run_deci.py` uses the built-in PyTorch fallback
- `08_gemma_causal_proposals.py` needs a working Ollama, Google AI, or HuggingFace backend

## How To Run

Run everything from the project root:

```powershell
python 01_audit.py
python 02_clean.py
python 03_build_column_mapping.py
python 04_forbidden_edges.py
python 05_run_baselines.py
python 06_run_notears.py
python 07_run_deci.py --epochs 200 --mode both
python 09_visualize_graphs.py
```

If your system uses `py` instead of `python`, replace accordingly.

## Current Default Data Paths

Configured in [config.py](/c:/Users/User/Desktop/go/config.py):

- raw input: `data/raw/df_asst_bnk_ecb.xlsx`
- clean output: `data/processed/data_clean.csv`
- causal-ready output: `data/processed/data_ready.csv`
- mapping file: `data/processed/column_mapping.csv`
- audit report: `reports/audit_report.txt`
- high-correlation report: `reports/high_correlation_pairs.csv`

## Main Outputs

Processed data:

- `data/processed/data_clean.csv`
- `data/processed/data_ready.csv`
- `data/processed/column_mapping.csv`

Reports:

- `reports/audit_report.txt`
- `reports/high_correlation_pairs.csv`

Graphs:

- `outputs/graphs/unconstrained_pc_adjacency.csv`
- `outputs/graphs/unconstrained_pc_graph.gml`
- `outputs/graphs/unconstrained_lingam_adjacency.csv`
- `outputs/graphs/unconstrained_lingam_weights.csv`
- `outputs/graphs/unconstrained_lingam_graph.gml`
- `outputs/graphs/constrained_pc_adjacency.csv`
- `outputs/graphs/constrained_pc_graph.gml`
- `outputs/graphs/notears_adjacency.csv`
- `outputs/graphs/notears_weights.csv`
- `outputs/graphs/notears_graph.gml`
- `outputs/graphs/deci_unconstrained_adjacency.csv`
- `outputs/graphs/deci_unconstrained_edge_probabilities.csv`
- `outputs/graphs/deci_unconstrained_graph.gml`
- `outputs/graphs/deci_constrained_adjacency.csv`
- `outputs/graphs/deci_constrained_edge_probabilities.csv`
- `outputs/graphs/deci_constrained_graph.gml`

Metrics and figures:

- `outputs/metrics/run_log.csv`
- `outputs/figures/comparison_grid.png`
- `outputs/figures/jaccard_heatmap.png`
- `outputs/figures/edge_count_comparison.png`
- `outputs/figures/constraint_impact.png`

## Latest Full Run Snapshot

Latest full rerun on the current 12-variable dataset produced:

- Unconstrained PC: `16` edges
- Constrained PC: `12` edges
- DirectLiNGAM: `9` edges
- NOTEARS: `7` edges
- DECI unconstrained: `62` edges
- DECI constrained: `66` edges

The constrained PC run removed `4` edges relative to unconstrained PC on the latest rerun.

## Constraints

The constrained models use ontology-guided prior knowledge from:

- [04_forbidden_edges.py](/c:/Users/User/Desktop/go/04_forbidden_edges.py)
- [forbidden_edges.py](/c:/Users/User/Desktop/go/forbidden_edges.py)

These currently encode domain-informed rules such as:

- emissions should not cause policy adoption
- emissions should not drive board-level strategy
- reporting quality should not cause core ESG performance variables
- board ESG oversight should drive policy, reporting, and diversity outcomes

## Current Known Limitations

`08_gemma_causal_proposals.py`
- is not fully runnable by default in this environment yet
- the latest attempt reached the Ollama endpoint but returned `HTTP 404`
- that usually means the Ollama service is up but the requested model tag is not available locally

`07_run_deci.py`
- currently runs through the built-in PyTorch fallback unless `causica` is installed
- the fallback is useful for experimentation, but it is not the official Causica backend

Financial text extraction
- `02_clean.py` now keeps the finance-like text columns by extracting numeric magnitudes
- mixed currencies are currently treated as comparable magnitudes, not normalized to a common currency

## Suggested Next Steps

- normalize currency units if cross-country financial comparability matters
- refine `08_gemma_causal_proposals.py` backend handling and model selection
- evaluate whether the constrained DECI behavior should be tightened, since it is currently denser than unconstrained DECI
- decide whether generated outputs in `outputs/` should remain version-controlled long term

## Git / Version Notes

The repo currently tracks code plus selected generated outputs and figures.
The raw input workbook itself is still ignored by `.gitignore`.
