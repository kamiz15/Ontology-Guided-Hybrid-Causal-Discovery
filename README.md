# Ontology-Guided Hybrid Causal Discovery For ESG Data

This repository contains the experiment pipeline for a thesis on
ontology-guided hybrid causal discovery for ESG and finance data. The project
compares unconstrained causal discovery with ontology-constrained variants
across an advisor-provided dummy dataset and a real ECB case-study dataset.

The current thesis-safe framing is:

- The advisor dummy dataset is the official dummy-data experiment.
- No explicit advisor-provided causal generation DAG/rule file was found.
- Advisor-dummy F1 and SHD are evaluated against an
  ontology-derived reference DAG, not an experimentally known causal mechanism.
- Real ECB results are a case study only; they report literature alignment,
  edge counts, stability, and forbidden-edge violations, not F1/SHD.

## Current Inputs

Advisor dummy files:

```text
data/advisor_dummy/ESG-Finance_dummy_data.csv
data/advisor_dummy/ESG-Finance_Metadata.xlsx
data/advisor_dummy/Dummy_dataset_ESG.txt
```

Real ECB modeling data:

```text
data/processed/data_ready.csv
```

The original raw ECB workbook workflow is still present, but the final
advisor-dummy experiment uses the advisor-provided CSV and metadata directly.

## Main Algorithms

The current experiment runner supports:

- PC: constraint-based, uses causal-learn background knowledge where available.
- LiNGAM: functional causal model, constraints applied by post-processing.
- NOTEARS: continuous optimization, constraints applied by post-processing
  because gCastle `Notears.learn()` ignores prior-knowledge kwargs.
- GES: classical score-based baseline through causal-learn GES, constraints
  applied by post-processing and labeled `ges_postproc`.
- DECI/Causica: deep generative baseline, native Causica constraints where
  supported; treated as exploratory because of data-size and runtime
  sensitivity.

## Constraint Modes

For `advisor_dummy`, the important constraint modes are:

- `none`: unconstrained baseline.
- `forbidden_only`: main thesis-safe constrained setting.
- `required_light`: secondary setting with a small number of high-confidence
  definitional required edges.
- `full_reference_sanity`: appendix-only constraint-enforcement check where
  required edges equal the reference DAG. This must not be presented as
  independent discovery evidence.

The main comparison is `none` versus `forbidden_only`.

## Key Scripts

```text
scripts/experiments/05_build_reference_dag_from_dummy.py     Build advisor-dummy registry, cleaned data, reference DAG, constraints
scripts/experiments/08_run_ges.py                            Run causal-learn GES and post-process constraints
scripts/figures/09_make_final_figures.py                 Generate final thesis figures and captions
scripts/constraints/14_constraint_adapter.py                 Convert shared constraints into algorithm-specific formats
scripts/evaluation/20_create_final_result_tables.py         Create final consolidated result tables and markdown summary
scripts/evaluation/21_significance_tests.py                 Wilcoxon signed-rank tests: constrained vs unconstrained per algorithm
scripts/experiments/22_snr_sensitivity_sweep.py              SNR sensitivity sweep on causal dummy v2 data (known ground-truth DAG)
scripts/experiments/generate_causal_dummy.py                 Generate ESG dummy data with actual structural equations (causal v2)
scripts/experiments/run_all.py                               Main experiment runner
scripts/experiments/deci_ablation.py                         DECI ablation, diagnostics, selected-config runs
scripts/experiments/07_run_deci.py                           DECI/Causica runner and fallback path
```

Legacy and supporting scripts for cleaning, Gemma checks, synthetic data,
and visualizations are still included in the repository.

## Quick Start

Create and activate the project-local virtual environment:

```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Rebuild Advisor-Dummy Reference Artifacts

Build the cleaned advisor-dummy modeling matrix, metadata registry,
ontology-derived reference DAG, and constraint files:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/05_build_reference_dag_from_dummy.py
```

This writes the main advisor-dummy artifacts:

```text
data/processed/advisor_dummy_ready.csv
outputs/experiments/advisor_dummy_data_audit.csv
outputs/experiments/advisor_dummy_metadata_registry.csv
outputs/experiments/advisor_dummy_reference_dag_edges.csv
outputs/experiments/advisor_dummy_reference_dag_adjacency.csv
outputs/experiments/advisor_dummy_reference_dag_validation.md
outputs/experiments/advisor_dummy_constraints_forbidden.csv
outputs/experiments/advisor_dummy_constraints_required_light.csv
outputs/experiments/advisor_dummy_constraints_full_reference_required.csv
```

The builder does not silently regenerate the advisor dummy data. Development
fallback from `Dummy_dataset_ESG.txt` is only allowed with:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/05_build_reference_dag_from_dummy.py --allow-dummy-regeneration
```

## Run Experiments

Advisor dummy, main constrained setting:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset advisor_dummy --algorithms pc,notears,lingam,ges,deci --constraint-mode forbidden_only
```

Advisor dummy, constraint ablation:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset advisor_dummy --algorithms pc,notears,lingam,ges,deci --constraint-mode none
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset advisor_dummy --algorithms pc,notears,lingam,ges,deci --constraint-mode forbidden_only
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset advisor_dummy --algorithms pc,notears,lingam,ges,deci --constraint-mode required_light
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset advisor_dummy --algorithms pc,notears,lingam,ges,deci --constraint-mode full_reference_sanity
```

Real ECB case-study runs:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset real --algorithms pc,notears,lingam,ges --skip-deci
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --only-deci --deci-selected-only --dataset real
```

Focused GES runs:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset advisor_dummy --algorithm ges --constraint-mode forbidden_only
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --dataset real --algorithm ges
```

DECI ablation and Linux helper:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --only-deci --deci-ablation --dataset synthetic
.\.venv\Scripts\python.exe scripts/experiments/run_all.py --only-deci --deci-selected-only --dataset real
```

For Linux/WSL:

```bash
bash run_deci_linux.sh
```

## Final Result Tables

Generate thesis-ready consolidated tables:

```powershell
.\.venv\Scripts\python.exe scripts/evaluation/20_create_final_result_tables.py
```

Main outputs:

```text
outputs/experiments/final_algorithm_comparison_advisor_dummy.csv
outputs/experiments/final_algorithm_comparison_advisor_dummy_sanity_check.csv
outputs/experiments/final_algorithm_comparison_real_ecb.csv
outputs/experiments/final_results_summary.md
```

Advisor dummy headline results, main `forbidden_only` comparison:

```text
PC:      violations 21.80 -> 0.00, F1 0.0127 -> 0.0552
LiNGAM:  violations 0.80 -> 0.00, F1 0.0000 -> 0.0000
NOTEARS: violations 0.00 -> 0.00, F1 0.0000 -> 0.0000
GES:     violations 4.80 -> 0.00, F1 0.0105 -> 0.0110
DECI:    violations 3.40 -> 0.00, F1 0.0180 -> 0.0110
```

Real ECB case-study headline results:

```text
PC:      alignment 0.000 -> 0.333, violations 0.333 -> 0.000
LiNGAM:  alignment 0.400 -> 0.400, violations 0.200 -> 0.000
NOTEARS: alignment 0.000 -> 0.000, violations 0.200 -> 0.000
GES:     alignment 0.300 -> 0.400, violations 0.600 -> 0.000
DECI:    alignment 0.000 -> 0.000, violations 0.000 -> 0.000
```

## Final Thesis Figures

Generate final figures and captions:

```powershell
.\.venv\Scripts\python.exe scripts/figures/09_make_final_figures.py
```

Outputs:

```text
outputs/figures/pipeline_overview.png
outputs/figures/constraint_pipeline.png
outputs/figures/advisor_dummy_f1_forbidden_only.png
outputs/figures/advisor_dummy_shd_forbidden_only.png
outputs/figures/advisor_dummy_violations_forbidden_only.png
outputs/figures/real_ecb_graph_selected.png
outputs/figures/real_ecb_stability.png
outputs/figures/full_reference_sanity_appendix.png
outputs/figures/figure_captions.md
```

The selected real ECB case-study graph uses GES forbidden-only because PC
graph CSVs were not available in `outputs/experiments/graphs/`.

## Current Final Interpretation

The main result is constraint compliance: forbidden-only constraints eliminate
or reduce ontology violations across the main algorithm set.

The clearest advisor-dummy reference-DAG alignment improvement is PC:

```text
F1 0.0127 -> 0.0552
SHD 127.60 -> 124.20
violations 21.80 -> 0.00
```

GES is a useful classical score-based baseline. It improves constraint
compliance, but reference-DAG F1 remains low, suggesting that the advisor
dummy data may not contain strong statistical signal matching the
ontology-derived reference DAG.

DECI native constraints are promising in secondary and sanity-check settings,
but DECI remains sensitive to data size, thresholding, and configuration. It
is treated as exploratory rather than the main thesis claim.

Real ECB results are case-study results only. The real dataset has no
reference DAG, so the project does not report real-data F1 or SHD.

## Project Layout

```text
go/
|-- scripts/experiments/05_build_reference_dag_from_dummy.py
|-- scripts/experiments/07_run_deci.py
|-- scripts/experiments/08_run_ges.py
|-- scripts/figures/09_make_final_figures.py
|-- scripts/constraints/14_constraint_adapter.py
|-- scripts/evaluation/20_create_final_result_tables.py
|-- scripts/experiments/deci_ablation.py
|-- scripts/experiments/run_all.py
|-- run_deci_linux.sh
|-- scripts/core/config.py
|-- data/
|   |-- advisor_dummy/
|   |   |-- ESG-Finance_dummy_data.csv
|   |   |-- ESG-Finance_Metadata.xlsx
|   |   `-- Dummy_dataset_ESG.txt
|   |-- processed/
|   |   |-- advisor_dummy_ready.csv
|   |   `-- data_ready.csv
|   `-- synthetic/
|-- outputs/
|   |-- experiments/
|   |-- figures/
|   |-- graphs/
|   `-- metrics/
|-- reports/
|-- docs/
`-- scripts/
```

## Important Output Files

Advisor dummy and reference-DAG artifacts:

```text
outputs/experiments/advisor_dummy_data_audit.csv
outputs/experiments/advisor_dummy_metadata_registry.csv
outputs/experiments/advisor_dummy_reference_dag_edges.csv
outputs/experiments/advisor_dummy_reference_dag_validation.md
outputs/experiments/advisor_dummy_constraint_ablation_summary.csv
outputs/experiments/advisor_dummy_final_report.md
```

Final thesis artifacts:

```text
outputs/experiments/final_algorithm_comparison_advisor_dummy.csv
outputs/experiments/final_algorithm_comparison_real_ecb.csv
outputs/experiments/final_results_summary.md
outputs/figures/figure_captions.md
```

DECI diagnostics:

```text
outputs/experiments/deci_diagnostics.csv
outputs/experiments/deci_threshold_sweep.csv
outputs/experiments/deci_stable_edges.csv
outputs/experiments/deci_real_selected_config.csv
outputs/experiments/deci_report.md
```

GES outputs:

```text
outputs/experiments/graphs/ges_<dataset>_<constraint_label>_seed<S>.csv
outputs/experiments/ges_stable_edges.csv
```

Significance test outputs:

```text
outputs/experiments/significance_tests.csv
outputs/experiments/significance_tests.md
```

SNR sensitivity sweep outputs:

```text
outputs/experiments/snr_sweep_results.csv
outputs/figures/snr_f1_sweep.png
outputs/figures/snr_shd_sweep.png
```

## Causal Dummy Dataset (v2)

`scripts/experiments/generate_causal_dummy.py` generates `ESG-Finance_dummy_data_causal_v2.csv`:
a version of the advisor dummy dataset where 46 variables are produced from
structural causal equations that match the ontology-derived reference DAG.

This is distinct from the original advisor CSV, which is randomly generated
with no causal structure. The v2 dataset has a known ground-truth DAG, making
F1 and SHD interpretable as causal recovery rather than ontology alignment.

```powershell
# Default: n=3000, seed=42, snr=0.6
.\.venv\Scripts\python.exe scripts/experiments/generate_causal_dummy.py

# Weaker signal (harder for algorithms)
.\.venv\Scripts\python.exe scripts/experiments/generate_causal_dummy.py --snr 0.3

# Without data quality injection (for controlled experiments)
.\.venv\Scripts\python.exe scripts/experiments/generate_causal_dummy.py --snr 0.6
```

Generated files:

```text
data/advisor_dummy/ESG-Finance_dummy_data_causal_v2.csv
data/advisor_dummy/causal_dummy_ground_truth_dag.csv
data/advisor_dummy/causal_dummy_ground_truth_edges.csv
```

## Statistical Significance Tests

`scripts/evaluation/21_significance_tests.py` runs one-sided paired Wilcoxon signed-rank tests
on the advisor_dummy `forbidden_only` constraint condition. Tests whether
constrained F1 > unconstrained F1 and constrained SHD < unconstrained SHD.

```powershell
.\.venv\Scripts\python.exe scripts/evaluation/21_significance_tests.py
```

Key results (n=5 seeds, forbidden_only, advisor_dummy):

```text
PC:     Delta F1 = +0.043, p = 0.031 *    Delta SHD = -3.4, p = 0.031 *
GES:    Delta F1 = +0.000, p = n/a         Delta SHD = -4.8, p = 0.031 *
LiNGAM: Delta F1 = 0.000, p = n/a         Delta SHD = -0.8, p = n/a
NOTEARS: Delta F1 = 0.000, p = n/a        Delta SHD = 0.0,  p = n/a
DECI:   Delta F1 = -0.007, p = 0.750      Delta SHD = +9.4, p = 0.844
```

Note: with n=5, the minimum achievable p-value is 1/32 â‰ˆ 0.031. PC and GES
show statistically significant SHD improvement. LiNGAM and NOTEARS are
untestable because all paired differences are zero (no signal in the data).

## SNR Sensitivity Sweep

`scripts/experiments/22_snr_sensitivity_sweep.py` tests the thesis claim that ontology-guided
constraints improve causal recovery more at low SNR than at high SNR. Uses
the causal v2 data (known ground-truth DAG), no data quality injection.

```powershell
# Full sweep (5 SNR values, 3 seeds, PC + LiNGAM + GES)
.\.venv\Scripts\python.exe scripts/experiments/22_snr_sensitivity_sweep.py

# Custom grid
.\.venv\Scripts\python.exe scripts/experiments/22_snr_sensitivity_sweep.py --snr-grid 0.2,0.5,0.8 --seeds 42,43
```

Sweep results (mean F1, 3 seeds, known ground-truth DAG):

```text
SNR   PC unconstrained  PC constrained  LiNGAM unc  LiNGAM con  GES unc  GES con
0.2        0.176            0.692          0.661       0.706       0.534    0.606
0.4        0.318            0.809          0.691       0.731       0.531    0.599
0.6        0.424            0.836          0.609       0.649       0.523    0.582
0.8        0.419            0.869          0.416       0.482       0.563    0.616
1.0        0.501            0.895          0.436       0.506       0.561    0.617
```

Constrained consistently outperforms unconstrained at all SNR levels for all
three algorithms. The gap is largest at low SNR (PC: +0.516 at SNR=0.2) and
narrows at high SNR (PC: +0.394 at SNR=1.0), supporting the thesis claim that
constraints are most valuable in the low-data, high-noise ESG regime.

## Thesis-Safe Wording

Use:

- ontology-derived reference DAG
- reference-DAG alignment
- ontology violations
- constraint compliance
- real-data case study

Avoid stronger causal-recovery wording for advisor-dummy or real ECB results
unless an explicit advisor-provided causal rule file is later added. In
particular, do not describe real-data results with F1/SHD.

## Notes And Limitations

- The advisor dummy dataset is randomly generated with injected data quality
  issues and no explicit causal equations. The reference DAG is reconstructed
  from metadata and conservative ontology/domain rules. F1/SHD on this dataset
  measure ontology alignment, not causal recovery.
- The causal v2 dataset (scripts/experiments/generate_causal_dummy.py) has a known ground-truth DAG
  and is the correct dataset for claiming causal structure recovery.
- `full_reference_sanity` is useful only as a constraint-enforcement check.
- NOTEARS and GES constrained variants are post-processed, not native
  constraint-aware learning.
- Real ECB has a small sample size and no reference graph. Results are
  descriptive case-study evidence.
- PC can fail on some real-data bootstrap seeds due to singular covariance in
  Fisher-Z tests; failures are recorded rather than hidden.
- Significance tests use n=5 seeds; the minimum achievable p-value is 0.031.
  Results should be interpreted with this limitation in mind.
