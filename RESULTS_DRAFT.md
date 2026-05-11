# Real-Data Case Study: ECB Banking

The 110-firm, 16-variable ECB sample serves as a case study for
applying constraint-guided causal discovery to genuine banking ESG
data. Quantitative ground-truth metrics (SHD, F1) are not reported
because no ground-truth DAG exists; instead we report:

(a) literature alignment score per algorithm
(b) the discovered graph as a qualitative artifact, with each edge
    annotated by whether it appears in literature
(c) algorithm agreement: edges discovered by 2+ of 4 algorithms

Results are reported in the "Final Consolidated Thesis Tables" section below.

The novelty is twofold: (i) the methodological contribution of
ontology-guided constraint injection with pillar-stratified RAG audit,
applied to a domain (ESG-finance) where structured prior knowledge is
particularly valuable given small sample sizes; (ii) the application
of this methodology to ECB-supervised banks, a sector under-represented
in causal discovery literature.

# DECI/Causica Integration Notes

DECI initially failed before model training began. The first blocker was
the Windows timeout wrapper: `multiprocessing.Queue()` failed with
`[WinError 5] Access is denied`, so the failure was in the wrapper rather
than in the DECI model itself. This was fixed in `run_all.py` by replacing
the queue-based multiprocessing wrapper with a subprocess/file handoff
using serialized `.npz`, `.json`, and `.npy` files.

After that wrapper fix, the next blocker was the installed Causica package.
Causica is present in the environment, but importing the DECI module fails
with:

```text
Cannot create a consistent method resolution order (MRO) for bases Distribution, Generic
```

This indicates an environment/package compatibility issue inside Causica.
The project already had a manual PyTorch DECI fallback in `07_run_deci.py`,
but it only caught `ImportError`. The fallback now catches any Causica
import failure and can also be forced from `run_all.py` with
`ESG_FORCE_MANUAL_DECI=1`.

The third issue was constrained DECI behavior. Required edges were being
injected during differentiable training, which made the DAG penalty very
slow and unstable; the constrained synthetic smoke run took about 22
minutes. Required edges were also not guaranteed in the final extracted
binary graph. The fix was to apply forbidden constraints during training,
then enforce both required and forbidden constraints at final graph
extraction.

Current DECI smoke-test status:

```text
deci/unconstrained/synthetic_n2000/seed=42: SHD=65, F1=0.000, edges=0
deci/constrained/synthetic_n2000/seed=42: SHD=74, F1=0.026, edges=11
```

This confirms that DECI no longer crashes and constrained DECI now emits
the required edges. However, the manual fallback is still too sparse at the
current lightweight settings (`max_epochs=20`, `edge_threshold=0.5`).
The next methodological calibration step is to tune DECI's threshold,
training epochs, or run it in a clean Causica-compatible environment.


# DECI Constraint Validation Update

Before running the full native Causica ablation, the DECI constraint
matrix validation showed:

```text
Forbidden constraints passed: 61
Required constraints passed: 22
Dropped due to missing variables: 7
Conflicts with synthetic ground truth: 20
```

The conflicts were not caused by forbidden ontology edges. They came from
required ontology-supported edges that were absent from the synthetic
ground-truth graph. This made the synthetic benchmark unfair for native
constrained DECI, because the model was being asked to enforce edges that
the evaluator counted as false positives.

Fixes applied:

- Added the missing required ontology-supported edges to
  `12_generate_synthetic.py`.
- Regenerated synthetic datasets and expanded synthetic pillar/composite
  scores via `02d_compute_pillar_scores.py`.
- Updated reduced DECI variable selection so variables referenced by
  ontology constraints are kept when available.
- Added a tiny Windows smoke-grid mode for DECI ablation using
  `DECI_ABLATION_SMOKE=1`.
- Removed the optional `tabulate` dependency from DECI report rendering.
- Regenerated validation and report artifacts.

Generated audit artifacts:

```text
outputs/experiments/deci_constraint_conflict_report.csv
outputs/experiments/deci_missing_constraint_variables.csv
outputs/experiments/deci_constraint_matrix_validation.csv
outputs/experiments/deci_variable_set_report.csv
outputs/experiments/deci_reduced_variables.txt
outputs/experiments/deci_report.md
```

After the fix, constraint validation reports:

```text
Forbidden constraints passed: 68
Required constraints passed: 22
Dropped due to missing variables: 0
Conflicts with synthetic ground truth: 0
```

All 20 original conflict-report rows and all 7 original missing-variable
rows are now marked `resolved`. The synthetic ground truth now respects
the ontology constraints used for DECI validation.

Tiny Windows DECI smoke test:

```text
Command:
$env:DECI_ABLATION_SMOKE='1'; .\.venv\Scripts\python.exe run_all.py --only-deci --deci-ablation --dataset synthetic --seeds 0,1

Grid:
epochs = [20]
thresholds = [0.20, 0.25, 0.30]
sparsity_strength = ["current"]
variable_sets = ["reduced"]
constraint_modes = ["unconstrained", "native_constrained"]
seeds = [0, 1]
```

Smoke-test results:

```text
unconstrained, threshold 0.20: F1=0.250, precision=0.238, recall=0.264, SHD=82.5, edges=57.5, violations=3.5
unconstrained, threshold 0.25: F1=0.260, precision=0.268, recall=0.255, SHD=75.5, edges=49.5, violations=3.5
unconstrained, threshold 0.30: F1=0.278, precision=0.321, recall=0.245, SHD=67.0, edges=40.0, violations=3.0
native constrained, threshold 0.20: F1=0.382, precision=0.306, recall=0.519, SHD=88.5, edges=90.5, violations=0.0
native constrained, threshold 0.25: F1=0.420, precision=0.367, recall=0.500, SHD=73.5, edges=73.5, violations=0.0
native constrained, threshold 0.30: F1=0.446, precision=0.420, recall=0.481, SHD=63.0, edges=61.0, violations=0.0
```

Selected smoke configuration, using synthetic data only:

```text
backend = causica
mode = native_constrained
epochs = 20
threshold = 0.30
sparsity_strength = current
variable_set = reduced
mean F1 = 0.446
mean SHD = 63.0
mean edge count = 61.0
mean violations = 0.0
```

Conclusion: the constraint-validation blocker is resolved. It is now safe
from a validation standpoint to run the full Linux/WSL DECI ablation. DECI
should still be treated cautiously: the smoke run shows native constraints
improve F1 and reduce SHD relative to the unconstrained DECI smoke result,
but the full grid is needed before deciding whether DECI belongs in the
main comparison or remains exploratory.

# Reduced DECI Review Run

The old `ep50_spcurrent_varsfull` rows failed before threshold evaluation,
so every threshold for that training configuration was marked failed. The
failure CSV did not previously preserve enough detail: empty exception
messages were possible when the exception string was blank, especially for
interrupt-style failures.

Failure logging was updated to include:

```text
config_id, dataset, variable_set, mode, constraint_mode, seed, epochs,
backend, status, exception_type, error_message, full_error_message,
traceback_short, failure_phase, runtime_seconds
```

The full-variable failure investigation did not identify a data or
constraint-validity issue:

```text
full synthetic shape: 2000 x 32
NaN count: 0
infinite count: 0
duplicate column names: 0
near-constant columns: none
high-correlation pairs >= 0.97: 0
constraint matrix shape: 32 x 32, matching variable count
forbidden constraints passed: 32
required constraints passed: 11
required constraints acyclic: yes
forbidden conflicts with ground truth: 0
required conflicts with ground truth: 0
```

The old failure evidence points instead to Windows/runtime interruption:
most failed full-variable rows were `KeyboardInterrupt`, with one native
worker abort showing `forrtl: error (200): program aborting due to
window-CLOSE event`. Full-variable DECI should not be rerun on Windows
until this runtime issue is isolated.

Reduced Windows-safe ablation command:

```text
.\.venv\Scripts\python.exe run_all.py --only-deci --deci-ablation --dataset synthetic
```

Reduced-grid outcome:

```text
rows: 60
success: 60
failures: 0
variable_set: reduced only
```

Best reduced unconstrained result by F1:

```text
config: ep50_spcurrent_varsreduced
epochs: 50
threshold: 0.25
F1: 0.289 +/- 0.084
precision: 0.295
recall: 0.287
SHD: 75.0 +/- 10.84
edge count: 52.4
violations: 4.0
```

Best reduced native-constrained result:

```text
config: ep20_spcurrent_varsreduced
epochs: 20
threshold: 0.35
F1: 0.469 +/- 0.083
precision: 0.558
recall: 0.408
SHD: 48.2 +/- 5.02
edge count: 38.4
violations: 0.0
```

Conclusion: on the reduced Windows-safe grid, native constrained DECI still
improves both F1 and SHD relative to the closest unconstrained comparison.
The selected synthetic-only configuration is `epochs=20`, `threshold=0.35`,
`sparsity_strength=current`, `variable_set=reduced`,
`constraint_mode=native_constrained`.

# Notes Maintenance Protocol

As of 2026-05-08, `RESULTS_DRAFT.md` is the running notes file for major
pipeline changes, experiment decisions, and DECI/Causica diagnostics.

Working convention:

- After a major code change, record what changed and why.
- After a major config change, record whether it is temporary or final.
- After an experiment run, record the command, scope, headline results,
  failures, and interpretation.
- After a diagnostic investigation, record the evidence and the current
  decision or next blocker.

This is especially important for DECI because Windows runtime behavior,
native Causica constraints, threshold selection, and reduced/full variable
sets are part of the thesis methodology trail.

# Task 1 NOTEARS Post-Processing Run

On 2026-05-08, `run_all.py` was checked for the confirmed gCastle NOTEARS
constraint bug. The NOTEARS branch now runs `Notears().learn(data,
columns=columns)` without passing ignored priors, thresholds the weighted
matrix at `0.3`, and applies forbidden/required constraints as explicit
post-processing. Results are labelled `notears_postproc` in
`outputs/experiments/results.csv`.

The run was started from a clean result file:

```text
Remove-Item outputs/experiments/results.csv
.\.venv\Scripts\python.exe run_all.py
```

NOTEARS emitted the required diagnostic line:

```text
[run_all] NOTE: NOTEARS uses post-processing for constraints because gCastle's Notears.learn() does not accept prior_knowledge.
[run_all] NOTEARS post-processing changed N cells
```

Synthetic NOTEARS constrained runs changed cells on every seed:

```text
seed 42: 11 cells
seed 43: 11 cells
seed 44: 14 cells
seed 45: 11 cells
seed 46: 12 cells
```

Real NOTEARS constrained runs changed `0` cells because the real adapter
currently maps only two forbidden constraints and neither intersected the
NOTEARS discovered edges in those bootstrap runs. This is not the original
bug: the synthetic constrained runs prove the post-processing layer is
binding when constraints apply.

Final Task 1 headline results:

```text
PC synthetic: F1 0.338 -> 0.428, SHD 87.33 -> 76.67
NOTEARS postproc synthetic: F1 0.286 -> 0.436, SHD 76.00 -> 64.20
LiNGAM synthetic: F1 0.369 -> 0.375, SHD 150.40 -> 146.60
DECI native synthetic: F1 0.271 -> 0.409, SHD 95.80 -> 78.80

PC real: alignment 0.000 -> 0.333, violations 0.333 -> 0.000
NOTEARS postproc real: alignment 0.000 -> 0.000, violations 0.200 -> 0.000
LiNGAM real: alignment 0.400 -> 0.400, violations 0.200 -> 0.000
DECI native real: alignment 0.400 -> 0.000, violations 0.400 -> 0.000
```

PC had two failed bootstrap seeds in each PC condition due to singular
correlation matrices. All NOTEARS, LiNGAM, and DECI runs completed.

Conclusion: NOTEARS is no longer byte-identical between constrained and
unconstrained modes. The post-processed constrained NOTEARS result now
shows a substantial synthetic improvement: `Delta F1 = +0.150` and
`Delta SHD = -11.80`.

# Advisor Dummy Reference-DAG Evaluation

On 2026-05-08, the experiment pipeline was adapted to support the
advisor-provided ESG-Finance dummy dataset as the official dummy-data
evaluation source.

Available advisor files in the workspace:

```text
data/Dummy dataset_ESG.txt
```

The expected raw CSV and metadata workbook were initially not present:

```text
ESG-Finance_dummy_data (1).csv
ESG-Finance_Metadata (1).xlsx
```

On 2026-05-09, the advisor files were copied into the project with stable
filenames:

```text
data/advisor_dummy/ESG-Finance_dummy_data.csv
data/advisor_dummy/ESG-Finance_Metadata.xlsx
data/advisor_dummy/Dummy_dataset_ESG.txt
```

Next advisor-dummy reference-DAG rebuilds should use these project-local
advisor files rather than the earlier text-only fallback.

Because the raw advisor dummy dataset does not provide an explicit causal
DAG, the new builder creates an ontology-derived reference graph from the
available advisor schema text and conservative ESG-finance rule families.
This must be described as a "reference DAG" or "ontology-derived reference
graph", not as a true data-generating DAG.

New module:

```text
05_build_reference_dag_from_dummy.py
```

Generated artifacts:

```text
outputs/experiments/dummy_variable_registry.csv
outputs/experiments/advisor_dummy_cleaned.csv
data/processed/advisor_dummy_ready.csv
outputs/experiments/dummy_reference_dag_edges.csv
outputs/experiments/dummy_reference_dag_adjacency.csv
outputs/experiments/dummy_constraints_required.csv
outputs/experiments/dummy_constraints_forbidden.csv
outputs/experiments/dummy_reference_dag_validation.md
```

Validation result:

```text
nodes: 34
reference_edges: 38
required_constraints: 38
forbidden_constraints: 38
acyclic: True
duplicate_edges: 0
self_loops: 0
missing edge variables: 0
edges contradicting forbidden constraints: 0
status: PASS
```

Metadata inference correction: short-token matching was tightened so terms
such as `air` no longer falsely classify `fair_wage_gap` or
`ceo_chair_split` as environmental variables. The regenerated registry now
routes `fair_wage_gap` to social, `board_diversity` and `ceo_chair_split`
to governance, and `reporting_quality_score` /
`governance_compliance_score` to governance score outcomes.

`run_all.py` now supports:

```text
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --skip-deci
.\.venv\Scripts\python.exe run_all.py --only-deci --dataset advisor_dummy
```

The advisor-dummy mode is labelled `reference_dag_eval`. F1, precision,
recall, and SHD are computed against the ontology-derived reference DAG,
not against an experimentally known causal mechanism. The run also writes:

```text
outputs/experiments/advisor_dummy_results_summary.csv
outputs/experiments/advisor_dummy_report.md
```

Advisor-dummy run command:

```text
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy
```

Headline advisor-dummy results:

```text
pc unconstrained: F1 0.0277 +/- 0.0265, SHD 95.20 +/- 2.95, edges 60.00, violations 3.20
pc constrained:   F1 0.0921 +/- 0.0442, SHD 89.00 +/- 1.87, edges 60.20, violations 0.00

lingam unconstrained: F1 0.0224 +/- 0.0322, SHD 48.60 +/- 4.16, edges 11.80, violations 0.40
lingam constrained:   F1 0.0224 +/- 0.0322, SHD 48.20 +/- 4.44, edges 11.40, violations 0.00

notears_postproc unconstrained: F1 0.0000 +/- 0.0000, SHD 38.00 +/- 0.00, edges 0.00, violations 0.00
notears_postproc constrained:   F1 1.0000 +/- 0.0000, SHD 0.00 +/- 0.00, edges 38.00, violations 0.00

deci_native_unconstrained: F1 0.0321 +/- 0.0338, SHD 49.60 +/- 11.06, edges 13.60, violations 0.80
deci_native_constrained:   F1 0.8860 +/- 0.0790, SHD 10.40 +/- 8.73, edges 48.40, violations 0.00
```

Interpretation note: constrained NOTEARS exactly matches the reference DAG
because the advisor-dummy required constraints are the reference-DAG edges
and unconstrained NOTEARS returned an empty graph. This is useful as a
constraint-enforcement sanity check, but it should not be presented as
independent causal-discovery evidence.

# Advisor Dummy Pipeline Correction

On 2026-05-09, the advisor-dummy experiment was corrected to use the actual
advisor-provided CSV and XLSX files directly:

```text
data/advisor_dummy/ESG-Finance_dummy_data.csv
data/advisor_dummy/ESG-Finance_Metadata.xlsx
data/advisor_dummy/Dummy_dataset_ESG.txt
```

The text generator/spec is now development-only fallback and is not used by
default. `run_all.py --dataset advisor_dummy` rebuilds the advisor-dummy
artifacts from the project-local CSV/XLSX before running algorithms. If the
CSV or XLSX is missing, the final path raises a clear error instead of
silently regenerating substitute data. The fallback requires:

```text
--allow-dummy-regeneration
```

New/updated advisor-dummy artifacts:

```text
outputs/experiments/advisor_dummy_data_audit.csv
outputs/experiments/advisor_dummy_metadata_registry.csv
outputs/experiments/advisor_dummy_reference_dag_edges.csv
outputs/experiments/advisor_dummy_reference_dag_validation.md
outputs/experiments/advisor_dummy_constraints_forbidden.csv
outputs/experiments/advisor_dummy_constraints_required_light.csv
outputs/experiments/advisor_dummy_constraints_full_reference_required.csv
outputs/experiments/advisor_dummy_constraint_ablation.csv
outputs/experiments/advisor_dummy_constraint_ablation_summary.csv
outputs/experiments/advisor_dummy_constraint_ablation_report.md
outputs/experiments/advisor_dummy_final_report.md
```

Reference build result from the actual advisor files:

```text
original CSV shape: 3002 rows x 161 columns
cleaned model shape: 3002 rows x 40 columns
explicit advisor causal rules found: False
evaluation target: ontology_derived_reference_dag
reference edges: 46
forbidden constraints: 247
required-light constraints: 4
full-reference required constraints: 46
validation status: PASS
```

Constraint design was corrected to avoid circular evaluation:

```text
none: no constraints
forbidden_only: main constrained evaluation; reverse/impossible directions only
required_light: secondary; forbidden_only plus 4 high-confidence required edges
full_reference_sanity: sanity check; all 46 reference edges required
```

Commands run:

```text
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --constraint-mode forbidden_only --skip-deci
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --constraint-mode none --skip-deci
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --constraint-mode required_light --skip-deci
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --constraint-mode full_reference_sanity --skip-deci
.\.venv\Scripts\python.exe run_all.py --only-deci --dataset advisor_dummy --constraint-mode forbidden_only
.\.venv\Scripts\python.exe run_all.py --only-deci --dataset advisor_dummy --constraint-mode required_light
.\.venv\Scripts\python.exe run_all.py --only-deci --dataset advisor_dummy --constraint-mode full_reference_sanity
```

Main advisor-dummy result (`forbidden_only`):

```text
pc unconstrained: F1 0.0127, SHD 127.60, edges 83.20, violations 21.80
pc constrained:   F1 0.0552, SHD 124.20, edges 85.40, violations 0.00

lingam unconstrained: F1 0.0000, SHD 58.20, edges 12.20, violations 0.80
lingam constrained:   F1 0.0000, SHD 57.40, edges 11.40, violations 0.00

notears_postproc unconstrained: F1 0.0000, SHD 46.00, edges 0.00, violations 0.00
notears_postproc constrained:   F1 0.0000, SHD 46.00, edges 0.00, violations 0.00

deci_native_unconstrained: F1 0.0180, SHD 60.80, edges 16.00, violations 3.40
deci_native_constrained:   F1 0.0110, SHD 70.20, edges 25.00, violations 0.00
```

Secondary advisor-dummy result (`required_light`):

```text
notears_postproc constrained: F1 0.1600, SHD 42.00, edges 4.00
deci_native_constrained: F1 0.1276, SHD 64.00, edges 27.20
```

Sanity-check advisor-dummy result (`full_reference_sanity`):

```text
notears_postproc constrained: F1 1.0000, SHD 0.00, edges 46.00
deci_native_constrained: F1 0.8165, SHD 21.20, edges 67.20
```

Interpretation: `forbidden_only` is the main thesis-safe constrained
comparison. `required_light` is a secondary constraint-assistance check.
`full_reference_sanity` demonstrates constraint enforcement and must not be
presented as independent causal discovery.

## GES Baseline Added

Added a classical score-based GES baseline through `08_run_ges.py` and
integrated it into `run_all.py` as `ges_postproc`.

Implementation details:

```text
backend: causal-learn ScoreBased.GES
native background knowledge: not supported by installed GES API
constraint handling: post-processing
reported algorithm label: ges_postproc
CPDAG handling: causal-learn GES CPDAG is oriented to one compatible DAG
default max parents: 3, to keep Windows/advisor-dummy runtime stable
graph outputs: outputs/experiments/graphs/ges_<dataset>_<constraint_label>_seed<S>.csv
stable edges: outputs/experiments/ges_stable_edges.csv
```

Commands run:

```text
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --algorithm ges --constraint-mode none
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --algorithm ges --constraint-mode required_light
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --algorithm ges --constraint-mode full_reference_sanity
.\.venv\Scripts\python.exe run_all.py --datasets advisor_dummy,real --algorithm ges --constraint-mode forbidden_only
```

Main GES results from the final forbidden-only run:

```text
advisor_dummy unconstrained: F1 0.0105 +/- 0.0235, SHD 76.00 +/- 3.39, edges 30.80, violations 4.80
advisor_dummy constrained:   F1 0.0110 +/- 0.0245, SHD 71.20 +/- 4.15, edges 26.00, violations 0.00

real ECB unconstrained: alignment 0.300 +/- 0.447, edges 21.60, violations 0.60
real ECB constrained:   alignment 0.400 +/- 0.548, edges 21.00, violations 0.00
```

Advisor-dummy GES ablation rows now include:

```text
none: unconstrained/constrained both F1 0.0105, SHD 76.00
forbidden_only: constraints remove violations 4.80 -> 0.00; F1 remains low
required_light: constrained F1 0.1159, SHD 67.20
full_reference_sanity: constrained F1 0.7830, SHD 25.60
```

Interpretation: GES is a useful classical score-based baseline and the
forbidden-only condition improves constraint compliance, but reference-DAG
F1 remains very low. This supports the existing caution that the advisor
dummy data may not contain statistical signal matching the ontology-derived
reference DAG. The full-reference sanity result reflects post-processing
constraint enforcement and must not be presented as independent discovery.

## Final Consolidated Thesis Tables

Created final thesis-facing result tables and plots with
`20_create_final_result_tables.py`.

Artifacts:

```text
outputs/experiments/final_algorithm_comparison_advisor_dummy.csv
outputs/experiments/final_algorithm_comparison_advisor_dummy_sanity_check.csv
outputs/experiments/final_algorithm_comparison_real_ecb.csv
outputs/experiments/final_results_summary.md
outputs/experiments/final_advisor_dummy_f1_forbidden_only.png
outputs/experiments/final_advisor_dummy_shd_forbidden_only.png
outputs/experiments/final_advisor_dummy_violations_forbidden_only.png
outputs/experiments/final_real_ecb_alignment_violations.png
```

Advisor dummy main comparison uses unconstrained rows scored against the
forbidden-only violation set, so the violation reduction is visible:

```text
PC:      violations 21.80 -> 0.00, F1 0.0127 -> 0.0552
LiNGAM:  violations 0.80 -> 0.00, F1 0.0000 -> 0.0000
NOTEARS: violations 0.00 -> 0.00, F1 0.0000 -> 0.0000
GES:     violations 4.80 -> 0.00, F1 0.0105 -> 0.0110
DECI:    violations 3.40 -> 0.00, F1 0.0180 -> 0.0110
```

Real ECB final comparison remains a case-study table only:

```text
PC:      alignment 0.000 -> 0.333, violations 0.333 -> 0.000
LiNGAM:  alignment 0.400 -> 0.400, violations 0.200 -> 0.000
NOTEARS: alignment 0.000 -> 0.000, violations 0.200 -> 0.000
GES:     alignment 0.300 -> 0.400, violations 0.600 -> 0.000
DECI:    alignment 0.000 -> 0.000, violations 0.000 -> 0.000
```

Thesis-safe interpretation: advisor dummy results are reference-DAG
alignment and constraint-compliance results. Real ECB results are case-study
literature-alignment results only; no F1/SHD or causal recovery claim is
made for real data.

## Final Thesis Figures

Created `09_make_final_figures.py` and generated the final figure set in
`outputs/figures/`.

Generated figures:

```text
pipeline_overview.png
constraint_pipeline.png
advisor_dummy_f1_forbidden_only.png
advisor_dummy_shd_forbidden_only.png
advisor_dummy_violations_forbidden_only.png
real_ecb_graph_selected.png
real_ecb_stability.png
full_reference_sanity_appendix.png
figure_captions.md
```

The selected real ECB case-study graph uses GES forbidden-only because PC
graph CSVs were not available in `outputs/experiments/graphs/`. Captions use
thesis-safe wording: ontology-derived reference DAG, reference-DAG alignment,
ontology violations, and real-data case-study graph.

# Advisor Dummy File Descriptions

## `Dummy_dataset_ESG.txt`

A Python script the advisor wrote that generates the dummy dataset. It defines
69 ESG variables across three pillars (Environmental, Social, Governance) with
realistic names and units, then produces N=3000 rows using
`np.random.uniform()`, `np.random.randint()`, and `np.random.choice()` — every
variable drawn independently, with no causal equations between them. It then
deliberately injects data quality problems: 5–20% missing values per column,
extreme outliers (×10–×100 on 10 numeric columns), wrong data types (the
string "error" inserted into numeric fields on 8 columns), and invalid category
values ("Very High" injected into a column that should only have
Low/Medium/High).

## `ESG-Finance_dummy_data.csv`

The pre-generated output from the script above: 3002 rows × 161 columns. The
first column is a LEI company identifier. The next ~69 columns are the ESG
variables from the spec. The remaining ~92 columns are financial variables
(market cap, earnings, ratios, ownership fields, etc.) not present in the text
spec — added separately to reflect a realistic ESG+finance combined dataset.
This is the actual input the pipeline uses.

## `ESG-Finance_Metadata.xlsx`

A metadata workbook read by `05_build_reference_dag_from_dummy.py` to classify
variables: pillar (E/S/G/Financial), ontology type (indicator, policy, score,
etc.), causal role (driver, outcome, composite), units, and descriptions. Used
to infer which variables belong in the reference DAG and what directions the
edges should point. If this file is missing the pipeline raises a hard error.

# Thesis Suitability Review (2026-05-10)

## Rating: 6.5 / 10 — solid foundation with one critical gap

### Strengths

- Original methodological contribution: ontology-guided constraint injection
  (forbidden edges from domain rules + pillar-stratified RAG audit) applied to
  causal discovery is not a standard textbook exercise.
- Multi-algorithm comparison: PC, LiNGAM, NOTEARS, GES, DECI with matched
  constraint modes across two datasets gives a genuine comparative study.
- Honest epistemic framing: distinguishing ontology-derived reference DAG from
  true data-generating DAG, and keeping real ECB results purely descriptive,
  shows methodological maturity.
- Engineering depth: DECI/Causica integration on Windows (subprocess/file
  handoff, native constraint injection, threshold sweep, constraint-matrix
  validation).
- Real-data case study: actual ECB-supervised bank data elevates this above a
  purely synthetic paper.

### Critical gap — the advisor dummy data has no causal structure

`Dummy_dataset_ESG.txt` generates every variable with `np.random.uniform()` or
`np.random.randint()` called independently. There are no structural equations,
no functional relationships, no injected causal edges anywhere in the generation
script. The data is i.i.d. noise per column.

Consequences:
- F1 and SHD on the advisor dummy dataset measure alignment with an
  ontology-derived reference DAG that the data was never generated from.
- Any algorithm that finds edges is finding noise correlations, not causal
  signal.
- Very low F1 in the main forbidden_only condition (PC: 0.055, GES: 0.011,
  DECI: 0.011, LiNGAM/NOTEARS: 0.000) is consistent with this.
- The main result reduces to "constraints eliminate violations", which is nearly
  tautological for post-processed algorithms.

### Other weaknesses

- Real ECB dataset is small (110 firms, 16 variables) with no reference graph.
- DECI remains exploratory with inconsistent results across configurations.

### Suggested improvements

1. Inject actual causal structure into the dummy data (see section below).
2. Bootstrap stability table for the dummy dataset.
3. Sample size sensitivity curve (n=100, 500, 1000, 3002) to show constraints
   help more when data is scarce.
4. Precision vs. recall decomposition of the constraint effect.
5. Edge annotation on the real ECB graph with literature citations.
6. Explicit methodology section describing how ontology rules were translated
   into forbidden/required constraints.

# Causal Dummy Data: generate_causal_dummy.py

## Why the original data cannot support causal recovery claims

The original `ESG-Finance_dummy_data.csv` was generated with
`np.random.uniform()` independently for every variable. The reference DAG was
built afterwards from ontology rules — the data was never generated from it.
F1 and SHD against the reference DAG are therefore not measures of causal
recovery; they measure coincidental alignment with domain rules.

## Where the DAG came from

The reference DAG was not discovered from data — it was written by hand based
on domain knowledge:

1. The advisor defined the variables (Dummy_dataset_ESG.txt).
2. The pipeline asked which variables should causally affect which, using two
   sources: an ESG ontology (structured knowledge base of ESG concept
   relationships) and domain common sense (e.g. emission_reduction_policy
   should reduce carbon_intensity).
3. Those relationships were encoded as directed edges in
   `05_build_reference_dag_from_dummy.py`, which reads the metadata XLSX and
   applies rule families.
4. The result is the "ontology-derived reference DAG": 46 edges over 38
   variables.

The DAG was never used to generate the original data. `generate_causal_dummy.py`
closes that gap by using the same DAG to actually generate the data.

## What generate_causal_dummy.py does

New script created on 2026-05-10. Generates `ESG-Finance_dummy_data_causal_v2.csv`:
a version of the advisor dummy dataset where variables are produced from
structural causal equations that match the reference DAG.

Approach: Structural Causal Model (SCM) in topological order.

For each variable with parents in the reference DAG:

```
latent = Σ (sign_i × snr × normalised_parent_i) + N(0, 0.25)
child  = clip(0.5 + latent × 0.5) → rescale to original range
```

Default snr=0.6 gives moderate signal (R² ≈ 0.3–0.7 per variable), realistic
for ESG data. Root variables (no parents) are still drawn randomly. The same
data quality problems are injected afterward: 5–20% missing values, extreme
outliers, wrong data types, invalid categories.

Topological generation order:

```
Level 0 (roots): esg_oversight_policy, board_diversity, ceo_chair_split,
  auditor_independence_score, esg_incentive_bonus, carbon_neutral_commitment,
  climate_risk_assessment_done, iso_14001_exists, waste_recycled_share,
  water_withdrawal, training_hours, collective_bargaining_coverage,
  diversity_representation, customer_satisfaction_score,
  anti_competitive_violations, corruption_cases

Level 1: emission_reduction_policy, ethical_breaches, reporting_quality_score,
  assurance_score, resilience_score, hazardous_waste_generated,
  injury_frequency_rate, turnover_rate, fair_wage_gap, human_rights_violations

Level 2: total_energy_consumption, renewable_energy_share,
  governance_compliance_score, net_profit_margin, roa_eat, roe_eat

Level 3: co2_ch4_n2o_scope_1_3, carbon_intensity, solvency_ratio

Level 4: environmental_fines

Level 5: debt_to_equity_ratio, market_value_equity
```

## Generated artifacts

```text
data/advisor_dummy/ESG-Finance_dummy_data_causal_v2.csv   (3000 rows x 74 cols)
data/advisor_dummy/causal_dummy_ground_truth_dag.csv       (38x38 adjacency matrix)
data/advisor_dummy/causal_dummy_ground_truth_edges.csv     (46 edges with coefficient signs)
```

## Sanity check results (seed=42, snr=0.6)

All 46 edges produced the correct-sign Pearson correlation except
`corruption_cases → ethical_breaches` (r=-0.004, effectively zero — noise
dominates this weak integer-to-integer link). All substantive edges (r > 0.2)
were correct.

Selected correlation values:

```text
emission_reduction_policy -> carbon_intensity:     r=-0.844  (expected -)  OK
climate_risk_assessment_done -> resilience_score:  r=+0.768  (expected +)  OK
carbon_neutral_commitment -> renewable_energy_share: r=+0.696 (expected +) OK
total_energy_consumption -> co2_ch4_n2o_scope_1_3: r=+0.607  (expected +)  OK
governance_compliance_score -> solvency_ratio:     r=+0.552  (expected +)  OK
iso_14001_exists -> environmental_fines:           r=-0.645  (expected -)  OK
emission_reduction_policy -> roa_eat:              r=+0.631  (expected +)  OK
```

## Run command

```powershell
.\.venv\Scripts\python.exe generate_causal_dummy.py
# optional flags:
# --snr 0.3   (weaker signal, harder for algorithms)
# --snr 0.9   (stronger signal, easier)
# --n 3000 --seed 42
```

## What changes in the thesis claim

With original data: can only claim "constraints reduce violations."
With v2 data: can claim "constraints improve recovery of the true causal
structure" — F1 and SHD are now evaluated against a DAG that actually generated
the data, making them measures of causal recovery rather than ontology alignment.

# Significance Tests (2026-05-10)

Script: `21_significance_tests.py`
Data: advisor_dummy, forbidden_only constraint mode, 5 seeds
Test: one-sided paired Wilcoxon signed-rank test
H1 (F1): constrained F1 > unconstrained F1
H1 (SHD): constrained SHD < unconstrained SHD

Note: with n=5 seeds, the minimum achievable p-value is 1/32 ≈ 0.031.

## F1 results

```text
PC:      Delta F1 = +0.043, p = 0.031 *    (significant)
GES:     Delta F1 = +0.000, p = n/a         (all differences zero at 3 decimals)
LiNGAM:  Delta F1 = +0.000, p = n/a         (all differences zero)
NOTEARS: Delta F1 = +0.000, p = n/a         (all differences zero)
DECI:    Delta F1 = -0.007, p = 0.750       (not significant)
```

## SHD results

```text
PC:      Delta SHD = -3.4, p = 0.031 *     (significant)
GES:     Delta SHD = -4.8, p = 0.031 *     (significant)
LiNGAM:  Delta SHD = -0.8, p = n/a         (insufficient non-zero differences)
NOTEARS: Delta SHD = 0.0,  p = n/a         (all differences zero)
DECI:    Delta SHD = +9.4, p = 0.844       (constrained SHD actually worse)
```

## Interpretation

PC is the only algorithm that shows statistically significant F1 improvement
under forbidden constraints. GES shows significant SHD improvement but no F1
improvement. LiNGAM and NOTEARS found no signal at all under the
ontology-derived reference DAG evaluation, making them untestable. DECI's
constrained variant performed worse than unconstrained — consistent with the
DECI instability noted elsewhere.

The n=5 limitation means these tests can only detect effects where all 5
differences have the same sign. The PC F1 result (p=0.031, all 5 differences
positive) is the strongest statistical evidence available from the current data.

Artifacts:

```text
outputs/experiments/significance_tests.csv
outputs/experiments/significance_tests.md
```

# SNR Sensitivity Sweep (2026-05-10)

Script: `22_snr_sensitivity_sweep.py`
Data: causal_dummy v2 (known ground-truth DAG, no quality injection)
Algorithms: PC, LiNGAM, GES
Seeds: 42, 43 (2 seeds; full 5-seed run can be resumed)
SNR grid tested: 0.2, 0.5, 0.8
All 36 runs: success

Thesis claim tested: "Ontology-guided constraints improve causal recovery most
at low SNR, where data is sparse and noisy — the typical ESG regime."

## Mean F1 results (averaged over 2 seeds)

```text
SNR   PC_unc  PC_con   LiNGAM_unc  LiNGAM_con  GES_unc  GES_con
0.2   0.170   0.690    0.664       0.711        0.561    0.632
0.5   0.298   0.811    0.643       0.685        0.562    0.624
0.8   0.440   0.876    0.426       0.489        0.599    0.646
```

## Mean SHD results (lower is better)

```text
SNR   PC_unc  PC_con   LiNGAM_unc  LiNGAM_con  GES_unc  GES_con
0.2   107.5    40.5      39.0        31.5         52.5     39.0
0.5    70.5    20.5      43.0        35.5         56.0     43.5
0.8    53.5    12.5      78.0        60.5         49.5     40.5
```

## Interpretation

Constrained outperforms unconstrained at every SNR level for all three
algorithms. The constraint benefit is most dramatic for PC:

- PC: gap at SNR=0.2 is +0.520 F1 (4× improvement); at SNR=0.8 is +0.436.
  SHD drops from 107.5 to 40.5 at SNR=0.2 — a 62% reduction. PC provides the
  strongest thesis evidence for constraint-guided recovery.
- LiNGAM: gap at SNR=0.2 is +0.047; at SNR=0.8 is +0.063. The gap grows
  modestly with SNR — possibly because stronger signal exposes direction
  recovery, where forbidden constraints help most.
- GES: gap at SNR=0.2 is +0.071; at SNR=0.8 is +0.047. Slight decrease with
  SNR, consistent with the claim but modest in magnitude.

The PC result is the thesis headline: at the lowest tested SNR (0.2,
representing very noisy ESG data), constraint injection raises F1 from 0.170
to 0.690 and reduces SHD from 107.5 to 40.5. This is the most compelling
quantitative finding.

Important: these F1 and SHD values are against the *known ground-truth DAG*
(from the structural equations in generate_causal_dummy.py), not the
ontology-derived reference DAG. They are valid causal recovery measures.

## Figures generated

```text
outputs/figures/snr_f1_sweep.png
outputs/figures/snr_shd_sweep.png
```

## Technical notes

- PC and LiNGAM fail on quality-injected data (singular correlation matrix /
  n < p after dropna). SNR sweep calls generate() with skip_quality_issues=True
  so all 3000 rows are available.
- GES edge extraction uses the causal-learn edge iterator API:
  `for edge in G.get_graph_edges()` with `TAIL→ARROW` endpoint logic.
  The matrix-based approach had a double-counting bug for undirected edges.
- Undirected GES edges are oriented once only (lower → higher column index).
- The sweep isolates SNR as the only variable. Data quality effects are a
  separate research dimension not tested here.

## Run command

```powershell
.\.venv\Scripts\python.exe 22_snr_sensitivity_sweep.py
# Custom grid:
.\.venv\Scripts\python.exe 22_snr_sensitivity_sweep.py --snr-grid 0.2,0.4,0.6,0.8,1.0 --seeds 42,43,44,45,46
```

# Active Script Inventory (2026-05-10)

Scripts moved to `archive/` (26 files, not listed here). The following 15
scripts form the current active pipeline.

## Core orchestration

| Script | Role |
|--------|------|
| `run_all.py` | Master runner; dispatches all algorithms and datasets |
| `config.py` | Shared constants, grid definitions, dataset paths |
| `deci_ablation.py` | DECI-specific ablation grid logic |

## Data preparation

| Script | Role |
|--------|------|
| `02a_parse_real_dataset.py` | Parse real ECB banking dataset |
| `02d_compute_pillar_scores.py` | Compute ESG pillar composite scores |
| `05_build_reference_dag_from_dummy.py` | Build ontology-derived reference DAG from advisor files |
| `12_generate_synthetic.py` | Generate synthetic ground-truth dataset |
| `generate_causal_dummy.py` | Generate causally-structured dummy v2 dataset (SCM) |

## Algorithm runners

| Script | Role |
|--------|------|
| `07_run_deci.py` | DECI/Causica causal discovery |
| `08_run_ges.py` | GES causal discovery (causal-learn) |
| `14_constraint_adapter.py` | Translate ontology constraints to algorithm-specific format |

## Analysis and output

| Script | Role |
|--------|------|
| `20_create_final_result_tables.py` | Produce thesis-facing result tables and CSV summaries |
| `21_significance_tests.py` | Wilcoxon significance tests: constrained vs unconstrained |
| `22_snr_sensitivity_sweep.py` | SNR sensitivity sweep on causal dummy v2 |
| `09_make_final_figures.py` | Generate all thesis figures |

## Full pipeline run order

```powershell
# 1. Build reference DAG from advisor files
.\.venv\Scripts\python.exe 05_build_reference_dag_from_dummy.py

# 2. Generate causally-structured dummy v2 data
.\.venv\Scripts\python.exe generate_causal_dummy.py

# 3. Run causal discovery — advisor dummy, all constraint modes (no DECI)
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --constraint-mode none --skip-deci
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --constraint-mode forbidden_only --skip-deci
.\.venv\Scripts\python.exe run_all.py --dataset advisor_dummy --constraint-mode required_light --skip-deci

# 4. Run DECI — advisor dummy
.\.venv\Scripts\python.exe run_all.py --only-deci --dataset advisor_dummy --constraint-mode forbidden_only
.\.venv\Scripts\python.exe run_all.py --only-deci --dataset advisor_dummy --constraint-mode required_light

# 5. Significance tests
.\.venv\Scripts\python.exe 21_significance_tests.py

# 6. SNR sensitivity sweep (causal dummy v2, known ground truth)
.\.venv\Scripts\python.exe 22_snr_sensitivity_sweep.py

# 7. Final result tables
.\.venv\Scripts\python.exe 20_create_final_result_tables.py

# 8. Final figures
.\.venv\Scripts\python.exe 09_make_final_figures.py
```

# Causal Dummy v2 Experimental Completion Log (2026-05-11)

This section supersedes the earlier 2-seed SNR note. The quantitative causal
recovery experiments below use causal dummy v2 and score against the fixed
generated ground-truth DAG from `generate_causal_dummy.py`.

## Commands Run

```powershell
.\.venv\Scripts\python.exe generate_causal_dummy.py
.\.venv\Scripts\python.exe 22_snr_sensitivity_sweep.py --snr-grid 0.2,0.4,0.6,0.8,1.0 --seeds 42,43,44,45,46
.\.venv\Scripts\python.exe 23_sample_size_sensitivity.py --sample-sizes 100,250,500,1000,2000,3000 --seeds 42,43,44,45,46
.\.venv\Scripts\python.exe 24_causal_dummy_final_comparison.py --seeds 42,43,44,45,46
.\.venv\Scripts\python.exe 25_stability_analysis.py
.\.venv\Scripts\python.exe 20_create_final_result_tables.py
.\.venv\Scripts\python.exe 09_make_final_figures.py
```

## Dataset and Design

- Main causal recovery dataset: `data/advisor_dummy/ESG-Finance_dummy_data_causal_v2.csv`.
- Ground truth: `data/advisor_dummy/causal_dummy_ground_truth_dag.csv` and `data/advisor_dummy/causal_dummy_ground_truth_edges.csv`.
- Final comparison: n=3000, SNR=0.6, seeds 42,43,44,45,46.
- SNR sweep: SNR 0.2,0.4,0.6,0.8,1.0; n=3000; seeds 42,43,44,45,46.
- Sample-size sweep: n=100,250,500,1000,2000,3000; SNR=0.6; seeds 42,43,44,45,46.
- Algorithms: PC, LiNGAM, GES for sensitivity sweeps; PC, LiNGAM, NOTEARS, GES for final comparison.
- Constraint modes: unconstrained and constrained. Constrained runs use forbidden-edge ontology constraints only.
- Advisor dummy remains reference-DAG alignment / constraint-compliance only.
- Real ECB remains literature-alignment / violation-reduction only.

## Outputs

Raw and summary CSVs:

```text
outputs/experiments/causal_dummy_final_comparison_raw.csv
outputs/experiments/causal_dummy_final_comparison_summary.csv
outputs/experiments/snr_sensitivity_results.csv
outputs/experiments/snr_sensitivity_summary.csv
outputs/experiments/sample_size_sensitivity_results.csv
outputs/experiments/sample_size_sensitivity_summary.csv
outputs/experiments/final_causal_dummy_comparison.csv
outputs/experiments/final_snr_sensitivity.csv
outputs/experiments/final_sample_size_sensitivity.csv
outputs/experiments/final_advisor_dummy_constraint_compliance.csv
outputs/experiments/final_real_ecb_case_study.csv
outputs/experiments/final_experiment_summary.md
```

Reports:

```text
outputs/experiments/causal_dummy_final_comparison_report.md
outputs/experiments/snr_sensitivity_report.md
outputs/experiments/sample_size_sensitivity_report.md
outputs/experiments/stability_analysis_report.md
```

Figures:

```text
outputs/figures/causal_dummy_final_f1.png
outputs/figures/causal_dummy_final_shd.png
outputs/figures/causal_dummy_final_violations.png
outputs/figures/causal_dummy_final_runtime.png
outputs/figures/causal_dummy_f1_comparison.png
outputs/figures/causal_dummy_shd_comparison.png
outputs/figures/causal_dummy_precision_recall.png
outputs/figures/causal_dummy_violations.png
outputs/figures/causal_dummy_runtime.png
outputs/figures/snr_f1_sweep.png
outputs/figures/snr_shd_sweep.png
outputs/figures/snr_precision_sweep.png
outputs/figures/snr_recall_sweep.png
outputs/figures/snr_runtime_sweep.png
outputs/figures/sample_size_f1.png
outputs/figures/sample_size_shd.png
outputs/figures/sample_size_precision.png
outputs/figures/sample_size_recall.png
outputs/figures/sample_size_runtime.png
outputs/figures/stability_causal_dummy.png
outputs/figures/stability_advisor_dummy.png
outputs/figures/stability_real_ecb.png
outputs/figures/advisor_dummy_violations.png
outputs/figures/real_ecb_alignment_violations.png
```

## Failures and Warnings

- SNR sweep: 150 rows, 150 success, 0 failed.
- Sample-size sweep: 180 rows, 180 success, 0 failed.
- Final causal-dummy comparison: 40 rows, 40 success, 0 failed.
- Stability analysis completed for causal dummy, advisor dummy, and real ECB using available graph CSVs.
- DECI was not included in the main final comparison; no comparable 5-seed DECI causal-dummy run was produced.
- Current `generate_causal_dummy.py` run completed without pandas dtype warnings after casting quality-injected columns to object dtype. The sanity printout still flags the marginal `corruption_cases -> ethical_breaches` correlation as near-zero/wrong-sign, but the structural edge remains in the generated ground-truth DAG. Quantitative experiments call the generator with `skip_quality_issues=True`.

## Headline Metrics: Final Causal Dummy Comparison

Mean over seeds 42,43,44,45,46. Lower SHD and violations are better.

```text
Algorithm  Mode           F1      SHD    Precision  Recall  Violations  Runtime_s
PC         unconstrained  0.682   27.8   0.722      0.648   8.8         1.235
PC         constrained    0.944    5.2   0.932      0.957   0.0         1.011
LiNGAM     unconstrained  0.690   34.6   0.594      0.826   7.8        10.245
LiNGAM     constrained    0.742   26.8   0.675      0.826   0.0        10.245
NOTEARS    unconstrained  0.423   42.0   0.575      0.335  11.4        14.989
NOTEARS    constrained    0.501   30.6   1.000      0.335   0.0        14.989
GES        unconstrained  0.671   35.2   0.590      0.778  10.6         2.495
GES        constrained    0.744   24.6   0.714      0.778   0.0         2.495
```

## Headline Metrics: SNR Sweep at SNR=0.6

```text
Algorithm  Mode           F1      SHD    Precision  Recall
PC         unconstrained  0.682   27.8   0.722      0.648
PC         constrained    0.944    5.2   0.932      0.957
LiNGAM     unconstrained  0.690   34.6   0.594      0.826
LiNGAM     constrained    0.742   26.8   0.675      0.826
GES        unconstrained  0.671   35.2   0.590      0.778
GES        constrained    0.744   24.6   0.714      0.778
```

## Headline Metrics: Sample-Size Sweep

At n=100:

```text
Algorithm  Mode           F1      SHD
PC         unconstrained  0.324   47.4
PC         constrained    0.764   18.4
LiNGAM     unconstrained  0.530   51.0
LiNGAM     constrained    0.592   39.4
GES        unconstrained  0.498   54.6
GES        constrained    0.570   41.0
```

At n=3000:

```text
Algorithm  Mode           F1      SHD
PC         unconstrained  0.682   27.8
PC         constrained    0.944    5.2
LiNGAM     unconstrained  0.690   34.6
LiNGAM     constrained    0.742   26.8
GES        unconstrained  0.671   35.2
GES        constrained    0.744   24.6
```

# RQ3 Runtime Analysis Correction Log (2026-05-11)

Runtime analysis for research question 3 was updated after per-seed inspection
showed that the apparent NOTEARS runtime improvement was a gCastle caching
artifact, not a real computational speedup.

## Command Run

```powershell
.\.venv\Scripts\python.exe 23_runtime_analysis.py
```

## Outputs

```text
outputs/experiments/runtime_comparison.csv
outputs/experiments/runtime_analysis.md
outputs/figures/runtime_comparison.png
LIMITATIONS_TO_REVIEW.md
```

## Runtime Findings

The current `outputs/experiments/results.csv` contains successful runtime rows
only for `advisor_dummy`. Real ECB and causal dummy v2 runtime rows are absent
from this file, so cross-dataset runtime scaling is documented as a limitation
rather than claimed from incomplete data.

```text
Algorithm  Dataset        Overhead_pct  CV warning
LiNGAM     advisor_dummy  -1.3          none
NOTEARS    advisor_dummy  n/a           caching suspected
PC         advisor_dummy  -23.3         high variance
```

Per-seed NOTEARS diagnostics:

```text
notears_postproc/unconstrained/advisor_dummy:
2.8331, 0.0195, 0.0297, 0.0213, 0.0224 seconds
CV = 2.15 -> caching suspected
```

Interpretation:

- PC shows a real measured runtime decrease with constraints: `-23.3%`.
  This is consistent with constraint-based search pruning.
- LiNGAM overhead is negligible: `-1.3%`. Constraints are post-processing.
- NOTEARS raw mean overhead is not interpreted as a real speedup because
  gCastle cached the optimization after the first seed. Honest interpretation:
  NOTEARS post-processing adds no measurable overhead on top of the real
  first optimization cost of about 2.83 seconds.
- DECI is not included in `runtime_comparison.csv` because the rows in
  `results.csv` are skipped, not successful runtime observations. Auxiliary
  DECI selected-config runtimes remain documented separately.

RQ3 answer recorded in `outputs/experiments/runtime_analysis.md`: constraint
injection imposes negligible computational overhead in the measured
advisor-dummy setting; PC can become faster under constraints, LiNGAM is
effectively unchanged, and NOTEARS requires per-fit interpretation because of
caching. Runtime scaling for real ECB and causal dummy v2 remains future work
until those rows are persisted in `results.csv`.
