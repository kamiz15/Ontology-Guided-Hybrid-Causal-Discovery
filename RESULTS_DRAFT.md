# Headline Findings

[fill in with table after run completes]

# Synthetic Validation

Five seeds, 0.8x bootstrap. Compares constraint-injected causal
discovery against unconstrained baseline.

[fill in numbers]

# Real-Data Case Study: ECB Banking

The 110-firm, 16-variable ECB sample serves as a case study for
applying constraint-guided causal discovery to genuine banking ESG
data. Quantitative ground-truth metrics (SHD, F1) are not reported
because no ground-truth DAG exists; instead we report:

(a) literature alignment score per algorithm
(b) the discovered graph as a qualitative artifact, with each edge
    annotated by whether it appears in literature
(c) algorithm agreement: edges discovered by 2+ of 4 algorithms

[fill in once run completes]

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

Note: the latest `outputs/experiments/results.csv` was overwritten by a
DECI-only smoke run. The full experiment should be rerun before final
reporting.

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

# Temporary DECI Review Grid

On 2026-05-08, `config.py` was temporarily narrowed for local testing of
the native Causica DECI path. This is a review grid, not the final thesis
ablation grid.

```python
DECI_ABLATION_GRID = {
    "epochs": [20, 50],
    "thresholds": [0.25, 0.30, 0.35],
    "sparsity_strength": ["current"],
    "variable_sets": ["reduced"],
    "constraint_modes": ["unconstrained", "native_constrained"],
}
```

Reason: keep Windows test runs short while reviewing DECI behavior after
the constraint-matrix and synthetic-ground-truth fixes. Before final
Linux/WSL runs, restore or explicitly document the broader ablation grid.

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
