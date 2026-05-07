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
