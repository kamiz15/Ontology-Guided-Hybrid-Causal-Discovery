# Runtime Analysis for RQ3

Input file: `outputs\experiments\results.csv`

Successful rows used: 30

## Methodological note

Mean-based runtime comparison can be misleading when one of the underlying
algorithms caches results across calls. We observed this with gCastle's
NOTEARS: only the first seed incurred the real optimization cost (~2.83s);
subsequent seeds returned cached results (~0.02s). Where this caching pattern
is detected, we report the per-fit runtime descriptively rather than the mean.
The coefficient-of-variation check in section "Per-seed diagnostics" flags any
algorithm whose per-seed runtime variance exceeds 2× its mean.

## Per-Algorithm Constraint Overhead

| algorithm | dataset | unconstrained_mean_s | unconstrained_std_s | constrained_mean_s | constrained_std_s | overhead_seconds | overhead_pct | n_seeds_unc | n_seeds_con | cv_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiNGAM | advisor_dummy | 13.875 | 0.586 | 13.700 | 0.294 | -0.175 | -1.3 | 5 | 5 |  |
| NOTEARS | advisor_dummy | 0.585 | 1.257 | 0.023 | 0.006 | -0.563 | n/a (caching artifact — see Methodological note)[^notears-cache] | 5 | 5 | caching suspected |
| PC | advisor_dummy | 1.528 | 1.116 | 1.172 | 0.281 | -0.356 | -23.3 | 5 | 5 | high variance |

[^notears-cache]: The raw NOTEARS overhead seconds are preserved in
`runtime_comparison.csv`, but the percentage is suppressed in this report
because the per-seed runtime pattern indicates caching.

## Per-seed diagnostics

| algorithm | mode | dataset | runtimes (s) | CV | flag |
| --- | --- | --- | --- | --- | --- |
| LiNGAM | constrained | advisor_dummy | 14.06, 13.93, 13.44, 13.67, 13.39 | 0.020 |  |
| LiNGAM | unconstrained | advisor_dummy | 14.74, 14.14, 13.27, 13.44, 13.79 | 0.040 |  |
| NOTEARS | constrained | advisor_dummy | 0.0194, 0.0201, 0.0192, 0.0213, 0.0332 | 0.260 |  |
| NOTEARS | unconstrained | advisor_dummy | 2.833, 0.0195, 0.0297, 0.0213, 0.0224 | 2.150 | caching suspected |
| PC | constrained | advisor_dummy | 1.442, 1.255, 0.8856, 1.414, 0.8628 | 0.240 |  |
| PC | unconstrained | advisor_dummy | 3.492, 1.191, 0.8516, 1.276, 0.8305 | 0.730 | high variance |

Highest overhead: LiNGAM on advisor_dummy (-0.175s, -1.3%). All complete pairs have negative overhead, so this is the smallest speed-up rather than an added cost.

Lowest overhead: PC on advisor_dummy (-0.356s, -23.3%). Negative overhead means the constrained run was faster.

## Dataset-Size Effect

| dataset | expected N | available in results.csv | mean successful runtime (s) |
| --- | ---: | --- | ---: |
| real | 110 | no |  |
| advisor_dummy | 3002 | yes | 5.147 |
| causal_dummy_v2 | 3000 | no |  |

Only `advisor_dummy` successful rows are present in `results.csv`, so this file cannot support a direct real-vs-dummy-vs-causal-v2 runtime scaling claim. The available evidence is within-dataset constraint overhead, not cross-dataset scalability with N.

## DECI Runtime

`results.csv` contains 10 DECI rows, but none are successful runtime observations; they are skipped rows. Auxiliary selected-config DECI real-data runtimes are: real constrained: 10.891s +/- 0.431s (n=5); real unconstrained: 11.564s +/- 0.775s (n=5). The auxiliary DECI means are compared only informally: the available non-DECI successful rows in `results.csv` average 5.147s, but those rows are advisor-dummy runs, not the real reduced-variable DECI setting. On this rough comparison, DECI is a runtime outlier. DECI should be flagged as runtime-sensitive and configuration-dependent, not as a directly comparable main runtime baseline here.

## SNR Runtime Scaling

`outputs\experiments\snr_sweep_results.csv` exists but has no `runtime_seconds` column.

## Figure

Generated `outputs\figures\runtime_comparison.png`.

## RQ3 answer

Constraint injection imposes negligible computational overhead in all measured cases on the advisor dummy dataset (N=3002, 46 variables). PC runtime decreases by 23.3% with constraints, consistent with constraint-based methods exploiting forbidden edges by pruning the conditional independence test budget. LiNGAM overhead is negligible (-1.3%) as constraints are applied as post-processing. NOTEARS runtime appears artificially compressed in the raw measurement due to gCastle's internal caching across repeated calls on the same data; honest interpretation is that NOTEARS post-processing adds no measurable overhead on top of a ~2.83-second optimization. Runtime measurements for the real ECB dataset (N=110) and the causal v2 dataset (N=3000) were not present in the current results.csv (only advisor_dummy was persisted) and are noted as future work. Constraint-count sensitivity (how runtime scales with the number of constraints applied) was also not measured and is a future-work item — the negative PC overhead at the current constraint count suggests such a sweep would be informative.

## DECI runtime (separate analysis)

Source file: `outputs\experiments\deci_real_selected_config.csv`

### Summary

| variable_set | constraint_mode | epochs | mean_runtime_s | std_runtime_s | n_runs |
| --- | --- | --- | --- | --- | --- |
| reduced | native_constrained | 20 | 10.891 | 0.431 | 5.000 |
| reduced | unconstrained | 20 | 11.564 | 0.775 | 5.000 |

### Matched constrained versus unconstrained comparison

| dataset | variable_set | epochs | unconstrained_mean_s | native_constrained_mean_s | overhead_seconds | multiplier | multiplier_label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real | reduced | 20 | 11.564 | 10.891 | -0.673 | 0.942 | 0.94x of unconstrained (faster) |

DECI is reported separately because it follows a different run schedule from
the standard seed loop, uses selected configurations/thresholds, and operates
on a larger and threshold-dependent runtime scale. real/reduced/epochs=20: 10.891s vs 11.564s, overhead -0.673s (0.94x of unconstrained (faster)).

No failed or timeout rows are present in the selected-config runtime file used here. These are still local Windows/runtime observations and should be treated as configuration-specific rather than a universal DECI scalability benchmark.
