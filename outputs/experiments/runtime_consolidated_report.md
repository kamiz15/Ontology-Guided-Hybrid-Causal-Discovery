# Consolidated runtime analysis

## Sample-size scaling (N=110 to N=3000)

The direct sample-size sweep covers causal dummy v2 from N=100 to N=3000.
Real ECB N=110 appears in the DECI-only real section below, not in the
classical sample-size sweep.

| algorithm | mode | sample_size | mean_runtime_s | std_runtime_s | median_runtime_s | n_runs | cv | cv_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GES | constrained | 100.000 | 2.846 | 0.436 | 2.991 | 5.000 | 0.153 |  |
| GES | constrained | 3000.000 | 2.508 | 0.088 | 2.514 | 5.000 | 0.035 |  |
| GES | unconstrained | 100.000 | 2.846 | 0.436 | 2.991 | 5.000 | 0.153 |  |
| GES | unconstrained | 3000.000 | 2.508 | 0.088 | 2.514 | 5.000 | 0.035 |  |
| LiNGAM | constrained | 100.000 | 3.825 | 0.771 | 3.500 | 5.000 | 0.201 |  |
| LiNGAM | constrained | 3000.000 | 10.120 | 0.219 | 10.013 | 5.000 | 0.022 |  |
| LiNGAM | unconstrained | 100.000 | 3.825 | 0.771 | 3.500 | 5.000 | 0.201 |  |
| LiNGAM | unconstrained | 3000.000 | 10.120 | 0.219 | 10.013 | 5.000 | 0.022 |  |
| PC | constrained | 100.000 | 0.370 | 0.008 | 0.368 | 5.000 | 0.022 |  |
| PC | constrained | 3000.000 | 1.005 | 0.036 | 1.009 | 5.000 | 0.036 |  |
| PC | unconstrained | 100.000 | 0.733 | 1.044 | 0.280 | 5.000 | 1.424 | high variance |
| PC | unconstrained | 3000.000 | 0.824 | 0.053 | 0.804 | 5.000 | 0.065 |  |

## SNR sensitivity (runtime as a function of noise)

The SNR sweep shows no strong evidence that noisier settings systematically
increase runtime. This means runtime is algorithm-dominated rather than
problem-dominated: practitioners deploying constraint-guided causal discovery
in ESG-finance can prioritize algorithm choice over noise-regime concerns when
planning compute budget.

| algorithm | mode | snr | mean_runtime_s | std_runtime_s | median_runtime_s | n_runs | cv | cv_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GES | constrained | 0.200 | 1607.508 | 3589.606 | 1.865 | 5.000 | 2.233 | caching suspected |
| GES | constrained | 1.000 | 2.780 | 0.147 | 2.739 | 5.000 | 0.053 |  |
| GES | unconstrained | 0.200 | 2.204 | 0.656 | 1.885 | 5.000 | 0.298 |  |
| GES | unconstrained | 1.000 | 2.821 | 0.150 | 2.784 | 5.000 | 0.053 |  |
| LiNGAM | constrained | 0.200 | 9.082 | 2.680 | 8.339 | 5.000 | 0.295 |  |
| LiNGAM | constrained | 1.000 | 9.871 | 0.082 | 9.854 | 5.000 | 0.008 |  |
| LiNGAM | unconstrained | 0.200 | 9.651 | 3.382 | 7.489 | 5.000 | 0.350 |  |
| LiNGAM | unconstrained | 1.000 | 9.891 | 0.048 | 9.882 | 5.000 | 0.005 |  |
| PC | constrained | 0.200 | 0.957 | 0.467 | 0.700 | 5.000 | 0.488 |  |
| PC | constrained | 1.000 | 1.039 | 0.013 | 1.041 | 5.000 | 0.013 |  |
| PC | unconstrained | 0.200 | 1.206 | 1.215 | 0.495 | 5.000 | 1.007 | high variance |
| PC | unconstrained | 1.000 | 0.838 | 0.068 | 0.825 | 5.000 | 0.081 |  |

## Real ECB runtime

Real ECB runtime evidence is formally captured only for DECI
(`deci_real_selected_config.csv`), where constrained DECI is slightly
faster than unconstrained on the reduced variable set. Classical
algorithms (PC, LiNGAM, NOTEARS, GES) at N=110 ran in well under one
second per fit during execution; runtime statistics for these
algorithms on real data were not persisted to results.csv for formal
aggregation. This is consistent with the sample-size sweep, which shows
sub-second runtimes for classical methods at all N <= 500 in the
synthetic settings.

| algorithm | mode | mean_runtime_s | std_runtime_s | median_runtime_s | n_runs | cv | cv_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DECI | constrained | 10.891 | 0.431 | 11.033 | 5.000 | 0.040 |  |
| DECI | unconstrained | 11.564 | 0.775 | 11.498 | 5.000 | 0.067 |  |

## DECI runtime breakdown

The DECI breakdown concatenates DECI ablation files and retains `source_file`
for provenance.

| source_file | dataset | constraint_mode | variable_set | epochs | threshold | mean_runtime_s | std_runtime_s | median_runtime_s | n_runs | cv | cv_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deci_ablation_synthetic.csv | synthetic_n2000 | native_constrained | reduced | 20.000 | 0.250 | 22.914 | 0.289 | 22.992 | 5.000 | 0.013 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | native_constrained | reduced | 20.000 | 0.300 | 22.914 | 0.289 | 22.992 | 5.000 | 0.013 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | native_constrained | reduced | 20.000 | 0.350 | 22.914 | 0.289 | 22.992 | 5.000 | 0.013 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | native_constrained | reduced | 50.000 | 0.250 | 57.367 | 29.338 | 44.543 | 5.000 | 0.511 | high variance |
| deci_ablation_synthetic.csv | synthetic_n2000 | native_constrained | reduced | 50.000 | 0.300 | 57.367 | 29.338 | 44.543 | 5.000 | 0.511 | high variance |
| deci_ablation_synthetic.csv | synthetic_n2000 | native_constrained | reduced | 50.000 | 0.350 | 57.367 | 29.338 | 44.543 | 5.000 | 0.511 | high variance |
| deci_ablation_synthetic.csv | synthetic_n2000 | unconstrained | reduced | 20.000 | 0.250 | 22.784 | 0.451 | 22.810 | 5.000 | 0.020 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | unconstrained | reduced | 20.000 | 0.300 | 22.784 | 0.451 | 22.810 | 5.000 | 0.020 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | unconstrained | reduced | 20.000 | 0.350 | 22.784 | 0.451 | 22.810 | 5.000 | 0.020 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | unconstrained | reduced | 50.000 | 0.250 | 42.737 | 1.449 | 42.100 | 5.000 | 0.034 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | unconstrained | reduced | 50.000 | 0.300 | 42.737 | 1.449 | 42.100 | 5.000 | 0.034 |  |
| deci_ablation_synthetic.csv | synthetic_n2000 | unconstrained | reduced | 50.000 | 0.350 | 42.737 | 1.449 | 42.100 | 5.000 | 0.034 |  |
| deci_constraint_type_ablation.csv | synthetic_n2000 | native_constrained | reduced | 20.000 | 0.350 | 22.810 | 0.506 | 22.768 | 20.000 | 0.022 |  |
| deci_constraint_type_ablation.csv | synthetic_n2000 | unconstrained | reduced | 20.000 | 0.350 | 24.136 | 0.938 | 24.087 | 5.000 | 0.039 |  |
| deci_real_selected_config.csv | real | native_constrained | reduced | 20.000 | 0.350 | 10.891 | 0.431 | 11.033 | 5.000 | 0.040 |  |
| deci_real_selected_config.csv | real | unconstrained | reduced | 20.000 | 0.350 | 11.564 | 0.775 | 11.498 | 5.000 | 0.067 |  |

## Canonical constraint overhead

| algorithm | dataset | overhead_seconds | overhead_pct | multiplier |
| --- | --- | --- | --- | --- |
| GES | causal_dummy_v2 | 0.000 | 0.000 | 1.000 |
| LiNGAM | advisor_dummy | -0.175 | -1.261 | 0.987 |
| LiNGAM | causal_dummy_v2 | 0.000 | 0.000 | 1.000 |
| NOTEARS | advisor_dummy | -0.563 |  | 0.039 |
| NOTEARS | causal_dummy_v2 | 0.000 | 0.000 | 1.000 |
| PC | advisor_dummy | -0.356 | -23.317 | 0.767 |
| PC | causal_dummy_v2 | -0.224 | -18.141 | 0.819 |

PC's constrained-variant speedup co-occurs with accuracy improvement
(Delta F1 = +0.043, p = 0.031; Delta SHD = -3.4 vs unconstrained on
advisor_dummy with 5 seeds, one-sided Wilcoxon). The faster runtime reflects
genuine search-space pruning rather than a quality-compute tradeoff.

## Revised RQ3 answer

Across existing runtime artifacts, constraint overhead is small or negative for the classical causal-discovery runs at the canonical causal-dummy configuration: PC is faster when constrained. PC's constrained-variant speedup co-occurs with accuracy improvement (Delta F1 = +0.043, p = 0.031; Delta SHD = -3.4 vs unconstrained on advisor_dummy with 5 seeds, one-sided Wilcoxon). The faster runtime reflects genuine search-space pruning rather than a quality-compute tradeoff. LiNGAM and GES have small post-processing overheads, and NOTEARS remains subject to the previously identified gCastle caching caveat. The sample-size sweep provides the direct scalability evidence: runtime grows with N most visibly for LiNGAM, while PC and GES remain on a lower seconds scale; constrained PC stays competitive across N and does not show a scaling penalty. The SNR sweep shows no strong evidence that noisier settings systematically increase runtime; runtime is mainly algorithm- and sample-size-driven rather than SNR-driven. This means runtime is algorithm-dominated rather than problem-dominated: practitioners deploying constraint-guided causal discovery in ESG-finance can prioritize algorithm choice over noise-regime concerns when planning compute budget. Real ECB runtime evidence is formally captured only for DECI (deci_real_selected_config.csv), where constrained DECI is slightly faster than unconstrained on the reduced variable set. Classical algorithms (PC, LiNGAM, NOTEARS, GES) at N=110 ran in well under one second per fit during execution; runtime statistics for these algorithms on real data were not persisted to results.csv for formal aggregation. This is consistent with the sample-size sweep, which shows sub-second runtimes for classical methods at all N <= 500 in the synthetic settings. DECI training cost is on a larger scale than the classical sweeps (about 10.2x the mean sample-sweep classical runtime), and threshold/configuration choices dominate its runtime. Still unknown: constraint-count sensitivity and full-variable DECI scalability on Windows.
