# Final Experiment Summary

## Generated Tables

- `outputs\experiments\final_causal_dummy_comparison.csv`
- `outputs\experiments\final_snr_sensitivity.csv`
- `outputs\experiments\final_sample_size_sensitivity.csv`
- `outputs\experiments\final_advisor_dummy_constraint_compliance.csv`
- `outputs\experiments\final_real_ecb_case_study.csv`

## Causal Dummy v2

Main causal-recovery metrics are computed against the generated ground-truth DAG.

| algorithm | constraint_mode | F1_mean | SHD_mean | precision_mean | recall_mean | violations_mean |
| --- | --- | --- | --- | --- | --- | --- |
| PC | constrained | 0.9443 | 5.2000 | 0.9324 | 0.9565 | 0.0000 |
| GES | constrained | 0.7444 | 24.6000 | 0.7139 | 0.7783 | 0.0000 |
| LiNGAM | constrained | 0.7418 | 26.8000 | 0.6751 | 0.8261 | 0.0000 |
| LiNGAM | unconstrained | 0.6901 | 34.6000 | 0.5942 | 0.8261 | 7.8000 |
| PC | unconstrained | 0.6824 | 27.8000 | 0.7216 | 0.6478 | 8.8000 |

## SNR Sensitivity

| algorithm | constraint_mode | snr | f1_mean_std | shd_mean_std | precision_mean_std | recall_mean_std | runtime_seconds_mean_std | successful_runs | failed_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GES | constrained | 0.2000 | 0.7281 +/- 0.0452 | 23.2000 +/- 3.4205 | 0.7884 +/- 0.0409 | 0.6783 +/- 0.0623 | 1607.5082 +/- 3589.6057 | 5.0000 | 0.0000 |
| GES | constrained | 0.4000 | 0.7643 +/- 0.0393 | 22.2000 +/- 3.7014 | 0.7470 +/- 0.0392 | 0.7826 +/- 0.0407 | 2.8719 +/- 0.3587 | 5.0000 | 0.0000 |
| GES | constrained | 0.6000 | 0.7444 +/- 0.0482 | 24.6000 +/- 4.7223 | 0.7139 +/- 0.0502 | 0.7783 +/- 0.0519 | 2.4989 +/- 0.0822 | 5.0000 | 0.0000 |
| GES | constrained | 0.8000 | 0.7238 +/- 0.0317 | 26.4000 +/- 2.9665 | 0.6976 +/- 0.0303 | 0.7522 +/- 0.0364 | 2.4833 +/- 0.0749 | 5.0000 | 0.0000 |
| GES | constrained | 1.0000 | 0.7259 +/- 0.0112 | 26.6000 +/- 1.5166 | 0.6908 +/- 0.0211 | 0.7652 +/- 0.0097 | 2.7797 +/- 0.1473 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.2000 | 0.6265 +/- 0.0566 | 37.2000 +/- 5.6303 | 0.5821 +/- 0.0524 | 0.6783 +/- 0.0623 | 2.2037 +/- 0.6558 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.4000 | 0.6874 +/- 0.0427 | 32.8000 +/- 4.9193 | 0.6130 +/- 0.0434 | 0.7826 +/- 0.0407 | 2.8286 +/- 0.3383 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.6000 | 0.6709 +/- 0.0508 | 35.2000 +/- 5.8052 | 0.5898 +/- 0.0503 | 0.7783 +/- 0.0519 | 2.5026 +/- 0.0966 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.8000 | 0.6594 +/- 0.0378 | 35.8000 +/- 4.4385 | 0.5872 +/- 0.0393 | 0.7522 +/- 0.0364 | 2.5459 +/- 0.0825 | 5.0000 | 0.0000 |
| GES | unconstrained | 1.0000 | 0.6631 +/- 0.0142 | 35.8000 +/- 2.0494 | 0.5851 +/- 0.0190 | 0.7652 +/- 0.0097 | 2.8212 +/- 0.1499 | 5.0000 | 0.0000 |

## Sample-Size Sensitivity

| algorithm | constraint_mode | n | f1_mean_std | shd_mean_std | precision_mean_std | recall_mean_std | runtime_seconds_mean_std | successful_runs | failed_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GES | constrained | 100.0000 | 0.5699 +/- 0.0207 | 41.0000 +/- 3.5355 | 0.5563 +/- 0.0474 | 0.5913 +/- 0.0583 | 2.8460 +/- 0.4357 | 5.0000 | 0.0000 |
| GES | constrained | 250.0000 | 0.6879 +/- 0.0397 | 31.0000 +/- 5.0990 | 0.6456 +/- 0.0601 | 0.7391 +/- 0.0344 | 2.7760 +/- 0.3278 | 5.0000 | 0.0000 |
| GES | constrained | 500.0000 | 0.7210 +/- 0.0192 | 26.4000 +/- 1.5166 | 0.7030 +/- 0.0293 | 0.7435 +/- 0.0563 | 2.4625 +/- 0.1707 | 5.0000 | 0.0000 |
| GES | constrained | 1000.0000 | 0.7312 +/- 0.0246 | 25.4000 +/- 2.0736 | 0.7129 +/- 0.0263 | 0.7522 +/- 0.0451 | 2.7102 +/- 0.4523 | 5.0000 | 0.0000 |
| GES | constrained | 2000.0000 | 0.7407 +/- 0.0267 | 24.6000 +/- 2.3022 | 0.7193 +/- 0.0286 | 0.7652 +/- 0.0471 | 2.4392 +/- 0.1703 | 5.0000 | 0.0000 |
| GES | constrained | 3000.0000 | 0.7444 +/- 0.0482 | 24.6000 +/- 4.7223 | 0.7139 +/- 0.0502 | 0.7783 +/- 0.0519 | 2.5080 +/- 0.0878 | 5.0000 | 0.0000 |
| GES | unconstrained | 100.0000 | 0.4984 +/- 0.0302 | 54.6000 +/- 3.1305 | 0.4320 +/- 0.0217 | 0.5913 +/- 0.0583 | 2.8460 +/- 0.4357 | 5.0000 | 0.0000 |
| GES | unconstrained | 250.0000 | 0.6258 +/- 0.0388 | 40.8000 +/- 5.4498 | 0.5433 +/- 0.0442 | 0.7391 +/- 0.0344 | 2.7760 +/- 0.3278 | 5.0000 | 0.0000 |
| GES | unconstrained | 500.0000 | 0.6488 +/- 0.0434 | 37.0000 +/- 4.4721 | 0.5759 +/- 0.0377 | 0.7435 +/- 0.0563 | 2.4625 +/- 0.1707 | 5.0000 | 0.0000 |
| GES | unconstrained | 1000.0000 | 0.6553 +/- 0.0370 | 36.4000 +/- 3.9115 | 0.5807 +/- 0.0336 | 0.7522 +/- 0.0451 | 2.7102 +/- 0.4523 | 5.0000 | 0.0000 |

## Advisor Dummy

Advisor-dummy rows are reference-DAG alignment and constraint-compliance outputs only.

| algorithm | mode | constraint_mode | reference_dag_f1_mean | reference_dag_f1_std | reference_dag_shd_mean | reference_dag_shd_std | violations_mean | violations_std | edge_count_mean | runtime_seconds_mean | successful_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DECI | constrained | forbidden_only | 0.0110 | 0.0152 | 70.2000 | 9.0388 | 0.0000 | 0.0000 | 25.0000 | 30.5500 | 5.0000 |
| DECI | unconstrained | forbidden_only | 0.0180 | 0.0247 | 60.8000 | 12.7161 | 3.4000 | 2.6077 | 16.0000 | 33.4507 | 5.0000 |
| DECI | constrained | full_reference_sanity | 0.8165 | 0.0622 | 21.2000 | 8.6139 | 0.0000 | 0.0000 | 67.2000 | 28.8246 | 5.0000 |
| DECI | unconstrained | full_reference_sanity | 0.0180 | 0.0247 | 60.8000 | 12.7161 | 3.4000 | 2.6077 | 16.0000 | 27.7554 | 5.0000 |
| DECI | constrained | required_light | 0.1276 | 0.0267 | 64.0000 | 11.5326 | 0.0000 | 0.0000 | 27.2000 | 28.6316 | 5.0000 |
| DECI | unconstrained | required_light | 0.0180 | 0.0247 | 60.8000 | 12.7161 | 3.4000 | 2.6077 | 16.0000 | 29.3297 | 5.0000 |
| GES | constrained | forbidden_only | 0.0110 | 0.0245 | 71.2000 | 4.1473 | 0.0000 | 0.0000 | 26.0000 | 1.0196 | 5.0000 |
| GES | unconstrained | forbidden_only | 0.0105 | 0.0235 | 76.0000 | 3.3912 | 4.8000 | 2.1679 | 30.8000 | 1.0938 | 5.0000 |
| GES | constrained | full_reference_sanity | 0.7830 | 0.0266 | 25.6000 | 3.9749 | 0.0000 | 0.0000 | 71.6000 | 1.0343 | 5.0000 |
| GES | unconstrained | full_reference_sanity | 0.0105 | 0.0235 | 76.0000 | 3.3912 | 4.8000 | 2.1679 | 30.8000 | 1.0893 | 5.0000 |

## Real ECB Case Study

Real ECB rows are literature-alignment and violation-reduction outputs only. No F1 or SHD is reported here.

| algorithm | constraint_mode | constraint_handling | alignment_mean | alignment_std | edge_count_mean | violations_mean | stable_edges_60 | stable_edges_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC | none | native | 0.0000 | 0.0000 | 16.3333 | 0.3333 |  |  |
| PC | forbidden_only | background_knowledge | 0.3333 | 0.5774 | 16.6667 | 0.0000 |  |  |
| LiNGAM | none | native | 0.4000 | 0.5477 | 24.4000 | 0.2000 |  |  |
| LiNGAM | forbidden_only | postproc | 0.4000 | 0.5477 | 24.2000 | 0.0000 |  |  |
| NOTEARS | none | native | 0.0000 | 0.0000 | 17.2000 | 0.2000 |  |  |
| NOTEARS | forbidden_only | postproc | 0.0000 | 0.0000 | 17.0000 | 0.0000 |  |  |
| GES | none | native | 0.3000 | 0.4472 | 21.6000 | 0.6000 | 14.0000 | 8.0000 |
| GES | forbidden_only | postproc | 0.4000 | 0.5477 | 21.0000 | 0.0000 | 14.0000 | 8.0000 |
| DECI | none | native | 0.0000 | 0.0000 | 10.4000 | 0.0000 | 0.0000 | 0.0000 |
| DECI | forbidden_only | native | 0.0000 | 0.0000 | 11.4000 | 0.0000 | 0.0000 | 0.0000 |
