# Stability Analysis Report

`25_stability_analysis.py`

## causal_dummy
| dataset | algorithm | constraint_mode | n_graphs | n_unique_edges | mean_edge_count | stable_edges_60 | jaccard_mean | jaccard_std | stable_true_positive_edges | stable_false_positive_edges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| causal_dummy | GES | constrained | 5.0000 | 82.0000 | 50.2000 | 44.0000 | 0.6347 | 0.0819 | 36.0000 | 8.0000 |
| causal_dummy | GES | unconstrained | 5.0000 | 99.0000 | 60.8000 | 54.0000 | 0.6213 | 0.0874 | 36.0000 | 18.0000 |
| causal_dummy | LiNGAM | constrained | 5.0000 | 94.0000 | 56.8000 | 50.0000 | 0.5752 | 0.0806 | 38.0000 | 12.0000 |
| causal_dummy | LiNGAM | unconstrained | 5.0000 | 109.0000 | 64.6000 | 57.0000 | 0.5587 | 0.0831 | 38.0000 | 19.0000 |
| causal_dummy | NOTEARS | constrained | 5.0000 | 18.0000 | 15.4000 | 15.0000 | 0.8475 | 0.0669 | 15.0000 | 0.0000 |
| causal_dummy | NOTEARS | unconstrained | 5.0000 | 32.0000 | 26.8000 | 27.0000 | 0.8141 | 0.0879 | 15.0000 | 12.0000 |
| causal_dummy | PC | constrained | 5.0000 | 60.0000 | 47.2000 | 44.0000 | 0.8733 | 0.0168 | 44.0000 | 0.0000 |
| causal_dummy | PC | unconstrained | 5.0000 | 83.0000 | 41.4000 | 37.0000 | 0.4391 | 0.0992 | 33.0000 | 4.0000 |

## advisor_dummy
| dataset | algorithm | constraint_mode | n_graphs | n_unique_edges | mean_edge_count | stable_edges_60 | jaccard_mean | jaccard_std | stable_true_positive_edges | stable_false_positive_edges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advisor_dummy | GES | forbidden_only | 5.0000 | 108.0000 | 26.0000 | 5.0000 | 0.0613 | 0.0519 |  |  |
| advisor_dummy | GES | full_reference_sanity | 5.0000 | 152.0000 | 71.6000 | 51.0000 | 0.5204 | 0.0447 |  |  |
| advisor_dummy | GES | required_light | 5.0000 | 112.0000 | 30.0000 | 9.0000 | 0.1324 | 0.0513 |  |  |
| advisor_dummy | GES | unconstrained | 5.0000 | 127.0000 | 30.8000 | 6.0000 | 0.0654 | 0.0495 |  |  |

## real_ecb
| dataset | algorithm | constraint_mode | n_graphs | n_unique_edges | mean_edge_count | stable_edges_60 | jaccard_mean | jaccard_std | stable_true_positive_edges | stable_false_positive_edges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_ecb | GES | forbidden_only | 5.0000 | 56.0000 | 21.0000 | 14.0000 | 0.2836 | 0.0647 |  |  |
| real_ecb | GES | unconstrained | 5.0000 | 58.0000 | 21.6000 | 14.0000 | 0.2773 | 0.0616 |  |  |
