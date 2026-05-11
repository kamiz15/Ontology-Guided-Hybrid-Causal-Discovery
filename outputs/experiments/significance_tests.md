# Significance Tests: Constrained vs Unconstrained (forbidden_only)

Dataset: advisor_dummy (ontology-derived reference DAG evaluation)
Test: one-sided paired Wilcoxon signed-rank test, n=5 seeds
H1 (F1): constrained F1 > unconstrained F1
H1 (SHD): constrained SHD < unconstrained SHD
Significance: * p<0.05, ** p<0.01, *** p<0.001

Note: with n=5, the minimum achievable p-value is 1/32 ≈ 0.031.
When all 5 differences have the same sign and are non-zero, p ≈ 0.031.
Results marked 'n/a' indicate all paired differences were zero
(algorithm produced identical output regardless of constraints).

## F1 Results

| Algorithm | F1 unconstrained | F1 constrained | Delta F1 | p-value |
|-----------|-----------------|----------------|----------|---------|
| GES | 0.0105 | 0.0110 | +0.0004 | n/a |
| LiNGAM | 0.0000 | 0.0000 | +0.0000 | n/a |
| NOTEARS | 0.0000 | 0.0000 | +0.0000 | n/a |
| PC | 0.0127 | 0.0552 | +0.0425 | 0.031 * |
| DECI | 0.0180 | 0.0110 | -0.0070 | 0.750 |

## SHD Results (lower is better)

| Algorithm | SHD unconstrained | SHD constrained | Delta SHD | p-value |
|-----------|------------------|-----------------|-----------|---------|
| GES | 76.0 | 71.2 | -4.8 | 0.031 * |
| LiNGAM | 58.2 | 57.4 | -0.8 | n/a |
| NOTEARS | 46.0 | 46.0 | +0.0 | n/a |
| PC | 127.6 | 124.2 | -3.4 | 0.031 * |
| DECI | 60.8 | 70.2 | +9.4 | 0.844 |

## Notes

Algorithms with zero F1 in both conditions (LiNGAM, NOTEARS) cannot
be tested — the test requires at least one non-zero difference.
This confirms these algorithms found no signal on this dataset
regardless of constraint mode, consistent with the low-signal nature
of the ontology-derived reference DAG evaluation.