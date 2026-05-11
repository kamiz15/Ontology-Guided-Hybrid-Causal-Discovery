# Causal Dummy v2 Final Algorithm Comparison

## Command

`C:/Users/User/Desktop/go/.venv/Scripts/python.exe 24_causal_dummy_final_comparison.py --seeds 42,43,44,45,46`

## Design

- Dataset: causal dummy v2.
- Sample size: n=3000.
- SNR: 0.6.
- Seeds: 42, 43, 44, 45, 46.
- Algorithms: PC, LiNGAM, NOTEARS, GES.
- Constraint modes: unconstrained and constrained.
- Metrics are computed against the generated ground-truth DAG.
- DECI is excluded from the main table unless explicitly run and complete.

## Outputs

- `outputs\experiments\causal_dummy_final_comparison_raw.csv`
- `outputs\experiments\causal_dummy_final_comparison_summary.csv`
- `outputs\experiments\causal_dummy_final_comparison_report.md`
- `outputs\figures\causal_dummy_final_f1.png`
- `outputs\figures\causal_dummy_final_shd.png`
- `outputs\figures\causal_dummy_final_violations.png`
- `outputs\figures\causal_dummy_final_runtime.png`

## Failures

No failed rows.

## Headline Metrics

| algorithm | constraint_mode | f1_mean | f1_std | shd_mean | shd_std | precision_mean | recall_mean | violations_mean | runtime_seconds_mean | successful_runs | failed_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GES | constrained | 0.7444 | 0.0482 | 24.6000 | 4.7223 | 0.7139 | 0.7783 | 0.0000 | 2.4946 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.6709 | 0.0508 | 35.2000 | 5.8052 | 0.5898 | 0.7783 | 10.6000 | 2.4946 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 0.7418 | 0.0660 | 26.8000 | 8.0747 | 0.6751 | 0.8261 | 0.0000 | 10.2449 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 0.6901 | 0.0655 | 34.6000 | 9.2358 | 0.5942 | 0.8261 | 7.8000 | 10.2449 | 5.0000 | 0.0000 |
| NOTEARS | constrained | 0.5009 | 0.0369 | 30.6000 | 1.5166 | 1.0000 | 0.3348 | 0.0000 | 14.9887 | 5.0000 | 0.0000 |
| NOTEARS | unconstrained | 0.4232 | 0.0433 | 42.0000 | 3.3166 | 0.5752 | 0.3348 | 11.4000 | 14.9887 | 5.0000 | 0.0000 |
| PC | constrained | 0.9443 | 0.0085 | 5.2000 | 0.8367 | 0.9324 | 0.9565 | 0.0000 | 1.0112 | 5.0000 | 0.0000 |
| PC | unconstrained | 0.6824 | 0.0790 | 27.8000 | 7.0143 | 0.7216 | 0.6478 | 8.8000 | 1.2353 | 5.0000 | 0.0000 |
