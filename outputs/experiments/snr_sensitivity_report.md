# SNR Sensitivity Experiment

## Command

`C:/Users/User/Desktop/go/.venv/Scripts/python.exe 22_snr_sensitivity_sweep.py --snr-grid 0.2,0.4,0.6,0.8,1 --seeds 42,43,44,45,46`

## Design

- Dataset: causal dummy v2 generated from a fixed structural DAG.
- Sample size per run: n=3000.
- SNR grid: 0.2, 0.4, 0.6, 0.8, 1.0.
- Seeds: 42, 43, 44, 45, 46.
- Algorithms: PC, LiNGAM, GES.
- Constraint modes: unconstrained and constrained.
- Metrics are computed against the generated ground-truth DAG.

## Outputs

- `outputs\experiments\snr_sensitivity_results.csv`
- `outputs\experiments\snr_sensitivity_summary.csv`
- `outputs\experiments\snr_sensitivity_report.md`
- `outputs\figures\snr_f1_sweep.png`
- `outputs\figures\snr_shd_sweep.png`
- `outputs\figures\snr_precision_sweep.png`
- `outputs\figures\snr_recall_sweep.png`
- `outputs\figures\snr_runtime_sweep.png`

## Failures

No failed rows.

## Headline Metrics

| algorithm | constraint_mode | snr | f1_mean | f1_std | shd_mean | shd_std | precision_mean | recall_mean | runtime_seconds_mean | successful_runs | failed_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GES | constrained | 0.2000 | 0.7281 | 0.0452 | 23.2000 | 3.4205 | 0.7884 | 0.6783 | 1607.5082 | 5.0000 | 0.0000 |
| GES | constrained | 0.4000 | 0.7643 | 0.0393 | 22.2000 | 3.7014 | 0.7470 | 0.7826 | 2.8719 | 5.0000 | 0.0000 |
| GES | constrained | 0.6000 | 0.7444 | 0.0482 | 24.6000 | 4.7223 | 0.7139 | 0.7783 | 2.4989 | 5.0000 | 0.0000 |
| GES | constrained | 0.8000 | 0.7238 | 0.0317 | 26.4000 | 2.9665 | 0.6976 | 0.7522 | 2.4833 | 5.0000 | 0.0000 |
| GES | constrained | 1.0000 | 0.7259 | 0.0112 | 26.6000 | 1.5166 | 0.6908 | 0.7652 | 2.7797 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.2000 | 0.6265 | 0.0566 | 37.2000 | 5.6303 | 0.5821 | 0.6783 | 2.2037 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.4000 | 0.6874 | 0.0427 | 32.8000 | 4.9193 | 0.6130 | 0.7826 | 2.8286 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.6000 | 0.6709 | 0.0508 | 35.2000 | 5.8052 | 0.5898 | 0.7783 | 2.5026 | 5.0000 | 0.0000 |
| GES | unconstrained | 0.8000 | 0.6594 | 0.0378 | 35.8000 | 4.4385 | 0.5872 | 0.7522 | 2.5459 | 5.0000 | 0.0000 |
| GES | unconstrained | 1.0000 | 0.6631 | 0.0142 | 35.8000 | 2.0494 | 0.5851 | 0.7652 | 2.8212 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 0.2000 | 0.8661 | 0.0133 | 11.6000 | 0.8944 | 0.9222 | 0.8174 | 9.0821 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 0.4000 | 0.7382 | 0.0673 | 27.0000 | 8.0312 | 0.6749 | 0.8174 | 10.0197 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 0.6000 | 0.7418 | 0.0660 | 26.8000 | 8.0747 | 0.6751 | 0.8261 | 10.0445 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 0.8000 | 0.6054 | 0.0494 | 38.6000 | 6.3087 | 0.5766 | 0.6391 | 9.8932 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 1.0000 | 0.5722 | 0.0534 | 42.6000 | 8.0498 | 0.5392 | 0.6130 | 9.8714 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 0.2000 | 0.7915 | 0.0283 | 19.8000 | 2.6833 | 0.7675 | 0.8174 | 9.6510 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 0.4000 | 0.6795 | 0.0716 | 36.0000 | 9.7724 | 0.5834 | 0.8174 | 9.9400 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 0.6000 | 0.6901 | 0.0655 | 34.6000 | 9.2358 | 0.5942 | 0.8261 | 10.0169 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 0.8000 | 0.5166 | 0.0449 | 55.4000 | 7.6026 | 0.4345 | 0.6391 | 9.9833 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 1.0000 | 0.4885 | 0.0468 | 59.6000 | 9.3968 | 0.4074 | 0.6130 | 9.8913 | 5.0000 | 0.0000 |
| ... | 10 more rows | | | | | | | | | | |
