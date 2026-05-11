# Sample-Size Sensitivity Experiment

## Command

`C:/Users/User/Desktop/go/.venv/Scripts/python.exe 23_sample_size_sensitivity.py --sample-sizes 100,250,500,1000,2000,3000 --seeds 42,43,44,45,46`

## Design

- Dataset: causal dummy v2 generated from the same fixed DAG at every sample size.
- Sample sizes: 100, 250, 500, 1000, 2000, 3000.
- Seeds: 42, 43, 44, 45, 46.
- SNR: 0.6.
- Algorithms: PC, LiNGAM, GES.
- Constraint modes: unconstrained and constrained.
- The design tests whether constraints help more when sample size is limited.

## Outputs

- `outputs\experiments\sample_size_sensitivity_results.csv`
- `outputs\experiments\sample_size_sensitivity_summary.csv`
- `outputs\experiments\sample_size_sensitivity_report.md`
- `outputs\figures\sample_size_f1.png`
- `outputs\figures\sample_size_shd.png`
- `outputs\figures\sample_size_precision.png`
- `outputs\figures\sample_size_recall.png`
- `outputs\figures\sample_size_runtime.png`

## Failures

No failed rows.

## Headline Metrics

| algorithm | constraint_mode | n | f1_mean | f1_std | shd_mean | shd_std | precision_mean | recall_mean | runtime_seconds_mean | successful_runs | failed_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GES | constrained | 100.0000 | 0.5699 | 0.0207 | 41.0000 | 3.5355 | 0.5563 | 0.5913 | 2.8460 | 5.0000 | 0.0000 |
| GES | constrained | 250.0000 | 0.6879 | 0.0397 | 31.0000 | 5.0990 | 0.6456 | 0.7391 | 2.7760 | 5.0000 | 0.0000 |
| GES | constrained | 500.0000 | 0.7210 | 0.0192 | 26.4000 | 1.5166 | 0.7030 | 0.7435 | 2.4625 | 5.0000 | 0.0000 |
| GES | constrained | 1000.0000 | 0.7312 | 0.0246 | 25.4000 | 2.0736 | 0.7129 | 0.7522 | 2.7102 | 5.0000 | 0.0000 |
| GES | constrained | 2000.0000 | 0.7407 | 0.0267 | 24.6000 | 2.3022 | 0.7193 | 0.7652 | 2.4392 | 5.0000 | 0.0000 |
| GES | constrained | 3000.0000 | 0.7444 | 0.0482 | 24.6000 | 4.7223 | 0.7139 | 0.7783 | 2.5080 | 5.0000 | 0.0000 |
| GES | unconstrained | 100.0000 | 0.4984 | 0.0302 | 54.6000 | 3.1305 | 0.4320 | 0.5913 | 2.8460 | 5.0000 | 0.0000 |
| GES | unconstrained | 250.0000 | 0.6258 | 0.0388 | 40.8000 | 5.4498 | 0.5433 | 0.7391 | 2.7760 | 5.0000 | 0.0000 |
| GES | unconstrained | 500.0000 | 0.6488 | 0.0434 | 37.0000 | 4.4721 | 0.5759 | 0.7435 | 2.4625 | 5.0000 | 0.0000 |
| GES | unconstrained | 1000.0000 | 0.6553 | 0.0370 | 36.4000 | 3.9115 | 0.5807 | 0.7522 | 2.7102 | 5.0000 | 0.0000 |
| GES | unconstrained | 2000.0000 | 0.6678 | 0.0372 | 35.0000 | 3.8730 | 0.5927 | 0.7652 | 2.4392 | 5.0000 | 0.0000 |
| GES | unconstrained | 3000.0000 | 0.6709 | 0.0508 | 35.2000 | 5.8052 | 0.5898 | 0.7783 | 2.5080 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 100.0000 | 0.5919 | 0.0574 | 39.4000 | 6.1074 | 0.5692 | 0.6217 | 3.8253 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 250.0000 | 0.6804 | 0.0422 | 33.4000 | 6.7676 | 0.6157 | 0.7652 | 3.9089 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 500.0000 | 0.6925 | 0.0729 | 32.2000 | 9.0111 | 0.6255 | 0.7783 | 4.4729 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 1000.0000 | 0.7336 | 0.0836 | 26.8000 | 9.2574 | 0.6916 | 0.7870 | 5.6762 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 2000.0000 | 0.6987 | 0.0738 | 32.0000 | 9.5131 | 0.6245 | 0.7957 | 7.7756 | 5.0000 | 0.0000 |
| LiNGAM | constrained | 3000.0000 | 0.7418 | 0.0660 | 26.8000 | 8.0747 | 0.6751 | 0.8261 | 10.1203 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 100.0000 | 0.5298 | 0.0656 | 51.0000 | 8.8600 | 0.4642 | 0.6217 | 3.8253 | 5.0000 | 0.0000 |
| LiNGAM | unconstrained | 250.0000 | 0.6258 | 0.0343 | 42.4000 | 6.6558 | 0.5316 | 0.7652 | 3.9089 | 5.0000 | 0.0000 |
| ... | 16 more rows | | | | | | | | | | |
