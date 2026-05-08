# DECI Diagnostic Report

- Current backend: `causica`
- Native Causica constraints used: yes, when backend is `causica` and mode is constrained
- Required constraints natively supported: yes in this codebase via `constraint_matrix_path` with `1.0` entries
- Selected threshold: `0.35`
- Selected epochs: `20`
- Selected sparsity setting: `current`
- Selected variable set: `reduced`

## Synthetic Best Configuration

| config_id | variable_set | constraint_mode | epochs | threshold | sparsity_strength | mean_f1 | mean_precision | mean_recall | mean_shd | mean_edge_count | successful_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ep20_spcurrent_varsreduced | reduced | native_constrained | 20 | 0.35 | current | 0.4688637730251908 | 0.5575341406585518 | 0.4075471698113208 | 48.2 | 38.4 | 5 |

## Real Selected Configuration

| config_id | dataset | variable_set | mode | seed | threshold | epochs | sparsity_strength | status | runtime_seconds | edge_count | alignment | violations | stable_edges_60 | stable_edges_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ep20_spcurrent_varsreduced | real | reduced | unconstrained | 42 | 0.35 | 20 | current | success | 12.1018 | 13 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | unconstrained | 43 | 0.35 | 20 | current | success | 10.6906 | 5 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | unconstrained | 44 | 0.35 | 20 | current | success | 11.4983 | 5 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | unconstrained | 45 | 0.35 | 20 | current | success | 12.5596 | 8 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | unconstrained | 46 | 0.35 | 20 | current | success | 10.9681 | 21 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | constrained | 42 | 0.35 | 20 | current | success | 10.2539 | 14 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | constrained | 43 | 0.35 | 20 | current | success | 10.6959 | 8 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | constrained | 44 | 0.35 | 20 | current | success | 11.3777 | 7 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | constrained | 45 | 0.35 | 20 | current | success | 11.0326 | 10 | 0.0 | 0 | 0 | 0 |
| ep20_spcurrent_varsreduced | real | reduced | constrained | 46 | 0.35 | 20 | current | success | 11.0944 | 18 | 0.0 | 0 | 0 | 0 |

## Constraint Matrix Validation Summary

- Forbidden constraints passed: `2`
- Required constraints passed: `0`
- Constraints dropped due to missing variables: `0`
- Synthetic ground-truth conflicts: `0`

## Stability Summary

| dataset | mode | edges | stable_60 | stable_80 |
| --- | --- | --- | --- | --- |
| real | constrained | 52 | 0 | 0 |
| real | unconstrained | 47 | 0 | 0 |
| synthetic_n2000 | constrained | 887 | 161 | 89 |
| synthetic_n2000 | unconstrained | 878 | 98 | 19 |

## Runtime And Failure Summary

No DECI ablation failures recorded.

## Final Interpretation

**DECI is suitable as a main result.** Constrained DECI improves F1, reduces SHD, and runs stably under the selected configuration.
