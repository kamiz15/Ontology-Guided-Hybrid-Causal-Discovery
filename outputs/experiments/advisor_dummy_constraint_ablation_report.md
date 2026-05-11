# Advisor Dummy Constraint Ablation Report

The advisor-provided dummy dataset is evaluated against an
ontology-derived reference DAG unless an explicit advisor causal rule file is
present.

- `forbidden_only` is the main constrained experiment.
- `required_light` is a secondary experiment with only a small set of
  high-confidence required edges.
- `full_reference_sanity` is only a constraint-enforcement sanity check.
- Near-perfect results under `full_reference_sanity` reflect constraint
  enforcement, not independent discovery.

## No Constraints

| constraint mode | algorithm | mode | successful runs | F1 directed mean +/- std | SHD mean +/- std | mean edge count | mean violations |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| none | ges_postproc | constrained | 5 | 0.0105 +/- 0.0235 | 76.00 +/- 3.39 | 30.80 | 0.00 |
| none | ges_postproc | unconstrained | 5 | 0.0105 +/- 0.0235 | 76.00 +/- 3.39 | 30.80 | 0.00 |
| none | lingam | constrained | 5 | 0.0000 +/- 0.0000 | 58.20 +/- 5.45 | 12.20 | 0.00 |
| none | lingam | unconstrained | 5 | 0.0000 +/- 0.0000 | 58.20 +/- 5.45 | 12.20 | 0.00 |
| none | notears_postproc | constrained | 5 | 0.0000 +/- 0.0000 | 46.00 +/- 0.00 | 0.00 | 0.00 |
| none | notears_postproc | unconstrained | 5 | 0.0000 +/- 0.0000 | 46.00 +/- 0.00 | 0.00 | 0.00 |
| none | pc | constrained | 5 | 0.0127 +/- 0.0139 | 127.60 +/- 7.33 | 83.20 | 0.00 |
| none | pc | unconstrained | 5 | 0.0127 +/- 0.0139 | 127.60 +/- 7.33 | 83.20 | 0.00 |

## Main: Forbidden Only

| constraint mode | algorithm | mode | successful runs | F1 directed mean +/- std | SHD mean +/- std | mean edge count | mean violations |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| forbidden_only | deci_native_constrained | constrained | 5 | 0.0110 +/- 0.0152 | 70.20 +/- 9.04 | 25.00 | 0.00 |
| forbidden_only | deci_native_unconstrained | unconstrained | 5 | 0.0180 +/- 0.0247 | 60.80 +/- 12.72 | 16.00 | 3.40 |
| forbidden_only | ges_postproc | constrained | 5 | 0.0110 +/- 0.0245 | 71.20 +/- 4.15 | 26.00 | 0.00 |
| forbidden_only | ges_postproc | unconstrained | 5 | 0.0105 +/- 0.0235 | 76.00 +/- 3.39 | 30.80 | 4.80 |
| forbidden_only | lingam | constrained | 5 | 0.0000 +/- 0.0000 | 57.40 +/- 4.83 | 11.40 | 0.00 |
| forbidden_only | lingam | unconstrained | 5 | 0.0000 +/- 0.0000 | 58.20 +/- 5.45 | 12.20 | 0.80 |
| forbidden_only | notears_postproc | constrained | 5 | 0.0000 +/- 0.0000 | 46.00 +/- 0.00 | 0.00 | 0.00 |
| forbidden_only | notears_postproc | unconstrained | 5 | 0.0000 +/- 0.0000 | 46.00 +/- 0.00 | 0.00 | 0.00 |
| forbidden_only | pc | constrained | 5 | 0.0552 +/- 0.0220 | 124.20 +/- 7.40 | 85.40 | 0.00 |
| forbidden_only | pc | unconstrained | 5 | 0.0127 +/- 0.0139 | 127.60 +/- 7.33 | 83.20 | 21.80 |

## Secondary: Required Light

| constraint mode | algorithm | mode | successful runs | F1 directed mean +/- std | SHD mean +/- std | mean edge count | mean violations |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| required_light | deci_native_constrained | constrained | 5 | 0.1276 +/- 0.0267 | 64.00 +/- 11.53 | 27.20 | 0.00 |
| required_light | deci_native_unconstrained | unconstrained | 5 | 0.0180 +/- 0.0247 | 60.80 +/- 12.72 | 16.00 | 3.40 |
| required_light | ges_postproc | constrained | 5 | 0.1159 +/- 0.0230 | 67.20 +/- 4.15 | 30.00 | 0.00 |
| required_light | ges_postproc | unconstrained | 5 | 0.0105 +/- 0.0235 | 76.00 +/- 3.39 | 30.80 | 4.80 |
| required_light | lingam | constrained | 5 | 0.0000 +/- 0.0000 | 57.40 +/- 4.83 | 11.40 | 0.00 |
| required_light | lingam | unconstrained | 5 | 0.0000 +/- 0.0000 | 58.20 +/- 5.45 | 12.20 | 0.80 |
| required_light | notears_postproc | constrained | 5 | 0.1600 +/- 0.0000 | 42.00 +/- 0.00 | 4.00 | 0.00 |
| required_light | notears_postproc | unconstrained | 5 | 0.0000 +/- 0.0000 | 46.00 +/- 0.00 | 0.00 | 0.00 |
| required_light | pc | constrained | 5 | 0.0552 +/- 0.0220 | 124.20 +/- 7.40 | 85.40 | 0.00 |
| required_light | pc | unconstrained | 5 | 0.0127 +/- 0.0139 | 127.60 +/- 7.33 | 83.20 | 21.80 |

## Sanity Check: Full Reference Required

| constraint mode | algorithm | mode | successful runs | F1 directed mean +/- std | SHD mean +/- std | mean edge count | mean violations |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| full_reference_sanity | deci_native_constrained | constrained | 5 | 0.8165 +/- 0.0622 | 21.20 +/- 8.61 | 67.20 | 0.00 |
| full_reference_sanity | deci_native_unconstrained | unconstrained | 5 | 0.0180 +/- 0.0247 | 60.80 +/- 12.72 | 16.00 | 3.40 |
| full_reference_sanity | ges_postproc | constrained | 5 | 0.7830 +/- 0.0266 | 25.60 +/- 3.97 | 71.60 | 0.00 |
| full_reference_sanity | ges_postproc | unconstrained | 5 | 0.0105 +/- 0.0235 | 76.00 +/- 3.39 | 30.80 | 4.80 |
| full_reference_sanity | lingam | constrained | 5 | 0.0000 +/- 0.0000 | 57.40 +/- 4.83 | 11.40 | 0.00 |
| full_reference_sanity | lingam | unconstrained | 5 | 0.0000 +/- 0.0000 | 58.20 +/- 5.45 | 12.20 | 0.80 |
| full_reference_sanity | notears_postproc | constrained | 5 | 1.0000 +/- 0.0000 | 0.00 +/- 0.00 | 46.00 | 0.00 |
| full_reference_sanity | notears_postproc | unconstrained | 5 | 0.0000 +/- 0.0000 | 46.00 +/- 0.00 | 0.00 | 0.00 |
| full_reference_sanity | pc | constrained | 5 | 0.0552 +/- 0.0220 | 124.20 +/- 7.40 | 85.40 | 0.00 |
| full_reference_sanity | pc | unconstrained | 5 | 0.0127 +/- 0.0139 | 127.60 +/- 7.33 | 83.20 | 21.80 |

