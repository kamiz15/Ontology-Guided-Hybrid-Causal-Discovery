# Final Consolidated Results Summary

## Dataset Roles

The advisor dummy dataset is the official dummy-data experiment because it
uses the advisor-provided CSV and metadata workbook. No explicit
advisor-provided causal generation rule file was found, so advisor-dummy F1
and SHD are computed against an ontology-derived reference DAG. These metrics
measure reference-DAG alignment, not recovery of an experimentally known
causal mechanism.

The real ECB dataset is used as a real-data case study. It has no reference
DAG, so F1 and SHD are not reported. Real-data results summarize discovered
edge counts, literature alignment, and forbidden-edge violations only.

## Constraint Settings

`forbidden_only` is the main constrained setting because it prevents
ontology-impossible or reverse-supported directions without giving the
algorithm the complete reference structure. This makes the comparison closer
to constraint-guided discovery.

`required_light` is a secondary constraint-assistance condition with only a
small set of high-confidence definitional edges.

`full_reference_sanity` is not main discovery evidence. It supplies all
reference-DAG edges as required constraints and therefore checks whether the
constraint adapters and post-processing layers obey the reference structure.

## Main Findings

Across algorithms, forbidden-only constraints eliminate or reduce
forbidden-edge violations. The strongest advisor-dummy violation reductions
are visible for PC, GES, DECI, and LiNGAM.

PC shows the clearest main-mode reference-DAG alignment improvement (F1 0.0127 to 0.0552).

GES is a classical score-based post-processing baseline. Its forbidden-only
condition improves constraint compliance on advisor_dummy and real ECB, but
advisor-dummy F1 remains low. This suggests that the advisor dummy data may
not contain statistical signal matching the ontology-derived reference DAG.

DECI native constraints are promising in secondary and sanity-check settings,
but the model remains sensitive to data size, thresholding, and configuration.
DECI should be treated as exploratory rather than the main thesis claim.

## Advisor Dummy Main Table

| algorithm | algorithm_family | constraint_mode | constraint_handling | F1_mean | F1_std | SHD_mean | SHD_std | edge_count_mean | violations_mean | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC | constraint-based | none | native | 0.0127 | 0.0139 | 127.6000 | 7.3348 | 83.2000 | 21.8000 | Unconstrained reference-DAG alignment baseline. |
| PC | constraint-based | forbidden_only | background_knowledge | 0.0552 | 0.0220 | 124.2000 | 7.3959 | 85.4000 | 0.0000 | Forbidden-only constraints remove violations, but reference-DAG alignment remains low. |
| PC | constraint-based | required_light | background_knowledge | 0.0552 | 0.0220 | 124.2000 | 7.3959 | 85.4000 | 0.0000 | Secondary light-required constraint-assistance result. |
| LiNGAM | functional causal model | none | native | 0.0000 | 0.0000 | 58.2000 | 5.4498 | 12.2000 | 0.8000 | Unconstrained reference-DAG alignment baseline. |
| LiNGAM | functional causal model | forbidden_only | postproc | 0.0000 | 0.0000 | 57.4000 | 4.8270 | 11.4000 | 0.0000 | Forbidden-only constraints remove violations, but reference-DAG alignment remains low. |
| LiNGAM | functional causal model | required_light | postproc | 0.0000 | 0.0000 | 57.4000 | 4.8270 | 11.4000 | 0.0000 | Secondary light-required constraint-assistance result. |
| NOTEARS | continuous optimization | none | native | 0.0000 | 0.0000 | 46.0000 | 0.0000 | 0.0000 | 0.0000 | Unconstrained reference-DAG alignment baseline. |
| NOTEARS | continuous optimization | forbidden_only | postproc | 0.0000 | 0.0000 | 46.0000 | 0.0000 | 0.0000 | 0.0000 | Forbidden-only constraints remove violations, but reference-DAG alignment remains low. |
| NOTEARS | continuous optimization | required_light | postproc | 0.1600 | 0.0000 | 42.0000 | 0.0000 | 4.0000 | 0.0000 | Secondary light-required constraint-assistance result. |
| GES | score-based | none | native | 0.0105 | 0.0235 | 76.0000 | 3.3912 | 30.8000 | 4.8000 | Unconstrained reference-DAG alignment baseline. |
| GES | score-based | forbidden_only | postproc | 0.0110 | 0.0245 | 71.2000 | 4.1473 | 26.0000 | 0.0000 | Forbidden-only constraints remove violations, but reference-DAG alignment remains low. |
| GES | score-based | required_light | postproc | 0.1159 | 0.0230 | 67.2000 | 4.1473 | 30.0000 | 0.0000 | Secondary light-required constraint-assistance result. |
| DECI | deep generative | none | native | 0.0180 | 0.0247 | 60.8000 | 12.7161 | 16.0000 | 3.4000 | Unconstrained reference-DAG alignment baseline. |
| DECI | deep generative | forbidden_only | native | 0.0110 | 0.0152 | 70.2000 | 9.0388 | 25.0000 | 0.0000 | Forbidden-only constraints remove violations, but reference-DAG alignment remains low. |
| DECI | deep generative | required_light | native | 0.1276 | 0.0267 | 64.0000 | 11.5326 | 27.2000 | 0.0000 | Secondary light-required constraint-assistance result. |

## Advisor Dummy Full-Reference Sanity Appendix

| algorithm | algorithm_family | constraint_mode | constraint_handling | F1_mean | F1_std | SHD_mean | SHD_std | edge_count_mean | violations_mean | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC | constraint-based | full_reference_sanity | sanity_check | 0.0552 | 0.0220 | 124.2000 | 7.3959 | 85.4000 | 0.0000 | Full-reference sanity check; high alignment reflects constraint enforcement rather than independent discovery. |
| LiNGAM | functional causal model | full_reference_sanity | sanity_check | 0.0000 | 0.0000 | 57.4000 | 4.8270 | 11.4000 | 0.0000 | Full-reference sanity check; high alignment reflects constraint enforcement rather than independent discovery. |
| NOTEARS | continuous optimization | full_reference_sanity | sanity_check | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 46.0000 | 0.0000 | Full-reference sanity check; high alignment reflects constraint enforcement rather than independent discovery. |
| GES | score-based | full_reference_sanity | sanity_check | 0.7830 | 0.0266 | 25.6000 | 3.9749 | 71.6000 | 0.0000 | Full-reference sanity check; high alignment reflects constraint enforcement rather than independent discovery. |
| DECI | deep generative | full_reference_sanity | sanity_check | 0.8165 | 0.0622 | 21.2000 | 8.6139 | 67.2000 | 0.0000 | Full-reference sanity check; high alignment reflects constraint enforcement rather than independent discovery. |

## Real ECB Case-Study Table

| algorithm | algorithm_family | constraint_mode | constraint_handling | alignment_mean | alignment_std | edge_count_mean | violations_mean | stable_edges_60 | stable_edges_80 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC | constraint-based | none | native | 0.0000 | 0.0000 | 16.3333 | 0.3333 |  |  | Real-data case-study baseline; no F1/SHD is defined. |
| PC | constraint-based | forbidden_only | background_knowledge | 0.3333 | 0.5774 | 16.6667 | 0.0000 |  |  | Constraints remove violations while preserving some literature alignment. |
| LiNGAM | functional causal model | none | native | 0.4000 | 0.5477 | 24.4000 | 0.2000 |  |  | Real-data case-study baseline; no F1/SHD is defined. |
| LiNGAM | functional causal model | forbidden_only | postproc | 0.4000 | 0.5477 | 24.2000 | 0.0000 |  |  | Constraints remove violations while preserving some literature alignment. |
| NOTEARS | continuous optimization | none | native | 0.0000 | 0.0000 | 17.2000 | 0.2000 |  |  | Real-data case-study baseline; no F1/SHD is defined. |
| NOTEARS | continuous optimization | forbidden_only | postproc | 0.0000 | 0.0000 | 17.0000 | 0.0000 |  |  | Constraints remove violations; real-data alignment remains descriptive. |
| GES | score-based | none | native | 0.3000 | 0.4472 | 21.6000 | 0.6000 | 14.0000 | 8.0000 | Real-data case-study baseline; no F1/SHD is defined. |
| GES | score-based | forbidden_only | postproc | 0.4000 | 0.5477 | 21.0000 | 0.0000 | 14.0000 | 8.0000 | Constraints remove violations while preserving some literature alignment. |
| DECI | deep generative | none | native | 0.0000 | 0.0000 | 10.4000 | 0.0000 | 0.0000 | 0.0000 | DECI real-data selected-config run is exploratory; no F1/SHD is defined. |
| DECI | deep generative | forbidden_only | native | 0.0000 | 0.0000 | 11.4000 | 0.0000 | 0.0000 | 0.0000 | DECI real-data selected-config run is exploratory; no F1/SHD is defined. |

## Plot Files

- `outputs/experiments/final_advisor_dummy_f1_forbidden_only.png`
- `outputs/experiments/final_advisor_dummy_shd_forbidden_only.png`
- `outputs/experiments/final_advisor_dummy_violations_forbidden_only.png`
- `outputs/experiments/final_real_ecb_alignment_violations.png`
## Recommended thesis figures

- `outputs/figures/pipeline_overview.png`: end-to-end pipeline overview linking advisor dummy evaluation and the real ECB case-study branch.
- `outputs/figures/constraint_pipeline.png`: constraint construction and injection/post-processing workflow.
- `outputs/figures/advisor_dummy_f1_forbidden_only.png`: main advisor-dummy reference-DAG F1 comparison, excluding full-reference sanity checks.
- `outputs/figures/advisor_dummy_shd_forbidden_only.png`: main advisor-dummy structural-distance comparison; lower is better.
- `outputs/figures/advisor_dummy_violations_forbidden_only.png`: ontology-violation reduction under forbidden-only constraints.
- `outputs/figures/real_ecb_graph_selected.png`: selected real ECB case-study graph, using GES forbidden-only because PC graph CSVs were not available.
- `outputs/figures/real_ecb_stability.png`: real ECB edge stability across runs.
- `outputs/figures/full_reference_sanity_appendix.png`: appendix-only sanity check; required edges equal the reference DAG and should not be used as main discovery evidence.

