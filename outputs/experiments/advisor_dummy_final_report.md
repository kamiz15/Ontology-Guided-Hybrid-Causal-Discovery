# Advisor Dummy Final Report

## Source Files

- CSV: `data/advisor_dummy/ESG-Finance_dummy_data.csv`
- Metadata workbook: `data/advisor_dummy/ESG-Finance_Metadata.xlsx`
- Text generator/spec: `data/advisor_dummy/Dummy_dataset_ESG.txt`

## Data

- Original CSV shape: 3002 rows x 161 columns
- Cleaned model dataset shape: 3002 rows x 40 columns
- Metadata XLSX loaded: True
- Explicit advisor causal rules found: False
- Evaluation target: `ontology_derived_reference_dag`

## Constraints

- Forbidden constraints: 247
- Required-light constraints: 4
- Full-reference required constraints: 46

## GES Baseline

GES is a classical score-based baseline. In the main `forbidden_only` comparison, constraints reduce forbidden-edge violations from 4.80 to 0.00, while F1 remains low (0.0105 to 0.0110). This is consistent with the advisor dummy data containing limited statistical signal matching the ontology-derived reference DAG; it should not be overclaimed as causal recovery.


## Methodological Wording

The raw advisor dummy data does not include explicit causal equations, a
causal edge list, or a causal DAG. Because no explicit advisor causal rule
file was found, the current graph is an ontology-derived reference DAG.
F1 and SHD therefore measure reference-DAG alignment, not recovery of an
experimentally known causal mechanism.

## Validation Snapshot

```text
# Advisor Dummy Reference DAG Validation

The advisor-provided dummy CSV and metadata workbook are used as the official
dummy-data source for this evaluation.

No explicit advisor-provided causal generation DAG/rule file was found. The current graph is an ontology-derived reference DAG reconstructed from metadata and domain rules. It should not be described as a true data-generating DAG.

- Data source: `data\advisor_dummy\ESG-Finance_dummy_data.csv`
- Metadata source: `data\advisor_dummy\ESG-Finance_Metadata.xlsx`
- Explicit rule file: `not found`
- Graph source mode: `metadata_reference`
- Evaluation target: `ontology_derived_reference_dag`
- Original CSV shape: 3002 rows x 161 columns
- Cleaned model dataset: `C:/Users/User/Desktop/go/data/processed/advisor_dummy_ready.csv`
- Cleaned model shape: 3002 rows x 40 columns
- Nodes included: 40
- Reference edges: 46
- Forbidden constraints: 247
- Acyclic: True
- Duplicate edges: 0
- Self-loops: 0
- Missing edge variables from cleaned data: 0
- Edges contradicting forbidden constraints: 0
- Validation status: PASS

## Notes

F1 and SHD for `advisor_dummy` are computed against the evaluation target
listed above. In metadata-reference mode, these metrics indicate alignment
with an ontology-derived reference DAG, not recovery of an experimentally
known causal mechanism.
```

## Main Results: Forbidden Only

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

## Secondary Results: Required Light

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

## Sanity Results: Full Reference Required

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


## Limitations

The main comparison should be unconstrained versus `forbidden_only`, because
that does not give algorithms the full reference structure as required edges.
The `full_reference_sanity` condition is useful only to verify that adapters
and post-processing honor constraints.
