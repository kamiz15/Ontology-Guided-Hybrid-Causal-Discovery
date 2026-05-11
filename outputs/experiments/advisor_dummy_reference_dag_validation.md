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
