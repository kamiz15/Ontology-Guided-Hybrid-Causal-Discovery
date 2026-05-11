# Figure Captions

## pipeline_overview.png
Suggested caption: Overview of the ontology-guided ESG causal discovery pipeline, showing the advisor dummy experiment and the real ECB case-study branch.

Interpretation: The advisor dummy path supports reference-DAG evaluation, while the real ECB path supports case-study literature alignment and constraint-compliance analysis.

Caveat: The advisor dummy graph is an ontology-derived reference DAG, not an explicit data-generating graph.

## constraint_pipeline.png
Suggested caption: Constraint construction pipeline from ontology rules and literature-audited relations to algorithm-specific constraint injection or post-processing.

Interpretation: `forbidden_only` is the main thesis-safe setting; `required_light` is secondary; `full_reference_sanity` is a sanity check.

Caveat: Full-reference constraints should not be interpreted as independent discovery evidence.

## advisor_dummy_f1_forbidden_only.png
Suggested caption: Advisor dummy reference-DAG F1 for unconstrained and forbidden-only constrained runs.

Interpretation: PC shows the clearest main-mode reference-DAG alignment improvement, while other algorithms mainly improve constraint compliance.

Caveat: F1 is measured against an ontology-derived reference DAG.

## advisor_dummy_shd_forbidden_only.png
Suggested caption: Structural distance to the advisor-dummy ontology-derived reference DAG.

Interpretation: Lower values indicate closer reference-DAG alignment.

Caveat: SHD is reported as distance to the ontology-derived reference DAG.

## advisor_dummy_violations_forbidden_only.png
Suggested caption: Mean ontology violations under unconstrained and forbidden-only constrained runs.

Interpretation: Forbidden-only constraints eliminate violations across the main algorithm set.

Caveat: Constraint compliance is distinct from causal accuracy.

## real_ecb_graph_selected.png
Suggested caption: Selected real ECB case-study graph using the GES forbidden-only run.

Interpretation: The graph visualizes discovered directions on real data; thicker edges indicate higher bootstrap stability where available.

Caveat: This is a descriptive real-data case-study graph.

## real_ecb_stability.png
Suggested caption: Number of real ECB edges appearing in at least 60% or 80% of runs.

Interpretation: Stability summarizes how reproducible discovered edges are across bootstrap seeds.

Caveat: Stable edges can still be descriptive rather than causal.

## full_reference_sanity_appendix.png
Suggested caption: Full-reference sanity check where required edges equal the ontology-derived reference DAG.

Interpretation: High F1 in this condition demonstrates constraint enforcement.

Caveat: This is not main discovery evidence.
