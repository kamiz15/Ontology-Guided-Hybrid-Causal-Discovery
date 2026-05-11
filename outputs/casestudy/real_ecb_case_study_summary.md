# Real ECB Case Study Graph Summary

## Edge counts

- Total discovered edges: 12
- Directed edges: 10
- Undirected CPDAG edges: 2

## Annotation distribution

- literature_supported: 0
- literature_silent: 12
- literature_forbidden: 0

## Strongest literature-supported edges

- None detected in the discovered graph.

## Literature-forbidden edges

No literature-forbidden directed edges were discovered.

## Template interpretation

The real ECB case-study graph is a qualitative constraint-guided discovery artifact rather than a ground-truth recovery test. The discovered CPDAG respects the audited literature constraints in the fitted graph: no literature-forbidden directed edge is present, while supported edges are explicitly marked when the discovered relation aligns with the reviewed evidence. The literature-constraint footprint on the real 16-variable schema is small: only 2 of 68 audited constraints map to the available real-data variables, so most discovered edges remain literature-silent rather than contradicted or confirmed. This makes the case study useful for inspecting plausible ESG-finance structure and constraint compliance, but not for estimating causal recovery accuracy. The quantitative evidence base for recovery and scalability remains the causal-dummy, SNR-sensitivity, and sample-size-sensitivity experiments.
