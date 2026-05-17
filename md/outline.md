
# Thesis Outline — Ontology-Guided Hybrid Causal Discovery in ESG Data

**Working title (suggested):** _Ontology-Guided Hybrid Causal Discovery in ESG Data: Integrating Domain Knowledge from ESGOnt into Constraint-Based Causal Structure Learning_

**Template basis:** primarily `Thesis_structure_ML_Data_Analytics.pdf` (because this is an empirical ML/causal-inference study), with journal-quality structural elements (comparison tables, sharp gap statement, error typology, SHD/SID metrics) borrowed from `Thesis_structure_LLM_to_generate_ontology_or_causal_graph.pdf`.

**Target length:** ~30–35 pages (bachelor) 

---

## 1 Introduction (3–5 pages)

### 1.1 Background and problem context

- ESG data is exploding (regulatory drivers: EU CSRD, NFRD, IFRS S1/S2, GRI, ESRS), but ESG analytics today are dominated by **correlation-based** predictive models that cannot answer _intervention_ questions ("if we raise renewable energy share, does compliance improve?").
- Two parallel research streams have grown up independently:
    1. **ESG ontologies** (ESGOnt, ESGMKG, etc.) that formalize _what_ an ESG concept is and how indicators relate semantically.
    2. **Causal discovery algorithms** (PC, FCI, GES, LiNGAM, NOTEARS, DECI) that learn cause–effect structure from observational data.
- Each stream alone has known weaknesses: ontologies are static and don't quantify effects; pure data-driven causal discovery on observational ESG data is noisy, prone to spurious edges, and frequently produces directions that **contradict domain logic** (e.g., an effect pointing to its cause).

### 1.2 Importance / impact

- Reliable causal structure → better intervention design for sustainability managers, more defensible disclosures, and regulatory audit support (SDG 13 / SDG 12 alignment).
- Hybrid (ontology + data) discovery promises causal graphs that are **both data-consistent and domain-plausible** — a precondition for ESG decision support that practitioners will actually trust.

### 1.3 Research gap (what's missing in the literature)

- Most causal-discovery benchmarks use synthetic or biomedical data; **ESG has been almost entirely absent** from causal-discovery evaluation.
- Existing ESG ontology work (ESGOnt, ESGMKG, RSO) stops at schema/KG validation — it does not feed into structure-learning algorithms.
- The few "knowledge-guided causal discovery" papers (e.g., LLM-as-prior, expert-elicited tiers) do not derive constraints **systematically from a formal OWL/RDF ontology**, and do not study ESG.

### 1.4 Research objective

> To design, implement, and evaluate a hybrid causal discovery pipeline that systematically translates ESGOnt ontology knowledge into machine-readable causal constraints (forbidden edges, required edges, temporal/tier orderings) for constraint-based structure learning, and to quantify the benefit of those constraints relative to a purely data-driven baseline on ESG data.

**Sub-objectives:**

1. Formalize a constraint-extraction procedure from ESGOnt (classes, object properties, SDG mappings) into `FORBIDDEN_EDGES` / `REQUIRED_EDGES` / tier orderings consumable by `causal-learn`.
2. Build a reproducible pipeline (audit → clean → map → constrain → discover → evaluate).
3. Compare ontology-constrained vs. unconstrained causal graphs on a controlled ESG dataset.
4. Validate plausibility against an expert-curated ground-truth subgraph.

### 1.5 Research questions

- **RQ1 (Plausibility):** Do ontology constraints reduce the rate of domain-implausible edges (e.g., outcome → driver) produced by the PC algorithm on ESG data?
- **RQ2 (Structural accuracy):** Compared with an expert-curated reference subgraph, does the ontology-guided variant achieve better edge precision/recall, lower SHD, and lower SID than the unconstrained baseline?
- **RQ3 (Stability):** Are ontology-guided graphs more stable across bootstrap resamples and hyperparameter settings (PC alpha, independence test) than unconstrained ones?
- **RQ4 (Cost of constraints):** Where ontology constraints disagree with data signals, what types of disagreement arise, and how should they be adjudicated?

### 1.6 Key contributions

1. **A systematic mapping** from ESGOnt classes/properties → causal constraint primitives usable by `causal-learn`'s `BackgroundKnowledge` API.
2. **An open, reproducible pipeline** (the GitHub repo: `01_audit.py` → `02_clean.py` → `03_build_column_mapping.py` → `04_forbidden_edges.py` → discovery → evaluation scripts).
3. **An ESG-specific causal benchmark** (variables, expert-curated reference subgraph, evaluation harness) — currently missing from the literature.
4. **Empirical evidence** on whether ontology guidance helps in a real ESG-style observational setting.

### 1.7 Practical and scientific implications

- _Practitioners_ (sustainability officers, ESG analysts): graphs they can actually use for "what-if" reasoning.
- _Researchers_ in causal discovery: a new domain benchmark + a worked example of formal-ontology-to-constraint translation.
- _Standards bodies_: a basis for embedding causal structure into ESG reporting frameworks.

### 1.8 Scope and limitations

- Cross-sectional, not (yet) time-series.
- One ontology (ESGOnt) — not a comparison of ontologies.
- Constraint-based (PC) is the primary algorithm; LiNGAM / GES considered as secondary.
- Ground truth is an _expert-curated_ subgraph derived from ESGOnt and prior literature, not a randomized intervention study.
- "Real ESG data" if supervisor data lands in time, otherwise the realistic synthetic dataset already in `Dummy_dataset_ESG.txt` (with the synthetic-vs-real risk discussed openly in §5.5).

### 1.9 Thesis structure

One short paragraph: Chapter 2 reviews causal discovery and ESG ontologies and positions the gap; Chapter 3 describes the methodology and pipeline; Chapter 4 reports the experiments; Chapter 5 interprets the findings; Chapter 6 concludes.

---

## 2 Literature Review and Conceptual Background (6–10 pages)

### 2.1 Theoretical foundations (2–3 pages)

#### 2.1.1 Causal inference vs. statistical association

- Pearl's ladder of causation; why correlation-based ML cannot answer interventional questions.
- DAGs, SCMs, d-separation, Markov equivalence classes (CPDAGs).
- Identifiability assumptions (causal sufficiency, faithfulness, acyclicity).

#### 2.1.2 Causal discovery algorithm families

- **Constraint-based:** PC, FCI (uses conditional-independence tests; outputs a CPDAG / PAG).
- **Score-based:** GES, NOTEARS.
- **Functional / asymmetry-based:** LiNGAM, ANM, IGCI.
- **Deep / amortized:** DECI, AVICI.
- Pick **PC + LiNGAM** (already in the repo plan) as the focus; justify briefly.

#### 2.1.3 Ontologies as formal domain knowledge

- OWL/RDF, classes, object properties, axioms.
- ESGOnt's four modules (Impact Mapper, Data Integrator, Performance Analyzer, Maturity Evaluator) and its alignment with SDGs, GRI, ESRS.

#### 2.1.4 Background knowledge in causal discovery

- Tier orderings, forbidden / required edges, prior adjacency matrices.
- The `causal-learn` `BackgroundKnowledge` API — what it can and cannot enforce.

### 2.2 Related work (3 pages, grouped + table)

Group prior work into four streams:

1. **Pure data-driven causal discovery on tabular data** (PC, GES, LiNGAM benchmarks)
2. **Knowledge-guided / expert-prior causal discovery** (tiered ordering, LLM-as-prior, expert-elicited adjacency)
3. **ESG ontologies and knowledge graphs** (ESGOnt, ESGMKG, RSO, ESGont)
4. **Causal / quantitative ESG analytics** (ESG factor models, predictive ML on ESG, very limited causal work)

**Comparison table** (mandatory — strong journal signal):

| Ref.                              | Objective                                   | Algorithm / Approach                          | Domain knowledge used?        | Dataset / Domain                     | Evaluation                                    | Limitation vs. this thesis     |
| --------------------------------- | ------------------------------------------- | --------------------------------------------- | ----------------------------- | ------------------------------------ | --------------------------------------------- | ------------------------------ |
| Spirtes et al. (PC)               | Constraint-based discovery                  | PC                                            | None                          | Synthetic / biomed                   | SHD, edge F1                                  | No domain prior; no ESG        |
| Shimizu et al. (LiNGAM)           | Linear non-Gaussian discovery               | LiNGAM                                        | None                          | Synthetic / fMRI                     | SHD                                           | No ESG, no ontology            |
| Glymour et al. (causal-learn)     | Library + BackgroundKnowledge               | PC w/ BK                                      | Manual edges                  | Mixed                                | Mixed                                         | No systematic ontology mapping |
| Vijaya et al. (ESGOnt)            | ESG-SDG ontology                            | OWL/RDF                                       | —                             | ESG schema                           | Competency questions                          | No causal discovery            |
| Wicaksono et al. (ESGMKG)         | ESG metric KG                               | Ontology                                      | —                             | ESG metrics                          | Schema compliance                             | No causal layer                |
| Yu et al. (OntoMetric)            | LLM extraction from ESG regs                | LLM + ontology prompt                         | Ontology as prompt constraint | 5 ESG standards                      | Semantic accuracy                             | Builds KG, not causal DAG      |
| Long et al. (LLM as causal prior) | LLM-elicited expert prior                   | PC + LLM prior                                | LLM (informal)                | Mixed                                | SHD, orientation acc.                         | No formal ontology             |
| **This thesis**                   | **Ontology-guided causal discovery on ESG** | **PC (+LiNGAM) + ESGOnt BackgroundKnowledge** | **Formal OWL ontology**       | **ESG (real / realistic synthetic)** | **SHD, SID, P/R/F1, plausibility, stability** | —                              |

Spend one paragraph per row explaining the cell content, then 2–3 paragraphs synthesizing the streams.

### 2.3 Research gap (½ page, sharp)

> _"While ESG ontologies are increasingly mature and causal-discovery algorithms increasingly powerful, the two have not been connected: no prior work systematically translates a formal ESG ontology (ESGOnt) into machine-readable structural constraints (forbidden / required edges, tier orderings) for constraint-based causal discovery, nor evaluates the resulting hybrid against a pure data-driven baseline on ESG data with a domain-expert reference graph. This thesis closes that gap."_

---

## 3 Methodology (7–10 pages) — **the heart of the thesis**

### 3.1 Overview of the concept

- Restate objective in one sentence.
- **Figure 1: Pipeline overview** — boxes: `raw ESG data` → `audit` → `clean` → `column mapping (ontology)` → `constraint extraction (ESGOnt → forbidden/required edges)` → `PC discovery (baseline + constrained)` → `evaluation (vs. reference DAG)` → `interpretation`.
- Tools: Python 3.11, `causal-learn`, `networkx`, `pandas`, `scikit-learn`, Protégé (for ontology inspection), Neo4j / NetworkX for graph storage.

### 3.2 Dataset

- If real supervisor data: source, year, organizations covered, granularity (firm-year).
- If realistic synthetic (`Dummy_dataset_ESG.txt`): N=3000 rows, 60+ ESG variables across E/S/G, with deliberate missingness, outliers, type errors injected for realism.
- Variable inventory by ESG pillar (Environmental: 26 vars, Social: 23, Governance: 14) — full table in an appendix; summary stats in the body.
- Discuss synthetic-vs-real caveat openly.

### 3.3 EDA and data preprocessing (`01_audit.py`, `02_clean.py`)

- Per-column missingness, near-constant flagging, type coercion errors, outlier ID.
- Imputation policy (median for ≤30% missing; drop above; flag near-constant by `NEAR_CONSTANT_STD_RATIO`).
- Correlation screen (drop one of any pair with |ρ| > 0.97 via `HIGH_CORR_THRESHOLD`).
- Exclusion of administrative / metadata columns (`incorporation_year`, `nace`, z-score sub-components, etc.).
- Outputs: `data_clean.csv`, `data_ready.csv`, `reports/audit_report.txt`, `reports/high_correlation_pairs.csv`.

### 3.4 ESGOnt → variable mapping (`03_build_column_mapping.py`)

- For every dataset column, record: ESG pillar (E/S/G), ontology class (`Indicator` / `PerformanceIndicator` / `ESG_Metric`), unit, _causal role hint_ (`candidate cause` / `candidate effect`), and rationale.
- This mapping is the bridge between data and ontology.
- Output: `column_mapping.csv`.

### 3.5 Ontology-to-constraint translation (`04_forbidden_edges.py`) — **the novel core**

- **Rule-set 1 — Outcome cannot cause its inputs:** composite scores (`governance_compliance_score`, `resource_efficiency_index`, `resilience_score`, `reporting_quality_score`) cannot point _back into_ their drivers.
- **Rule-set 2 — Failure-outcome variables are sinks:** counts of failures (`ethical_breaches`, `corruption_cases`, `toxic_spills`, `human_rights_violations`) cannot cause their preventive drivers.
- **Rule-set 3 — Required edges (very sparingly, only where empirically and theoretically strong):** e.g., `emission_reduction_policy → co2_ch4_n2o_scope_1_3`, `renewable_energy_share → total_energy_consumption`, `training_hours → injury_frequency_rate`.
- **Rule-set 4 (optional) — Pillar tier ordering:** e.g., governance indicators temporally precede environmental and social outcomes within a reporting year.
- Translation into the `causal-learn` `BackgroundKnowledge` object via `build_background_knowledge()`.
- **Document every rule with an ontology / literature citation** — this is the auditability story for the journal version.

### 3.6 Causal discovery experiments

- **Algorithm A — PC unconstrained (baseline).**
- **Algorithm B — PC + ESGOnt `BackgroundKnowledge` (proposed).**
- **Algorithm C (sensitivity) — LiNGAM** with/without ordering prior.
- Hyperparameter grid: alpha ∈ {0.01, 0.05, 0.1}; independence test = Fisher-Z (continuous) / chi-square (mixed).
- Repeated bootstrap (B=100) for edge-stability estimates.

### 3.7 Reference graph (ground truth) construction

- Two complementary sources:
    1. **Ontology-derived expected edges** from ESGOnt object properties (`hasCategory`, `hasCause`, `impacts`, `consumesEnergy`, `generateWaste`, etc.) instantiated on the variable set.
    2. **Literature-derived expected edges** from ESG / sustainability research (cite 10–20 papers; tabulate as a "Hypothesis Table" similar to the example in the LLM template).
- Adjudicate disagreements with supervisor / domain expert.
- Optional: small inter-rater agreement check on a 20-edge subset (Cohen's κ).
- Output: `reference_dag.csv` (edge list with provenance per edge).

### 3.8 Evaluation metrics

**Structural (vs. reference DAG):**

- Edge precision / recall / F1 (skeleton).
- Edge precision / recall / F1 (oriented).
- **Structural Hamming Distance (SHD).**
- **Structural Intervention Distance (SID).**
- Number of v-structures recovered correctly.

**Plausibility (vs. ontology constraints):**

- # of forbidden-edge violations.
    
- # of required-edge recoveries.
    
- "Domain-implausibility rate" = forbidden / total edges produced.

**Stability:**

- Bootstrap edge-stability score (fraction of bootstraps in which each edge appears).
- Hyperparameter sensitivity (Jaccard between graphs across alpha values).

**Qualitative error categorization** (drives §4.3):

- Direction reversal (cause/effect swapped).
- Spurious edge (no domain plausibility).
- Missing edge (well-supported in ontology, absent in output).
- Confounded edge (true confounder not in dataset).
- Disagreement between ontology and data (the interesting cases).

### 3.9 Reproducibility plan

- Repo: `https://github.com/<you>/Ontology-Guided-Hybrid-Causal-Discovery` (link your README).
- Pinned environment (`requirements.txt` / `environment.yml`).
- All seeds fixed; one-command rerun (`make all` or `python run_pipeline.py`).
- Every figure and table in the thesis has a numbered script that produces it.

---

## 4 Results (7–10 pages — make it visual, journal-style)

### 4.1 Descriptive overview

- Final variable count after preprocessing.
- Reference DAG: # nodes, # edges, edge density, edges per pillar, # ontology-derived vs. literature-derived.
- 1 figure of the reference DAG.

### 4.2 RQ1 — Plausibility (forbidden-edge violations)

- Table: # forbidden-edge violations, baseline vs. ontology-guided, across alpha values.
- Figure: violation-rate bar chart by ESG pillar.

### 4.3 RQ2 — Structural accuracy vs. reference DAG

- Table: precision, recall, F1 (skeleton + oriented), SHD, SID for baseline vs. ontology-guided.
- Figure: side-by-side graph visualizations (baseline | ontology-guided | reference) for one illustrative subgraph (e.g., the emissions cluster: `emission_reduction_policy`, `renewable_energy_share`, `total_energy_consumption`, `carbon_intensity`, `co2_ch4_n2o_scope_1_3`).

### 4.4 RQ3 — Stability

- Bootstrap edge-stability heatmap.
- Hyperparameter sensitivity table (Jaccard across alpha settings).

### 4.5 RQ4 — Disagreement cases (error analysis)

- Walk through ~5 representative disagreements between data signal and ontology constraint, classify them (true confounder vs. data noise vs. ontology over-constraint), and discuss adjudication.

### 4.6 Summary of main findings

- One short synthesis paragraph that crisply answers each RQ.

---

## 5 Discussion (4–6 pages — interpret, don't repeat)

### 5.1 Interpretation of findings

- Why ontology guidance helps (or doesn't) — likely: helps most on orientation, less on skeleton.
- Which constraint families pay off most (forbidden vs. required vs. tier).
- Where the dataset (synthetic vs. real) matters.

### 5.2 Comparison with prior literature

- Vs. unconstrained causal-discovery benchmarks: how does ESG behave?
- Vs. LLM-as-prior approaches: structured ontology may be more auditable but less flexible.
- Vs. pure ESG ontology work (ESGOnt, ESGMKG, OntoMetric): you extend their static schema with quantitative causal estimation.

### 5.3 Theoretical implications

- Formal ontologies and causal discovery are _complementary_, not competing.
- Background knowledge in causal discovery should be sourced **systematically** (from an OWL ontology) rather than ad-hoc.

### 5.4 Practical implications

- For sustainability officers: causal graphs that practitioners can defend in audit.
- For standards bodies: a path toward causally-grounded ESG metrics.
- For ESG tooling vendors: an integration pattern between ESG knowledge graphs and causal analytics.

### 5.5 Limitations

- Cross-sectional only; no temporal causal discovery (yet).
- Synthetic / dummy dataset risk if real data not delivered in time.
- Reference DAG is _expert-curated_, not from randomized experiments — circular-validation risk acknowledged.
- One ontology, one algorithm family (PC) — generalization claims accordingly scoped.

### 5.6 Future work

- Time-series ESG causal discovery (PCMCI, Granger-DAG hybrids).
- Multi-ontology robustness (ESGOnt vs. ESGMKG vs. domain-specific extensions).
- LLM ↔ ontology hybrid priors (combine OntoMetric-style LLM extraction with this thesis's structure-learning).
- Integration into a Neo4j-backed ESG decision-support tool with intervention queries.
- Causal effect estimation (DoWhy / EconML) on the validated DAG — moving from discovery to inference.

---

## 6 Conclusion (1–2 pages)

- Restate the gap, the contribution, and the headline result in three short paragraphs.
- Close with a forward-looking sentence:
    
    > _"Causal discovery on ESG data becomes meaningful only when domain knowledge enters the loop. Formal ontologies are the natural carrier of that knowledge; this thesis demonstrates a concrete, reproducible path from ESGOnt to causal DAGs and shows that the hybrid is more plausible, more stable, and more decision-relevant than either ingredient alone."_
    

---

## Appendices

- **A.** Full ESG variable inventory and ontology mapping (the `column_mapping.csv`).
- **B.** Full forbidden-edge / required-edge rule list with ontology / literature provenance per rule.
- **C.** Reference DAG edge list (`reference_dag.csv`).
- **D.** Hyperparameter grid and full result tables.
- **E.** Code listing / repository structure (`README.md`, `config.py`, `01_audit.py` … `04_forbidden_edges.py`, and any discovery / eval scripts you add).

---

## Checklist for journal-readiness (parallel publication path)

- [ ] Comparison table (§2.2) covers ≥6 prior works across ≥3 streams.
- [ ] Gap statement (§2.3) is sharp, falsifiable, and 2–4 sentences.
- [ ] Methodology (§3) has a single overview figure and reproducible pipeline.
- [ ] Every constraint in `04_forbidden_edges.py` has a cited justification.
- [ ] Reference DAG construction has documented provenance per edge.
- [ ] Results (§4) include both structural and plausibility metrics, plus stability.
- [ ] Discussion (§5) explicitly answers each RQ.
- [ ] Repo link + DOI / Zenodo archive for the version-of-record.