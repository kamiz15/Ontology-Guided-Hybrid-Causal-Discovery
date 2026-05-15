# Limitations To Review Before Submission

This file tracks only unresolved methodological caveats that still need a
submission decision. Resolved implementation notes and completed experiment
tasks have been removed.

---

## 1. LLM-assisted ordinal coding of qualitative ESG disclosures

**Affected files:** data/processed/data_real_parsed.csv,
reports/ordinal_parsing_unmapped.csv, reports/llm_scoring_log.csv

**Affected columns:** emission_reduction_policy_score, health_safety_score,
board_strategy_esg_oversight_score, reporting_quality_score

**Current state:** Qualitative ESG cells that could not be parsed
deterministically were scored with Gemma 3 27B using a fixed 1-5 rubric.
All filled cells have confidence >= 0.7, but they have not been independently
validated by a human coder.

**Remaining concern:** The real-data case study uses these ordinal columns, so
the thesis should be transparent that this is an AI-assisted preprocessing
step rather than fully manual coding.

**Action before submission:**
- [ ] Spot-check at least 20% of the LLM-coded cells against the rubric
- [ ] Manually review any values used in prominent case-study interpretation
- [ ] Document the rubric and coding provenance in the methods or appendix
- [ ] Add inter-rater agreement only if a second coder becomes available

---

## 2. Real-data parsing residuals

**Affected files:** data/processed/data_real_parsed.csv,
data/processed/data_ready.csv, reports/currency_parsing_warnings.csv,
reports/fx_conversions_applied.csv, reports/fx_unsupported_currencies.csv

**Current state:** Locale-aware currency parsing and single-date FX conversion
are implemented and frozen for the experiment phase. The parser handled the
mixed currency notation well enough for the real ECB case study, and remaining
edge cases are now analysis/reporting caveats rather than parser bugs.

**Known residuals:**
- 102/110 banks have valid total_revenue_eur; 8 rows remain NaN because of
  unsupported COP values, failed parses, or deliberate exclusions.
- The minimum valid revenue value, about 3.4M EUR, is implausibly small for an
  ECB-supervised entity and should be verified or treated cautiously.
- The maximum revenue value, about 724B EUR, is at the upper edge of
  plausibility and may reflect a source-field mismatch such as assets reported
  as revenue.
- One ECB reference date is used for FX conversion across the cross-sectional
  sample rather than matching rates to each bank's reporting period.

**Action before submission:**
- [ ] Decide whether to leave the 3.4M EUR and 724B EUR rows in the real case
      study or flag them in the narrative
- [ ] Document the single-date FX choice as a cross-sectional simplification
- [ ] Avoid making strong quantitative claims from real ECB magnitudes; use
      the real dataset as a qualitative case study

---

## 3. Paper inventory cleanup

**Affected files:** paper_inverntory.md, archive/organize_papers.py,
C:/Users/User/Desktop/Outline and Materials needed/

**Current state:** The paper inventory was synced against the source PDF
folder, duplicate downloads were treated as the same paper, and rows marked
for removal were removed. The organizer helper has been archived with the
other development utilities.

**Remaining concern:** Some paper classifications came from manual grouping
notes rather than a fully audited metadata pass.

**Action before submission:**
- [ ] Fill any remaining blank classifications in paper_inverntory.md
- [ ] Check that cited paper IDs in reports and tables resolve to the final
      inventory
- [ ] Keep the archived organizer as a utility only; do not describe it as
      part of the thesis method

---

## 4. Runtime measurement limitations (RQ3)

**Current state:** Advisor-dummy runtime, causal-dummy runtime,
sample-size scaling, SNR sensitivity, and DECI runtime breakdowns have now
been consolidated in outputs/experiments/runtime_consolidated_report.md.

**Remaining concerns:**
- gCastle's NOTEARS caches optimization results across repeated calls on the
  same data. The runtime scripts now flag this via coefficient-of-variation
  diagnostics, but the NOTEARS mean runtime should still be interpreted
  descriptively rather than as a clean repeated-fit estimate.
- Real ECB runtime is formally persisted only for DECI. Classical algorithms
  ran in under one second during execution, but those real-data runtime rows
  were not persisted for formal aggregation.
- Constraint-count sensitivity, meaning runtime as a function of the number of
  constraints applied, was not measured. This remains future work.
- DECI runtime is not directly comparable to PC, LiNGAM, NOTEARS, and GES
  because it uses a different training schedule and threshold-dependent
  configuration grid.

**Action before submission:**
- [ ] In the RQ3 text, cite the consolidated runtime report rather than the
      older advisor-only runtime table
- [ ] State that NOTEARS has a caching caveat
- [ ] State that full real-data classical runtime aggregation and
      constraint-count sensitivity are future-work items
