# Limitations To Review Before Submission

This file tracks methodological shortcuts taken during development that need
revisiting before thesis submission. Each entry: what was done, why it was a
shortcut, what the rigorous alternative is, and a checkbox.

---

## 1. LLM-assisted ordinal coding of qualitative ESG disclosures

**Date introduced:** 2026-05-04
**Affected files:** data/processed/data_real_parsed.csv,
                    reports/ordinal_parsing_unmapped.csv,
                    reports/llm_scoring_log.csv
**Affected columns:** emission_reduction_policy_score, health_safety_score,
                      board_strategy_esg_oversight_score, reporting_quality_score

**What was done:** ~110 cells across 4 ordinal ESG columns could not be parsed
by deterministic keyword matching (output of 02a). Instead of manually coding
them, I used Gemma (via Google AI Studio API) with a fixed rubric to assign
1-5 scores. Per-cell confidence is logged.

**Why this is a shortcut:** The thesis claims to use ontology-grounded,
literature-derived methodology. Having an LLM score qualitative ESG
disclosures introduces an additional AI-assisted step that has not been
validated against human coders. A reviewer or examiner may reasonably ask
why the qualitative data was scored by the same kind of model whose causal
proposals the thesis is testing.

**Rigorous alternative:** Manually code each unmapped cell using the rubric
in this file (Appendix B in thesis). Estimated time: ~90 minutes for ~110
cells. Recommended at minimum for low-confidence LLM scores
(confidence < 0.7).

**Action before submission:**
- [ ] Manually review all LLM scores with confidence < 0.7
- [ ] Spot-check 20% of high-confidence scores against rubric
- [ ] Document inter-rater agreement if a second coder is available
- [ ] Update parsing_decisions.md to reflect final coding source
- [ ] Add Appendix B to thesis showing the rubric and decision examples

**Rubric used:**
5 - Excellent / external verification / SBTi-validated / specific
    quantified targets with audit
4 - Very Good / documented commitment with structure / third-party verified /
    GRI/TCFD aligned / specific numeric targets without independent verification
3 - Good / documented framework without verification / general commitments
    with some specificity
2 - Moderate / adequate / generic descriptions of activity without commitment
3
1 - Limited / weak / vague mention only
NaN - "Qualitative" alone with no further content; non-applicable

---

## 2. Locale-aware currency parsing

**Date introduced:** 2026-05-04
**Status:** Resolved, with residual warnings to review
**Affected files:** data/processed/data_real_parsed.csv,
                    reports/currency_parsing_warnings.csv,
                    reports/currency_parsing_failures.csv,
                    reports/revenue_parsing_comparison.csv
**Affected columns:** community_investment_eur, green_financing_eur,
                      total_revenue_eur

**What was done:** The deterministic currency parser was rewritten to handle
both US/UK notation and European decimal notation. Ambiguous single-separator
cases are resolved using suffix and magnitude reasonableness, and suspicious
results are logged instead of silently corrected.

**Why this was a shortcut:** The initial parser assumed dot/comma separators
could be normalized with a simple rule. That was too weak for ECB bank reports,
where EUR, USD, SEK, HUF, COP and local-language suffixes appear together.

**Rigorous alternative:** Manually review every row in
reports/currency_parsing_warnings.csv and convert non-EUR currencies using
consistent exchange rates before final thesis submission.

**Action before submission:**
- [ ] Review all rows in reports/currency_parsing_warnings.csv
- [ ] Decide whether non-EUR currency values need FX conversion or exclusion
- [ ] Review reports/currency_parsing_failures.csv for qualitative placeholders
- [ ] Confirm total_revenue_eur min/max are defensible for the ECB sample

**Residual concern:** The parser now catches locale notation, but the source
spreadsheet still mixes currencies and scale conventions. Rule 4 warnings
should be reviewed before final submission.

---

## 3. Single-date FX conversion

**Date introduced:** 2026-05-04
**Status:** Implemented, with review needed for unsupported or implicit
currency labels
**Affected files:** data/processed/data_real_parsed.csv,
                    reports/fx_conversions_applied.csv,
                    reports/fx_unsupported_currencies.csv,
                    reports/currency_parsing_warnings.csv
**Affected columns:** community_investment_eur, green_financing_eur,
                      total_revenue_eur

**What was done:** Non-EUR currency amounts detected in the raw spreadsheet
are converted to EUR inside the parser using the most recent rate in
exchange_rates/ecb_rates_2025.csv. Values without a currency token default
to EUR; detected currencies missing from the FX file are mapped to NaN and
logged.

**Why this is a shortcut:** The conversion uses one ECB reference date for
the full cross-sectional dataset rather than matching rates to each bank's
reporting period. Some raw cells also use implicit or non-standard labels
such as `Ps.` that are not covered by the explicit detection list.

**Rigorous alternative:** Time-match FX rates to each bank's reporting period
and manually review unsupported or implicit currency labels before final
analysis.

**Action before submission:**
- [ ] Review reports/fx_unsupported_currencies.csv
- [ ] Review large revenue outliers in reports/currency_parsing_warnings.csv
- [ ] Decide how to treat implicit currency labels such as `Ps.`
- [ ] Document the FX reference date and limitation in the methods chapter

**Residual concern:** Single-date FX conversion is acceptable for a
cross-sectional sample, but it should be revisited if the analysis is
extended to panel data with mixed reporting periods.

---
## Real-data parsing — finalized [today's date]

**Status:** Frozen. No further parser iteration before submission.

**State at freeze:**
- 102/110 banks have valid total_revenue_eur (8 NaN: COP unsupported,
  failed parses, deliberate exclusions)
- Revenue median: 3.07B EUR, max: 724B EUR, min: 3.4M EUR
- 53 rows FX-converted from 7 currencies (USD, GBP, SEK, DKK, HUF, CZK, BGN)
- 3 manual overrides applied for green_financing_eur (rows 63, 82, 102)
- 6 rows in currency_parsing_warnings.csv reviewed and accepted as
  legitimate large-bank values (rows 33, 43, 52, 66, 89, 101)
- All ordinal qualitative scores filled by LLM with Gemma 3 27B,
  confidence ≥ 0.7 on all 113 cells (see entry above for caveats)

**Known residuals not addressed:**
- Min revenue 3.4M EUR is implausibly small for an ECB-supervised entity;
  likely a small investment vehicle in the sample. Verify or NaN before
  final reporting.
- Max revenue 724B EUR is at the upper edge of plausibility; if this
  represents total assets mis-labeled as revenue in the source, real
  revenue is likely ~70B EUR. Verify against bank's annual report.
- Single-date FX rate (2025-12-31) applied to all rows regardless of
  reporting period.

**Decision rationale:** Further parser iteration produces diminishing
returns. Remaining edge cases are best handled at analysis time
(sensitivity analysis: re-run with these rows excluded) rather than at
parse time. Synthetic-data experiments are the headline thesis result;
real-data is the case study chapter.