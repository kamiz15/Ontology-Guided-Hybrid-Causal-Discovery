---
name: academic-writing
description: >
  Academic writing, research methodology, and scholarly communication for
  computer science researchers and students. Use this skill whenever the user
  is writing or revising a thesis, paper, or any section thereof — introduction,
  related work, methodology, results, discussion, conclusion, or abstract. Also
  trigger for literature review construction, citation management (IEEE style),
  research question formulation, contribution framing, writing style feedback,
  benchmarking and evaluation design, reproducibility considerations, and
  defense preparation. Essential for CS BSc/MSc thesis writers. If the user
  mentions "my thesis", "my paper", "related work", "research question",
  "contribution", "writing feedback", "cite", "benchmark", "evaluate", or
  "literature review", use this skill immediately, even if the request seems simple.
---

# Academic Writing Skill

A skill for producing rigorous, well-structured academic writing in computer
science. Grounded in Zobel's Writing for Computer Science (3rd ed.), Nijssen's
CS thesis writing guidance, and empirical work on evaluation quality in CS research.

---

## Quick Reference

| User says...                                        | Go to section              |
|-----------------------------------------------------|----------------------------|
| "Help me structure my thesis"                       | § 1 Thesis Structure       |
| "Write / improve my intro / related work / etc."    | § 2 Section Guides         |
| "Help me with literature review"                    | § 3 Literature Review      |
| "Find / format citations"                           | § 4 IEEE Citations         |
| "Is my writing clear / academic?"                   | § 5 Style & Clarity        |
| "Common writing mistakes"                           | § 6 Writing Problems       |
| "What's my research question / contribution?"       | § 7 Contribution Framing   |
| "How should I evaluate / benchmark?"                | § 8 Evaluation Quality     |
| "Prepare for peer review / defense"                 | § 9 Review Preparation     |

---

## 1. Thesis Structure (Constructor University BSc CS)

The Constructor University template defines this structure, which should be
respected unless the supervisor specifies otherwise:


Abstract                    (separate audience: non-specialist; 15–20 lines)
Statutory Declaration
Table of Contents

1. Introduction             (1–2 pages)
   - Connect to core concepts in the field
   - State the research/engineering question
   - Outline the rest of the thesis

2. Statement and Motivation of Research   (5–10 pages)
   - Exact research question
   - Why the project is relevant/interesting
   - Background and literature review
   - Gap analysis: where does this extend the state of the art?
   - Weaknesses in prior approaches you hope to overcome
   - Preliminary experiments if any

3. Description of the Investigation       (5–10 pages)
   - Technical core: how you answered the research question
   - Design of experiments or simulations
   - Difficulties encountered and how addressed

4. Evaluation of the Investigation        (5–10 pages)
   - Evaluation criteria
   - Comparison to published / state-of-the-art results

5. Conclusions                            (~0.5 page)
   - Summary of main results
   - Direct answers to research questions stated in §1

References
Appendices (code, proofs, extended tables, typesetting examples)


*WHY → WHAT → HOW ordering* (Nijssen): at every level — chapter, section,
paragraph — always establish why before what, and what before how.
Never describe an implementation before the reader knows what problem it solves
and why that problem matters.

*Thesis vs. report* (Nijssen): a report is written for 1–2 people who know
the topic. A thesis is a public document that must be understandable to any CS
graduate. Write for the broader audience.

---

## 2. Section-by-Section Writing Guides

### Abstract
The abstract addresses a different audience than the rest of the thesis: a
non-specialist decision-maker, not a technical expert. Do not use jargon.
Motivate the work and highlight its importance.

Constructor University target: 15–20 lines. IEEE papers: single paragraph, 150–250 words.

Four-sentence template (write last):
1. Context — what field/problem?
2. Gap — what is missing?
3. Method — what did you do?
4. Result — what did you find/contribute?

### Introduction
*Goal*: connect to the field, state the research question, outline the thesis.

Structure (funnel):
1. Broad context — why does this topic matter?
2. Narrow to the specific problem
3. Gap — what do we not yet know?
4. Research question / engineering question (be precise)
5. Contributions preview — why does this work matter?
6. Outline — one sentence per chapter

*Do not open with* "In recent years..." — this is a cliché. Start with a
concrete claim or observation.

Contributions must be *specific and verifiable*, not vague:
- ✗ "We investigate the performance of X"
- ✓ "We show that X outperforms Y on benchmark Z by N%"

### Statement and Motivation of Research (Background + Related Work)
*Goal*: establish the problem precisely, review prior work, and identify the gap.

Structure:
1. Formal or precise definition of the problem (use mathematics where it
   clarifies; avoid it where it obscures)
2. Related work — grouped by *theme*, not chronology
3. For each theme: summarise the thrust, then identify what it leaves open
4. Explicit gap statement: "Unlike [A, B, C], this thesis addresses..."

*Synthesis, not summary*: do not write "Paper A does X. Paper B does Y."
Write: "Approaches to X fall into two camps: [A, B] share limitation L;
[C] addresses L but introduces M."

*Gap identification template*:
> "While [existing work] has demonstrated [achievement], these approaches
> [limitation]. In particular, no prior work has addressed [gap], which matters
> because [reason]. This thesis addresses this gap by [approach]."

### Description of the Investigation (Methodology)
*Goal*: enable exact reproducibility. Justify every design choice.

*Reproducibility test* (Zobel; Collberg & Proebsting): *Could a researcher
outside your group replicate this experiment from this section alone?*
Collberg et al. found that only ~32% of surveyed CS papers met even weak
repeatability criteria. Write methods sections that beat that standard.

Must include:
- Model / system: exact name, version, checkpoint, parameter count (with citation)
- Dataset: name, split sizes, pre-processing pipeline, augmentations
- Experimental setup: framework and version, hardware (GPU model, VRAM), random seeds
- Hyperparameters: all of them — learning rate, batch size, optimiser, scheduler
- Evaluation: metric definitions (formal for non-standard metrics),
  statistical testing method

*Difficulties*: the Constructor University template explicitly asks you to
describe difficulties encountered and how you addressed them. Include this —
it demonstrates research maturity and honesty.

### Evaluation of the Investigation (Results + Discussion)
*Goal*: compare to state-of-the-art; present and interpret findings.

*Evaluation criteria first*: before showing numbers, state what criteria a
good result must satisfy, and why. This prevents the reader from questioning
your choice of metrics.

See § 8 for a full benchmarking checklist. Key points:
- Use an appropriate baseline (typically the original system with default settings)
- Report variance (std deviation or confidence intervals), not just point estimates
- Use *geometric mean* to average overhead ratios, not arithmetic mean
- Do not select a benchmark subset without explicit justification

*Report, then interpret*: present findings neutrally in Results; interpret
in Discussion/Evaluation.

### Conclusions
*Goal*: concise summary + forward vision. Target ~0.5 pages.

Rules:
- Do not introduce new information
- Directly answer each research question stated in the Introduction
- Mirror the contributions listed in the Introduction
- Future work: be specific — say what and why, not "future work could explore X"

---

## 3. Literature Review

### Workflow
1. *Seed*: 3–5 highly cited anchor papers → extract their reference lists
2. *Forward search*: Google Scholar "Cited by" on anchor papers
3. *Venue sweep*: check last 2–3 years of relevant venues (NeurIPS, ICML,
   ICLR, CVPR, ACM CCS, USENIX Security, VLDB, SOSP, etc. — depending on sub-area)
4. *Snowball* until saturation — new papers stop introducing new concepts

### Synthesis (Zobel, Nijssen)
- Do NOT summarise papers one by one
- DO group by theme, state the collective thrust, then identify the gap
- End with an explicit positioning statement

### Gap Identification Template
> "While [existing body of work] has demonstrated [achievement], these approaches
> [limitation]. In particular, no prior work has addressed [specific gap], which
> is critical because [reason]. This thesis addresses this gap by [approach]."

---

## 4. IEEE Citation Style

### Key Rules
- Citations are *numbered in order of first appearance*: [1], [2], [3]...
- Use the same number every time the same source recurs
- References list is ordered by citation number, NOT alphabetically

### Format Templates

*Conference paper:*
> [1] A. Author and B. Author, "Title of paper," in Proc. Conf. Name (ABBREV),
> City, Country, Year, pp. ZZ–ZZ.

*Journal article:*
> [2] A. Author, B. Author, and C. Author, "Title," Journal Name, vol. X,
> no. Y, pp. ZZ–ZZ, Mon. Year. doi: 10.xxxx/xxxxx.

*arXiv preprint:*
> [3] A. Author, "Title," arXiv:XXXX.XXXXX [cs.LG], Year. [Online]. Available:
> https://arxiv.org/abs/XXXX.XXXXX

*Book:*
> [4] A. Author, Title of Book, Xth ed. City: Publisher, Year.

*Software / GitHub:*
> [5] A. Author, "Tool Name," Version X.Y, GitHub, Year. [Online]. Available:
> https://github.com/user/repo

### Citation Patterns by Context

| Context | Pattern |
|---|---|
| Introducing a concept | "Sparse autoencoders [1] are used to..." |
| Supporting a claim | "ViTs have been shown to outperform CNNs [1], [4], [7]." |
| Contrasting approaches | "While [1] uses gradient attribution, [2] employs causal intervention." |
| Methodology reference | "Following the protocol of [3], we..." |
| Direct quote (rare in CS) | The authors note that "exact phrase" [1, p. X]. |

### Common Pitfalls
- Do NOT use author names as the primary reference handle: ✗ "Smith et al. [1]
  showed..." → ✓ "As shown in [1]..."
  - Exception: when the author is the subject: "The method in [1]..." is fine
- arXiv papers: always include the arXiv ID, not just the URL
- If a preprint was later published at a conference, cite the conference version

For extended IEEE formatting examples (edge cases: >6 authors, workshop papers,
theses, datasets), load references/ieee-examples.md.

---

## 5. Writing Style & Clarity

### The Core Principle (Zobel, Nijssen)
*Be economical and simple — but not informal.* Simple ≠ informal.

| Informal ✗ | Simple and formal ✓ |
|---|---|
| "We couldn't think of a good reason for these results." | "The explanation for these results remains an open question." |
| "We first tried X but it didn't work, then Y worked better." | "We evaluated approaches X and Y; Y performed better." |

### Paragraph Structure (PEEL)
- *Point* — topic sentence stating the paragraph's claim
- *Evidence* — data, citation, or demonstration
- *Explain* — how the evidence supports the claim
- *Link* — transition to the next paragraph or section

One idea per sentence. One idea per paragraph. One idea per section.

### Academic Register Checklist
- [ ] No contractions ("don't" → "do not")
- [ ] No colloquialisms ("a lot of" → "numerous")
- [ ] No clichés ("In recent years...", "rapidly evolving field...")
- [ ] No buzzwords used to camouflage weak claims
- [ ] Passive voice used sparingly — acceptable in Methods, avoid in
      Introduction and Conclusions
- [ ] Hedging where appropriate ("suggests", "indicates")
- [ ] Precise quantification ("several" → "three"; "fast" → specific measurement)
- [ ] Acronyms defined on first use, then used consistently

### Sentence-Level Fixes
| Weak phrasing | Stronger alternative |
|---|---|
| "It can be seen that..." | State the finding directly |
| "In order to..." | "To..." |
| "Due to the fact that..." | "Because..." |
| "Utilize" | "Use" |
| "Novel" (overused) | Describe what is new specifically |
| "State of the art" as noun | Use as adjective or cite a specific benchmark |
| "It is important to note that..." | Delete; state the finding |

*Obfuscation* (Nijssen): if a sentence requires three readings, rewrite it.
Long sentences that join independent clauses with commas should be split.
Do not use jargon to imply the reader should already know something.

### CS-Specific Precision (Zobel)
- Name the *exact model* (not "a transformer" → "ViT-B/16 [X]")
- Distinguish *architecture* from *weights* from *checkpoint*
- Distinguish *correlation, **association, and **causation*
- Figures: every axis must be labelled with units; every figure needs a
  self-contained caption

---

## 6. Common Writing Problems & Fixes

### Wordiness
❌ "It is important to note that the results demonstrate..."
✅ "The results demonstrate..."

❌ "In order to evaluate the performance of the model..."
✅ "To evaluate the model..."

❌ "A total of three baseline models were used..."
✅ "Three baseline models were used..."

❌ "In this chapter, we will use the experience gained from prior chapters to
    formulate guidelines for researchers who wish to build X."
✅ "This chapter formulates guidelines for building X."

### Weak Verbs & Passive Overuse
❌ "The experiment was conducted by us to evaluate..."
✅ "We conducted the experiment to evaluate..."

❌ "There was a significant improvement observed in accuracy"
✅ "Accuracy improved significantly"

> *Note*: passive is acceptable in Methods ("Weights were initialised using...").
> Avoid it where authorial agency matters.

### Vague Language
❌ "Several studies have shown that transformers generalise well..."
✅ "Three studies [1]–[3] demonstrated that ViT-B/16 generalises to
    out-of-distribution data..."

❌ "The model performed well"
✅ "The model achieved 91.3% top-1 accuracy on ImageNet-1K [X]"

❌ "The results were significant"
✅ "The results were statistically significant (p < 0.01, Cohen's d = 0.61)"

### Overclaiming vs. Hedging
| Overclaim ❌ | Appropriate hedge ✅ |
|---|---|
| "This proves that..." | "This suggests that..." |
| "Our method is superior" | "Our method outperforms [X] on [benchmark] by [Y]" |
| "Results conclusively demonstrate..." | "Results indicate..." |
| "The explanation is..." | "One plausible explanation is..." |

### Citation Problems (IEEE)
❌ Citing a survey for a result that originates in a primary source
✅ Find and cite the primary source directly

❌ Undifferentiated string: "Many works address this [1][2][3][4][5]"
✅ Group by contribution: "Gradient-based methods [1][2] and causal methods
    [3][4] both address X, differing in..."

### Tense by Section

| Section | Tense | Example |
|---|---|---|
| Abstract | Past (methods/results); Present (conclusions) | "We trained... This suggests..." |
| Introduction | Present (established knowledge) | "Transformers achieve state-of-the-art results [1]." |
| Related Work | Past (what prior work did) | "Wang et al. [2] identified..." |
| Methodology | Past | "We fine-tuned the model for 10 epochs..." |
| Results / Evaluation | Past | "The model achieved 91.3% accuracy." |
| Discussion | Present + Past | "These results indicate... We found that..." |
| Conclusions | Present (contributions); Future (next steps) | "This work demonstrates... Future work should..." |

### Transition Words by Function

*Addition:* furthermore, moreover, additionally, in addition
*Contrast:* however, nevertheless, conversely, in contrast, yet
*Cause/effect:* therefore, consequently, as a result, thus, hence
*Illustration:* for instance, specifically, to illustrate, concretely
*Sequence:* first, subsequently, finally, thereafter
*Concession:* although, while, despite this, even so
*Summary:* in summary, taken together, overall, collectively

---

## 7. Contribution Framing

A strong CS contribution is (Zobel):
1. *Specific* — describes exactly what artifact or finding you produce
2. *Novel* — distinguishes from prior work
3. *Verifiable* — can be confirmed or falsified

### Contribution Types

| Type | Phrasing template |
|---|---|
| System / tool | "We implement X, a system that does Y, demonstrating Z" |
| Empirical result | "We show empirically that X, across N models/datasets" |
| Theoretical insight | "We prove / formalise that X under assumptions Y" |
| Negative result | "We demonstrate that X, contrary to prior belief, does not Y" |
| Methodology | "We introduce a method for X that improves over Y by Z" |
| Survey / analysis | "We survey N papers and identify M systematic patterns in..." |

---

## 8. Evaluation Quality & Benchmarking

This section draws on van der Kouwe et al. ("Benchmarking Crimes", 2018), a
survey of 50 systems security papers at top venues. They found an average of
5 benchmarking crimes per paper, and only 1 paper in 50 with none. While their
focus is systems security, the taxonomy applies broadly to CS evaluation.

### The Four Requirements of a Good Evaluation
- *Completeness* — verifies all claimed contributions
- *Relevance* — results actually tell the reader something meaningful
- *Soundness* — numbers measure what is intended, with repeatability
- *Reproducibility* — enough information to replicate the study

### Common Benchmarking Errors to Avoid

*Selective benchmarking*
- Not evaluating performance degradation on all workload dimensions the system
  could plausibly affect
- Selecting a benchmark subset without justification — especially problematic
  if the omitted benchmarks stress your system most
- Presenting an incomplete suite's average as representative

*Improper result handling*
- Using microbenchmarks as evidence of overall system performance
- Reporting throughput degradation as overhead when the CPU was not fully loaded
  (systematically underestimates true overhead)
- Incorrect averaging: use *geometric mean* for overhead ratios, not arithmetic
- Not reporting variance: always include std deviation or confidence intervals

*Wrong benchmarks*
- Benchmarking a simplified or virtualised system when the real system is available
- Using IO-bound workloads to evaluate a CPU-bound technique
- Train/test leakage: using the same data for calibration and evaluation

*Improper comparisons*
- No proper baseline (baseline = original system with default settings)
- Only comparing against your own earlier work, not the state of the art
- Unfair competitor benchmarking (using non-optimal competitor configurations)

*Omissions*
- Not evaluating all claimed contributions
- Measuring only runtime — memory, binary size, startup costs are often relevant too
- Not testing false positive/negative rates where applicable

*Missing information (reproducibility)*
- Missing hardware specification (CPU, GPU model, VRAM, memory)
- Missing software versions (OS, compiler, framework, library)
- Not listing individual subbenchmark results — only the aggregate
- Presenting relative overhead only, without absolute numbers

### Reproducibility (Collberg & Proebsting)
A survey of 601 CS papers found only 32% met weak repeatability (code available
and buildable in ≤30 minutes). To write an evaluation section that holds up:
- Provide exact software versions, hardware specs, and random seeds
- Archive code and data (GitHub + Zenodo DOI, or similar)
- In your thesis, state explicitly which artifacts are available and where

---

## 9. Peer Review & Defense Preparation

### Anticipating Reviewer Questions
After each section, ask:
- *Significance*: "Why does this matter?"
- *Novelty*: "What is new vs. [closest prior work]?"
- *Rigour*: "Are the experiments sufficient to support this claim?"
- *Clarity*: "Could a reader in an adjacent subfield follow this?"

### Common Reviewer Objections in CS
- "The evaluation baseline is insufficient or misconfigured"
- "Variance is not reported; the result may not be reliable"
- "Not all contributions listed in the introduction are evaluated"
- "The related work section omits [key paper]"
- "The comparison to prior work is unfair"

Pre-empt these in your Evaluation and Conclusions sections.

### Defense Checklist
- [ ] Can you state your research question in one sentence?
- [ ] Can you explain your methodology to a CS graduate outside your sub-area?
- [ ] Do you know every number in your results tables?
- [ ] Do you have answers for the 3 hardest questions a reviewer could ask?
- [ ] Is your conclusion directly traceable back to your stated contributions?
- [ ] Can you defend every design choice in the Investigation section?

---

## 10. Workflow Integration

### Thesis Writing Session Template
When the user asks for help on a specific section:
1. Ask: what draft (if any) exists? What feedback have they received?
2. Identify the section type → apply the relevant guide above
3. For writing tasks: produce a draft, then annotate weaknesses inline with
   <!-- REVIEW: ... --> comments
4. For review tasks: apply the Style Checklist + PEEL check + precision audit
5. Always preserve the user's chosen structure unless asked to restructure;
   flag any structural concern before acting on it

### Citation Lookup
When asked to format a citation:
- Ask for: author(s), title, venue/journal, year, DOI or URL
- Format per IEEE templates above
- If only a paper title is given, note that bibliographic details cannot be
  verified — recommend the user confirm via IEEE Xplore or the publisher page

---

## Reference Files

- references/interpretability-papers.md — Curated reading list for mechanistic
  interpretability and vision transformers (load when user asks for literature
  recommendations in this area)
- references/ieee-examples.md — Extended IEEE formatting examples with edge
  cases (load when user has complex citation formatting questions)