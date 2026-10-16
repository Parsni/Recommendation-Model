# Academic Report Template — Business Analytics DA

> Placeholders in `[Square brackets]` must be filled from REAL runs on the
> chosen dataset. Do not fill any of them with synthetic-demo numbers or
> invented findings. Every claim must trace to an artifact in `outputs/`.
> The bundled `data/prompt_rec/` trial ratings are deterministic placeholders as
> well - `docs/domain_prompt_recommendation.md` section 5 lists the artifacts they
> produce, and none of those numbers belongs in this report.

---

## 1. Title
Prompt-to-Model Matching: a recommender system for choosing which AI tool to run
a prompt on — [refine wording, add your course/module identifier]

## 2. Executive summary
[150-250 words, written last: problem, method, headline measured results,
key recommendation]

## 3. Problem statement and context
- Real-world problem: users facing many AI tools (chat assistants, reasoning models,
  code assistants, image generators) do not know which one handles a given prompt
  best; trial-and-error costs time and credits, and public benchmarks rarely match an
  individual's task mix and writing style.
- Business/analytical objective: turn each user's own logged trials (prompt, tool,
  rated output) into ranked tool suggestions for new prompts, and quantify how much
  better - or no better - the personalized ranking is than "use whatever is most used
  and well rated overall".
- Research questions: [RQ1 - does the hybrid beat the popularity baseline on held-out
  trials? RQ2 - which task categories show the largest tool-performance differences?
  RQ3 - how much prompt-text similarity is needed before a suggestion is trustworthy?
  Finalise wording and, if you pre-register targets, state them before evaluating]
- Scope and success metric agreed BEFORE evaluation: [NDCG@4 and hit-rate@4 vs the
  popularity baseline on your own trials; state the improvement you will treat as
  meaningful and why]

## 4. Data
- Source, period, sampling unit: [your own AI trials; one row = one prompt run on one
  model on one date, rated 1-5 against the rubric in
  `data/prompt_rec/README_DATA_COLLECTION.md`. Logging them inside the app
  (**My trial log -> Log a trial**) produces exactly this schema.]
- Schema mapping used (config): `config/prompt_config.json` —
  prompt_id → user_id, model_id → item_id, quality_score → interaction_value,
  timestamp → timestamp; prompts.csv carries task_category/language/prompt_text,
  models.csv carries provider_label/best_for/modality/pricing
- Validation outcomes: [paste real ValidationReport lines from the app's
  "My trial log → Data & uploads" section]
- Data-quality report: [quality_report + chart from outputs/]
- Known data limitations: [single rater, no inter-rater reliability check, self-selected
  tool set, prompt phrasing not randomised, catalog size — list what actually applies]

## 5. Methodology
- System architecture: [link/summarize docs/architecture.md]
- Algorithms eligible for this data and why: [real eligibility table from a
  run on the chosen dataset]
- Preprocessing decisions and justifications: [list]
- Hybrid configuration: [weights chosen on validation + tuning trials]
- Statistical methods and the reason each test was chosen (variable types,
  assumptions, sampling unit): [list]

## 6. Statistical analysis of the problem domain
[Only analyses actually executed via src/statistics/ on the real data. For
each: question → test → assumptions checked → statistic, CI, effect size →
interpretation WITH its caveats. Never present a p-value without its context.]

## 7. Model evaluation
- Split strategy and why it prevents leakage: [describe the real split]
- n eval users, K, metric definitions: [from docs/evaluation.md]
- Comparison table (baseline vs each eligible method vs hybrid): [paste from
  outputs/tables/... generated on real data]
- Uncertainty: [standard errors / CIs; significance tests if run]
- Which method wins on which metric and what that means for the problem:
  [analysis]

## 8. Findings
[Findings strictly supported by Sections 6-7. State effect directions and
magnitudes, not just "significant".]

## 9. Recommendations
[Actionable recommendations, each tied to a finding, including confidence
and prerequisites]

## 10. Limitations and threats to validity
- General: [evaluation caveats, data caveats - copy real ones from outputs]
- Fallback events during this run: [from outputs/logs/fallback_log.jsonl]
- What was NOT possible and why: [state precisely]

## 11. Reproducibility appendix
- Environment: [python -V, pip freeze subset]
- Config snapshot used: [config file version / diff from default]
- Random seed: [value]
- Commands executed: [exact list]
- Test run reference: [pytest summary]

## 12. References
[Real citations only — literature actually consulted for methods/context.]

---

### Pre-submission checklist
- [ ] No `[...]` placeholders remain
- [ ] Every number traceable to an `outputs/` artifact from a real-data run
- [ ] Scores never described as probabilities unless calibrated
- [ ] Fallback log reviewed and reflected in limitations section
- [ ] Statistical assumptions documented alongside every test result
