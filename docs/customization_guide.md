# Customization guide

How to adapt this framework once the real-world DA problem is chosen.
This document deliberately proposes no problem or industry.

---

## 1. Components intended to remain stable (do not rewrite)

| component | why it stays |
|---|---|
| `recommenders/base.py` interface | all algorithms, the ranker and the UI depend only on `fit/recommend/check_suitability` |
| algorithm registry (`recommenders/__init__.py`) | add/remove entries; nothing else changes |
| `ranking/hybrid.py` | normalization/weights/dedup/diversity are source-agnostic |
| `fallback/manager.py`, `utils/helpers.FailureLog` | failure handling is generic |
| `selection/eligibility.py` | reads registry + suitability, generic |
| `evaluation/` metrics & splits | definitions are standard regardless of domain |
| `app.py` framework | sections and wiring stay; labels/copy may change |
| `reporting/exporter.py` | table/figure export is generic |

## 2. Components expected to change with the problem

### a. Data adapters and column mappings
- Edit `config/default_config.json → column_mapping` to your CSV headers
  (see `docs/data_schema.md` for an example).
- If the real table cannot be expressed as user_id/item_id/value/timestamp at
  all, write a small adapter function producing that shape and pass it via
  `RawData` to `engine.load(raw=...)`.

### b. Domain-specific feature engineering
- Keep `features/preprocess.py` generic. Add a `src/features/<domain>.py`
  module producing additional item/user columns *before* `prepare()`; the
  field-type inference and TF-IDF then pick up your text/categorical features
  automatically (config `items.text_fields` etc.).

### c. Relevant algorithms
- Disable irrelevant ones per run: `engine.build(algorithm_names=[...])`, or
  exclude via config thresholds.
- Add a new plugin:

```python
# src/recommenders/my_method.py
from src.recommenders.base import BaseRecommender, Recommendation, SuitabilityResult

class MyMethodRecommender(BaseRecommender):
    name = "my_method"
    requires = {"interactions": True, "item_metadata": False}

    def check_suitability(self, data):
        return SuitabilityResult(condition, "reason shown in UI/logs")

    def fit(self, data):
        self.data = data
        ...                       # train only on data given
        self.is_fitted = True
        return self

    def recommend(self, user_id, top_k=10, exclude_seen=True, candidate_ids=None):
        scores = {...}            # dict item -> raw score
        return self._rank(scores, user_id, top_k, exclude_seen,
                          explanation_fn=..., score_kind="ranking_score",
                          limitations="state what the score is and is not")

# register it - eligibility, hybrid blending, UI lists and fallbacks
# pick it up with no other edits:
RECOMMENDER_CLASSES[MyMethodRecommender.name] = MyMethodRecommender
```
- Write its unit tests alongside the existing ones (known-input mini dataset
  with hand-computed expected ranking; cold-user case; suitability failure).

### d. Eligibility and ranking constraints
- Business constraints (stock availability, region, freshness) belong in a
  filter applied to `candidate_ids` when calling `engine.recommend(...)` -
  the engine already filters, dedupes and re-ranks candidates safely.
- Tune thresholds in config (`thresholds.*`) instead of touching code.

### e. Statistical tests
- Delete/ignore unused statistics functions; the report should describe only
  the tests actually run for the chosen questions, following the selection
  guidance in `docs/statistical_methodology.md`.

### f. Evaluation strategy and success metric
- Swap `split_strategy` semantics by choosing another function in
  `evaluation/splits.py` (chronological vs user-holdout) or adding a rolling
  origin variant; set `evaluation.top_k` to the K that matters for the task.
- Define the DA success metric (e.g. hit rate@10 improvement over baseline)
  and record it in the report template before running, not after.

### g. UI language and visuals
- `app.py` uses generic labels ("user", "item", "interaction"); rename copy
  per domain. Add domain charts in new tabs; keep engine calls unchanged.
- Keep the engine calls and the honesty contract: a score bar must be labelled
  with its `score_kind`, never presented as a probability, and a card may state
  only reasons that exist as rows (`src/domain/prompt_matcher.py` pattern).
- Layout that has worked: guided steps in the sidebar, tabs per concern, a
  horizontal browse panel before the query, a card grid for comparison, one
  reason chip per card, secondary controls collapsed behind one "refine" toggle,
  and one-tap feedback that writes a real data row. Avoid modals - expanders and
  tabs keep the flow.
- Store-style density: route every on-screen message through small presentation
  helpers (`chip`, `section`, `line`, `fineprint` in `app_prompt.py`) instead of
  scattering `st.info` / `st.success` / `st.warning` boxes. Caveats and honesty
  statements survive as a chip and as fine print; the long explanation goes in one
  collapsed "how this works" expander plus tooltips. Keep one such caveat visible
  whenever the loaded dataset is not real data.
- Colours, fonts and the toolbar belong in `.streamlit/config.toml` (`[theme]`,
  `[client] toolbarMode`), not in per-page CSS. Streamlit reads that file at
  server start, so restart `streamlit run` after editing it. Repeat only shape
  (radius, chips, badges) in the page stylesheet.
- Escape any user-supplied text placed in `st.markdown(..., unsafe_allow_html=True)`.
- Streamlit 1.55 `AppTest` cannot drive `st.pills` or `st.segmented_control` (both
  read back empty / `None` headless). Use them for decoration only; keep every
  functional filter a `selectbox`/`checkbox`/`toggle` so the flow stays testable.

### h. Worked example already in this repository

The chosen DA problem is *prompt → which AI model should run it*, adapted exactly
along the lines above (full write-up: `docs/domain_prompt_recommendation.md`):

- mapping only: `config/prompt_config.json` renames the contract columns
  (prompt = user, AI model = item, your 1-5 trial rating = interaction_value);
- one adapter module: `src/domain/prompt_matcher.py` - TF-IDF neighbour blending for
  brand-new prompts. No recommender, ranker, selection, evaluation or fallback file
  was edited;
- domain choices in config: `exclude_seen: false` (a model you already trialled is
  still a valid suggestion for a new prompt), `min_basket_value: 3.0` so association
  learns from good trials only, relaxed CF floors for a small catalog;
- a hard candidate filter exposed in the UI (model `modality`) rather than a coded
  assumption inside the ranker;
- domain UI in task language: `app_prompt.py` (guided three steps, tabbed, an
  app-style front page - hero search, "Recommended for you" browse carousel, a
  comparison card grid with one reason chip per card, filters collapsed behind
  **Refine ▾**, one-tap 👍/👎 feedback - themed through `.streamlit/config.toml`,
  with an in-app trial logger writing through `src/domain/trial_log.py` to new
  files only; "Switch the app onto my log" reuses the same mapping so every layer
  revalidates on the user's own rows).

Reproduce its artifacts with `python run_prompt_demo_outputs.py`; the domain test
files are the assertions that must keep holding if you change the mapping.

## 3. Revalidation protocol on real data (required, do not skip)

1. **Validate ingestion** on a real sample: check the validation report
   counts against manual spot checks before trusting anything downstream.
2. **Re-run eligibility**: real volumes will enable/disable different
   algorithms; read the excluded-reasons table.
3. **Re-run the chronological/user-holdout comparison** against the baseline
   with a real timestamp column; all demo numbers are void outside the demo.
4. **Re-tune weights on real validation data** (`tune_weights`), then look at
   test once.
5. **Inspect explanations** on sampled outputs for plausibility against domain
   knowledge; honest limitations strings must survive into the report.
6. **Re-run fallback drill**: feed a corner of real data (a day with sparse
   logs, metadata join failures) and confirm the fallback log records what
   happens.
7. Freeze config + seed, save `outputs/logs`, and cite those artifacts in the
   academic report.

## 4. What NOT to change casually

- Score-kind labelling rules (never call uncalibrated scores probabilities).
- Leakage discipline in splits (fit on train PreparedData objects only).
- The baseline-retention rule in `select_eligible`.
