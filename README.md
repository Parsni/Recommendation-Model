# Universal Recommendation Engine

A domain-agnostic, locally-runnable recommendation platform built for a
Business Analytics DA. It ingests any user-item interaction CSV, validates it,
automatically selects which recommendation algorithms can run on the data,
blends them into ranked recommendations with honest explanations, evaluates
them with leakage-safe splits, and ships reusable statistical-analysis
utilities. When the real-world problem is chosen later, only the data mapping,
features, constraints and analysis need adapting - see
`docs/customization_guide.md`.

**Status: implemented and tested.** 91 automated tests pass on Python 3.10.11
(full log: `docs/test_results.md`, raw output: `outputs/logs/pytest_verbose.txt`).

**Chosen DA topic: prompt → AI model recommendation** ("which AI tool should I run
this prompt on to get the best result?"). The engine was tailored for that without
changing the generic core: `app_prompt.py` plus `config/prompt_config.json`,
`src/domain/prompt_matcher.py` and `data/prompt_rec/`. See
`docs/domain_prompt_recommendation.md`.

---

## Requirements

- Python 3.10+ (tested on 3.10.11, Windows)
- Free, open-source libraries only: pandas, numpy, scipy, scikit-learn,
  streamlit, matplotlib, pytest. No paid APIs, no cloud, no LLM dependency.

## Setup (beginner step-by-step)

```bash
cd "D:\Recommendation Model\recommendation_engine"

# 1. create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# (or: source .venv/bin/activate on Mac/Linux)

# 2. install dependencies
pip install -r requirements.txt

# 3. run the test suite (optional sanity check)
python -m pytest tests -q
```

## Launch the applications

```bash
streamlit run app_prompt.py            # tailored: prompt -> which AI model to use
streamlit run app.py                   # generic domain-agnostic engine
```

`app_prompt.py` is the one for this DA. It is organised as three guided steps -
**🔍 Find a model** → **📝 My trial log** → **📊 Report & checks** - each with
tabs, so the input screen, your logged evidence and the report numbers never mix
on one page. Step 1 is app-style: a hero search box, a
**"Recommended for you"** browse carousel (one card per screen, ◀ ▶ plus dots, no
modal), and a **"Best match for your prompt"** comparison grid on request. Every
card carries a single blue **reason chip** quoting an existing trial row, a grey
fine-print line naming the algorithms behind it, one-tap **👍 / 👎** feedback that
writes a real 1-5 trial row, and **Not now** for a session-only display filter
(**Undo n** restores it). Secondary controls are collapsed behind **Refine ▾**
(hard model-type filter, result count, hide-models-I-rated), and the explanatory
prose lives in tooltips plus one **"How these rankings work"** panel rather than
on the canvas. The demo-data caveat stays visible as an amber sidebar chip - step
1 renders no coloured callout boxes. Step 2 also logs trials from inside the app
(`src/domain/trial_log.py`, appended to new files only) and can switch the whole
app onto your own log. `app.py` keeps the generic terminology ("user", "item",
"interaction") and runs on the synthetic demo dataset; upload your own CSVs in
either app. If both run at once, Streamlit puts the second one on the next free
port (8501 → 8502).

Colours, fonts and the minimal toolbar come from `.streamlit/config.toml`; that
file is read when the server starts, so restart `streamlit run` after editing it.

## Generate the example outputs (tables/figures/logs)

```bash
python run_demo_outputs.py             # generic demo deliverables
python run_prompt_demo_outputs.py      # prompt-domain tables/figure/log
```

Writes real computed artifacts to `outputs/tables`, `outputs/figures`
and `outputs/logs`.

## Project layout

```
recommendation_engine/
  app.py                  Streamlit UI (generic terminology, no domain bias)
  app_prompt.py           Streamlit UI for the chosen DA topic (prompt -> AI model)
  run_demo_outputs.py     regenerates verified example deliverables
  run_prompt_demo_outputs.py  same, for the prompt domain
  requirements.txt
  .streamlit/config.toml      light app theme, minimal toolbar, no usage stats
  config/default_config.json   column mapping, weights, thresholds, seed
  config/prompt_config.json    domain column mapping + domain thresholds
  data/synthetic_demo/    deterministic demo CSVs + KNOWN_FACTS.md + generator
  data/prompt_rec/        sample prompts/models/trials CSVs + data-collection protocol
  src/
    engine.py             orchestration facade (load -> build -> recommend)
    domain/prompt_matcher.py  cold-start adapter: TF-IDF prompt similarity + blend
    domain/trial_log.py        append-only personal trial logger (writes new files)
    data/                 CSV ingestion + schema validation (never mutates files)
    features/             preparation, sparse matrix, data-quality report
    recommenders/         popularity, content_based, item_similarity,
                          collaborative, association  (+ registry)
    ranking/hybrid.py     normalization, weights, dedup, diversity, explanations
    selection/            automatic algorithm eligibility
    evaluation/           metrics, chronological/user splits, comparison runner
    statistics/           descriptive / inferential / modelling utilities
    fallback/             failure detection, correction, alternative + logging
    reporting/            table/figure export helpers
    utils/                config, logging, paths
  tests/                  91 deterministic tests (60 core + 31 domain)
  outputs/                figures, tables, logs (generated by real runs)
  docs/                   architecture, algorithms, evaluation, schema,
                          statistics, domain_prompt_recommendation, customization
                          guide, report template
```

## What the scores mean (anti-fooling guide)

| score_kind              | meaning                                      | is it a probability? |
|-------------------------|----------------------------------------------|----------------------|
| frequency_count         | distinct users who interacted with the item  | no                   |
| similarity              | TF-IDF or co-occurrence cosine [0,1]         | no                   |
| ranking_score           | latent-factor reconstruction value           | no                   |
| confidence              | empirical co-occurrence rule confidence      | no                   |
| normalized_ranking_score| hybrid weighted blend of per-source rankscores| no                  |

Nothing in this system is calibrated; no uncalibrated score is ever presented
as a probability.

## Using the synthetic demo honestly

The demo dataset is **placeholder data to prove functionality** (its exact
ground truth is in `data/synthetic_demo/KNOWN_FACTS.md`). Metrics measured on
it - e.g. content-based NDCG@5 = 0.967 vs popularity 0.0 under the
chronological split - demonstrate that the pipeline *works and measures*,
not that any algorithm would be effective on a real problem.

## Known limitations

- No neural / deep-learning recommenders (none were demonstrated to be needed).
- Hybrid weights are tuned only among 3 candidate configurations (validation
  data), not optimized globally.
- Association rules use max-confidence aggregation; a full Apriori-style
  rule-set explorer is not included.
- Streamlit UI is verified by automated AppTest flows and an HTTP 200 server
  boot, not by manual visual QA (browser tooling unavailable in the build
  environment).
- Evaluation reports point estimates + standard errors; no formal significance
  testing between methods yet (paired comparisons need bootstrap - see
  customization guide).

See `docs/` for full architecture, algorithm, evaluation, statistical and
customization documentation.
