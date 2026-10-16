# Architecture

## Design goal

A reusable, domain-agnostic recommendation platform: every layer knows only
generic structures (`PreparedData`, `Recommendation`, the algorithm registry).
No module contains retail/education/finance logic.

## Data flow

```
CSV files ──> src/data/loader.py + schema.py      validation report (originals untouched)
        └──> src/features/preprocess.py           PreparedData: cleaned log,
        └──> src/features/quality.py              metadata typing, sparse matrix
                                              │
                src/selection/eligibility.py    │ per-algorithm suitability checks
                        │                       ▼
        src/fallback/manager.py  ── fit each eligible recommender
                        │           (exploding algorithm → logged fallback
                        │            to popularity baseline → or disabled)
                        ▼
        src/ranking/hybrid.py     per-user candidate lists from every fitted
                        │         source → candidate-relative min-max
                        │         normalization → weighted aggregation →
                        │         dedup → seen-filter → optional diversity →
                        │         merged explanations
                        ▼
        src/engine.py RecommendationEngine facade:
                        load() → build() → recommend(user) with cold-start
                        handling and notes
        │
        ├── src/evaluation/   chronological last-out split → per-method
        │                     Precision/Recall/Hit/NDCG@K, coverage,
        │                     validation-only weight tuning, test-once table
        ├── src/statistics/   descriptive / inferential / models (separate
        │                     from recommendation code, reusable later)
        └── src/reporting/    real outputs only: tables/figures/JSON logs
```

## Module responsibilities and boundaries

| Layer | Files | Knows about | Never does |
|---|---|---|---|
| Ingestion | `src/data/*` | column mapping, schema, malformed rows | modify source files |
| Preparation | `src/features/*` | dtypes, duplicates, sparse matrix, field typing | domain transformations |
| Algorithms | `src/recommenders/*` | PreparedData interface | each other, the UI |
| Hybrid | `src/ranking/hybrid.py` | per-source recommendation lists | retraining models |
| Selection | `src/selection/eligibility.py` | volumes, sparsity, metadata presence | fitting |
| Fallback | `src/fallback/manager.py` | exceptions, failure log | hiding failures |
| Evaluation | `src/evaluation/*` | splits, metrics, comparability | touching test data during tuning |
| Statistics | `src/statistics/*` | plain DataFrames | recommendation internals |
| UI | `app.py` | engine facade API | algorithm logic |

## Key invariants

1. **Baseline always survives**: popularity is force-included whenever any
   candidate item exists (`select_eligible`).
2. **No silent data loss**: dropped/dupe/coerced records are counted and
   reported (`ValidationReport`).
3. **No leakage by construction**: evaluation fits only on `train`
   PreparedData objects rebuilt from the split (`evaluation/splits.py`).
4. **Scores are labelled**: every `Recommendation` carries `score_kind`;
   normalization happens per source before blending; the hybrid output is
   explicitly a normalized ranking score, never a probability.
5. **Failures are recorded**: `FailureLog` JSON-lines in
   `outputs/logs/fallback_log.jsonl`.

## Domain layer (leaf, additive)

`src/domain/prompt_matcher.py` is the only domain-specific module: it wraps the
engine instead of modifying it, treating a prompt as the "user" and an AI model as
the "item", and answering brand-new prompts (which have no trial history) by
TF-IDF-matching them to logged prompts and blending those neighbours' rankings.
`src/domain/trial_log.py` is the only writer: one-tap 👍/👎 feedback and the
in-app form append rows to new files under `data/prompt_rec/uploaded/`, which the
engine reads only when the user switches onto them - source CSVs stay untouched.
`app_prompt.py` is their UI. Nothing in `src/recommenders/`, `src/ranking/`,
`src/selection/`, `src/evaluation/` or `src/fallback/` changed for the domain, and
`app.py` keeps generic terminology - the adaptation is a column mapping plus this
one adapter (`docs/domain_prompt_recommendation.md`,
`docs/customization_guide.md` section 2h).

## Reproducibility

- Single `config/default_config.json` drives mapping, thresholds, weights,
  evaluation K and `random_seed` (used by SVD, KMeans, CV folds).
- Synthetic demo is generator-script deterministic (no RNG at all).
