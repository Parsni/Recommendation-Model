# Algorithm documentation

All algorithms implement `BaseRecommender`
(`src/recommenders/base.py`): `check_suitability()`, `fit()`,
`recommend(user_id, top_k, exclude_seen, candidate_ids)` returning
`list[Recommendation]` with item id, score, score kind, rank, contributing
sources, explanation and limitations.

---

## 1. popularity (`PopularityRecommender`)
- **Needs**: interaction log only (user_id, item_id). Works with zero metadata.
- **Method**: score = count of distinct users who interacted with the item
  (mean interaction value reported alongside). Ties broken by item id.
- **Output kind**: `frequency_count`.
- **Suitability**: any dataset with ≥1 usable interaction.
- **Limitations**: not personalized; popularity bias; it is the system's
  permanent baseline and cold-start answer.
- **Tests**: known top-5 order on demo; dedup-count correctness; empty-data raise.

## 2. content_based (`ContentBasedRecommender`)
- **Needs**: item metadata with ≥1 text field (TF-IDF) and/or categorical
  fields (tokenized); user history to build a profile.
- **Method**: sublinear TF-IDF over item descriptions; user profile =
  value-weighted mean of history vectors; score = cosine(profile, item).
- **Output kind**: `similarity` [0,1].
- **Suitability check**: fails cleanly when `items` absent or no text/cat
  fields.
- **Explainability**: lists top profile terms and matched terms per item
  (derived from actual TF-IDF contributions, not invented).
- **Limitations**: only as good as metadata text; cold users have no profile
  (returns empty → hybrid uses other sources).

## 3. item_similarity (`ItemSimilarityRecommender`)
- **Needs**: interaction matrix with overlapping items across users.
- **Method**: cosine similarity between item columns of the user-item matrix;
  user seeds = up to `hybrid.seed_top_n` highest value×recency items;
  score = weight-normalized mean similarity to seeds.
- **Output kind**: `similarity` [0,1].
- **Suitability**: ≥2 items, at least one co-occurring pair, ≤3000 items
  (dense-similarity feasibility).
- **Limitations**: sparse logs give weak neighbours; popularity leakage into
  co-occurrence.

## 4. collaborative (`CollaborativeRecommender`)
- **Needs**: ≥10 users, ≥10 items, ≥20 interactions, density ≤0.9
  (configurable `thresholds`).
- **Method**: TruncatedSVD factorization (n_components =
  `collaborative.n_components`, default 8, seed fixed); score = latent-factor
  reconstruction value for unseen cells.
- **Output kind**: `ranking_score` (NOT calibrated; not a predicted rating).
- **Limitations**: unstable on tiny data; cold users/items produce nothing;
  factors are not per-feature explainable - explanation states this honestly.
- **Demo-data note**: in the synthetic demo each user consumes every item of
  their taste group, so full-history CF often returns empty for warm users;
  under the train/test split it does score held-out items. Both behaviours
  are exercised by tests.

## 5. association (`AssociationRecommender`)
- **Needs**: basket-like logs; ≥10 users; ≥1 user with ≥2 distinct items;
  values optional (with values, items at/above `min_basket_value` form the
  basket).
- **Method**: support = share of baskets containing item; confidence(a→b) =
  co-occurrence(a,b)/count(a); user score = max confidence over their basket
  items; lift reported in the explanation. Implemented directly on sparse
  matrix arithmetic - no mlxtend dependency.
- **Output kind**: `confidence` [0,1] - an empirical co-occurrence rate,
  explicitly not P(user likes item).
- **Limitations**: confounds popularity; no causality; min-confidence filter
  configurable.

## 6. hybrid (`HybridRanker`)
- Collects top-`pool` candidates from every fitted source (per user).
- **Candidate-relative min-max normalization** per source (so excluded
  high-scoring items cannot compress eligible candidates toward 0 - a bug
  caught by tests during development).
- Weighted sum of normalized scores (config weights; equal weights are only
  the initial configuration; three weight sets are compared on validation
  data in `evaluation/compare.py::tune_weights`).
- Aggregation, duplicate removal (dict-keyed), invalid/unseen filtering,
  top-K cut, deterministic tie-break by item id.
- Optional greedy diversification using the first categorical item field.
- Merged per-source explanations + per-source limitations on every output row.

## Automatic selection & fallback

`selection/eligibility.py` runs each algorithm's suitability check; failures
exclude the algorithm with a human-readable reason (shown in the UI and
status). `fallback/manager.py` wraps fitting: one `retry_simple()` correction
attempt when available, otherwise the configured alternative (popularity),
logging every event. Nothing is claimed to have run unless it did.

## Cold start

Unknown user id → popularity baseline with an explicit "no history" note
(`engine.recommend` returns `cold_start: True`). Item never seen in training
→ no interaction-based score exists for it; only content-based can surface
it, which is demonstrated (and explained) in the demo evaluation table.
