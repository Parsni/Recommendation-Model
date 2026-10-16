# Model evaluation documentation

## Strategy (implemented and enforced in code)

**Per-user chronological last-out** (`src/evaluation/splits.py`):

1. Each user's interactions are sorted by timestamp.
2. The last event = test target; the second-last = validation target;
   everything earlier = train.
3. Users with fewer than `min_train_per_user + 2` events are excluded
   (reported count: `n_eval_users`).
4. All recommender fitting, similarity computation, association mining,
   factorization and weight tuning happen on the **train object only** -
   a separate `PreparedData` rebuilt from training rows, so held-out events
   cannot be seen even indirectly.
5. Hybrid weights are compared across exactly 3 candidate configurations on
   **validation** targets; the winner is then applied to the test evaluation.
   Test outcomes are never consulted during tuning.

A `user_holdout` split is also provided for datasets without timestamps
(entire users removed from training; measures cold-start baseline behaviour).

## Metrics (`src/evaluation/metrics.py`)

| metric | definition | verified against |
|---|---|---|
| Precision@K | relevant hits in top-K / K | hand example (0.4, 0.5) |
| Recall@K | hits / total relevant | hand example |
| Hit Rate@K | 1 if any hit in top-K | hand example + per-user mean, binomial-style SE reported |
| NDCG@K | binary and graded (gain-weighted log discount / ideal DCG) | hand-computed 1/log2(3) case; graded ordering case |
| Catalog coverage@K | distinct recommended items / catalog size | hand example 0.6 |
| Intra-list diversity | distinct share within lists (simple honest proxy) | - |
| Shannon entropy | exposure evenness across lists | uniform = 1 bit check |

## Interpretation rules followed

- Every method (all eligible algorithms + hybrid) is compared **against the
  popularity baseline on the same evaluation users**.
- Results carry sample size and a standard error on hit rate; no significance
  claims are made without tests.
- The demo run (synthetic data) showed content_based NDCG@5 = 0.9667 vs
  popularity 0.0 and hybrid 0.2579 - an artifact of the synthetic structure
  (held-out items are absent from the interaction-based methods' training
  view, while metadata allows content-based generalization). This is exactly
  the kind of result the framework must not over-interpret, and it is
  documented as such in `outputs/logs/demo_run_summary.json`.
- "Most complicated model is best" is never assumed; the comparison table
  decides, and weight tuning reverts to whatever validation prefers.

## Limitations of the current evaluation layer

- Single chronological split (no rolling-origin backtest yet).
- Implicit-feedback relevance: any held-out interaction counts as relevant;
  unobserved items are neither positives nor negatives (standard caveat).
- No cross-method significance test (paired bootstrap is the natural next
  step, sketched in the customization guide).
- NDCG uses binary gains by default; graded gains supported when values exist.
