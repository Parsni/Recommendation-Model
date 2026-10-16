"""Item-to-item similarity recommender (cosine on the interaction matrix).

Input requirements : interaction matrix with enough overlap between users.
Suitable for       : datasets where users share items; medium sparsity OK.
Output             : score = weighted sum of cosine similarities between the
                     candidate item and the user's seed items. Similarity in
                     [0, 1] range, NOT a probability.
Limitations        : quality depends on overlap; weak for very sparse logs;
                     biased toward commonly co-consumed items.
"""
from __future__ import annotations

import numpy as np

from src.features.preprocess import PreparedData
from src.recommenders.base import BaseRecommender, Recommendation, SuitabilityResult

MAX_ITEMS_FOR_DENSE_SIM = 3000


class ItemSimilarityRecommender(BaseRecommender):
    name = "item_similarity"
    requires = {"interactions": True, "item_metadata": False, "user_metadata": False,
                "shared_items_across_users": True}

    def check_suitability(self, data: PreparedData) -> SuitabilityResult:
        if data.n_interactions == 0:
            return SuitabilityResult(False, "no usable interactions")
        if data.n_items < 2:
            return SuitabilityResult(False, "need at least 2 items")
        if data.n_items > MAX_ITEMS_FOR_DENSE_SIM:
            return SuitabilityResult(False,
                                     f"catalog too large for dense similarity ({data.n_items} items)")
        co = (data.matrix > 0).toarray().astype(float)
        pairs = co.T @ co
        shared = int(((np.diag(pairs) > 0)[:, None] & (np.diag(pairs) > 0)[None, :]
                      & (pairs >= 2)).sum()) // 2
        if shared == 0:
            return SuitabilityResult(False, "no item pair co-occurs for any user")
        return SuitabilityResult(True, f"{shared} co-occurring item pairs available")

    def fit(self, data: PreparedData) -> "ItemSimilarityRecommender":
        self.data = data
        if data.n_interactions == 0:
            raise ValueError("Cannot fit item similarity on empty interactions")
        M = data.matrix.toarray().astype(float)
        norms = np.linalg.norm(M, axis=0)
        norms[norms == 0] = 1e-12
        item_vecs = M / norms                      # columns as unit vectors
        self.sim = item_vecs.T @ item_vecs         # cosine similarity matrix
        np.fill_diagonal(self.sim, 0.0)
        self.is_fitted = True
        return self

    def _seed_items(self, user_id: str, n_seeds: int):
        hist = self.data.user_history(user_id)
        if hist.empty:
            return []
        weights = _seed_weights(hist, self.data.has_values)
        order = np.argsort(-weights, kind="stable")
        out = []
        for pos in order:
            item = str(hist.iloc[pos]["item_id"])
            j = self.data.index_of(item_id=item)
            if j is not None:
                out.append((j, float(weights[pos])))
            if len(out) >= n_seeds:
                break
        return out

    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: bool = True,
                  candidate_ids: list | None = None) -> list[Recommendation]:
        self._require_fit()
        n_seeds = int(self.config.get("hybrid", {}).get("seed_top_n", 8))
        seeds = self._seed_items(user_id, n_seeds)
        if not seeds:
            return []  # cold user: no personal similarity signal
        weights = np.array([w for _, w in seeds])
        raw = self.sim[[j for j, _ in seeds]].T @ weights / max(float(weights.sum()), 1e-12)
        cand = self._items_from_candidates(candidate_ids)
        idx = {i: k for k, i in enumerate(self.data.item_ids)}
        item_scores = {}
        for i in cand:
            j = idx.get(i)
            if j is not None and raw[j] > 0:
                item_scores[i] = float(raw[j])
        if not item_scores:
            return []
        seed_by_idx = {j: w for j, w in seeds}
        limitations = ("Similarity-based score: reflects co-consumption overlap with the "
                       "user's history, not an estimated probability of engagement.")

        def explain(item, score, rank):
            j = idx[item]
            best_j, best_s = max(((sj, self.sim[sj, j]) for sj in seed_by_idx),
                                 key=lambda t: t[1])
            best_item = self.data.item_ids[best_j]
            return (f"Most similar (cosine {best_s:.2f}) to {best_item} from this user's "
                    f"history; aggregate similarity {score:.2f} across {len(seed_by_idx)} seed items.")

        return self._rank(item_scores, user_id, top_k, exclude_seen, explain,
                          score_kind="similarity", limitations=limitations)


def _seed_weights(hist, has_values: bool) -> np.ndarray:
    """Weight = recorded value (or 1) with a mild recency boost."""
    if has_values:
        vals = hist["interaction_value"].fillna(1.0).to_numpy(dtype=float)
    else:
        vals = np.ones(len(hist))
    recency = np.linspace(0.5, 1.5, len(vals)) if len(vals) > 1 else np.ones(len(vals))
    return vals * recency
