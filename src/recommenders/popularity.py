"""Popularity / frequency baseline recommender.

Input requirements : interactions only (user_id, item_id). Values optional.
Suitable for       : any dataset that has at least one interaction; the
                     always-available fallback and cold-start baseline.
Output             : score = number of distinct users who interacted with the
                     item (optionally value-weighted). NOT a probability.
Limitations        : not personalized; biased toward globally frequent items;
                     explains nothing about the individual user.
"""
from __future__ import annotations

import numpy as np

from src.features.preprocess import PreparedData
from src.recommenders.base import BaseRecommender, Recommendation, SuitabilityResult


class PopularityRecommender(BaseRecommender):
    name = "popularity"
    requires = {"interactions": True, "item_metadata": False, "user_metadata": False}

    def check_suitability(self, data: PreparedData) -> SuitabilityResult:
        if data.n_interactions == 0:
            return SuitabilityResult(False, "no usable interactions available")
        return SuitabilityResult(True, "popularity needs only the interaction log")

    def fit(self, data: PreparedData) -> "PopularityRecommender":
        self.data = data
        if data.n_interactions == 0:
            raise ValueError("Cannot fit popularity on an empty interaction set")
        counts = np.asarray((data.matrix > 0).sum(axis=0)).ravel().astype(float)  # distinct users
        self.popularity = dict(zip(data.item_ids, counts))
        if data.has_values:
            dense_sum = np.asarray(data.matrix.sum(axis=0)).ravel()
            nonzero = np.clip(counts, 1e-12, None)
            self.mean_value = dict(zip(data.item_ids, dense_sum / nonzero))
        else:
            self.mean_value = {i: 1.0 for i in data.item_ids}
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: bool = True,
                  candidate_ids: list | None = None) -> list[Recommendation]:
        self._require_fit()
        candidates = self._items_from_candidates(candidate_ids)
        scores = {i: self.popularity[i] for i in candidates if self.popularity[i] > 0}
        if not scores:
            scores = {i: 0.0 for i in candidates}  # degenerate catalog, still deterministic
        limitations = ("Popularity baseline: not personalized; reflects group-wide "
                       "frequency, not this user's taste.")

        def explain(item, score, rank):
            n_users = int(self.popularity[item])
            mv = self.mean_value.get(item, float("nan"))
            if self.data.has_values:
                return (f"Item {item} was interacted with by {n_users} distinct users "
                        f"(mean recorded value {mv:.2f}); ranked #{rank} by frequency.")
            return (f"Item {item} was interacted with by {n_users} distinct users; "
                    f"ranked #{rank} by frequency.")

        return self._rank(scores, user_id, top_k, exclude_seen, explain,
                          score_kind="frequency_count", limitations=limitations)
