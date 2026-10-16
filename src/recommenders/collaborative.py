"""Collaborative filtering via truncated matrix factorisation (SVD).

Input requirements : user-item interaction matrix with enough users, items
                     and overlap (see check_suitability thresholds).
Suitable for       : datasets where user similarity cannot be captured by
                     item-item similarity alone; needs minimum data volume.
Output             : score = latent-factor reconstruction value for the
                     (user, item) cell. This is a RANKING SCORE from matrix
                     factorisation - NOT a calibrated rating or probability.
Limitations        : unstable on tiny/sparse data; cold users and cold items
                     get no score; scores are not interpretable per feature.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.features.preprocess import PreparedData
from src.recommenders.base import BaseRecommender, Recommendation, SuitabilityResult


class CollaborativeRecommender(BaseRecommender):
    name = "collaborative"
    requires = {"interactions": True, "min_users": True, "min_items": True,
                "user_overlap": True}

    def check_suitability(self, data: PreparedData) -> SuitabilityResult:
        t = data.config.get("thresholds", {})
        min_u = int(t.get("min_users_for_cf", 10))
        min_i = int(t.get("min_items_for_cf", 10))
        min_int = int(t.get("min_total_interactions", 20))
        max_density = float(t.get("max_matrix_density_for_cf", 0.9))
        if data.n_users < min_u:
            return SuitabilityResult(False, f"only {data.n_users} users (< {min_u})")
        if data.n_items < min_i:
            return SuitabilityResult(False, f"only {data.n_items} items (< {min_i})")
        if data.n_interactions < min_int:
            return SuitabilityResult(False,
                                     f"only {data.n_interactions} interactions (< {min_int})")
        if data.density > max_density:
            return SuitabilityResult(False,
                                     f"matrix too dense ({data.density:.2f} > {max_density}) "
                                     "for meaningful similarity")
        return SuitabilityResult(True,
                                 f"{data.n_users} users x {data.n_items} items, "
                                 f"density {data.density:.3f}")

    def fit(self, data: PreparedData) -> "CollaborativeRecommender":
        self.data = data
        if data.matrix is None or data.matrix.nnz == 0:
            raise ValueError("collaborative filtering needs a non-empty matrix")
        n_cap = int(self.config.get("collaborative", {}).get("n_components", 8))
        n_comp = int(min(n_cap, data.n_users - 1, data.n_items - 1))
        if n_comp < 2:
            raise ValueError("matrix too small for meaningful factorisation")
        svd = TruncatedSVD(n_components=n_comp, random_state=int(data.config.get("random_seed", 42)))
        user_factors = svd.fit_transform(data.matrix)
        self.reconstructed = user_factors @ svd.components_  # users x items scores
        self.n_comp = n_comp
        self.explained_ratio = float(svd.explained_variance_ratio_.sum())
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: bool = True,
                  candidate_ids: list | None = None) -> list[Recommendation]:
        self._require_fit()
        i = self.data.index_of(user_id=user_id)
        if i is None:
            return []  # cold user
        cand = self._items_from_candidates(candidate_ids)
        idx = {it: k for k, it in enumerate(self.data.item_ids)}
        scores = {}
        for item in cand:
            j = idx.get(item)
            if j is None:
                continue
            s = float(self.reconstructed[i, j])
            if s > 1e-9:
                scores[item] = s
        if not scores:
            return []
        limitations = ("Latent-factor score from matrix reconstruction; not calibrated and "
                       "not directly explainable by item features.")

        def explain(item, score, rank):
            return (f"Users with similar interaction histories to this user show latent-factor "
                    f"affinity {score:.2f} for this item ({self.n_comp} factors, "
                    f"{self.explained_ratio:.0%} of variance).")

        return self._rank(scores, user_id, top_k, exclude_seen, explain,
                          score_kind="ranking_score", limitations=limitations)
