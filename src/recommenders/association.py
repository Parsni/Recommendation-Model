"""Association / co-occurrence recommender (support-confidence rules).

Implemented directly on the interaction matrix - no external rule-mining
dependency required (mlxtend is not needed).

Input requirements : interaction log that can be read as per-user baskets.
                     If values exist, items with value >= min_value form the
                     basket (e.g. positive engagements); otherwise every
                     interacted item enters the basket.
Suitable for       : basket-style logs with repeated co-occurrence.
Output             : score = max confidence(a -> candidate) over the user's
                     basket items a. Confidence in [0, 1]; it is an
                     empirical co-occurrence rate, NOT P(user likes item).
Limitations        : confounds popularity with association; no direction or
                     causality; unstable with few baskets.
"""
from __future__ import annotations

import numpy as np

from src.features.preprocess import PreparedData
from src.recommenders.base import BaseRecommender, Recommendation, SuitabilityResult

MAX_ITEMS_FOR_PAIRS = 5000


class AssociationRecommender(BaseRecommender):
    name = "association"
    requires = {"interactions": True, "basket_like_logs": True}

    def check_suitability(self, data: PreparedData) -> SuitabilityResult:
        t = data.config.get("thresholds", {})
        if data.n_interactions == 0:
            return SuitabilityResult(False, "no usable interactions")
        if data.n_items > MAX_ITEMS_FOR_PAIRS:
            return SuitabilityResult(False, f"catalog too large for pair mining ({data.n_items})")
        min_baskets = int(t.get("min_baskets_for_association", 10))
        if data.n_users < min_baskets:
            return SuitabilityResult(False, f"only {data.n_users} baskets (< {min_baskets})")
        basket_sizes = data.interactions.groupby("user_id")["item_id"].nunique()
        if basket_sizes.max() < 2:
            return SuitabilityResult(False, "all baskets hold a single item - "
                                            "no co-occurrence information exists")
        return SuitabilityResult(True, f"{data.n_users} baskets, {data.n_items} items")

    def fit(self, data: PreparedData) -> "AssociationRecommender":
        self.data = data
        t = data.config.get("thresholds", {})
        B = data.matrix.toarray().astype(float)
        if data.has_values:
            min_value = float(t.get("min_basket_value", 3.0))
            presence = B >= min_value if np.nanmax(B) >= min_value else (B > 0)
        else:
            presence = B > 0
        presence = presence.astype(float)
        n_baskets = presence.shape[0]
        counts = presence.sum(axis=0)                       # item support counts
        cooc = presence.T @ presence                        # pairwise counts
        support = counts / max(n_baskets, 1)
        min_item_support = float(t.get("min_item_support", 0.02))
        keep = counts >= max(2.0, min_item_support * n_baskets)
        with np.errstate(divide="ignore", invalid="ignore"):
            conf = np.where(keep[:, None], cooc / np.maximum(counts[:, None], 1e-12), 0.0)
        conf = np.where(keep[None, :], conf, 0.0)
        np.fill_diagonal(conf, 0.0)
        self.conf, self.support, self.counts, self.cooc = conf, support, counts, cooc
        self.min_confidence = float(t.get("min_confidence", 0.1))
        self.n_baskets = n_baskets
        self.is_fitted = True
        return self

    def _basket(self, user_id: str):
        i = self.data.index_of(user_id=user_id)
        if i is None:
            return []
        B = (self.data.matrix.toarray()[i].astype(float))
        if self.data.has_values:
            min_value = float(self.data.config.get("thresholds", {}).get("min_basket_value", 3.0))
            present = B >= min_value if (B >= min_value).any() else B > 0
        else:
            present = B > 0
        return [k for k, v in enumerate(present) if v]

    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: bool = True,
                  candidate_ids: list | None = None) -> list[Recommendation]:
        self._require_fit()
        basket = self._basket(user_id)
        if not basket:
            return []
        agg = self.conf[basket].max(axis=0)                  # max-conf rule per target
        best_src = np.argmax(self.conf[basket], axis=0)
        cand = self._items_from_candidates(candidate_ids)
        idx = {it: k for k, it in enumerate(self.data.item_ids)}
        scores = {}
        for item in cand:
            j = idx.get(item)
            if j is None or agg[j] < self.min_confidence:
                continue
            scores[item] = float(agg[j])
        if not scores:
            return []
        limitations = ("Association confidence is an observed co-occurrence rate in training "
                       "baskets; it implies no causation and mixes in item popularity.")

        def explain(item, score, rank):
            a = self.data.item_ids[basket[int(best_src[idx[item]])]]
            pair = int(self.cooc[idx[a], idx[item]])
            lift = score / max(self.support[idx[item]], 1e-12)
            return (f"In training baskets, {item} followed {a} with confidence {score:.2f} "
                    f"(co-occurred in {pair}/{self.n_baskets} baskets, lift {lift:.2f}).")

        return self._rank(scores, user_id, top_k, exclude_seen, explain,
                          score_kind="confidence", limitations=limitations)
