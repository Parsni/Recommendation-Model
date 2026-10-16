"""Content-based recommender: TF-IDF over item metadata + cosine to user profile.

Input requirements : item metadata with at least one text field or usable
                     categorical fields; interaction history for the user.
Suitable for       : catalogs with descriptive item attributes.
Output             : score = cosine similarity between the user's TF-IDF
                     profile and the candidate item vector. Similarity in
                     [0, 1], NOT a probability.
Limitations        : depends on metadata quality; can only match what the
                     profile words describe; cold users have no profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.features.preprocess import PreparedData
from src.recommenders.base import BaseRecommender, Recommendation, SuitabilityResult


class ContentBasedRecommender(BaseRecommender):
    name = "content_based"
    requires = {"interactions": True, "item_metadata": True,
                "text_or_categorical_item_fields": True}

    def check_suitability(self, data: PreparedData) -> SuitabilityResult:
        if data.items is None:
            return SuitabilityResult(False, "no item metadata table supplied")
        if not (data.item_text_fields or data.item_categorical_fields):
            return SuitabilityResult(False,
                                     "item metadata has no text or categorical fields to index")
        if data.n_interactions == 0:
            return SuitabilityResult(False, "no interactions to build any user profile")
        return SuitabilityResult(True,
                                 f"item text fields {data.item_text_fields}, "
                                 f"categorical fields {data.item_categorical_fields}")

    def fit(self, data: PreparedData) -> "ContentBasedRecommender":
        self.data = data
        if data.items is None:
            raise ValueError("content_based requires item metadata")
        fields = data.item_text_fields + data.item_categorical_fields
        docs = []
        for item_id in data.item_ids:
            meta = data.item_metadata(item_id)
            parts = [str(meta.get(f, "") or "") for f in fields]
            if not any(p.strip() for p in parts):
                parts = [str(item_id)]  # token fallback keeps index alignment
            docs.append(" ".join(parts).lower())
        self.docs = docs
        vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z0-9]+",
                                     min_df=1, sublinear_tf=True)
        self.tfidf = vectorizer.fit_transform(docs)
        self.feature_names = vectorizer.get_feature_names_out()
        self.item_pos = {i: k for k, i in enumerate(data.item_ids)}
        self.profiles = {}
        self.profile_items = {}
        for uid, hist in data.interactions.groupby("user_id"):
            vecs, weights = [], []
            for _, row in hist.iterrows():
                j = self.item_pos.get(str(row["item_id"]))
                if j is None:
                    continue
                v = float(row["interaction_value"]) if data.has_values and \
                    pd.notna(row["interaction_value"]) else 1.0
                vecs.append(self.tfidf[j].toarray().ravel())
                weights.append(max(v, 0.0))
            if not vecs:
                continue
            profile = np.average(np.vstack(vecs), axis=0,
                                 weights=np.array(weights) + 1e-12)
            norm = np.linalg.norm(profile)
            if norm > 0:
                self.profiles[str(uid)] = profile / norm
                self.profile_items[str(uid)] = [str(r["item_id"]) for _, r in hist.iterrows()]
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: bool = True,
                  candidate_ids: list | None = None) -> list[Recommendation]:
        self._require_fit()
        profile = self.profiles.get(str(user_id))
        if profile is None:
            return []  # cold user - honest empty, fallback layer decides
        sims = self.tfidf @ profile
        cand = self._items_from_candidates(candidate_ids)
        scores = {}
        for i in cand:
            j = self.item_pos.get(i)
            if j is not None and sims[j] > 0:
                scores[i] = float(sims[j])
        if not scores:
            return []
        top_terms = np.argsort(-profile)[:3]
        term_str = ", ".join(str(self.feature_names[t]) for t in top_terms
                             if profile[t] > 0) or "(no discriminative terms)"
        limitations = ("Content similarity reflects shared descriptive metadata with the "
                       "user's history; it is not evidence of quality or fit beyond the text.")

        def explain(item, score, rank):
            j = self.item_pos[item]
            contrib = np.argsort(-(profile * self.tfidf[j].toarray().ravel()))[:3]
            words = [str(self.feature_names[t]) for t in contrib if
                     profile[t] > 0 and self.tfidf[j, t] > 0]
            matched = ", ".join(words) if words else term_str
            return (f"Description matches this user's profile (top terms: {term_str}); "
                    f"shared terms with item: {matched}; cosine {score:.2f}.")

        return self._rank(scores, user_id, top_k, exclude_seen, explain,
                          score_kind="similarity", limitations=limitations)
