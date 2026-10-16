"""Common recommender interface shared by all algorithms.

Every recommender must provide:
- name              : registry key
- requires          : explicit input requirements (documented per class)
- check_suitability : can this dataset feed this algorithm?
- fit               : train on PreparedData (training data only at eval time)
- recommend         : consistent list[Recommendation] output, or raise
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.features.preprocess import PreparedData


@dataclass
class Recommendation:
    item_id: str
    score: float
    score_kind: str = "ranking_score"   # ranking_score | similarity | normalized_score
    rank: int = 0
    sources: list = field(default_factory=list)
    explanation: str = ""
    limitations: str = ""

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "score": round(self.score, 6),
                "score_kind": self.score_kind, "rank": self.rank,
                "sources": ";".join(self.sources), "explanation": self.explanation,
                "limitations": self.limitations}


@dataclass
class SuitabilityResult:
    suitable: bool
    reason: str


class BaseRecommender(ABC):
    name: str = "base"
    requires: dict = {}

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def check_suitability(self, data: PreparedData) -> SuitabilityResult:
        """Explicit, cheap eligibility check. Must not fit anything."""

    @abstractmethod
    def fit(self, data: PreparedData) -> "BaseRecommender":
        """Train on the given data. Raise on unusable input."""

    @abstractmethod
    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: bool = True,
                  candidate_ids: list | None = None) -> list[Recommendation]:
        """Return ranked Recommendations. Empty list is valid; errors raise."""

    # ---- shared helpers -----------------------------------------------------
    def _items_from_candidates(self, candidate_ids: list | None) -> list:
        if candidate_ids is None:
            return list(self.data.item_ids)
        valid = set(self.data.item_ids)
        return [str(c) for c in candidate_ids if str(c) in valid]

    def _rank(self, scores: dict, user_id: str, top_k: int, exclude_seen: bool,
              explanation_fn=None, score_kind: str = "ranking_score",
              limitations: str = "") -> list[Recommendation]:
        items = list(scores.keys())
        if exclude_seen:
            seen = self.data.seen_items(str(user_id))
            items = [i for i in items if i not in seen]
        items.sort(key=lambda i: (-scores[i], i))  # deterministic tie-break by item_id
        out = []
        for rank, item in enumerate(items[:top_k], start=1):
            expl = explanation_fn(item, scores[item], rank) if explanation_fn else ""
            out.append(Recommendation(item_id=item, score=float(scores[item]),
                                      score_kind=score_kind, rank=rank,
                                      sources=[self.name], explanation=expl,
                                      limitations=limitations))
        return out

    def _require_fit(self):
        if not self.is_fitted:
            raise RuntimeError(f"Recommender '{self.name}' used before fit().")
