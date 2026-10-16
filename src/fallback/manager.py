"""Fallback and recovery manager.

Strategy (per the project rules): detect + record a failure, attempt at most
one targeted correction, otherwise switch to a defensible simpler
alternative, verify the alternative still functions, and record the reduced
functionality. Failures are never concealed.
"""
from __future__ import annotations

from src.features.preprocess import PreparedData
from src.recommenders.base import BaseRecommender
from src.utils.helpers import FailureLog


class FallbackManager:
    def __init__(self, log: FailureLog | None = None):
        self.log = log or FailureLog()
        self.events: list[dict] = []

    def safe_fit(self, recommender: BaseRecommender, data: PreparedData,
                alt_factory=None) -> BaseRecommender | None:
        """Fit one recommender with one correction attempt and an alternative.

        alt_factory: callable(data) -> a simpler fitted-or-fittable
        recommender used if this one cannot fit. Returns a fitted
        recommender (possibly the alternative) or None.
        """
        name = recommender.name
        try:
            recommender.fit(data)
            return recommender
        except Exception as exc:
            self._record(name, "fit_failed", f"{type(exc).__name__}: {exc}")
        # one targeted correction: drop the most demanding requirement by
        # re-checking suitability on a filtered view is rarely safe, so the
        # documented correction is: re-fit at reduced complexity when the
        # recommender exposes `retry_simple()`.
        if hasattr(recommender, "retry_simple"):
            try:
                recommender.retry_simple()
                recommender.fit(data)
                self._record(name, "corrected", "refit succeeded after retry_simple()")
                return recommender
            except Exception as exc:
                self._record(name, "correction_failed", f"{type(exc).__name__}: {exc}")
        if alt_factory is not None:
            try:
                alt = alt_factory(data)
                if not alt.is_fitted:
                    alt.fit(data)
                self._record(name, "fallback_used",
                             f"switched to '{alt.name}'; {name} disabled for this run")
                return alt
            except Exception as exc:
                self._record(name, "fallback_failed", f"{type(exc).__name__}: {exc}")
        self._record(name, "disabled", f"{name} cannot operate on this data")
        return None

    def safe_recommend(self, recommender: BaseRecommender, user_id: str, top_k: int,
                       **kwargs) -> list:
        """Recommend with recovery: exceptions degrade to empty list + log."""
        try:
            return recommender.recommend(user_id, top_k, **kwargs)
        except Exception as exc:
            self._record(recommender.name, "recommend_failed", f"{type(exc).__name__}: {exc}")
            return []

    def _record(self, component: str, event: str, detail: str):
        entry = self.log.record(component, event, detail)
        self.events.append(entry)

    def status_table(self):
        return [{"time": e["timestamp"], "component": e["component"],
                 "event": e["event"], "detail": e["detail"]} for e in self.events]
