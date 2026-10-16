"""RecommendationEngine: the single entry point that wires all layers.

Generic and domain-agnostic: it only knows PreparedData, the algorithm
registry, eligibility, fallbacks and hybrid ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.data.loader import load_dataset, RawData
from src.features.preprocess import prepare, PreparedData
from src.features.quality import quality_report
from src.fallback.manager import FallbackManager
from src.ranking.hybrid import HybridRanker
from src.recommenders.base import Recommendation
from src.recommenders.popularity import PopularityRecommender
from src.selection.eligibility import select_eligible
from src.utils.helpers import FailureLog, load_config


@dataclass
class EngineStatus:
    eligible: list = field(default_factory=list)
    excluded: dict = field(default_factory=dict)
    disabled_after_failure: dict = field(default_factory=dict)
    fallback_events: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    validation_reports: list = field(default_factory=list)

    def describe(self) -> str:
        lines = ["Enabled algorithms: " + (", ".join(self.eligible) or "none")]
        for name, reason in self.excluded.items():
            lines.append(f"Excluded {name}: {reason}")
        for name, reason in self.disabled_after_failure.items():
            lines.append(f"Disabled after failure {name}: {reason}")
        return "\n".join(lines)


class RecommendationEngine:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.failure_log = FailureLog()
        self.fallback = FallbackManager(self.failure_log)
        self.data: PreparedData | None = None
        self.ranker: HybridRanker | None = None
        self.recommenders: dict[str, object] = {}
        self.status = EngineStatus()

    # ------------------------------------------------------------------
    def load(self, interactions_path=None, users_path=None, items_path=None,
             raw: RawData | None = None) -> "RecommendationEngine":
        raw = raw or load_dataset(interactions_path, users_path, items_path, self.config)
        self.data = prepare(raw, self.config)
        self.status.validation_reports = raw.reports
        self.status.quality = quality_report(self.data)
        if self.data.n_interactions == 0:
            self.failure_log.record("engine", "no_data",
                                    "interaction set empty after cleaning; only "
                                    "structure-level features available", "error")
        return self

    def build(self, algorithm_names: list | None = None) -> "RecommendationEngine":
        """Select eligible algorithms, fit each under fallback management."""
        if self.data is None:
            raise RuntimeError("call load() before build()")
        selection = select_eligible(self.data, algorithm_names)
        self.status.excluded = dict(selection["excluded"])
        fitted = []
        for rec in selection["eligible"]:
            alt = (lambda d: PopularityRecommender(d.config).fit(d)) \
                if rec.name != "popularity" else None
            out = self.fallback.safe_fit(rec, self.data, alt_factory=alt)
            if out is None:
                self.status.disabled_after_failure[rec.name] = \
                    "failed to fit and no fallback accepted it (see fallback log)"
            else:
                if out.name != rec.name:
                    self.status.disabled_after_failure[rec.name] = \
                        f"replaced by '{out.name}' fallback (see log)"
                self.recommenders[out.name] = out
                fitted.append(out)
        self.ranker = HybridRanker(fitted, self.config)
        self.status.eligible = sorted(self.recommenders)
        self.status.fallback_events = self.fallback.status_table()
        return self

    # ------------------------------------------------------------------
    def recommend(self, user_id: str, top_k: int | None = None,
                  exclude_seen: bool | None = None,
                  candidate_ids: list | None = None,
                  diversify: bool | None = None) -> dict:
        """Returns {"recommendations": [...], "cold_start": bool, "notes": [...]}."""
        if self.ranker is None or self.data is None:
            raise RuntimeError("call load() and build() before recommend()")
        top_k = top_k or int(self.config["hybrid"]["top_k"])
        exclude_seen = self.config["hybrid"]["exclude_seen"] if exclude_seen is None else exclude_seen
        notes = []
        known = self.data.index_of(user_id=user_id) is not None
        cold = not known
        if cold:
            notes.append("Cold-start user: no training history exists for this id. "
                         "Serving popularity baseline only - personalization unavailable.")
            pop = self.recommenders.get("popularity") or PopularityRecommender(self.config).fit(self.data)
            rs = self.fallback.safe_recommend(pop, user_id, top_k, exclude_seen=exclude_seen,
                                              candidate_ids=candidate_ids)
            for r in rs:
                r.explanation = ("No history for this user yet; baseline popularity used. "
                                 + r.explanation)
            return {"recommendations": rs, "cold_start": True, "notes": notes}
        sources = self.ranker.source_scores(user_id, candidate_ids)
        personalized = {k: v for k, v in sources.items() if k != "popularity" and v}
        if not personalized:
            notes.append("No personalized source produced candidates for this user "
                         "(sparse or fully-consumed history); popularity baseline used.")
        try:
            rs = self.ranker.rank(user_id, top_k=top_k, exclude_seen=exclude_seen,
                                  candidate_ids=candidate_ids, diversify=diversify,
                                  sources_map=sources)
        except Exception as exc:
            self.fallback._record("hybrid", "rank_failed", f"{type(exc).__name__}: {exc}")
            pop = self.recommenders.get("popularity")
            rs = self.fallback.safe_recommend(pop, user_id, top_k, exclude_seen=exclude_seen,
                                              candidate_ids=candidate_ids) if pop else []
            notes.append("Hybrid ranking failed; fell back to the popularity baseline.")
        return {"recommendations": rs, "cold_start": False, "notes": notes}
