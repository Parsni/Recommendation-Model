"""Fallback triggering and recovery-after-failure tests."""
import pytest

from src.fallback.manager import FallbackManager
from src.recommenders.base import BaseRecommender, SuitabilityResult
from src.recommenders.popularity import PopularityRecommender
from src.utils.helpers import FailureLog


class ExplodingRecommender(BaseRecommender):
    name = "exploding"
    requires = {}

    def check_suitability(self, data):
        return SuitabilityResult(True, "test stub")

    def fit(self, data):
        raise RuntimeError("simulated training crash")

    def recommend(self, user_id, top_k=10, exclude_seen=True, candidate_ids=None):
        raise RuntimeError("simulated serving crash")


class BrokenThenFixed(BaseRecommender):
    name = "flaky"
    requires = {}

    def __init__(self, config=None):
        super().__init__(config)
        self.attempts = 0

    def check_suitability(self, data):
        return SuitabilityResult(True, "stub")

    def retry_simple(self):
        self.attempts += 1

    def fit(self, data):
        if self.attempts == 0:
            raise ValueError("needs reduced complexity")
        self.is_fitted = True
        return self

    def recommend(self, user_id, top_k=10, exclude_seen=True, candidate_ids=None):
        return []


@pytest.fixture()
def fb(tmp_path):
    return FallbackManager(FailureLog(tmp_path / "log.jsonl"))


def test_fit_failure_falls_back_to_popularity(demo_data, fb):
    out = fb.safe_fit(ExplodingRecommender(), demo_data,
                      alt_factory=lambda d: PopularityRecommender(d.config))
    assert out is not None and out.name == "popularity" and out.is_fitted
    events = [e["event"] for e in fb.events]
    assert "fit_failed" in events and "fallback_used" in events
    assert fb.log.read_all(), "failure must be persisted to the log"


def test_no_alternative_disables_algorithm(demo_data, fb):
    out = fb.safe_fit(ExplodingRecommender(), demo_data, alt_factory=None)
    assert out is None
    assert any(e["event"] == "disabled" for e in fb.events)


def test_single_correction_then_success(demo_data, fb):
    rec = BrokenThenFixed()
    out = fb.safe_fit(rec, demo_data)
    assert out is rec and rec.is_fitted
    assert any(e["event"] == "corrected" for e in fb.events)


def test_recommend_failure_returns_empty_and_logs(demo_data, fb):
    rec = ExplodingRecommender()
    rec.is_fitted = True
    rec.data = demo_data
    assert fb.safe_recommend(rec, "U001", 3) == []
    assert any(e["event"] == "recommend_failed" for e in fb.events)


def test_failing_source_cannot_break_hybrid(demo_data):
    """A source that raises during recommend contributes nothing; hybrid still ranks."""
    from src.ranking.hybrid import HybridRanker
    pop = PopularityRecommender(demo_data.config).fit(demo_data)
    boom = ExplodingRecommender()
    boom.data = demo_data
    boom.is_fitted = True
    ranker = HybridRanker([pop, boom], demo_data.config)
    rs = ranker.rank("U001", top_k=3)
    assert rs and all(r.item_id for r in rs)
