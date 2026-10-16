"""Eligibility selection, hybrid ranking and engine integration tests."""
import pandas as pd
import pytest

from src.engine import RecommendationEngine
from src.data.loader import RawData
from src.features.preprocess import prepare
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.item_similarity import ItemSimilarityRecommender
from src.ranking.hybrid import HybridRanker, minmax_normalize
from src.selection.eligibility import select_eligible


def test_minmax_normalize_cases():
    assert minmax_normalize({}) == {}
    assert minmax_normalize({"a": 5.0}) == {"a": 1.0}
    assert minmax_normalize({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}
    assert minmax_normalize({"a": 0.0, "b": 10.0, "c": 5.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}


def test_metadata_absent_disables_content_based(demo_raw, demo_config):
    raw_no_items = RawData(interactions=demo_raw.interactions, users=demo_raw.users, items=None)
    data = prepare(raw_no_items, demo_config)
    sel = select_eligible(data)
    names = {r.name for r in sel["eligible"]}
    assert "content_based" not in names
    assert "content_based" in sel["excluded"]
    assert "popularity" in names  # baseline retained


def test_insufficient_volume_disables_cf_and_baskets(demo_raw, demo_config):
    small = demo_raw.interactions.groupby("user_id").head(1).reset_index(drop=True)
    data = prepare(RawData(interactions=small, users=None, items=None), demo_config)
    sel = select_eligible(data)
    names = {r.name for r in sel["eligible"]}
    assert "collaborative" not in names and "association" not in names
    assert "popularity" in names


def test_hybrid_dedup_and_bounds(demo_data):
    pop = PopularityRecommender(demo_data.config).fit(demo_data)
    sim = ItemSimilarityRecommender(demo_data.config).fit(demo_data)
    ranker = HybridRanker([pop, sim], demo_data.config)
    rs = ranker.rank("U001", top_k=6)
    ids = [r.item_id for r in rs]
    assert len(ids) == len(set(ids)), "duplicate candidates must be removed"
    assert all(0.0 <= r.score <= 1.0 for r in rs)
    assert all(r.score_kind == "normalized_ranking_score" for r in rs)
    assert all(str(r.rank) for r in rs)
    assert [r.rank for r in rs] == list(range(1, len(rs) + 1))
    assert all("probability" not in r.limitations.split("not a")[0] for r in rs)


def test_engine_end_to_end_demo():
    eng = RecommendationEngine().load(
        "data/synthetic_demo/interactions.csv", "data/synthetic_demo/users.csv",
        "data/synthetic_demo/items.csv").build()
    assert set(eng.recommenders) == {"popularity", "content_based", "item_similarity",
                                     "collaborative", "association"}
    res = eng.recommend("U001", top_k=3)
    assert len(res["recommendations"]) == 3
    assert res["cold_start"] is False
    assert all(r.explanation for r in res["recommendations"])
    cold = eng.recommend("does_not_exist", top_k=2)
    assert cold["cold_start"] is True
    assert [r.item_id for r in cold["recommendations"]] == ["I013", "I001"]
    assert any("Cold-start" in n for n in cold["notes"])


def test_engine_candidate_filter_and_exclude_seen(demo_data):
    eng = RecommendationEngine(demo_data.config)
    eng.data = demo_data
    eng.build()
    res = eng.recommend("U001", top_k=5,
                        candidate_ids=["I005", "I006", "I007"])
    assert {r.item_id for r in res["recommendations"]} <= {"I005", "I006", "I007"}
    res2 = eng.recommend("U001", top_k=10, exclude_seen=True)
    seen = demo_data.seen_items("U001")
    assert not (seen & {r.item_id for r in res2["recommendations"]})


def test_unseen_zero_popularity_items_excluded_from_popularity_source(demo_data):
    eng = RecommendationEngine(demo_data.config)
    eng.data = demo_data
    eng.build()
    pop = eng.recommenders["popularity"]
    ids = {r.item_id for r in pop.recommend("U001", top_k=24, exclude_seen=True)}
    assert "I014" not in ids  # zero-interaction catalog items are not 'popular'
