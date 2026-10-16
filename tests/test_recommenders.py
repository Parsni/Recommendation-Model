"""Per-algorithm unit tests with controlled mini-datasets and known outputs."""
import pandas as pd
import pytest

from src.data.loader import RawData
from src.features.preprocess import prepare
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.item_similarity import ItemSimilarityRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.association import AssociationRecommender
from src.utils.helpers import load_config


def mk(inter_rows, items=None, config=None):
    df = pd.DataFrame(inter_rows)
    if "interaction_value" in df:
        df["interaction_value"] = pd.to_numeric(df["interaction_value"], errors="coerce")
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    raw = RawData(interactions=df, users=None,
                  items=pd.DataFrame(items) if items else None)
    return prepare(raw, config or load_config())


# ---------------- popularity ----------------
def test_popularity_known_order():
    data = mk([
        {"user_id": "u1", "item_id": "a"}, {"user_id": "u2", "item_id": "a"},
        {"user_id": "u3", "item_id": "a"}, {"user_id": "u1", "item_id": "b"},
        {"user_id": "u2", "item_id": "b"}, {"user_id": "u1", "item_id": "c"},
    ])
    rec = PopularityRecommender().fit(data)
    assert [r.item_id for r in rec.recommend("u9", 3)] == ["a", "b", "c"]
    assert [r.score for r in rec.recommend("u9", 3)] == [3.0, 2.0, 1.0]


def test_popularity_demo_top5_matches_known_facts(demo_data):
    rec = PopularityRecommender(demo_data.config).fit(demo_data)
    assert [r.item_id for r in rec.recommend("COLD", 5)] == \
        ["I013", "I001", "I002", "I003", "I004"]


def test_popularity_empty_data_raises():
    data = mk([{"user_id": "x", "item_id": "y"}]).interactions.iloc[0:0]
    from src.data.loader import RawData as R
    empty = prepare(R(interactions=pd.DataFrame(columns=["user_id", "item_id"]),
                       users=None, items=None))
    pop = PopularityRecommender()
    assert not pop.check_suitability(empty).suitable
    with pytest.raises(ValueError):
        pop.fit(empty)


# ---------------- item similarity ----------------
def test_item_similarity_recommends_paired_item():
    data = mk([
        {"user_id": "u1", "item_id": "a"}, {"user_id": "u1", "item_id": "b"},
        {"user_id": "u2", "item_id": "a"}, {"user_id": "u2", "item_id": "b"},
        {"user_id": "u3", "item_id": "a"}, {"user_id": "u3", "item_id": "c"},
    ])
    rec = ItemSimilarityRecommender().fit(data)
    top = rec.recommend("u1", 3)
    assert top[0].item_id == "c"          # u1 saw a,b; c co-occurs with a
    assert 0.0 < top[0].score <= 1.0      # bounded weighted-mean similarity


def test_item_similarity_cold_user_empty():
    data = mk([{"user_id": "u1", "item_id": "a"}, {"user_id": "u2", "item_id": "a"},
               {"user_id": "u2", "item_id": "b"}])
    rec = ItemSimilarityRecommender().fit(data)
    assert rec.recommend("nobody", 3) == []


# ---------------- content based ----------------
def test_content_based_matches_shared_words():
    items = [{"item_id": "a", "description": "quiet wireless garden lamp"},
             {"item_id": "b", "description": "solar powered garden lamp outdoor"},
             {"item_id": "c", "description": "steel espresso coffee machine"}]
    data = mk([{"user_id": "u1", "item_id": "a", "interaction_value": 5}], items=items)
    rec = ContentBasedRecommender().fit(data)
    top = rec.recommend("u1", 2)
    assert top[0].item_id == "b"          # shares 'garden lamp'
    assert "garden" in top[0].explanation or "lamp" in top[0].explanation


def test_content_based_requires_metadata(demo_data):
    class NoItems:
        def __init__(self, d):
            for k, v in d.__dict__.items():
                setattr(self, k, v)
            self.items = None
    data = NoItems(demo_data)
    assert not ContentBasedRecommender().check_suitability(data).suitable


# ---------------- collaborative ----------------
CF_CONFIG = {"thresholds": {"min_users_for_cf": 3, "min_items_for_cf": 3,
                           "min_total_interactions": 6},
             "collaborative": {"n_components": 2}, "random_seed": 42,
             "preprocessing": {"deduplicate_key": ["user_id", "item_id"]},
             "column_mapping": {"items": {}, "users": {}}}


def test_collaborative_finds_block_neighbour():
    rows = []
    for u in ["u1", "u2", "u3"]:
        for i in ["a", "b", "c"]:
            if (u, i) != ("u1", "c"):
                rows.append({"user_id": u, "item_id": i, "interaction_value": 5})
    for u in ["u4", "u5", "u6"]:
        for i in ["x", "y", "z"]:
            rows.append({"user_id": u, "item_id": i, "interaction_value": 4})
    data = mk(rows, config=CF_CONFIG)
    rec = CollaborativeRecommender(CF_CONFIG).fit(data)
    top = rec.recommend("u1", 3)
    assert "c" in [r.item_id for r in top]   # block peer behaviour recovered
    assert "a" not in [r.item_id for r in top]


def test_collaborative_insufficient_data_excluded(demo_data):
    small = demo_data.interactions.iloc[:8]
    from src.data.loader import RawData
    data = prepare(RawData(interactions=small.reset_index(drop=True), users=None, items=None),
                   demo_data.config)
    assert not CollaborativeRecommender(demo_data.config).check_suitability(data).suitable


# ---------------- association ----------------
def test_association_confidence_known_value():
    rows = []
    for u in ["u1", "u2", "u3"]:
        rows += [{"user_id": u, "item_id": "a"}, {"user_id": u, "item_id": "b"}]
    rows += [{"user_id": "u4", "item_id": "a"}, {"user_id": "u4", "item_id": "c"}]
    data = mk(rows)
    rec = AssociationRecommender(data.config).fit(data)
    top = rec.recommend("u4", 2)
    assert top[0].item_id == "b" and abs(top[0].score - 0.75) < 1e-9  # 3/4 baskets with a had b
    assert "confidence" in top[0].score_kind or top[0].score_kind == "confidence"


def test_association_demo_rule_truth(demo_data):
    rec = AssociationRecommender(demo_data.config).fit(demo_data)
    i1, i2 = demo_data.index_of(item_id="I001"), demo_data.index_of(item_id="I002")
    assert rec.conf[i1, i2] == pytest.approx(1.0)
    assert rec.support[i1] == pytest.approx(10 / 30, abs=0.02)
