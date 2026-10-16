"""Evaluation metric arithmetic (hand-checkable), split integrity, leakage
checks, and reproducibility."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (precision_at_k, recall_at_k, hit_rate_at_k,
                                    ndcg_at_k, catalog_coverage, shannon_entropy)
from src.evaluation.splits import chronological_last_out
from src.evaluation.compare import run_chronological_comparison


# ---------------- metric arithmetic vs hand-computed values ----------------
def test_precision_recall_hit_known_values():
    ranked = ["a", "x", "c", "y", "z"]
    rel = {"a", "c"}
    assert precision_at_k(ranked, rel, 5) == pytest.approx(0.4)
    assert precision_at_k(ranked, rel, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, rel, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, rel, 5) == pytest.approx(1.0)
    assert hit_rate_at_k(ranked, rel, 1) == 1.0
    assert hit_rate_at_k(["x"], rel, 1) == 0.0


def test_ndcg_known_values_binary():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    # [b(0 gain), a(hit)] -> DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == pytest.approx(0.0)


def test_ndcg_graded_gains():
    gains = {"a": 3, "b": 1}
    # ranking a first: DCG = 3/log2(2) + 1/log2(3), ideal identical -> 1.0
    assert ndcg_at_k(["a", "b"], gains, 2) == pytest.approx(1.0)
    val = ndcg_at_k(["b", "a"], gains, 2)
    expected = (1 / np.log2(2) + 3 / np.log2(3)) / (3 / np.log2(2) + 1 / np.log2(3))
    assert val == pytest.approx(expected)


def test_coverage_and_entropy():
    assert catalog_coverage([["a", "b"], ["b", "c"]], 5) == pytest.approx(0.6)
    assert catalog_coverage([], 5) == 0.0
    assert shannon_entropy(["a", "a"]) == 0.0
    assert shannon_entropy(["a", "b"]) == pytest.approx(1.0)


# ---------------- split integrity & leakage ----------------
def test_chronological_split_matches_known_last_items(demo_data):
    split = chronological_last_out(demo_data, min_train=3)
    assert split["test"]["U001"] == "I001" and split["val"]["U001"] == "I002"
    assert split["test"]["U002"] == "I005" and split["val"]["U002"] == "I006"
    assert len(split["test"]) == 30


def test_no_leakage_train_precedes_heldout(demo_data):
    """Every training event for a user must be strictly at/ before that user's
    earliest held-out (validation) event - the real leakage guarantee."""
    split = chronological_last_out(demo_data, min_train=3)
    train = split["train"].interactions.dropna(subset=["timestamp"])
    for uid in split["test"]:
        user_train_ts = train.loc[train["user_id"] == uid, "timestamp"]
        if user_train_ts.empty:
            continue
        hist = demo_data.interactions[demo_data.interactions["user_id"] == uid]
        hist = hist.dropna(subset=["timestamp"]).sort_values("timestamp")
        heldout_start = hist.iloc[-2]["timestamp"]
        assert user_train_ts.max() <= heldout_start, \
            f"leakage: {uid} train contains events after validation time"


def test_train_excludes_heldout_pairs(demo_data):
    split = chronological_last_out(demo_data, min_train=3)
    train_pairs = set(zip(split["train"].interactions["user_id"],
                          split["train"].interactions["item_id"]))
    for uid, target in split["test"].items():
        if (uid, target) in train_pairs and uid != "U007":
            pytest.fail(f"held-out test pair {uid,target} present in train")


# ---------------- comparison pipeline ----------------
def test_comparison_runs_and_is_reproducible(demo_data):
    out1 = run_chronological_comparison(demo_data)
    out2 = run_chronological_comparison(demo_data)
    assert "test_table" in out1 and len(out1["test_table"]) > 1
    assert out1["n_eval_users"] == 30
    pd.testing.assert_frame_equal(out1["test_table"], out2["test_table"])
    # popularity baseline evaluated on same users, and content beats it here
    table = out1["test_table"]
    kcol = [c for c in table.columns if c.startswith("ndcg")][0]
    pop = float(table.loc[table["method"] == "popularity", kcol].iloc[0])
    content = float(table.loc[table["method"] == "content_based", kcol].iloc[0])
    assert content > pop
    assert out1["beats_baseline"] is True
