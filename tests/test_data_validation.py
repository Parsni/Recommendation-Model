"""Data ingestion and validation tests against the seeded demo defects."""
import io

import pandas as pd
import pytest

from src.data.loader import load_dataset, read_csv
from src.data.schema import validate_interactions, validate_metadata
from src.features.preprocess import prepare


MAPPING = {"user_id": "user_id", "item_id": "item_id",
           "interaction_value": "interaction_value", "timestamp": "timestamp"}


def demo_paths(root):
    d = root / "data" / "synthetic_demo"
    return d / "interactions.csv", d / "users.csv", d / "items.csv"


def test_demo_validation_detects_seeded_defects(demo_raw):
    rep = demo_raw.reports[0]
    assert rep.rows_in == 154
    assert len(rep.malformed_rows) == 1          # blank user_id row dropped
    assert rep.duplicate_rows == 2               # two exact duplicates
    assert any("non-numeric" in s for s in rep.dtype_issues)
    assert rep.rows_usable == 153


def test_prepared_counts_match_known_facts(demo_data):
    assert demo_data.n_users == 30
    assert demo_data.n_items == 20               # full catalog incl. zero-interaction items
    inter = demo_data.interactions
    pairs = inter.groupby(["user_id", "item_id"]).ngroups
    assert pairs == 150                          # unique user-item pairs
    counts = inter.groupby("item_id")["user_id"].nunique()
    assert counts["I013"] == 30
    assert all(counts[f"I{n:03d}"] == 10 for n in range(1, 13))


def test_original_files_not_mutated(tmp_path):
    src = tmp_path / "inter.csv"
    df = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["a", "b"],
                       "interaction_value": [1, 2], "timestamp": ["2025-01-01", "2025-01-02"]})
    df.to_csv(src, index=False)
    before = src.read_text()
    load_dataset(src)
    assert src.read_text() == before


def test_missing_required_column_reported():
    df = pd.DataFrame({"user_id": ["u1"], "wrong_item": ["a"]})
    _, rep = validate_interactions(df, MAPPING)
    assert not rep.ok
    assert "item_id" in rep.missing_required_columns


def test_rows_with_missing_ids_dropped_and_recorded():
    df = pd.DataFrame({"user_id": ["u1", None, "u3", ""],
                       "item_id": ["a", "b", None, "d"],
                       "interaction_value": [1, 2, 3, 4],
                       "timestamp": ["2025-01-01"] * 4})
    clean, rep = validate_interactions(df, MAPPING)
    assert len(clean) == 1
    assert sorted(rep.malformed_rows) == [1, 2, 3]


def test_invalid_interaction_value_coerced_not_crash():
    df = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["a", "b"],
                       "interaction_value": ["excellent", "3"],
                       "timestamp": ["2025-01-01", "not-a-date"]})
    clean, rep = validate_interactions(df, MAPPING)
    assert pd.isna(clean["interaction_value"].iloc[0])
    assert float(clean["interaction_value"].iloc[1]) == 3.0
    assert any("timestamp" in s for s in rep.dtype_issues)


def test_empty_dataset_handled():
    df = pd.DataFrame({"user_id": [], "item_id": []})
    clean, rep = validate_interactions(df, MAPPING)
    assert rep.rows_in == 0 and rep.ok
    raw_mod = type(df)()
    from src.data.loader import RawData
    data = prepare(RawData(interactions=clean, users=None, items=None))
    assert data.n_interactions == 0


def test_unsupported_format_rejected(tmp_path):
    bad = tmp_path / "data.xlsx"
    bad.write_bytes(b"not really excel")
    with pytest.raises(ValueError):
        read_csv(bad)


def test_metadata_duplicate_ids_deduplicated():
    df = pd.DataFrame({"item_id": ["a", "a", "b"], "category": ["x", "x", "y"]})
    clean, rep = validate_metadata(df, {"item_id": "item_id", "_dataset": "items"}, "item_id")
    assert len(clean) == 2
    assert rep.duplicate_rows == 1


def test_insufficient_data_still_produces_structures():
    df = pd.DataFrame({"user_id": ["u1"], "item_id": ["a"],
                       "interaction_value": [5], "timestamp": ["2025-01-01"]})
    clean, _ = validate_interactions(df, MAPPING)
    from src.data.loader import RawData
    data = prepare(RawData(interactions=clean, users=None, items=None))
    assert data.n_users == 1 and data.n_items == 1
