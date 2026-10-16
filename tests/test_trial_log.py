"""Tests for the append-only personal trial logger (runs in tmp_path only)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.domain import trial_log


def test_first_trial_creates_both_files(tmp_path):
    row = trial_log.append_trial(tmp_path, "summarize this contract in five bullets",
                                 "research_summary", "M_claude", 4, date="2026-10-01")
    assert row["prompt_id"] == "MY001"
    trials = pd.read_csv(tmp_path / "my_trials.csv")
    prompts = pd.read_csv(tmp_path / "my_prompts.csv")
    assert list(trials.columns) == trial_log.TRIAL_COLUMNS
    assert trials.iloc[0].to_dict() == {"prompt_id": "MY001", "model_id": "M_claude",
                                        "quality_score": 4, "timestamp": "2026-10-01"}
    assert prompts.loc[0, "task_category"] == "research_summary"


def test_same_prompt_reuses_id_and_forms_one_basket(tmp_path):
    text = "write a python function to read a csv"
    a = trial_log.append_trial(tmp_path, text, "coding", "M_gpt", 3)
    b = trial_log.append_trial(tmp_path, "  " + text + " ", "coding", "M_claude", 5)
    c = trial_log.append_trial(tmp_path, "another prompt entirely", "coding", "M_gpt", 4)
    assert a["prompt_id"] == b["prompt_id"] != c["prompt_id"]
    trials = pd.read_csv(tmp_path / "my_trials.csv")
    assert len(trials) == 3
    assert len(pd.read_csv(tmp_path / "my_prompts.csv")) == 2
    assert c["prompt_id"] == "MY002"


def test_ids_never_reuse_after_deletions_of_higher_numbers(tmp_path):
    for i, text in enumerate(["p one", "p two", "p three"]):
        trial_log.append_trial(tmp_path, text, "coding", "M_gpt", 3 + (i % 3))
    kept = pd.read_csv(tmp_path / "my_prompts.csv")
    kept[kept["prompt_id"] != "MY003"].to_csv(tmp_path / "my_prompts.csv", index=False)
    nxt = trial_log.next_prompt_id(tmp_path)
    assert nxt == "MY004", "must not collide with the removed MY003 row's replacements"


def test_invalid_inputs_rejected(tmp_path):
    with pytest.raises(ValueError):
        trial_log.append_trial(tmp_path, "   ", "coding", "M_gpt", 4)
    with pytest.raises(ValueError):
        trial_log.append_trial(tmp_path, "ok prompt", "coding", "M_gpt", 7)


def test_logged_trials_view(tmp_path):
    trial_log.append_trial(tmp_path, "compare two spreadsheets", "data_analysis",
                           "M_gpt", 2, date="2026-09-30")
    trial_log.append_trial(tmp_path, "compare two spreadsheets", "data_analysis",
                           "M_deepseek", 5, date="2026-09-30")
    view = trial_log.logged_trials(tmp_path)
    assert len(view) == 2
    assert set(view["prompt_id"]) == {"MY001"}
    assert view["prompt_text"].nunique() == 1
    assert list(view["quality_score"]) == [2, 5]


def test_corrupt_existing_file_reports_the_column(tmp_path):
    (tmp_path / "my_trials.csv").write_text("prompt_id,model_id\nMY001,M_gpt\n",
                                            encoding="utf-8")
    with pytest.raises(ValueError, match="quality_score"):
        trial_log.logged_trials(tmp_path)
