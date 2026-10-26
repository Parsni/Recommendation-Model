"""Tests for honest dual-metric evaluation, domain detection, and real dataset."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.routing.domains import DOMAINS, detect_domain
from src.routing.evaluation import (
    RoutingAccuracyReport,
    empirical_best_by_domain,
    group_split_by_prompt,
    load_anchors,
    load_battles,
    overall_battle_win_rates,
    resolve_dataset_path,
    run_accuracy_evaluation,
    win_rate_tables,
)

MODELS = ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", "llama-3.1-70b"]


@pytest.fixture
def tiny_battles():
    rng = np.random.RandomState(7)
    rows = []
    prompts = (
        [f"Solve the integral of x^{i} dx from 0 to {i} with full proof step {i}." for i in range(20)]
        + [f"Write a python function helper_{i} to sort a list with multiprocessing." for i in range(20)]
    )
    for i, p in enumerate(prompts):
        for _ in range(2):
            a, b = rng.choice(MODELS, 2, replace=False)
            p_a_wins = 0.7 if a == "gpt-4o" else (0.7 if b != "gpt-4o" else 0.3)
            a_won = rng.rand() < p_a_wins
            rows.append({
                "prompt": p,
                "model_a": a,
                "model_b": b,
                "winner": "model_a" if a_won else "model_b",
                "user_id": f"u{i}",
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Domain detection (shared single source of truth)
# -----------------------------------------------------------------------------
def test_detect_domain_buckets():
    assert detect_domain("Solve the integral of x^2 from 0 to 5") == "math"
    assert detect_domain("Refactor this python function to use async/await") == "code"
    assert detect_domain("Write a scifi story about a protagonist on Saturn") == "creative"
    assert detect_domain("Translate this contract from English to Japanese") == "translation"
    assert detect_domain("Summarize this 100-page regulatory compliance PDF") == "doc"
    assert detect_domain("Draft a diplomatic vendor email update") == "comm"
    assert detect_domain("What is the capital of France?") == "qa"
    assert detect_domain("purple monkey dishwasher") == "general"
    assert detect_domain("") == "general"


def test_domain_vocabulary_stable():
    assert DOMAINS == ["math", "code", "creative", "translation", "doc", "comm", "qa", "general"]


# -----------------------------------------------------------------------------
# Splitting & win-rate tables
# -----------------------------------------------------------------------------
def test_group_split_by_prompt_no_leakage(tiny_battles):
    train_df, test_df = group_split_by_prompt(tiny_battles, test_frac=0.2, seed=42)
    train_prompts = set(train_df["prompt"])
    test_prompts = set(test_df["prompt"])
    assert not (train_prompts & test_prompts), "prompt leaked between train and test"
    assert len(train_df) + len(test_df) == len(tiny_battles)
    assert len(test_prompts) >= 1


def test_win_rate_tables_and_empirical_best(tiny_battles):
    dwr, dgames, owr, ogames = win_rate_tables(tiny_battles, MODELS, min_games=8.0)
    assert set(dwr.keys()) == set(DOMAINS)
    for m in MODELS:
        assert ogames[m] > 0
        assert 0.0 < owr[m] < 1.0
    best = empirical_best_by_domain(dwr)
    assert set(best.keys()) == set(DOMAINS)
    for d in DOMAINS:
        assert best[d] in MODELS


def test_overall_battle_win_rates_counts(tiny_battles):
    stats = overall_battle_win_rates(tiny_battles)
    total_appearances = sum(s["games"] for s in stats.values())
    assert total_appearances == 2 * len(tiny_battles)
    for s in stats.values():
        assert 0.0 <= s["win_rate"] <= 1.0


# -----------------------------------------------------------------------------
# Real arena dataset (shape only — no network)
# -----------------------------------------------------------------------------
def test_real_arena_dataset_shape():
    from src.utils.helpers import PROJECT_ROOT
    path = resolve_dataset_path()
    assert "arena_battles_real.csv" in path.name, "real arena dataset should be preferred"
    df = load_battles()
    assert len(df) >= 3000
    assert set(df["winner"].unique()) <= {"model_a", "model_b"}
    cfg_models = set(json.loads(
        (PROJECT_ROOT / "config" / "routing_config.json").read_text(encoding="utf-8")
    )["models"].keys())
    used = set(df["model_a"]) | set(df["model_b"])
    assert used == cfg_models, "dataset candidates must match the configured lineup"


# -----------------------------------------------------------------------------
# End-to-end evaluation on tiny data (smoke)
# -----------------------------------------------------------------------------
def test_run_accuracy_evaluation_smoke(tiny_battles, tmp_path):
    anchors = load_anchors()
    report = run_accuracy_evaluation(
        battles=tiny_battles,
        anchors=anchors,
        test_frac=0.25,
        seed=7,
        db_url=f"sqlite:///{tmp_path}/eval.db",
    )
    assert isinstance(report, RoutingAccuracyReport)
    assert 0.0 <= report.metric_a <= 1.0
    assert 0.0 <= report.metric_b <= 1.0
    assert report.n_train + report.n_test == len(tiny_battles)
    assert report.threshold_met == (report.metric_a >= report.threshold)
    # round-trip through dict/JSON
    restored = RoutingAccuracyReport(**report.to_dict())
    assert restored.metric_a == report.metric_a
    saved = report.save(tmp_path / "report.json")
    assert json.loads(saved.read_text(encoding="utf-8"))["metric_b"] == report.metric_b


def test_precomputed_report_consistency():
    """If the app's report JSON exists, it must be self-consistent."""
    from src.utils.helpers import PROJECT_ROOT
    path = PROJECT_ROOT / "outputs" / "accuracy" / "routing_accuracy_report.json"
    if not path.exists():
        pytest.skip("precomputed report not generated yet")
    report = RoutingAccuracyReport(**json.loads(path.read_text(encoding="utf-8")))
    assert report.threshold_met == (report.metric_a >= report.threshold)
    assert report.n_test > 0
    assert set(report.model_order) == set(report.overall_win_rates.keys())
