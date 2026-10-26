"""Honest dual-metric accuracy evaluation for the routing engine.

Metric A - Task-Category Agreement:
    The router's top-1 pick equals the *empirical best model* for the prompt's
    task category (domain). The empirical best is derived strictly from real
    LMArena battle win-rates observed on the training split (Beta(1,1) prior,
    shrunk toward the model's global win-rate for thin cells).

Metric B - Head-to-Head Preference Agreement:
    On a held-out battle between model X and Y (X won), the router's composite
    score ranks X above Y.

Both metrics are computed on held-out prompts only (group split by prompt text
so no prompt leaks between train and test). No metric is ever fabricated: if a
threshold is not met, ``threshold_met`` is False and the raw numbers stand.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.routing.domains import DOMAINS, detect_domain
from src.routing.engine import LLMSelectionRoutingEngine
from src.routing.scoring import RoutingPolicyWeights
from src.utils.helpers import PROJECT_ROOT

DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "accuracy" / "routing_accuracy_report.json"
ARENA_DATASET_PATH = PROJECT_ROOT / "data" / "routing" / "arena_battles_real.csv"
LEGACY_DATASET_PATH = PROJECT_ROOT / "data" / "routing" / "pairwise_comparisons.csv"
ANCHORS_PATH = PROJECT_ROOT / "data" / "routing" / "anchor_prompts.csv"

METRIC_A_THRESHOLD = 0.90


def resolve_dataset_path(config: Optional[dict] = None) -> Path:
    """Prefer the real Arena battles CSV; fall back to the legacy 24-row set."""
    rel = (config or {}).get("inference", {}).get("training_data")
    candidate = (PROJECT_ROOT / rel) if rel else ARENA_DATASET_PATH
    if candidate.exists():
        return candidate
    return LEGACY_DATASET_PATH


def load_battles(path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or resolve_dataset_path())
    df["winner"] = df["winner"].astype(str).str.strip().str.lower()
    return df


def load_anchors() -> pd.DataFrame:
    return pd.read_csv(ANCHORS_PATH)


def group_split_by_prompt(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group split: every row of a prompt lands entirely on one side."""
    prompts = df["prompt"].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(prompts)
    n_test = max(1, int(round(len(prompts) * test_frac)))
    test_prompts = set(prompts[:n_test])
    mask = df["prompt"].isin(test_prompts)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def _shrunk_wr(wins: float, games: float, prior_wr: float, min_games: float) -> float:
    """Beta(1,1) win-rate, shrunk toward ``prior_wr`` when games < min_games."""
    wr = (wins + 1.0) / (games + 2.0)
    if games < min_games:
        lam = (games / min_games) if games > 0.0 else 0.0
        wr = lam * wr + (1.0 - lam) * prior_wr
    return float(np.clip(wr, 1e-4, 1.0 - 1e-4))


def win_rate_tables(
    df: pd.DataFrame,
    model_names: List[str],
    min_games: float = 8.0,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]], Dict[str, float], Dict[str, int]]:
    """Return (domain_wr, domain_games, overall_wr, overall_games) for models in ``model_names``."""
    overall_wins = {m: 0.0 for m in model_names}
    overall_games = {m: 0 for m in model_names}
    dom_wins = {m: {d: 0.0 for d in DOMAINS} for m in model_names}
    dom_games = {m: {d: 0 for d in DOMAINS} for m in model_names}

    for _, row in df.iterrows():
        m_a = str(row["model_a"]).strip()
        m_b = str(row["model_b"]).strip()
        if m_a not in overall_games or m_b not in overall_games:
            continue
        winner = m_a if str(row["winner"]).lower() in ("model_a", m_a) else m_b
        domain = detect_domain(str(row["prompt"]))
        for m in (m_a, m_b):
            overall_games[m] += 1
            dom_games[m][domain] += 1
        overall_wins[winner] += 1.0
        dom_wins[winner][domain] += 1.0

    overall_wr = {
        m: _shrunk_wr(overall_wins[m], overall_games[m], 0.5, min_games)
        for m in model_names
    }
    domain_wr: Dict[str, Dict[str, float]] = {}
    domain_game_counts: Dict[str, Dict[str, int]] = {}
    for d in DOMAINS:
        domain_wr[d] = {
            m: _shrunk_wr(dom_wins[m][d], dom_games[m][d], overall_wr[m], min_games)
            for m in model_names
        }
        domain_game_counts[d] = {m: dom_games[m][d] for m in model_names}
    return domain_wr, domain_game_counts, overall_wr, overall_games


def empirical_best_by_domain(domain_wr: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Strongest model per domain by shrunk training-split win-rate."""
    return {
        d: max(wr_table.items(), key=lambda kv: kv[1])[0]
        for d, wr_table in domain_wr.items()
    }


@dataclass
class RoutingAccuracyReport:
    """Honest accuracy numbers for the routing engine (held-out evaluation)."""
    metric_a: float
    metric_b: float
    n_train: int
    n_test: int
    threshold: float
    threshold_met: bool
    model_order: List[str]
    empirical_best: Dict[str, str]
    per_domain_a: Dict[str, float]
    domain_test_counts: Dict[str, int]
    confusion: List[List[int]]
    domain_win_rates: Dict[str, Dict[str, float]]
    domain_game_counts: Dict[str, Dict[str, int]]
    overall_win_rates: Dict[str, float]
    overall_game_counts: Dict[str, int]
    weights: Dict[str, float]
    dataset: str
    seed: int
    metric_a_label: str = "Task-Category Agreement (held-out)"
    metric_b_label: str = "Head-to-Head Preference Agreement (held-out)"
    cv_metric_a_mean: Optional[float] = None
    cv_metric_a_std: Optional[float] = None
    cv_metric_b_mean: Optional[float] = None
    cv_metric_b_std: Optional[float] = None
    cv_folds: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path = DEFAULT_REPORT_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def evaluate_split(
    engine: LLMSelectionRoutingEngine,
    test_df: pd.DataFrame,
    empirical_best: Dict[str, str],
    weights: Optional[RoutingPolicyWeights] = None,
) -> Dict[str, object]:
    """Run Metric A + Metric B over a held-out battle frame using ``engine``."""
    model_order = sorted(engine.model_names)
    model_idx = {m: i for i, m in enumerate(model_order)}

    confusion = [[0 for _ in model_order] for _ in model_order]
    per_domain_correct: Dict[str, int] = {d: 0 for d in DOMAINS}
    per_domain_total: Dict[str, int] = {d: 0 for d in DOMAINS}
    a_correct = 0.0
    b_total = 0.0
    b_correct = 0.0

    for _, row in test_df.iterrows():
        prompt = str(row["prompt"])
        domain = detect_domain(prompt)
        expected = empirical_best.get(domain, max(engine.classifier_ensemble.empirical_win_rates,
                                                  key=engine.classifier_ensemble.empirical_win_rates.get))

        decision = engine.route(
            {"user_id": "usr_eval", "prompt": prompt},
            override_weights=weights,
            persist_telemetry=False,
        )

        predicted = decision.selected_model
        per_domain_total[domain] = per_domain_total.get(domain, 0) + 1
        if predicted in model_idx and expected in model_idx:
            confusion[model_idx[expected]][model_idx[predicted]] += 1
        if predicted == expected:
            a_correct += 1.0
            per_domain_correct[domain] = per_domain_correct.get(domain, 0) + 1

        # Metric B: composite ranking of the real battle winner vs loser
        m_a = str(row["model_a"]).strip()
        m_b = str(row["model_b"]).strip()
        winner = m_a if str(row["winner"]).lower() in ("model_a", m_a) else m_b
        loser = m_b if winner == m_a else m_a
        scores = {r.model_id: r.composite_score for r in decision.ranked_candidates}
        if winner in scores and loser in scores:
            b_total += 1.0
            diff = scores[winner] - scores[loser]
            if diff > 1e-9:
                b_correct += 1.0
            elif abs(diff) <= 1e-9:
                b_correct += 0.5

    metric_a = a_correct / len(test_df) if len(test_df) else 0.0
    metric_b = (b_correct / b_total) if b_total else 0.0
    per_domain_a = {
        d: (per_domain_correct[d] / per_domain_total[d]) if per_domain_total[d] else float("nan")
        for d in DOMAINS
    }
    return {
        "metric_a": round(metric_a, 4),
        "metric_b": round(metric_b, 4),
        "confusion": confusion,
        "per_domain_a": per_domain_a,
        "domain_test_counts": {d: per_domain_total[d] for d in DOMAINS if per_domain_total[d]},
    }


def run_accuracy_evaluation(
    battles: Optional[pd.DataFrame] = None,
    anchors: Optional[pd.DataFrame] = None,
    test_frac: float = 0.2,
    seed: int = 42,
    min_games: float = 8.0,
    weights: Optional[RoutingPolicyWeights] = None,
    config_path: Optional[Path] = None,
    db_url: Optional[str] = None,
    save: bool = False,
    dataset_label: Optional[str] = None,
) -> RoutingAccuracyReport:
    """Train on the train split, evaluate Metric A/B on the held-out split."""
    battles = battles if battles is not None else load_battles()
    anchors = anchors if anchors is not None else load_anchors()

    train_df, test_df = group_split_by_prompt(battles, test_frac=test_frac, seed=seed)

    engine = LLMSelectionRoutingEngine(config_path=config_path, db_url=db_url)
    engine.train(train_df, anchors)

    model_names = sorted(engine.model_names)
    domain_wr, domain_games, overall_wr, overall_games = win_rate_tables(
        train_df, model_names, min_games=min_games
    )
    empirical_best = empirical_best_by_domain(domain_wr)

    active_weights = weights or engine.optimizer.weights
    split_result = evaluate_split(engine, test_df, empirical_best, weights=active_weights)

    report = RoutingAccuracyReport(
        metric_a=float(split_result["metric_a"]),
        metric_b=float(split_result["metric_b"]),
        n_train=len(train_df),
        n_test=len(test_df),
        threshold=METRIC_A_THRESHOLD,
        threshold_met=float(split_result["metric_a"]) >= METRIC_A_THRESHOLD,
        model_order=model_names,
        empirical_best=empirical_best,
        per_domain_a={k: (None if isinstance(v, float) and math.isnan(v) else v)
                      for k, v in split_result["per_domain_a"].items()},
        domain_test_counts=split_result["domain_test_counts"],
        confusion=split_result["confusion"],
        domain_win_rates=domain_wr,
        domain_game_counts=domain_games,
        overall_win_rates=overall_wr,
        overall_game_counts=overall_games,
        weights={
            "w_success": active_weights.w_success,
            "w_similarity": active_weights.w_similarity,
            "w_cost": active_weights.w_cost,
            "w_latency": active_weights.w_latency,
        },
        dataset=dataset_label or str(resolve_dataset_path().name),
        seed=seed,
    )
    if save:
        report.save()
    return report


def cross_validate(
    battles: Optional[pd.DataFrame] = None,
    anchors: Optional[pd.DataFrame] = None,
    k_folds: int = 5,
    seed: int = 42,
    min_games: float = 8.0,
) -> Dict[str, float]:
    """Grouped k-fold CV over prompts; returns mean/std of Metric A and B."""
    battles = battles if battles is not None else load_battles()
    anchors = anchors if anchors is not None else load_anchors()

    prompts = battles["prompt"].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(prompts)
    folds = np.array_split(prompts, k_folds)

    a_scores: List[float] = []
    b_scores: List[float] = []
    for i, fold_prompts in enumerate(folds):
        test_set = set(fold_prompts)
        test_df = battles[battles["prompt"].isin(test_set)].reset_index(drop=True)
        train_df = battles[~battles["prompt"].isin(test_set)].reset_index(drop=True)

        engine = LLMSelectionRoutingEngine()  # default repo; CV never persists telemetry
        engine.train(train_df, anchors)
        model_names = sorted(engine.model_names)
        domain_wr, _, _, _ = win_rate_tables(train_df, model_names, min_games=min_games)
        empirical_best = empirical_best_by_domain(domain_wr)

        res = evaluate_split(engine, test_df, empirical_best, weights=engine.optimizer.weights)
        a_scores.append(float(res["metric_a"]))
        b_scores.append(float(res["metric_b"]))

    return {
        "metric_a_mean": round(float(np.mean(a_scores)), 4),
        "metric_a_std": round(float(np.std(a_scores)), 4),
        "metric_b_mean": round(float(np.mean(b_scores)), 4),
        "metric_b_std": round(float(np.std(b_scores)), 4),
        "folds": k_folds,
        "seed": seed,
    }


def overall_battle_win_rates(df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
    """Appearance-based real win-rate per model: {model: {"win_rate", "games"}}."""
    df = df if df is not None else load_battles()
    wins: Dict[str, float] = {}
    games: Dict[str, int] = {}
    for _, row in df.iterrows():
        m_a = str(row["model_a"]).strip()
        m_b = str(row["model_b"]).strip()
        winner = m_a if str(row["winner"]).lower() in ("model_a", m_a) else m_b
        for m in (m_a, m_b):
            games[m] = games.get(m, 0) + 1
            wins[m] = wins.get(m, 0.0) + (1.0 if m == winner else 0.0)
    return {
        m: {"win_rate": round(wins[m] / games[m], 4), "games": games[m]}
        for m in sorted(games)
    }
