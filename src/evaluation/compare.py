"""Model comparison runner: evaluates every eligible algorithm plus the
hybrid against the popularity baseline on comparable held-out cases.

Only TRAIN data is ever fitted on; validation targets are used for weight
selection; test targets are evaluated once at the end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine import RecommendationEngine
from src.evaluation.metrics import (catalog_coverage, hit_rate_at_k, ndcg_at_k,
                                    precision_at_k, recall_at_k)
from src.evaluation.splits import chronological_last_out
from src.features.preprocess import PreparedData


def _targets(targets: dict) -> dict:
    """Normalise {user: item | set} to {user: set}."""
    return {u: (v if isinstance(v, set) else {v}) for u, v in targets.items()}


def evaluate_sources(engine: RecommendationEngine, targets: dict, top_k: int) -> dict:
    """Run each fitted recommender + the hybrid on the held-out targets."""
    results = {name: [] for name in engine.recommenders}
    results["hybrid"] = []
    for user, rel in _targets(targets).items():
        for name, rec in engine.recommenders.items():
            rs = rec.recommend(user, top_k=top_k, exclude_seen=True)
            results[name].append([r.item_id for r in rs])
        hybrid_rs = engine.ranker.rank(user, top_k=top_k, exclude_seen=True)
        results["hybrid"].append([r.item_id for r in hybrid_rs])
    return results


def summarise(results: dict, targets: dict, top_k: int, catalog_size: int) -> pd.DataFrame:
    rel = _targets(targets)
    rows = []
    for name, lists in results.items():
        users = list(rel)
        prec = [precision_at_k(l, rel[u], top_k) for l, u in zip(lists, users)]
        rec = [recall_at_k(l, rel[u], top_k) if rel[u] else 0.0 for l, u in zip(lists, users)]
        hits = [hit_rate_at_k(l, rel[u], top_k) for l, u in zip(lists, users)]
        ndcg = [ndcg_at_k(l, rel[u], top_k) for l, u in zip(lists, users)]
        n = max(len(lists), 1)
        hr = sum(hits) / n
        se = float(np.std(hits, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append({
            "method": name, "n_eval_users": n,
            f"precision@{top_k}": round(sum(prec) / n, 4),
            f"recall@{top_k}": round(sum(rec) / n, 4),
            f"hit_rate@{top_k}": round(hr, 4),
            f"hit_rate_se": round(se, 4),
            f"ndcg@{top_k}": round(sum(ndcg) / n, 4),
            f"coverage@{top_k}": round(catalog_coverage(lists, catalog_size), 4),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(f"ndcg@{top_k}", ascending=False).reset_index(drop=True)


def tune_weights(engine: RecommendationEngine, val_targets: dict, top_k: int,
                 candidates: list[dict] | None = None) -> dict:
    """Evaluate a small set of hybrid weight configurations on VALIDATION data
    only. Returns best weights + a comparison table (no test access)."""
    names = sorted(engine.recommenders)
    if candidates is None:
        equal = {n: 1.0 / len(names) for n in names}
        pop_heavy = {n: (0.6 if n == "popularity" else 0.4 / max(len(names) - 1, 1))
                     for n in names}
        no_pop = ({n: 1.0 / max(len(names) - 1, 1) for n in names if n != "popularity"}
                  if len(names) > 1 else equal)
        candidates = [equal, pop_heavy, no_pop]
    original = dict(engine.config["hybrid"]["weights"])
    rows, best, best_score = [], None, -np.inf
    for w in candidates:
        engine.config["hybrid"]["weights"] = w
        engine.ranker.weights = w
        res = evaluate_sources(engine, val_targets, top_k)
        df = summarise(res, val_targets, top_k, engine.data.n_items)
        score = float(df.loc[df["method"] == "hybrid", f"ndcg@{top_k}"].iloc[0])
        rows.append({"weights": {k: round(v, 3) for k, v in w.items()}, "hybrid_ndcg": round(score, 4)})
        if score > best_score:
            best, best_score = w, score
    engine.config["hybrid"]["weights"] = original
    engine.ranker.weights = original
    return {"best_weights": best, "best_val_ndcg": round(float(best_score), 4),
            "trials": rows,
            "note": ("Weights selected on validation split only; test outcomes were "
                     "not consulted. Equal weights are a starting configuration, not "
                     "an optimum claim.")}


def run_chronological_comparison(data: PreparedData, top_k: int | None = None,
                                 do_tuning: bool = True) -> dict:
    """Full defensible evaluation pipeline on a prepared dataset."""
    if not data.has_timestamps:
        return {"error": "chronological evaluation needs a timestamp column; "
                         "supply timestamps or use user_holdout instead"}
    min_train = int(data.config.get("evaluation", {}).get("min_train_per_user", 3))
    top_k = top_k or int(data.config.get("evaluation", {}).get("top_k", 5))
    split = chronological_last_out(data, min_train=min_train)
    train = split["train"]
    engine = RecommendationEngine(data.config)
    engine.data = train
    engine.build()
    tuning = None
    if do_tuning and split["val"] and len(engine.recommenders) > 1:
        tuning = tune_weights(engine, split["val"], top_k)
        engine.config["hybrid"]["weights"] = tuning["best_weights"]
        engine.ranker.weights = tuning["best_weights"]
    test_results = evaluate_sources(engine, split["test"], top_k)
    table = summarise(test_results, split["test"], top_k, data.n_items)
    baseline = table[table["method"] == "popularity"]
    return {
        "strategy": (f"per-user chronological last-out: train excludes each user's two "
                     f"most recent interactions; validation = second-last, "
                     f"test = last; only users with >= {min_train + 2} interactions; "
                     f"K={top_k}"),
        "n_eval_users": int(len(split["test"])),
        "test_table": table,
        "weight_tuning": tuning,
        "beats_baseline": bool(len(table) > 1 and
                               table.iloc[0][f"ndcg@{top_k}"] >
                               float(baseline[f"ndcg@{top_k}"].iloc[0]) if len(baseline) else False),
        "limitations": ("Single-split evaluation on one dataset; hit-rate standard error "
                        "shown but no significance testing across methods; "
                        "synthetic-data results do not imply real-world effectiveness."),
    }
