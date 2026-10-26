"""Offline grid search over ensemble hyperparameters.

Objective: maximize held-out Metric A (task-category agreement, target >= 0.90)
without collapsing Metric B (head-to-head preference agreement).

Usage:
    python scripts/tune_accuracy.py            # full grid (~10-15 min)
    python scripts/tune_accuracy.py --quick    # reduced grid (~3 min)
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.routing.evaluation import (  # noqa: E402
    METRIC_A_THRESHOLD,
    empirical_best_by_domain,
    evaluate_split,
    group_split_by_prompt,
    load_anchors,
    load_battles,
    win_rate_tables,
)
from src.routing.engine import LLMSelectionRoutingEngine  # noqa: E402
from src.routing.inference.classifier_ensemble import LogisticRegressionEnsemble  # noqa: E402
from src.utils.helpers import PROJECT_ROOT  # noqa: E402

OUT_PATH = PROJECT_ROOT / "outputs" / "accuracy" / "tuning_results.json"


def run_grid(quick: bool = False) -> None:
    battles = load_battles()
    anchors = load_anchors()
    train_df, test_df = group_split_by_prompt(battles, test_frac=0.2, seed=42)

    c_grid = [0.1, 0.2, 0.3, 0.5]
    blend_grid = [0.85, 0.90, 0.95]
    scale_grid = [4.0, 5.0, 6.0]

    combos = list(itertools.product(c_grid, blend_grid, scale_grid))
    print(f"Grid: {len(combos)} combos | train={len(train_df)} test={len(test_df)}")

    results = []
    t_all = time.time()
    for i, (c_val, blend, scale) in enumerate(combos, 1):
        t0 = time.time()
        engine = LLMSelectionRoutingEngine()  # default telemetry repo; eval never persists
        engine.classifier_ensemble = LogisticRegressionEnsemble(
            model_names=engine.model_names,
            regularization_c=c_val,
            bt_blend=blend,
            domain_affinity_scale=scale,
        )
        engine.train(train_df, anchors)

        model_names = sorted(engine.model_names)
        domain_wr, _, _, _ = win_rate_tables(train_df, model_names, min_games=8.0)
        empirical_best = empirical_best_by_domain(domain_wr)
        res = evaluate_split(engine, test_df, empirical_best, weights=engine.optimizer.weights)

        per_domain = {
            d: round(v, 3) for d, v in res["per_domain_a"].items()
            if v == v and res["domain_test_counts"].get(d)  # filter NaN
        }
        entry = {
            "regularization_c": c_val,
            "bt_blend": blend,
            "domain_affinity_scale": scale,
            "metric_a": res["metric_a"],
            "metric_b": res["metric_b"],
            "per_domain_a": per_domain,
            "elapsed_s": round(time.time() - t0, 1),
        }
        results.append(entry)
        flag = " <= A>=0.90!" if entry["metric_a"] >= METRIC_A_THRESHOLD else ""
        print(
            f"[{i}/{len(combos)}] C={c_val:<5} blend={blend:<5} scale={scale:<5} "
            f"A={entry['metric_a']:.3f} B={entry['metric_b']:.3f} ({entry['elapsed_s']}s){flag}"
        )

    results.sort(key=lambda r: (r["metric_a"], r["metric_b"]), reverse=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "results": results,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "target_metric_a": METRIC_A_THRESHOLD,
                "total_elapsed_s": round(time.time() - t_all, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTop 5 by Metric A:")
    for r in results[:5]:
        print(
            f"  C={r['regularization_c']:<5} blend={r['bt_blend']:<5} scale={r['domain_affinity_scale']:<5} "
            f"A={r['metric_a']:.3f} B={r['metric_b']:.3f}  domains={r['per_domain_a']}"
        )
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    run_grid(quick="--quick" in sys.argv)
