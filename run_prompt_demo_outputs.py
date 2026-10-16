"""Generate real outputs for the prompt -> AI-model domain.

Runs the same code path as app_prompt.py on data/prompt_rec/*.csv and writes
tables, a figure and a timing log under outputs/. The quality_score column in
the bundled trials.csv is a deterministic placeholder pattern, NOT benchmark
data, so these numbers demonstrate the pipeline, not model quality.

Usage:  python run_prompt_demo_outputs.py
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.domain.prompt_matcher import PromptMatcher
from src.evaluation.compare import run_chronological_comparison
from src.reporting.exporter import recommendations_to_df, save_bar_chart, save_table
from src.utils.helpers import LOGS_DIR

EXAMPLE_PROMPTS = [
    "write a python script to clean and analyze survey data from a csv file",
    "create an isometric illustration of a cozy bookshop for a website banner",
    "translate this marketing email from English to Japanese and keep the tone formal",
]


def main() -> dict:
    t0 = time.perf_counter()
    matcher = PromptMatcher()
    build_ms = round((time.perf_counter() - t0) * 1000, 1)

    eng = matcher.engine
    summary = {
        "dataset": "data/prompt_rec (PLACEHOLDER trial ratings, not benchmark results)",
        "engine_build_ms": build_ms,
        "prompts": int(eng.data.n_users),
        "ai_models": int(eng.data.n_items),
        "trials": int(len(eng.data.interactions)),
        "matrix_density": round(eng.data.density, 4),
        "algorithms_enabled": list(eng.status.eligible),
        "algorithms_excluded": eng.status.excluded,
        "score_kinds": sorted({rec.score_kind
                               for pid in matcher.prompt_ids[:5]
                               for rec in eng.recommend(pid, top_k=4)["recommendations"]}),
    }

    rows, latencies = [], []
    for text in EXAMPLE_PROMPTS:
        t1 = time.perf_counter()
        res = matcher.recommend_for_prompt(text, top_k=4)
        latencies.append(round((time.perf_counter() - t1) * 1000, 1))
        df = recommendations_to_df(res["recommendations"])
        if df.empty:
            continue
        names = {mid: eng.data.item_metadata(mid).get("provider_label", mid)
                 for mid in df["item_id"]}
        df.insert(0, "query_prompt", text)
        df.insert(1, "ai_model", df["item_id"].map(names))
        df["best_neighbour_similarity"] = (round(res["matched_prompts"][0]["similarity"], 3)
                                          if res["matched_prompts"] else 0.0)
        df["notes"] = " | ".join(res["notes"])
        rows.append(df)

    if rows:
        suggestions = pd.concat(rows, ignore_index=True)
        summary["suggestions_table"] = save_table(suggestions, "prompt_ai_model_suggestions")
        summary["recommend_latency_ms_mean"] = round(sum(latencies) / len(latencies), 2)

    table, by_cat = matcher.stats_table()
    summary["model_trial_stats_table"] = save_table(
        table.reset_index().rename(columns={"item_id": "model_id", "count": "trials",
                                            "mean": "mean_quality_score"}),
        "prompt_model_trial_stats")
    if by_cat is not None:
        summary["model_by_category_table"] = save_table(
            by_cat.reset_index(), "prompt_model_mean_by_category")
        summary["chart"] = save_bar_chart(
            {k: float(v) for k, v in table["mean"].items()},
            "Mean placeholder trial score per AI model (NOT a benchmark)",
            "prompt_model_mean_scores", ylabel="mean quality_score (placeholder)")

    t2 = time.perf_counter()
    ev = run_chronological_comparison(eng.data, top_k=4, do_tuning=False)
    summary["evaluation_wall_ms"] = round((time.perf_counter() - t2) * 1000, 1)
    if "error" in ev:
        summary["evaluation"] = ev["error"]
    else:
        summary["evaluation"] = {
            "strategy": ev["strategy"],
            "evaluated_prompts": ev["n_eval_users"],
            "test_table": save_table(ev["test_table"], "prompt_eval_comparison"),
            "ndcg_by_method": {r["method"]: float(r[f"ndcg@4"])
                               for _, r in ev["test_table"].iterrows()},
            "limitations": ev["limitations"],
        }

    path = LOGS_DIR / "prompt_demo_run_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
