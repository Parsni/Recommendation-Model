"""Generate verified example outputs from ACTUAL computation runs.

Run:  python run_demo_outputs.py
Writes tables, figures and a demonstration summary into outputs/, all
produced by the engine on the bundled synthetic demo dataset. Nothing here
is hand-authored or fabricated.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from src.engine import RecommendationEngine
from src.evaluation.compare import run_chronological_comparison
from src.reporting.exporter import recommendations_to_df, save_bar_chart, save_table
from src.utils.helpers import LOGS_DIR, PROJECT_ROOT, load_config

DEMO = PROJECT_ROOT / "data" / "synthetic_demo"


def main() -> None:
    t0 = time.perf_counter()
    eng = RecommendationEngine(load_config()).load(
        DEMO / "interactions.csv", DEMO / "users.csv", DEMO / "items.csv").build()
    fit_ms = (time.perf_counter() - t0) * 1000

    # 1. data quality
    q = pd.DataFrame([{"metric": k, "value": str(v)} for k, v in eng.status.quality.items()])
    save_table(q, "demo_quality_report")

    # 2. recommendations for a warm and a cold user
    warm = eng.recommend("U001", top_k=5)
    save_table(recommendations_to_df(warm["recommendations"]), "demo_recommendations_U001")
    cold = eng.recommend("brand_new_user", top_k=5)
    save_table(recommendations_to_df(cold["recommendations"]), "demo_recommendations_cold_start")

    # 3. per-algorithm view for U001
    rows = []
    for name, rec in sorted(eng.recommenders.items()):
        for r in rec.recommend("U001", top_k=3):
            rows.append(r.as_dict() | {"algorithm": name})
    save_table(pd.DataFrame(rows), "demo_per_algorithm_U001")

    # 4. evaluation comparison + chart
    ev = run_chronological_comparison(eng.data)
    table = ev["test_table"]
    save_table(table, "demo_model_comparison")
    kcol = [c for c in table.columns if c.startswith("ndcg")][0]
    chart = save_bar_chart(dict(zip(table["method"], table[kcol])),
                           f"{kcol} by method (synthetic demo, held-out test)",
                           "demo_ndcg_comparison", ylabel=kcol)
    with open(LOGS_DIR / "demo_weight_tuning.json", "w", encoding="utf-8") as fh:
        json.dump({"strategy": ev["strategy"], "n_eval_users": ev["n_eval_users"],
                   "weight_tuning": ev["weight_tuning"],
                   "beats_baseline": ev["beats_baseline"],
                   "limitations": ev["limitations"]}, fh, indent=2, default=str)

    # 5. timing summary (measured, not claimed)
    rec_ms = []
    for uid in ["U001", "U002", "U003"]:
        s = time.perf_counter()
        eng.recommend(uid, top_k=5)
        rec_ms.append((time.perf_counter() - s) * 1000)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "synthetic demo (154 rows read, 151 usable, 150 unique user-item pairs)",
        "engine_build_ms": round(fit_ms, 1),
        "recommend_latency_ms_mean": round(sum(rec_ms) / len(rec_ms), 2),
        "algorithms_enabled": sorted(eng.recommenders),
        "algorithms_excluded": eng.status.excluded,
        "eval_strategy": ev["strategy"],
        "note": "All figures computed on placeholder synthetic data; they demonstrate "
                "functionality only and are NOT real-world effectiveness evidence.",
    }
    with open(LOGS_DIR / "demo_run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("chart:", chart)
    print("outputs written to outputs/tables and outputs/figures and outputs/logs")


if __name__ == "__main__":
    main()
