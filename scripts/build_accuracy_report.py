"""Regenerate the honest dual-metric accuracy report JSON used by the app.

Trains on the train split of the real LMArena battles, evaluates Metric A and
Metric B on the held-out split, and writes
``outputs/accuracy/routing_accuracy_report.json``.

Usage:
    python scripts/build_accuracy_report.py            # single held-out split
    python scripts/build_accuracy_report.py --with-cv  # + 5-fold grouped CV (~2 min)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.routing.evaluation import cross_validate, run_accuracy_evaluation  # noqa: E402


def main() -> None:
    t0 = time.time()
    report = run_accuracy_evaluation(save=True)
    print(f"Metric A (task-category agreement): {report.metric_a:.3f} "
          f"{'>= 0.90 TARGET MET' if report.threshold_met else '< 0.90 TARGET MISSED'}")
    print(f"Metric B (head-to-head agreement): {report.metric_b:.3f}")

    if "--with-cv" in sys.argv:
        cv = cross_validate(k_folds=5, seed=report.seed)
        report.cv_metric_a_mean = cv["metric_a_mean"]
        report.cv_metric_a_std = cv["metric_a_std"]
        report.cv_metric_b_mean = cv["metric_b_mean"]
        report.cv_metric_b_std = cv["metric_b_std"]
        report.cv_folds = cv["folds"]
        print(f"5-fold CV: A={cv['metric_a_mean']:.3f}±{cv['metric_a_std']:.3f} "
              f"B={cv['metric_b_mean']:.3f}±{cv['metric_b_std']:.3f}")

    path = report.save()
    print(f"train={report.n_train} test={report.n_test} dataset={report.dataset}")
    print(f"saved -> {path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
