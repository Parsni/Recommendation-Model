"""Smoke tests for the reworked app_router.py via streamlit AppTest.

The engine trains once on the real Arena battles (module-scoped fixture) and
subsequent tests reuse the same AppTest state.
"""
from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from src.routing import charts as ch
from src.routing.evaluation import RoutingAccuracyReport
from src.routing.scoring import RoutingPolicyWeights
from src.utils.helpers import PROJECT_ROOT


def test_all_chart_functions_render():
    report_path = PROJECT_ROOT / "outputs" / "accuracy" / "routing_accuracy_report.json"
    if not report_path.exists():
        pytest.skip("accuracy report not generated")
    report = RoutingAccuracyReport(**json.loads(report_path.read_text(encoding="utf-8")))

    class FakeProf:
        p_success = 0.7
        confidence_interval = (0.6, 0.8)

        def __init__(self, mid):
            self.model_id = mid

    class FakeRow:
        def __init__(self, mid, score, i):
            self.model_id = mid
            self.composite_score = score
            self.rank = i + 1
            self.success_prob = 0.6
            self.similarity = 0.4
            self.raw_cost_usd = 0.001 * (i + 1)
            self.raw_latency_ms = 200 + 50 * i
            self.cost_efficiency = 0.8
            self.latency_efficiency = 0.7

    probs = {m: FakeProf(m) for m in report.model_order}
    rows = [FakeRow(m, 0.7 - 0.05 * i, i) for i, m in enumerate(report.model_order)]
    affinities = {m: {"math": 0.5, "code": -0.2, "creative": 0.1, "general": 0.0} for m in report.model_order}
    weights = RoutingPolicyWeights(w_success=0.7, w_similarity=0.15, w_cost=0.08, w_latency=0.07)

    for fig in (
        ch.chart_success_bars(probs, report.model_order[0]),
        ch.chart_composite_bars(rows, report.model_order[0]),
        ch.chart_cost_latency_scatter(rows, report.model_order[0]),
        ch.chart_confusion_matrix(report.confusion, report.model_order),
        ch.chart_domain_accuracy(report.per_domain_a, report.domain_test_counts),
        ch.chart_win_rate_heatmap(report.domain_win_rates, report.model_order),
        ch.chart_overall_win_rates({
            m: {"win_rate": report.overall_win_rates[m], "games": report.overall_game_counts[m]}
            for m in report.model_order
        }),
        ch.chart_affinity_heatmap(affinities, report.model_order, highlight_domain="math"),
        ch.chart_contribution_breakdown(rows, weights, report.model_order[0]),
    ):
        assert fig is not None


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file("app_router.py", default_timeout=240)
    at.run()
    return at


def test_router_launches_without_exceptions(app):
    assert not app.exception, f"app raised: {[e.value for e in app.exception]}"


def test_prompt_entry_and_pipeline_run(app):
    at = app
    at.text_area[0].set_value(
        "Write a high-performance Python script to sort 10 million records with multiprocessing."
    )
    launch = next(b for b in at.button if b.label == "Launch Analysis Pipeline")
    launch.set_value(True).run()
    assert not app.exception, f"pipeline raised: {[e.value for e in app.exception]}"

    md = " ".join(str(m.value) for m in at.markdown)
    assert "Optimal Chatbot" in md, "winner reveal missing"
    assert "Honest Dual Metrics" in md, "accuracy section missing"
    assert "Decision Analytics" in md, "charts section missing"
    assert "Candidate Model Comparison Deck" in md, "comparison deck missing"
    assert "Ask the Router" in md, "chat section missing"
    # persistent pipeline trail remains visible after the animation
    assert "Bounded Linear Min-Max Composite Scoring Formula" in md, "pipeline trail not persisted"


def test_continue_to_feedback_navigates(app):
    at = app
    nav = next(b for b in at.button if b.label == "Continue to Feedback →")
    nav.set_value(True).run()
    assert not at.exception, f"feedback view raised: {[e.value for e in at.exception]}"
    md = " ".join(str(m.value) for m in at.markdown)
    assert "Ground Truth Feedback" in md
    assert "Decision under review" in md
    # navigate back
    back = next(b for b in at.button if b.label == "← Back to Results (skip)")
    back.set_value(True).run()
    assert not at.exception
    md2 = " ".join(str(m.value) for m in at.markdown)
    assert "Optimal Chatbot" in md2


# -----------------------------------------------------------------------------
# EXPERIENCE-LEVEL PANELS (Simple / Standard / Expert)
# These reuse the module-scoped app fixture (pipeline already evaluated).
# -----------------------------------------------------------------------------
def _mode_radio(at):
    return next(r for r in at.radio if r.key == "exp_mode_radio")


def test_sidebar_experience_level_exists(app):
    labels = {r.key for r in app.radio}
    assert "exp_mode_radio" in labels, "experience-level radio missing from sidebar"
    radio = _mode_radio(app)
    assert set(radio.options) == {"👶 Simple", "Standard", "🔧 Expert"}


def test_standard_mode_shows_under_hood_and_plain_card(app):
    at = app
    md = " ".join(str(m.value) for m in at.markdown)
    assert "In everyday terms" in md, "layman plain-summary card missing in Standard mode"
    assert "Under the Hood" in md, "under-the-hood section missing in Standard mode"
    assert "What-If Sandbox" in md, "sandbox missing in Standard mode"
    assert "Ensemble Internals" in md, "ensemble internals heading missing"
    # ensemble internals dataframe exposes the latent utility column
    df_cols = []
    for df in at.dataframe:
        try:
            df_cols.extend([str(c) for c in df.value.columns])
        except Exception:
            pass
    assert any("Latent utility" in c for c in df_cols), f"latent-utility column missing: {df_cols}"
    # sandbox sliders present
    slider_keys = {s.key for s in at.slider}
    assert {"sb_ws", "sb_wsim", "sb_wc", "sb_wl", "sb_crit"} <= slider_keys
    # hyperparameters panel is Expert-only
    assert "Hyperparameters & Provenance" not in md


def test_expert_mode_exposes_hyperparams_and_export(app):
    at = app
    _mode_radio(at).set_value("🔧 Expert").run()
    assert not at.exception, f"expert mode raised: {[e.value for e in at.exception]}"
    md = " ".join(str(m.value) for m in at.markdown)
    assert "Under the Hood" in md
    assert "Hyperparameters & Provenance" in md, "expert hyperparameters panel missing"
    assert "Decision Export" in md, "decision export panel missing"
    assert "Telemetry" in md, "telemetry panel missing"
    # st.download_button is not exposed as an AppTest widget; verify the
    # API-style payload preview (st.code) rendered the full decision payload
    code_texts = " ".join(str(getattr(c, "value", "")) for c in at.code)
    assert "decision_id" in code_texts and "selected_model" in code_texts, \
        "decision export payload preview missing"


def test_sandbox_cost_only_weights_flip_leader(app):
    at = app
    at.slider(key="sb_ws").set_value(0.0)
    at.slider(key="sb_wsim").set_value(0.0)
    at.slider(key="sb_wc").set_value(1.0)
    at.slider(key="sb_wl").set_value(0.0)
    at.run()
    assert not at.exception, f"sandbox re-rank raised: {[e.value for e in at.exception]}"
    msgs = " ".join(str(m.value) for m in at.success) + " " + " ".join(str(m.value) for m in at.info)
    assert "under your sandbox weights" in msgs.lower() or "leader" in msgs.lower(), \
        f"sandbox leader banner missing: {msgs[:200]}"
    # delta-rank table rendered
    found_delta = False
    for df in at.dataframe:
        try:
            if "Δ rank" in [str(c) for c in df.value.columns]:
                found_delta = True
                break
        except Exception:
            continue
    assert found_delta, "sandbox delta-rank table missing"


def test_simple_mode_shows_plain_card_hides_under_hood(app):
    at = app
    _mode_radio(at).set_value("👶 Simple").run()
    assert not at.exception, f"simple mode raised: {[e.value for e in at.exception]}"
    md = " ".join(str(m.value) for m in at.markdown)
    assert "In everyday terms" in md, "plain-summary card should stay visible in Simple mode"
    assert "Under the Hood" not in md, "under-the-hood should be hidden in Simple mode"
    assert "What-If Sandbox" not in md, "sandbox should be hidden in Simple mode"
    slider_keys = {s.key for s in at.slider}
    assert "sb_ws" not in slider_keys, "sandbox sliders should be gone in Simple mode"
    # restore Standard mode for any subsequent runs
    _mode_radio(at).set_value("Standard").run()
    assert not at.exception
