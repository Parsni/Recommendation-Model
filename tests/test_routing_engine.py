"""Comprehensive test suite for the Multi-Criteria LLM Selection & Contextual Routing Engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.routing.sanitizer import RequestSanitizer
from src.routing.featurizer import InFlightFeaturizer
from src.routing.clustering import IntentClusterer
from src.routing.anchor_similarity import AnchorSimilarityMatrix
from src.routing.context_bridge import ContextEnrichmentBridge
from src.routing.inference.classifier_ensemble import LogisticRegressionEnsemble
from src.routing.inference.resource_regressors import DualHeadRidgeRegressorNetwork
from src.routing.scoring import MultiCriteriaOptimizer, RoutingPolicyWeights
from src.routing.reconciliation import ReconciliationEngine
from src.routing.engine import LLMSelectionRoutingEngine
from src.routing.telemetry.db_repository import TelemetryRepository
from src.utils.helpers import PROJECT_ROOT


@pytest.fixture
def sample_comparisons():
    path = PROJECT_ROOT / "data" / "routing" / "pairwise_comparisons.csv"
    return pd.read_csv(path)


@pytest.fixture
def sample_anchors():
    path = PROJECT_ROOT / "data" / "routing" / "anchor_prompts.csv"
    return pd.read_csv(path)


@pytest.fixture
def trained_engine(sample_comparisons, sample_anchors, tmp_path):
    db_file = tmp_path / "test_telemetry.db"
    engine = LLMSelectionRoutingEngine(db_url=f"sqlite:///{db_file}")
    engine.train(sample_comparisons, sample_anchors)
    return engine


# -----------------------------------------------------------------------------
# Layer 1: Ingestion & Preprocessing
# -----------------------------------------------------------------------------
def test_sanitizer_valid_payload():
    sanitizer = RequestSanitizer(min_chars=3)
    req = sanitizer.sanitize({"user_id": "usr_test", "prompt": "Write a quicksort in Python."})
    assert req.is_compliant is True
    assert req.user_id == "usr_test"
    assert "quicksort" in req.sanitized_prompt
    assert req.char_length > 0


def test_sanitizer_injection_neutralization():
    sanitizer = RequestSanitizer()
    malicious = "<script>alert('xss')</script> ![img](javascript:evil()) Hello world"
    req = sanitizer.sanitize({"user_id": "usr_x", "prompt": malicious})
    assert "<script>" not in req.sanitized_prompt
    assert "javascript:" not in req.sanitized_prompt
    assert "Hello world" in req.sanitized_prompt


def test_sanitizer_bounds_rejection():
    sanitizer = RequestSanitizer(min_chars=5, max_bytes=100)
    # Too short
    req_short = sanitizer.sanitize({"prompt": "hi"})
    assert req_short.is_compliant is False
    assert any("below minimum" in v for v in req_short.violations)

    # Exceeds byte bound
    req_huge = sanitizer.sanitize({"prompt": "A" * 150})
    assert req_huge.is_compliant is False
    assert any("exceeds maximum" in v for v in req_huge.violations)


# -----------------------------------------------------------------------------
# Layer 2: Featurization & Parallel Tracks
# -----------------------------------------------------------------------------
def test_in_flight_featurizer_indicators():
    featurizer = InFlightFeaturizer()
    code_prompt = "def quicksort(arr): return arr if len(arr) <= 1 else arr"
    feats_code = featurizer.extract(code_prompt)
    assert feats_code.code_indicator > 0.0
    assert feats_code.estimated_tokens > 5
    assert feats_code.language == "en"

    math_prompt = "Solve the integral of x^2 + 3x from 0 to 5 equals what?"
    feats_math = featurizer.extract(math_prompt)
    assert feats_math.math_indicator > 0.0
    assert feats_math.question_indicator == 1.0


def test_intent_clusterer_fit_and_predict(sample_comparisons, sample_anchors):
    texts = list(sample_comparisons["prompt"]) + list(sample_anchors["prompt_text"])
    clusterer = IntentClusterer(n_clusters=4, random_state=42)
    clusterer.fit(texts)
    assert clusterer.is_fitted is True

    c_id, c_label, soft_probs = clusterer.predict("Write a python sorting algorithm")
    assert 0 <= c_id < 4
    assert isinstance(c_label, str)
    assert len(soft_probs) == 4
    assert np.isclose(np.sum(soft_probs), 1.0, atol=1e-3)


def test_anchor_similarity_matrix(sample_anchors):
    anchor_mat = AnchorSimilarityMatrix()
    anchor_mat.build_index(sample_anchors)
    assert anchor_mat.is_indexed is True

    res = anchor_mat.compute_similarity("Write an optimized python script for analytics")
    assert 0.0 <= res.max_similarity <= 1.0
    assert res.nearest_anchor_id.startswith("ANC_")
    assert len(res.similarity_vector) == len(sample_anchors)


def test_context_enrichment_bridge(sample_comparisons, sample_anchors):
    featurizer = InFlightFeaturizer()
    clusterer = IntentClusterer(n_clusters=4).fit(list(sample_comparisons["prompt"]))
    anchor_mat = AnchorSimilarityMatrix().build_index(sample_anchors)
    bridge = ContextEnrichmentBridge(clusterer, num_anchors=8)

    prompt = "Explain quantum entanglement to a child."
    struct = featurizer.extract(prompt)
    anch = anchor_mat.compute_similarity(prompt)
    ctx = bridge.build_context(prompt, "usr_test", struct, anch)

    assert isinstance(ctx.feature_vector, np.ndarray)
    assert ctx.feature_vector.ndim == 1
    assert not np.isnan(ctx.feature_vector).any()
    assert len(ctx.feature_vector) == len(ctx.feature_names)


# -----------------------------------------------------------------------------
# Layer 3: Cognitive Inference Layer
# -----------------------------------------------------------------------------
def test_classifier_ensemble(sample_comparisons, sample_anchors):
    models = ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", "llama-3.1-70b"]
    ensemble = LogisticRegressionEnsemble(model_names=models)

    # Dummy feature matrix
    X_dummy = np.random.RandomState(42).randn(len(sample_comparisons), 20).astype(np.float32)
    ensemble.fit(sample_comparisons, X_dummy)
    assert ensemble.is_fitted is True

    # gpt-4o won all 6 battles in sample dataset
    assert ensemble.empirical_win_rates["gpt-4o"] > ensemble.empirical_win_rates["gemini-1.5-pro"]

    probs = ensemble.predict_success_probabilities(X_dummy[0])
    for m in models:
        assert 0.05 <= probs[m].p_success <= 0.98
        assert probs[m].confidence_interval[0] <= probs[m].confidence_interval[1]


def test_resource_regressors():
    specs = {
        "gpt-4o": {"cost_input_per_million": 2.5, "cost_output_per_million": 10.0, "baseline_latency_ms": 400.0, "per_token_latency_ms": 12.0},
        "llama-3.1-70b": {"cost_input_per_million": 0.35, "cost_output_per_million": 0.40, "baseline_latency_ms": 220.0, "per_token_latency_ms": 7.0},
    }
    reg = DualHeadRidgeRegressorNetwork(model_specs=specs).fit_synthetic_baselines()
    featurizer = InFlightFeaturizer()
    struct = featurizer.extract("Write a comprehensive guide to data engineering.")
    forecasts = reg.forecast(struct, cluster_id=0)

    assert forecasts["gpt-4o"].predicted_cost_usd > forecasts["llama-3.1-70b"].predicted_cost_usd
    assert forecasts["gpt-4o"].predicted_latency_ms > forecasts["llama-3.1-70b"].predicted_latency_ms


# -----------------------------------------------------------------------------
# Layer 4: Multi-Criteria Optimizer & Scoring
# -----------------------------------------------------------------------------
def test_multi_criteria_optimizer():
    weights = RoutingPolicyWeights(w_success=0.70, w_similarity=0.10, w_cost=0.10, w_latency=0.10)
    optimizer = MultiCriteriaOptimizer(default_weights=weights)

    from src.routing.inference.classifier_ensemble import ModelSuccessProbability
    from src.routing.inference.resource_regressors import ResourceForecast

    succ_probs = {
        "gpt-4o": ModelSuccessProbability("gpt-4o", 0.90, 1.2, (0.85, 0.95)),
        "llama-3.1-70b": ModelSuccessProbability("llama-3.1-70b", 0.30, -0.8, (0.25, 0.35)),
    }
    forecasts = {
        "gpt-4o": ResourceForecast("gpt-4o", 0.002, 500.0, 50, 150),
        "llama-3.1-70b": ResourceForecast("llama-3.1-70b", 0.0001, 250.0, 50, 150),
    }

    # Under quality-focused weights, gpt-4o should win
    ranked_quality = optimizer.evaluate_candidates(succ_probs, forecasts, anchor_similarity=0.5)
    assert ranked_quality[0].model_id == "gpt-4o"
    assert ranked_quality[0].rank == 1

    # Under pure cost-saving weights, llama-3.1-70b should achieve higher composite score
    cost_weights = RoutingPolicyWeights(w_success=0.0, w_similarity=0.0, w_cost=1.0, w_latency=0.0)
    ranked_cost = optimizer.evaluate_candidates(succ_probs, forecasts, anchor_similarity=0.5, override_weights=cost_weights)
    assert ranked_cost[0].model_id == "llama-3.1-70b"
    assert ranked_cost[0].cost_efficiency == 1.0


# -----------------------------------------------------------------------------
# Layer 5: Reconciliation Engine & Failsafe Gate
# -----------------------------------------------------------------------------
def test_reconciliation_failsafe_gate():
    from src.routing.scoring import CandidateScoreRow
    rec = ReconciliationEngine(critical_success_threshold=0.40, fallback_model="gpt-4o")

    # Case A: Top candidate has success_prob 0.85 >= 0.40 -> No fallback
    cand_high = [
        CandidateScoreRow("claude-3-5-sonnet", 1, 0.75, 0.85, 0.6, 0.001, 500.0, 0.8, 0.8, True),
        CandidateScoreRow("gemini-1.5-pro", 2, 0.50, 0.45, 0.6, 0.0005, 400.0, 0.9, 0.9, False),
    ]
    dec_a = rec.reconcile_and_dispatch("req_1", "usr_1", "test prompt", "Coding", cand_high, 0.0)
    assert dec_a.is_fallback_triggered is False
    assert dec_a.selected_model == "claude-3-5-sonnet"

    # Case B: Top candidate has success_prob 0.25 < 0.40 -> Fallback activates!
    cand_low = [
        CandidateScoreRow("llama-3.1-70b", 1, 0.80, 0.25, 0.6, 0.0001, 200.0, 1.0, 1.0, True),
    ]
    dec_b = rec.reconcile_and_dispatch("req_2", "usr_2", "test prompt", "Coding", cand_low, 0.0)
    assert dec_b.is_fallback_triggered is True
    assert dec_b.selected_model == "gpt-4o"
    assert "below critical threshold" in dec_b.fallback_reason


# -----------------------------------------------------------------------------
# Layer 6: End-to-End Engine & Feedback Loop
# -----------------------------------------------------------------------------
def test_end_to_end_engine_and_feedback(trained_engine):
    payload = {"user_id": "usr_99", "prompt": "Write a python function to compute fibonacci numbers."}
    decision = trained_engine.route(payload, persist_telemetry=True)

    assert decision.selected_model in trained_engine.model_names
    assert len(decision.ranked_candidates) == len(trained_engine.model_names)
    assert decision.routing_latency_ms > 0.0
    assert "Coding" in decision.intent_cluster

    # Record user feedback
    fb_id = trained_engine.repository.record_user_feedback(
        decision_id=decision.decision_id,
        user_id="usr_99",
        is_positive=True,
        rating=5
    )
    assert fb_id is not None

    unprocessed = trained_engine.repository.get_unprocessed_feedback()
    assert any(item["feedback_id"] == fb_id for item in unprocessed)

    # Trigger retraining loop
    res = trained_engine.trigger_retraining()
    assert res.status == "SUCCESS"
    assert res.feedback_samples_incorporated >= 1

    # Verify marked processed
    remaining = trained_engine.repository.get_unprocessed_feedback()
    assert not any(item["feedback_id"] == fb_id for item in remaining)
