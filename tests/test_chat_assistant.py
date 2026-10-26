"""Tests for the offline rule-based chat assistant (no network, no LLM)."""
from __future__ import annotations

from src.routing.chat_assistant import (
    POLICY_BUDGET,
    POLICY_FAST,
    POLICY_QUALITY,
    ChatAction,
    parse_intent,
    respond,
)
from src.routing.reconciliation import RoutingDecision
from src.routing.scoring import CandidateScoreRow


def _decision() -> RoutingDecision:
    return RoutingDecision(
        decision_id="d1",
        request_id="r1",
        user_id="u1",
        selected_model="gpt-4o",
        is_fallback_triggered=False,
        fallback_reason=None,
        intent_cluster="Coding (K-4)",
        top_composite_score=0.71,
        top_success_prob=0.8,
        ranked_candidates=[
            CandidateScoreRow("gpt-4o", 1, 0.71, 0.80, 0.5, 0.002, 500, 0.8, 0.7, True),
            CandidateScoreRow("llama-3.1-70b", 2, 0.60, 0.55, 0.5, 0.0004, 250, 1.0, 1.0, False),
        ],
        routing_latency_ms=3.2,
    )


def test_parse_intent_help_and_greeting():
    assert parse_intent("help").kind == "help"
    assert parse_intent("hi there").kind == "help"
    assert parse_intent("what can you do?").kind == "help"


def test_parse_intent_why_metrics_feedback():
    assert parse_intent("why this model?").kind == "why"
    assert parse_intent("how accurate is the router?").kind == "metrics"
    assert parse_intent("open feedback").kind == "feedback"
    assert parse_intent("leave a rating").kind == "feedback"


def test_parse_intent_compare_detects_models():
    intent = parse_intent("compare claude vs gpt-4o")
    assert intent.kind == "compare"
    assert "claude-3-5-sonnet" in intent.params["models"]
    assert "gpt-4o" in intent.params["models"]


def test_parse_intent_switch_policy_synonyms():
    assert parse_intent("switch to cheapest").params["policy"] == POLICY_BUDGET
    assert parse_intent("use the fastest preset").params["policy"] == POLICY_FAST
    assert parse_intent("apply highest quality").params["policy"] == POLICY_QUALITY


def test_parse_intent_new_prompt_fallback():
    intent = parse_intent("how do I bake sourdough bread step by step")
    assert intent.kind == "new_prompt"
    assert "sourdough" in intent.params["prompt"]


def test_respond_why_with_decision():
    reply, action = respond(parse_intent("why did you pick that?"), decision=_decision(), profiles={
        "gpt-4o": {"name": "GPT-4o", "provider": "OpenAI", "tier": "Flagship", "strengths": ["reasoning"]},
    })
    assert "GPT-4o" in reply
    assert action.type == "none"


def test_respond_switch_policy_returns_action():
    reply, action = respond(parse_intent("switch to cheapest"))
    assert action.type == "switch_policy"
    assert action.payload["policy"] == POLICY_BUDGET
    assert POLICY_BUDGET in reply


def test_respond_metrics_without_report_is_safe():
    reply, action = respond(parse_intent("what's your accuracy?"), report=None)
    assert "not loaded" in reply.lower()
    assert isinstance(action, ChatAction)


def test_respond_new_prompt_routes():
    reply, action = respond(parse_intent("translate hello to french please"))
    assert action.type == "route_prompt"
    assert "translate" in action.payload["prompt"]
