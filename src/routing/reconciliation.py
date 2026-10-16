"""Layer 5: Reconciliation Engine & Dispatch.

Evaluates failsafe policies, executes failover routing when candidate reliability
falls below critical business thresholds, and constructs deployable dispatch payloads.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.routing.scoring import CandidateScoreRow


@dataclass
class RoutingDecision:
    """Final reconciliation and dispatch decision."""
    decision_id: str
    request_id: str
    user_id: str
    selected_model: str
    is_fallback_triggered: bool
    fallback_reason: Optional[str]
    intent_cluster: str
    top_composite_score: float
    top_success_prob: float
    ranked_candidates: List[CandidateScoreRow]
    routing_latency_ms: float
    audit_trail: Dict[str, Any] = field(default_factory=dict)


class ReconciliationEngine:
    """The Failsafe Gate and Policy Dispatch Reconciliation."""

    def __init__(
        self,
        critical_success_threshold: float = 0.40,
        critical_composite_threshold: float = 0.35,
        fallback_model: str = "gpt-4o"
    ) -> None:
        self.critical_success_threshold = critical_success_threshold
        self.critical_composite_threshold = critical_composite_threshold
        self.fallback_model = fallback_model

    def reconcile_and_dispatch(
        self,
        request_id: str,
        user_id: str,
        sanitized_prompt: str,
        intent_cluster: str,
        ranked_candidates: List[CandidateScoreRow],
        pipeline_start_time: float,
        critical_threshold_override: Optional[float] = None
    ) -> RoutingDecision:
        """Evaluate failsafe gate policies and dispatch target destination."""
        if not ranked_candidates:
            # Complete failure fallback
            elapsed_ms = (time.perf_counter() - pipeline_start_time) * 1000.0
            return RoutingDecision(
                decision_id=str(uuid.uuid4()),
                request_id=request_id,
                user_id=user_id,
                selected_model=self.fallback_model,
                is_fallback_triggered=True,
                fallback_reason="No candidate candidates available; triggered hard fallback default.",
                intent_cluster=intent_cluster,
                top_composite_score=0.0,
                top_success_prob=0.0,
                ranked_candidates=[],
                routing_latency_ms=round(elapsed_ms, 2),
                audit_trail={"status": "hard_failure_default"}
            )

        top_candidate = ranked_candidates[0]
        threshold = critical_threshold_override or self.critical_success_threshold

        is_fallback = False
        fallback_reason = None
        selected_model = top_candidate.model_id

        # Policy 1: Success Probability Failsafe Gate
        if top_candidate.success_prob < threshold:
            is_fallback = True
            fallback_reason = (
                f"Candidate '{top_candidate.model_id}' success probability ({top_candidate.success_prob}) "
                f"is below critical threshold ({threshold}). Bypassed to fallback '{self.fallback_model}'."
            )
            selected_model = self.fallback_model

        # Policy 2: Critical Composite Score Lower Bound
        elif top_candidate.composite_score < self.critical_composite_threshold:
            is_fallback = True
            fallback_reason = (
                f"Candidate '{top_candidate.model_id}' composite score ({top_candidate.composite_score}) "
                f"is below critical composite threshold ({self.critical_composite_threshold}). "
                f"Bypassed to fallback '{self.fallback_model}'."
            )
            selected_model = self.fallback_model

        elapsed_ms = (time.perf_counter() - pipeline_start_time) * 1000.0

        audit_trail = {
            "top_candidate": top_candidate.model_id,
            "top_composite_score": top_candidate.composite_score,
            "top_success_prob": top_candidate.success_prob,
            "threshold_applied": threshold,
            "fallback_model": self.fallback_model,
            "all_scores": {r.model_id: r.composite_score for r in ranked_candidates}
        }

        return RoutingDecision(
            decision_id=str(uuid.uuid4()),
            request_id=request_id,
            user_id=user_id,
            selected_model=selected_model,
            is_fallback_triggered=is_fallback,
            fallback_reason=fallback_reason,
            intent_cluster=intent_cluster,
            top_composite_score=top_candidate.composite_score,
            top_success_prob=top_candidate.success_prob,
            ranked_candidates=ranked_candidates,
            routing_latency_ms=round(elapsed_ms, 2),
            audit_trail=audit_trail
        )
