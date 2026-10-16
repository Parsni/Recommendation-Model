"""Layer 6: Telemetry Repository & Data Access Layer.

Handles storage of routing decisions, candidate scores, execution telemetry,
and user feedback records across SQLite (local) and PostgreSQL (production).
"""
from __future__ import annotations

import datetime
import os
import uuid
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, select, desc
from sqlalchemy.orm import sessionmaker, Session

from src.routing.telemetry.db_models import (
    Base,
    RoutingRequestModel,
    CandidateEvaluationModel,
    RoutingDecisionModel,
    ExecutionTelemetryModel,
    UserFeedbackModel,
    AnchorBaselineModel,
    RoutingPolicyModel,
)
from src.routing.sanitizer import SanitizedRequest
from src.routing.scoring import CandidateScoreRow, RoutingPolicyWeights
from src.routing.reconciliation import RoutingDecision


class TelemetryRepository:
    """Database repository for routing telemetry, anchor management, and feedback."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        if not db_url:
            # Default to SQLite file in outputs/logs/
            from src.utils.helpers import PROJECT_ROOT
            db_dir = PROJECT_ROOT / "outputs" / "telemetry"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "routing_telemetry.db"
            db_url = f"sqlite:///{db_path}"

        self.db_url = db_url
        self.engine = create_engine(
            self.db_url,
            echo=False,
            connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {}
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.init_db()

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    def log_full_routing_cycle(
        self,
        sanitized: SanitizedRequest,
        intent_cluster_id: int,
        intent_cluster_label: str,
        anchor_id: Optional[str],
        anchor_sim: float,
        decision: RoutingDecision
    ) -> str:
        """Atomically persist request, candidate scores, and final decision."""
        with self.SessionLocal() as session:
            # 1. Persist Request
            req_model = RoutingRequestModel(
                id=decision.request_id,
                user_id=sanitized.user_id,
                sanitized_prompt=sanitized.sanitized_prompt,
                char_length=sanitized.char_length,
                estimated_tokens=int(sanitized.char_length / 3.8),
                detected_language="en",
                intent_cluster_id=intent_cluster_id,
                intent_cluster_label=intent_cluster_label,
                nearest_anchor_id=anchor_id,
                anchor_similarity=float(anchor_sim),
                created_at=datetime.datetime.utcnow()
            )
            session.add(req_model)

            # 2. Persist Candidate Evaluations
            for cand in decision.ranked_candidates:
                is_sel = (cand.model_id == decision.selected_model)
                eval_model = CandidateEvaluationModel(
                    id=str(uuid.uuid4()),
                    request_id=decision.request_id,
                    model_id=cand.model_id,
                    rank=cand.rank,
                    composite_score=cand.composite_score,
                    success_prob=cand.success_prob,
                    similarity=cand.similarity,
                    predicted_cost_usd=cand.raw_cost_usd,
                    predicted_latency_ms=cand.raw_latency_ms,
                    cost_efficiency=cand.cost_efficiency,
                    latency_efficiency=cand.latency_efficiency,
                    is_selected=is_sel
                )
                session.add(eval_model)

            # 3. Persist Decision
            dec_model = RoutingDecisionModel(
                id=decision.decision_id,
                request_id=decision.request_id,
                selected_model=decision.selected_model,
                fallback_triggered=decision.is_fallback_triggered,
                fallback_reason=decision.fallback_reason,
                routing_latency_ms=decision.routing_latency_ms,
                audit_metadata=decision.audit_trail,
                dispatched_at=datetime.datetime.utcnow()
            )
            session.add(dec_model)
            session.commit()
            return decision.decision_id

    def record_execution_telemetry(
        self,
        decision_id: str,
        model_id: str,
        actual_latency_ms: float,
        actual_input_tokens: int,
        actual_output_tokens: int,
        actual_cost_usd: float,
        status_code: int = 200,
        error_message: Optional[str] = None
    ) -> str:
        """Capture downstream API actual performance."""
        with self.SessionLocal() as session:
            telem_id = str(uuid.uuid4())
            telem = ExecutionTelemetryModel(
                id=telem_id,
                decision_id=decision_id,
                model_id=model_id,
                actual_latency_ms=actual_latency_ms,
                actual_input_tokens=actual_input_tokens,
                actual_output_tokens=actual_output_tokens,
                actual_cost_usd=actual_cost_usd,
                status_code=status_code,
                error_message=error_message,
                recorded_at=datetime.datetime.utcnow()
            )
            session.add(telem)
            session.commit()
            return telem_id

    def record_user_feedback(
        self,
        decision_id: str,
        user_id: str,
        is_positive: bool,
        rating: Optional[int] = None,
        preferred_winner_model: Optional[str] = None,
        feedback_comment: Optional[str] = None
    ) -> str:
        """Store end-user ground truth feedback to close the re-training loop."""
        with self.SessionLocal() as session:
            fb_id = str(uuid.uuid4())
            fb = UserFeedbackModel(
                id=fb_id,
                decision_id=decision_id,
                user_id=user_id,
                rating=rating,
                is_positive=is_positive,
                preferred_winner_model=preferred_winner_model,
                feedback_comment=feedback_comment,
                used_for_retraining=False,
                created_at=datetime.datetime.utcnow()
            )
            session.add(fb)
            session.commit()
            return fb_id

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent routing decisions with candidate breakdown for UI terminal."""
        with self.SessionLocal() as session:
            stmt = select(RoutingDecisionModel).order_by(desc(RoutingDecisionModel.dispatched_at)).limit(limit)
            decisions = session.scalars(stmt).all()
            results = []
            for d in decisions:
                req = d.request
                results.append({
                    "decision_id": d.id,
                    "request_id": d.request_id,
                    "user_id": req.user_id if req else "unknown",
                    "prompt": req.sanitized_prompt if req else "",
                    "intent_cluster": req.intent_cluster_label if req else "General",
                    "selected_model": d.selected_model,
                    "fallback_triggered": d.fallback_triggered,
                    "fallback_reason": d.fallback_reason,
                    "routing_latency_ms": d.routing_latency_ms,
                    "dispatched_at": d.dispatched_at.strftime("%Y-%m-%d %H:%M:%S") if d.dispatched_at else "",
                })
            return results

    def get_unprocessed_feedback(self) -> List[Dict[str, Any]]:
        """Retrieve feedback entries that have not yet been consumed by the re-training loop."""
        with self.SessionLocal() as session:
            stmt = (
                select(UserFeedbackModel, RoutingDecisionModel, RoutingRequestModel)
                .join(RoutingDecisionModel, UserFeedbackModel.decision_id == RoutingDecisionModel.id)
                .join(RoutingRequestModel, RoutingDecisionModel.request_id == RoutingRequestModel.id)
                .where(UserFeedbackModel.used_for_retraining == False)
            )
            rows = session.execute(stmt).all()
            feedback_data = []
            for fb, dec, req in rows:
                feedback_data.append({
                    "feedback_id": fb.id,
                    "decision_id": fb.decision_id,
                    "prompt": req.sanitized_prompt,
                    "selected_model": dec.selected_model,
                    "is_positive": fb.is_positive,
                    "rating": fb.rating,
                    "preferred_winner": fb.preferred_winner_model,
                    "user_id": fb.user_id,
                })
            return feedback_data

    def mark_feedback_processed(self, feedback_ids: List[str]) -> None:
        """Mark feedback entries as incorporated into training models."""
        if not feedback_ids:
            return
        with self.SessionLocal() as session:
            stmt = select(UserFeedbackModel).where(UserFeedbackModel.id.in_(feedback_ids))
            items = session.scalars(stmt).all()
            for it in items:
                it.used_for_retraining = True
            session.commit()
