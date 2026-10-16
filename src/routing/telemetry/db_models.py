"""Layer 6: Telemetry & Logging - PostgreSQL / SQLAlchemy ORM Schema.

Complete production-grade ORM tables tracking:
1. Inbound Routing Requests & Sanitized Metadata
2. Multi-Candidate Model Evaluations (Scores, Costs, Latencies)
3. Final Routing Decisions & Failsafe Audit Trails
4. Downstream Execution Telemetry (Actual Token Usage, Cost, Latency)
5. User Feedback & Ground Truth Preferences for Continuous Re-training
6. Anchor Baseline Matrix
7. Routing Policy Weight Configurations
"""
from __future__ import annotations

import datetime
import uuid
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class RoutingRequestModel(Base):
    """Inbound request payload and in-flight featurization audit record."""
    __tablename__ = "routing_requests"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(64), nullable=False, index=True)
    sanitized_prompt = Column(Text, nullable=False)
    char_length = Column(Integer, nullable=False)
    estimated_tokens = Column(Integer, nullable=False)
    detected_language = Column(String(10), nullable=False, default="en")
    intent_cluster_id = Column(Integer, nullable=False)
    intent_cluster_label = Column(String(128), nullable=False)
    nearest_anchor_id = Column(String(64), nullable=True)
    anchor_similarity = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)

    # Relationships
    evaluations = relationship("CandidateEvaluationModel", back_populates="request", cascade="all, delete-orphan")
    decision = relationship("RoutingDecisionModel", back_populates="request", uselist=False, cascade="all, delete-orphan")


class CandidateEvaluationModel(Base):
    """Per-candidate score breakdown and predicted resource parameters."""
    __tablename__ = "candidate_evaluations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(36), ForeignKey("routing_requests.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = Column(String(64), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    composite_score = Column(Float, nullable=False)
    success_prob = Column(Float, nullable=False)
    similarity = Column(Float, nullable=False)
    predicted_cost_usd = Column(Float, nullable=False)
    predicted_latency_ms = Column(Float, nullable=False)
    cost_efficiency = Column(Float, nullable=False)
    latency_efficiency = Column(Float, nullable=False)
    is_selected = Column(Boolean, nullable=False, default=False)

    request = relationship("RoutingRequestModel", back_populates="evaluations")

    __table_args__ = (
        Index("ix_eval_request_rank", "request_id", "rank"),
    )


class RoutingDecisionModel(Base):
    """Reconciliation result, failsafe status, and dispatch metadata."""
    __tablename__ = "routing_decisions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(36), ForeignKey("routing_requests.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    selected_model = Column(String(64), nullable=False, index=True)
    fallback_triggered = Column(Boolean, nullable=False, default=False)
    fallback_reason = Column(Text, nullable=True)
    routing_latency_ms = Column(Float, nullable=False)
    audit_metadata = Column(JSON, nullable=True)
    dispatched_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)

    request = relationship("RoutingRequestModel", back_populates="decision")
    execution_telemetry = relationship("ExecutionTelemetryModel", back_populates="decision", uselist=False, cascade="all, delete-orphan")
    user_feedback = relationship("UserFeedbackModel", back_populates="decision", uselist=False, cascade="all, delete-orphan")


class ExecutionTelemetryModel(Base):
    """Downstream actual execution telemetry captured from LLM provider API."""
    __tablename__ = "execution_telemetry"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id = Column(String(36), ForeignKey("routing_decisions.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    model_id = Column(String(64), nullable=False)
    actual_latency_ms = Column(Float, nullable=False)
    actual_input_tokens = Column(Integer, nullable=False)
    actual_output_tokens = Column(Integer, nullable=False)
    actual_cost_usd = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False, default=200)
    error_message = Column(Text, nullable=True)
    recorded_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    decision = relationship("RoutingDecisionModel", back_populates="execution_telemetry")


class UserFeedbackModel(Base):
    """End-user ground truth feedback loop used for model retraining."""
    __tablename__ = "user_feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id = Column(String(36), ForeignKey("routing_decisions.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    rating = Column(Integer, nullable=True)  # 1 to 5 scale
    is_positive = Column(Boolean, nullable=False, default=True)
    preferred_winner_model = Column(String(64), nullable=True)
    feedback_comment = Column(Text, nullable=True)
    used_for_retraining = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    decision = relationship("RoutingDecisionModel", back_populates="user_feedback")


class AnchorBaselineModel(Base):
    """Production Anchor Baseline Prompts table."""
    __tablename__ = "anchor_baselines"

    id = Column(String(64), primary_key=True)
    category = Column(String(64), nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class RoutingPolicyModel(Base):
    """Strategic routing policy configurations and criteria weights."""
    __tablename__ = "routing_policies"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    policy_name = Column(String(64), nullable=False)
    w_success = Column(Float, nullable=False, default=0.45)
    w_similarity = Column(Float, nullable=False, default=0.15)
    w_cost = Column(Float, nullable=False, default=0.20)
    w_latency = Column(Float, nullable=False, default=0.20)
    failsafe_threshold = Column(Float, nullable=False, default=0.40)
    is_active = Column(Boolean, default=True, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)
