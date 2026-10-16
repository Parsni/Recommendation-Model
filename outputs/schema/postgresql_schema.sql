-- =====================================================================
-- Enterprise Multi-Criteria LLM Selection & Contextual Routing Engine
-- Target Database: PostgreSQL 14+ / 16+
-- Schema Definition: Telemetry, Routing Decisions, and Feedback Loop
-- =====================================================================

-- 1. Inbound Routing Requests & Sanitized Metadata
CREATE TABLE IF NOT EXISTS routing_requests (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    sanitized_prompt TEXT NOT NULL,
    char_length INTEGER NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    detected_language VARCHAR(10) NOT NULL DEFAULT 'en',
    intent_cluster_id INTEGER NOT NULL,
    intent_cluster_label VARCHAR(128) NOT NULL,
    nearest_anchor_id VARCHAR(64),
    anchor_similarity DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS ix_routing_requests_user_id ON routing_requests (user_id);
CREATE INDEX IF NOT EXISTS ix_routing_requests_created_at ON routing_requests (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_routing_requests_intent ON routing_requests (intent_cluster_id);

-- 2. Multi-Candidate Model Evaluations (Scores, Costs, Latencies)
CREATE TABLE IF NOT EXISTS candidate_evaluations (
    id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL REFERENCES routing_requests(id) ON DELETE CASCADE,
    model_id VARCHAR(64) NOT NULL,
    rank INTEGER NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    success_prob DOUBLE PRECISION NOT NULL,
    similarity DOUBLE PRECISION NOT NULL,
    predicted_cost_usd DOUBLE PRECISION NOT NULL,
    predicted_latency_ms DOUBLE PRECISION NOT NULL,
    cost_efficiency DOUBLE PRECISION NOT NULL,
    latency_efficiency DOUBLE PRECISION NOT NULL,
    is_selected BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS ix_candidate_evaluations_request_id ON candidate_evaluations (request_id);
CREATE INDEX IF NOT EXISTS ix_candidate_evaluations_model_id ON candidate_evaluations (model_id);
CREATE INDEX IF NOT EXISTS ix_candidate_evaluations_req_rank ON candidate_evaluations (request_id, rank);

-- 3. Final Routing Decisions & Failsafe Audit Trails
CREATE TABLE IF NOT EXISTS routing_decisions (
    id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL UNIQUE REFERENCES routing_requests(id) ON DELETE CASCADE,
    selected_model VARCHAR(64) NOT NULL,
    fallback_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    fallback_reason TEXT,
    routing_latency_ms DOUBLE PRECISION NOT NULL,
    audit_metadata JSONB,
    dispatched_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS ix_routing_decisions_selected_model ON routing_decisions (selected_model);
CREATE INDEX IF NOT EXISTS ix_routing_decisions_dispatched_at ON routing_decisions (dispatched_at DESC);
CREATE INDEX IF NOT EXISTS ix_routing_decisions_fallback ON routing_decisions (fallback_triggered);

-- 4. Downstream Execution Telemetry (Actual Token Usage, Cost, Latency)
CREATE TABLE IF NOT EXISTS execution_telemetry (
    id VARCHAR(36) PRIMARY KEY,
    decision_id VARCHAR(36) NOT NULL UNIQUE REFERENCES routing_decisions(id) ON DELETE CASCADE,
    model_id VARCHAR(64) NOT NULL,
    actual_latency_ms DOUBLE PRECISION NOT NULL,
    actual_input_tokens INTEGER NOT NULL,
    actual_output_tokens INTEGER NOT NULL,
    actual_cost_usd DOUBLE PRECISION NOT NULL,
    status_code INTEGER NOT NULL DEFAULT 200,
    error_message TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS ix_execution_telemetry_decision_id ON execution_telemetry (decision_id);
CREATE INDEX IF NOT EXISTS ix_execution_telemetry_model_id ON execution_telemetry (model_id);

-- 5. User Feedback & Ground Truth Preferences for Continuous Re-training
CREATE TABLE IF NOT EXISTS user_feedback (
    id VARCHAR(36) PRIMARY KEY,
    decision_id VARCHAR(36) NOT NULL UNIQUE REFERENCES routing_decisions(id) ON DELETE CASCADE,
    user_id VARCHAR(64) NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    is_positive BOOLEAN NOT NULL DEFAULT TRUE,
    preferred_winner_model VARCHAR(64),
    feedback_comment TEXT,
    used_for_retraining BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS ix_user_feedback_user_id ON user_feedback (user_id);
CREATE INDEX IF NOT EXISTS ix_user_feedback_retraining ON user_feedback (used_for_retraining) WHERE used_for_retraining = FALSE;

-- 6. Production Anchor Baseline Prompts
CREATE TABLE IF NOT EXISTS anchor_baselines (
    id VARCHAR(64) PRIMARY KEY,
    category VARCHAR(64) NOT NULL,
    prompt_text TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS ix_anchor_baselines_category ON anchor_baselines (category);

-- 7. Strategic Routing Policy Configurations
CREATE TABLE IF NOT EXISTS routing_policies (
    id VARCHAR(36) PRIMARY KEY,
    policy_name VARCHAR(64) NOT NULL,
    w_success DOUBLE PRECISION NOT NULL DEFAULT 0.45,
    w_similarity DOUBLE PRECISION NOT NULL DEFAULT 0.15,
    w_cost DOUBLE PRECISION NOT NULL DEFAULT 0.20,
    w_latency DOUBLE PRECISION NOT NULL DEFAULT 0.20,
    failsafe_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.40,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
