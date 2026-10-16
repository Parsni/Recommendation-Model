"""Multi-Criteria LLM Selection & Contextual Routing Engine Facade.

Integrates all 6 core layers into an enterprise-grade, high-throughput orchestrator.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from src.routing.sanitizer import RequestSanitizer, SanitizedRequest
from src.routing.featurizer import InFlightFeaturizer, StructuralFeatures
from src.routing.clustering import IntentClusterer
from src.routing.anchor_similarity import AnchorSimilarityMatrix, AnchorMatchResult
from src.routing.context_bridge import ContextEnrichmentBridge, EnrichedContext
from src.routing.inference.classifier_ensemble import LogisticRegressionEnsemble
from src.routing.inference.resource_regressors import DualHeadRidgeRegressorNetwork
from src.routing.scoring import MultiCriteriaOptimizer, RoutingPolicyWeights, CandidateScoreRow
from src.routing.reconciliation import ReconciliationEngine, RoutingDecision
from src.routing.telemetry.db_repository import TelemetryRepository
from src.routing.telemetry.retraining_loop import GroundTruthRetrainingLoop, RetrainingResult
from src.utils.helpers import PROJECT_ROOT


class LLMSelectionRoutingEngine:
    """Enterprise-grade Multi-Criteria LLM Selection & Contextual Routing Engine."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        db_url: Optional[str] = None
    ) -> None:
        cfg_file = config_path or (PROJECT_ROOT / "config" / "routing_config.json")
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        models_cfg = self.config.get("models", {
            "gpt-4o": {"cost_input_per_million": 2.5, "cost_output_per_million": 10.0, "baseline_latency_ms": 400.0, "per_token_latency_ms": 12.0},
            "claude-3-5-sonnet": {"cost_input_per_million": 3.0, "cost_output_per_million": 15.0, "baseline_latency_ms": 600.0, "per_token_latency_ms": 16.0},
            "gemini-1.5-pro": {"cost_input_per_million": 1.25, "cost_output_per_million": 5.0, "baseline_latency_ms": 480.0, "per_token_latency_ms": 13.5},
            "llama-3.1-70b": {"cost_input_per_million": 0.35, "cost_output_per_million": 0.40, "baseline_latency_ms": 220.0, "per_token_latency_ms": 7.0},
        })
        self.model_names = list(models_cfg.keys())

        # Layer 1: Ingestion Sanitizer
        san_cfg = self.config.get("sanitizer", {})
        self.sanitizer = RequestSanitizer(
            min_chars=san_cfg.get("min_chars", 3),
            max_bytes=san_cfg.get("max_bytes", 10 * 1024 * 1024),
            strip_markdown=san_cfg.get("strip_markdown_injections", True)
        )

        # Layer 2: Featurization & Parallel Tracks
        self.featurizer = InFlightFeaturizer()
        
        clust_cfg = self.config.get("clustering", {})
        self.clusterer = IntentClusterer(
            n_clusters=clust_cfg.get("n_clusters", 4),
            random_state=clust_cfg.get("random_state", 42)
        )
        self.anchor_matrix = AnchorSimilarityMatrix()
        self.context_bridge = ContextEnrichmentBridge(self.clusterer, num_anchors=8)

        # Layer 3: Cognitive Inference Layer
        self.classifier_ensemble = LogisticRegressionEnsemble(model_names=self.model_names, regularization_c=1.0)
        self.resource_regressors = DualHeadRidgeRegressorNetwork(model_specs=models_cfg, alpha=1.0)

        # Layer 4: Multi-Criteria Scoring & Optimizer
        weights_cfg = self.config.get("default_weights", {})
        init_weights = RoutingPolicyWeights(
            w_success=weights_cfg.get("w_success", 0.45),
            w_similarity=weights_cfg.get("w_similarity", 0.15),
            w_cost=weights_cfg.get("w_cost", 0.20),
            w_latency=weights_cfg.get("w_latency", 0.20),
        )
        self.optimizer = MultiCriteriaOptimizer(default_weights=init_weights)

        # Layer 5: Reconciliation & Failsafe Gate
        fail_cfg = self.config.get("failsafe", {})
        self.reconciliation = ReconciliationEngine(
            critical_success_threshold=fail_cfg.get("critical_success_threshold", 0.40),
            critical_composite_threshold=fail_cfg.get("critical_composite_threshold", 0.35),
            fallback_model=fail_cfg.get("fallback_model", "gpt-4o")
        )

        # Layer 6: Telemetry & Continuous Retraining Loop
        self.repository = TelemetryRepository(db_url=db_url)
        self.retraining_loop = GroundTruthRetrainingLoop(
            repo=self.repository,
            ensemble=self.classifier_ensemble
        )

        self.is_trained = False
        self.base_comparisons_df: Optional[pd.DataFrame] = None
        self.anchors_df: Optional[pd.DataFrame] = None

    def train(self, comparisons_df: pd.DataFrame, anchors_df: pd.DataFrame) -> "LLMSelectionRoutingEngine":
        """Train and index all components using historical pairwise dataset and anchor baselines."""
        self.base_comparisons_df = comparisons_df.copy().reset_index(drop=True)
        self.anchors_df = anchors_df.copy().reset_index(drop=True)

        # 1. Index Anchor Baselines
        self.anchor_matrix.build_index(self.anchors_df)

        # 2. Fit K-Means Intent Clusterer on corpus of prompts
        all_texts = list(self.base_comparisons_df["prompt"].dropna().unique()) + list(self.anchors_df["prompt_text"].dropna().unique())
        self.clusterer.fit(all_texts)

        # 3. Fit Resource Regressors (Cost & Latency)
        self.resource_regressors.fit_synthetic_baselines()

        # 4. Extract feature matrix for comparisons dataset
        feat_list = []
        for _, row in self.base_comparisons_df.iterrows():
            p_text = str(row["prompt"])
            u_id = str(row.get("user_id", "usr_train"))
            struct = self.featurizer.extract(p_text)
            anchor_match = self.anchor_matrix.compute_similarity(p_text)
            ctx = self.context_bridge.build_context(p_text, u_id, struct, anchor_match)
            feat_list.append(ctx.feature_vector)

        import numpy as np
        X_train = np.array(feat_list)

        # 5. Train Logistic Regression Ensemble
        self.classifier_ensemble.fit(self.base_comparisons_df, X_train)

        self.is_trained = True
        return self

    def route(
        self,
        payload: Dict[str, Any],
        override_weights: Optional[RoutingPolicyWeights] = None,
        critical_threshold_override: Optional[float] = None,
        persist_telemetry: bool = True
    ) -> RoutingDecision:
        """Execute end-to-end routing pipeline on an inbound request payload."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Layer 1: Ingestion & Preprocessing
        sanitized = self.sanitizer.sanitize(payload)
        if not sanitized.is_compliant:
            # Non-compliant request: dispatch to default fallback with audit explanation
            ranked = self.optimizer.evaluate_candidates(
                success_probs=self.classifier_ensemble.predict_success_probabilities(np.zeros(20)),
                resource_forecasts=self.resource_regressors.forecast(self.featurizer.extract(""), 0),
                anchor_similarity=0.0
            ) if self.is_trained else []
            
            decision = self.reconciliation.reconcile_and_dispatch(
                request_id=request_id,
                user_id=sanitized.user_id,
                sanitized_prompt=sanitized.sanitized_prompt,
                intent_cluster="Non-Compliant / Rejected",
                ranked_candidates=ranked,
                pipeline_start_time=start_time
            )
            decision.is_fallback_triggered = True
            decision.fallback_reason = f"Request sanitization failed: {'; '.join(sanitized.violations)}"
            return decision

        # Layer 2: In-Flight Featurization & Parallel Tracks
        structural = self.featurizer.extract(sanitized.sanitized_prompt)
        anchor_match = self.anchor_matrix.compute_similarity(sanitized.sanitized_prompt)
        enriched_ctx = self.context_bridge.build_context(
            sanitized_prompt=sanitized.sanitized_prompt,
            user_id=sanitized.user_id,
            structural=structural,
            anchor_match=anchor_match,
            user_meta=sanitized.metadata
        )

        # Layer 3: Cognitive Inference Layer
        success_probs = self.classifier_ensemble.predict_success_probabilities(
            prompt_feature_vector=enriched_ctx.feature_vector,
            prompt_text=sanitized.sanitized_prompt,
            intent_label=enriched_ctx.cluster_label
        )
        resource_forecasts = self.resource_regressors.forecast(structural, enriched_ctx.cluster_id)

        # Layer 4: Normalized Scoring Matrix & Multi-Criteria Optimizer
        ranked_candidates = self.optimizer.evaluate_candidates(
            success_probs=success_probs,
            resource_forecasts=resource_forecasts,
            anchor_similarity=anchor_match.max_similarity,
            override_weights=override_weights
        )

        # Layer 5: Reconciliation Engine & Dispatch
        decision = self.reconciliation.reconcile_and_dispatch(
            request_id=request_id,
            user_id=sanitized.user_id,
            sanitized_prompt=sanitized.sanitized_prompt,
            intent_cluster=enriched_ctx.cluster_label,
            ranked_candidates=ranked_candidates,
            pipeline_start_time=start_time,
            critical_threshold_override=critical_threshold_override
        )

        # Layer 6: Telemetry Persistence
        if persist_telemetry:
            try:
                self.repository.log_full_routing_cycle(
                    sanitized=sanitized,
                    intent_cluster_id=enriched_ctx.cluster_id,
                    intent_cluster_label=enriched_ctx.cluster_label,
                    anchor_id=anchor_match.nearest_anchor_id,
                    anchor_sim=anchor_match.max_similarity,
                    decision=decision
                )
            except Exception as e:
                # Telemetry logging failure must not break inline dispatch
                decision.audit_trail["telemetry_error"] = str(e)

        return decision

    def trigger_retraining(self) -> RetrainingResult:
        """Execute retraining loop using collected telemetry and feedback."""
        if self.base_comparisons_df is None or self.anchors_df is None:
            raise RuntimeError("Engine must be initially trained before retraining.")
        return self.retraining_loop.execute_retraining(
            base_comparisons_df=self.base_comparisons_df,
            context_bridge=self.context_bridge,
            featurizer=self.featurizer,
            anchor_matrix=self.anchor_matrix
        )
