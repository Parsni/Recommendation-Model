"""Layer 4: Normalized Scoring Matrix & Multi-Criteria Optimizer.

Implements Bounded Linear Min-Max Scaling to normalize heterogeneous metrics
(currency, latency milliseconds, probabilities) into strict [0.0, 1.0] bounds,
inverts resource penalties into efficiency scores, and calculates composite scores.

Final Score = (w1 * SuccessProb) + (w2 * Similarity) + (w3 * CostEff) + (w4 * LatencyEff)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from src.routing.inference.classifier_ensemble import ModelSuccessProbability
from src.routing.inference.resource_regressors import ResourceForecast


@dataclass
class CandidateScoreRow:
    """Complete multi-criteria evaluated metrics for a single LLM candidate."""
    model_id: str
    rank: int
    composite_score: float
    success_prob: float
    similarity: float
    raw_cost_usd: float
    raw_latency_ms: float
    cost_efficiency: float
    latency_efficiency: float
    is_top_choice: bool = False


@dataclass
class RoutingPolicyWeights:
    """Configurable multi-criteria decision weights."""
    w_success: float = 0.45
    w_similarity: float = 0.15
    w_cost: float = 0.20
    w_latency: float = 0.20

    def normalized(self) -> "RoutingPolicyWeights":
        """Ensure all weights strictly sum to 1.0."""
        total = self.w_success + self.w_similarity + self.w_cost + self.w_latency
        if total <= 0:
            return RoutingPolicyWeights(0.45, 0.15, 0.20, 0.20)
        return RoutingPolicyWeights(
            w_success=round(self.w_success / total, 4),
            w_similarity=round(self.w_similarity / total, 4),
            w_cost=round(self.w_cost / total, 4),
            w_latency=round(self.w_latency / total, 4)
        )


class MultiCriteriaOptimizer:
    """Bounded Linear Min-Max Scaler & Composite Weighted Scoring Engine."""

    def __init__(self, default_weights: Optional[RoutingPolicyWeights] = None) -> None:
        self.weights = (default_weights or RoutingPolicyWeights()).normalized()

    def update_weights(self, weights: RoutingPolicyWeights) -> None:
        """Update active strategic policy weights."""
        self.weights = weights.normalized()

    def evaluate_candidates(
        self,
        success_probs: Dict[str, ModelSuccessProbability],
        resource_forecasts: Dict[str, ResourceForecast],
        anchor_similarity: float,
        override_weights: Optional[RoutingPolicyWeights] = None
    ) -> List[CandidateScoreRow]:
        """Compute scaled bounds and composite scores across candidate models.
        
        Args:
            success_probs: Dict of ModelSuccessProbability per candidate
            resource_forecasts: Dict of ResourceForecast per candidate
            anchor_similarity: Cosine similarity against anchor baseline [0.0, 1.0]
            override_weights: Optional transient policy weights (e.g. from UI sliders)
            
        Returns:
            List of CandidateScoreRow sorted descending by composite_score.
        """
        weights = (override_weights or self.weights).normalized()
        model_ids = list(success_probs.keys())

        # Collect raw values
        costs = [resource_forecasts[m].predicted_cost_usd for m in model_ids]
        latencies = [resource_forecasts[m].predicted_latency_ms for m in model_ids]

        min_cost, max_cost = min(costs), max(costs)
        cost_range = max_cost - min_cost if (max_cost - min_cost) > 1e-9 else 1.0

        min_lat, max_lat = min(latencies), max(latencies)
        lat_range = max_lat - min_lat if (max_lat - min_lat) > 1e-9 else 1.0

        sim_val = max(0.0, min(1.0, anchor_similarity))

        rows: List[CandidateScoreRow] = []

        for m in model_ids:
            p_succ = max(0.0, min(1.0, success_probs[m].p_success))
            raw_c = resource_forecasts[m].predicted_cost_usd
            raw_l = resource_forecasts[m].predicted_latency_ms

            # Bounded Linear Min-Max Inversion: lower cost/latency -> higher efficiency [0, 1]
            cost_eff = (max_cost - raw_c) / cost_range
            lat_eff = (max_lat - raw_l) / lat_range

            cost_eff = float(np.clip(cost_eff, 0.0, 1.0))
            lat_eff = float(np.clip(lat_eff, 0.0, 1.0))

            # Composite Formula:
            # Final Score = (w1 * SuccessProb) + (w2 * Similarity) + (w3 * CostEff) + (w4 * LatencyEff)
            final_score = (
                (weights.w_success * p_succ) +
                (weights.w_similarity * sim_val) +
                (weights.w_cost * cost_eff) +
                (weights.w_latency * lat_eff)
            )

            rows.append(CandidateScoreRow(
                model_id=m,
                rank=0,
                composite_score=round(float(final_score), 4),
                success_prob=round(p_succ, 4),
                similarity=round(sim_val, 4),
                raw_cost_usd=round(raw_c, 6),
                raw_latency_ms=round(raw_l, 2),
                cost_efficiency=round(cost_eff, 4),
                latency_efficiency=round(lat_eff, 4),
                is_top_choice=False
            ))

        # Sort descending by composite_score
        rows.sort(key=lambda r: r.composite_score, reverse=True)

        # Assign ranks
        for idx, row in enumerate(rows):
            row.rank = idx + 1
            if idx == 0:
                row.is_top_choice = True

        return rows
