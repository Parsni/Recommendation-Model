"""Layer 3: Bifurcated Resource Regressor Network (Dual-Headed Ridge Regression).

Top-head calculates predicted computational cost in USD;
Bottom-head estimates network execution latency in milliseconds based on
prompt complexity and destination model parameters.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from typing import Dict, List, Any, Optional

from src.routing.featurizer import StructuralFeatures


@dataclass
class ResourceForecast:
    """Resource estimation for a candidate destination model."""
    model_id: str
    predicted_cost_usd: float
    predicted_latency_ms: float
    estimated_input_tokens: int
    estimated_output_tokens: int


class DualHeadRidgeRegressorNetwork:
    """Bifurcated Ridge Regression engine forecasting cost and latency in parallel."""

    def __init__(self, model_specs: Dict[str, Dict[str, Any]], alpha: float = 1.0) -> None:
        self.model_specs = model_specs
        self.alpha = alpha
        
        # Parallel Ridge Regression models per destination candidate
        self.cost_regressors: Dict[str, Ridge] = {}
        self.latency_regressors: Dict[str, Ridge] = {}
        
        for m in self.model_specs.keys():
            self.cost_regressors[m] = Ridge(alpha=alpha, fit_intercept=True)
            self.latency_regressors[m] = Ridge(alpha=alpha, fit_intercept=True)
            
        self.is_fitted = False

    def fit_synthetic_baselines(self) -> "DualHeadRidgeRegressorNetwork":
        """Calibrate regressors across prompt token sizes and task complexities."""
        # Generate calibration points spanning 10 tokens to 4000 tokens
        token_samples = np.array([10, 25, 50, 100, 200, 350, 500, 1000, 2000, 4000], dtype=np.float32)
        
        for model_id, spec in self.model_specs.items():
            cost_in_per_token = spec.get("cost_input_per_million", 2.0) / 1_000_000.0
            cost_out_per_token = spec.get("cost_output_per_million", 8.0) / 1_000_000.0
            base_latency = spec.get("baseline_latency_ms", 400.0)
            per_token_latency = spec.get("per_token_latency_ms", 12.0)

            X_calib = []
            y_cost = []
            y_lat = []

            for t_in in token_samples:
                # Expected output tokens scale with input complexity (log-linear with task bias)
                t_out = max(50.0, float(t_in * 1.5))
                
                # Feature vector: [norm_tokens, char_ratio, code_bias, math_bias]
                feat = [t_in / 1000.0, (t_in * 4.0) / 4000.0, 0.2, 0.1]
                
                cost = (t_in * cost_in_per_token) + (t_out * cost_out_per_token)
                lat = base_latency + (t_out * per_token_latency * 0.1)

                X_calib.append(feat)
                y_cost.append(cost)
                y_lat.append(lat)

            self.cost_regressors[model_id].fit(np.array(X_calib), np.array(y_cost))
            self.latency_regressors[model_id].fit(np.array(X_calib), np.array(y_lat))

        self.is_fitted = True
        return self

    def forecast(
        self,
        structural: StructuralFeatures,
        cluster_id: int
    ) -> Dict[str, ResourceForecast]:
        """Parallel dual-headed inference predicting cost & latency for each destination model."""
        if not self.is_fitted:
            self.fit_synthetic_baselines()

        # Task output token multiplier by intent cluster
        # Coding (0) & Creative (1) generate longer output; QA/Factual (3) generate shorter output
        cluster_multipliers = {
            0: 2.2,  # Coding needs comprehensive function + tests
            1: 2.5,  # Creative narratives produce rich copy
            2: 1.8,  # Step-by-step math reasoning
            3: 1.0,  # Factual / QA answers
        }
        mult = cluster_multipliers.get(cluster_id, 1.5)

        t_in = structural.estimated_tokens
        t_out = int(max(40, t_in * mult))

        feat = np.array([[
            t_in / 1000.0,
            structural.char_length / 4000.0,
            structural.code_indicator,
            structural.math_indicator
        ]], dtype=np.float32)

        results: Dict[str, ResourceForecast] = {}
        for model_id in self.model_specs.keys():
            pred_cost = float(self.cost_regressors[model_id].predict(feat)[0])
            pred_lat = float(self.latency_regressors[model_id].predict(feat)[0])

            # Ensure strict non-negativity
            pred_cost = max(0.00001, pred_cost)
            pred_lat = max(50.0, pred_lat)

            results[model_id] = ResourceForecast(
                model_id=model_id,
                predicted_cost_usd=round(pred_cost, 6),
                predicted_latency_ms=round(pred_lat, 2),
                estimated_input_tokens=t_in,
                estimated_output_tokens=t_out
            )

        return results
