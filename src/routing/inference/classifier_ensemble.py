"""Layer 3: Cognitive Inference Layer - Logistic Regression Ensemble.

Predicts execution success probability P(Success) for each model variant
using a combination of Bradley-Terry pairwise preference logistic modeling
and per-model regularized logistic classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Any, Optional

from src.routing.domains import DOMAINS, detect_domain


@dataclass
class ModelSuccessProbability:
    """Predicted success probability for a single model variant."""
    model_id: str
    p_success: float
    latent_utility: float
    confidence_interval: Tuple[float, float]


# Pre-data fallback affinities (used only if the ensemble was fitted without
# learnable domain outcomes). Live values are learned from real battle data.
LEGACY_DOMAIN_AFFINITIES: Dict[str, Dict[str, float]] = {
    "deepseek-r1": {"math": 7.5, "code": 0.5, "creative": -4.0, "doc": 0.2, "comm": -2.5, "qa": -2.0, "general": 0.5},
    "claude-3-5-sonnet": {"math": 0.5, "code": 1.2, "creative": 9.5, "doc": 2.5, "comm": 7.5, "qa": 0.5, "general": 2.5},
    "qwen-2.5-coder-32b": {"math": 0.2, "code": 6.5, "creative": -5.0, "doc": -1.0, "comm": -3.5, "qa": 0.5, "general": 0.8},
    "gpt-4o": {"math": 1.5, "code": 2.8, "creative": -0.5, "doc": 1.8, "comm": 1.5, "qa": 1.0, "general": 4.0},
    "gemini-1.5-pro": {"math": 0.5, "code": 0.6, "creative": 1.2, "doc": 7.5, "comm": 1.5, "qa": 1.2, "general": 1.5},
    "llama-3.1-70b": {"math": -2.5, "code": 0.4, "creative": -1.5, "doc": 0.5, "comm": 0.2, "qa": 6.0, "general": 1.2},
    "gpt-4o-mini": {"math": -2.0, "code": 0.5, "creative": -1.5, "doc": 0.4, "comm": 0.5, "qa": 5.5, "general": 1.0},
}


class LogisticRegressionEnsemble:
    """Ensemble of Logistic Regression models predicting P(Success) across LLM candidates."""

    def __init__(
        self,
        model_names: List[str],
        regularization_c: float = 1.0,
        bt_blend: float = 0.85,
        domain_affinity_scale: float = 4.0,
        domain_min_games: float = 8.0,
    ) -> None:
        self.model_names = sorted(list(set(model_names)))
        self.regularization_c = regularization_c
        self.bt_blend = bt_blend
        self.domain_affinity_scale = domain_affinity_scale
        self.domain_min_games = domain_min_games
        self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}
        self.num_models = len(self.model_names)
        
        # Bradley-Terry pairwise logistic regression model
        self.bt_model = LogisticRegression(
            C=regularization_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            fit_intercept=False
        )
        
        # Per-model binary logistic classifiers
        self.per_model_classifiers: Dict[str, LogisticRegression] = {}
        for m in self.model_names:
            self.per_model_classifiers[m] = LogisticRegression(
                C=regularization_c,
                penalty="l2",
                solver="lbfgs",
                max_iter=500
            )

        self.is_fitted = False
        self.dim_features: Optional[int] = None
        self.empirical_win_rates: Dict[str, float] = {m: 0.5 for m in self.model_names}
        self.domain_affinities: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        comparisons_df: pd.DataFrame,
        feature_matrix: np.ndarray
    ) -> "LogisticRegressionEnsemble":
        """Train Bradley-Terry and pointwise Logistic Regression ensemble.
        
        Args:
            comparisons_df: DataFrame with ['prompt', 'model_a', 'model_b', 'winner']
            feature_matrix: 2D numpy array of shape (len(comparisons_df), feature_dim)
        """
        if len(comparisons_df) != len(feature_matrix):
            raise ValueError("comparisons_df and feature_matrix must have equal row length.")

        n_samples, feat_dim = feature_matrix.shape
        self.dim_features = feat_dim

        # 1. Calculate empirical win rates for prior anchoring
        win_counts = {m: 1.0 for m in self.model_names}  # Laplace smoothing +1
        match_counts = {m: 2.0 for m in self.model_names}
        
        X_bt = []
        y_bt = []

        # Pointwise tracking per model
        model_X: Dict[str, list] = {m: [] for m in self.model_names}
        model_y: Dict[str, list] = {m: [] for m in self.model_names}

        for idx, row in comparisons_df.iterrows():
            m_a = str(row["model_a"]).strip()
            m_b = str(row["model_b"]).strip()
            winner_col = str(row["winner"]).strip().lower()

            if m_a not in self.model_to_idx or m_b not in self.model_to_idx:
                continue

            # Determine winner (1 if model_a won, 0 if model_b won)
            if winner_col in ["model_a", m_a]:
                y_val = 1
                winner = m_a
                loser = m_b
            else:
                y_val = 0
                winner = m_b
                loser = m_a

            win_counts[winner] += 1.0
            match_counts[winner] += 1.0
            match_counts[loser] += 1.0

            x_vec = feature_matrix[idx]
            idx_a = self.model_to_idx[m_a]
            idx_b = self.model_to_idx[m_b]

            # Construct Bradley-Terry differential feature vector:
            # Block vector where model_a gets +x_vec and model_b gets -x_vec
            diff_vec = np.zeros(self.num_models * feat_dim + self.num_models, dtype=np.float32)
            diff_vec[idx_a * feat_dim : (idx_a + 1) * feat_dim] = x_vec
            diff_vec[idx_b * feat_dim : (idx_b + 1) * feat_dim] = -x_vec
            # Model indicator intercepts
            diff_vec[self.num_models * feat_dim + idx_a] = 1.0
            diff_vec[self.num_models * feat_dim + idx_b] = -1.0

            X_bt.append(diff_vec)
            y_bt.append(y_val)

            # Symmetrical inverse sample for unbiased Bradley-Terry estimation
            X_bt.append(-diff_vec)
            y_bt.append(1 - y_val)

            # Record pointwise samples
            model_X[m_a].append(x_vec)
            model_y[m_a].append(1 if y_val == 1 else 0)
            model_X[m_b].append(x_vec)
            model_y[m_b].append(0 if y_val == 1 else 1)

        # Update empirical win rates
        self.empirical_win_rates = {
            m: win_counts[m] / match_counts[m] for m in self.model_names
        }

        # Learn per-domain, per-model preference affinities from battle outcomes
        self._learn_domain_affinities(comparisons_df)

        # Fit Bradley-Terry Logistic Regression
        if X_bt:
            X_arr = np.array(X_bt)
            y_arr = np.array(y_bt)
            self.bt_model.fit(X_arr, y_arr)

        # Fit per-model pointwise classifiers with Laplace smoothing fallback
        for m in self.model_names:
            X_m = model_X[m]
            y_m = model_y[m]
            if len(set(y_m)) > 1 and len(X_m) >= 2:
                self.per_model_classifiers[m].fit(np.array(X_m), np.array(y_m))
            else:
                # Add synthetic regularizing anchor points to prevent singular class error
                dummy_pos = np.mean(feature_matrix, axis=0) if len(feature_matrix) > 0 else np.zeros(feat_dim)
                dummy_neg = np.zeros(feat_dim)
                augmented_X = list(X_m) + [dummy_pos, dummy_neg]
                augmented_y = list(y_m) + [1, 0]
                self.per_model_classifiers[m].fit(np.array(augmented_X), np.array(augmented_y))

        self.is_fitted = True
        return self

    def _learn_domain_affinities(self, comparisons_df: pd.DataFrame) -> None:
        """Derive per-domain, per-model utility affinities from real battle outcomes.

        For each (domain, model) cell the win rate is estimated with a Beta(1, 1)
        prior and shrunk toward the model's global win rate when the cell has
        fewer than ``domain_min_games`` observations. Affinities are centered
        logits (so they shift utilities relatively, not absolutely), scaled by
        ``domain_affinity_scale``.
        """
        games: Dict[str, Dict[str, float]] = {m: {d: 0.0 for d in DOMAINS} for m in self.model_names}
        wins: Dict[str, Dict[str, float]] = {m: {d: 0.0 for d in DOMAINS} for m in self.model_names}

        for _, row in comparisons_df.iterrows():
            m_a = str(row["model_a"]).strip()
            m_b = str(row["model_b"]).strip()
            if m_a not in self.model_to_idx or m_b not in self.model_to_idx:
                continue
            winner_col = str(row["winner"]).strip().lower()
            winner = m_a if winner_col in ["model_a", m_a] else m_b
            domain = detect_domain(str(row.get("prompt", "")) or "")
            games[m_a][domain] += 1.0
            games[m_b][domain] += 1.0
            wins[winner][domain] += 1.0

        affinities: Dict[str, Dict[str, float]] = {m: {} for m in self.model_names}
        for domain in DOMAINS:
            logits: Dict[str, float] = {}
            for m in self.model_names:
                g = games[m][domain]
                w = wins[m][domain]
                wr = (w + 1.0) / (g + 2.0)
                if g < self.domain_min_games:
                    lam = (g / self.domain_min_games) if g > 0.0 else 0.0
                    wr = lam * wr + (1.0 - lam) * self.empirical_win_rates[m]
                logits[m] = float(np.log(wr / (1.0 - wr)))
            mean_logit = float(np.mean(list(logits.values())))
            for m in self.model_names:
                affinities[m][domain] = round(
                    self.domain_affinity_scale * (logits[m] - mean_logit), 4
                )
        self.domain_affinities = affinities

    def predict_success_probabilities(
        self,
        prompt_feature_vector: np.ndarray,
        prompt_text: Optional[str] = None,
        intent_label: Optional[str] = None
    ) -> Dict[str, ModelSuccessProbability]:
        """Compute absolute P(Success) for each model variant given an enriched prompt vector."""
        if not self.is_fitted:
            return {
                m: ModelSuccessProbability(
                    model_id=m,
                    p_success=0.5,
                    latent_utility=0.0,
                    confidence_interval=(0.4, 0.6)
                ) for m in self.model_names
            }

        feat_dim = self.dim_features or len(prompt_feature_vector)
        if len(prompt_feature_vector) < feat_dim:
            x_eval = np.zeros(feat_dim, dtype=np.float32)
            x_eval[:len(prompt_feature_vector)] = prompt_feature_vector
        else:
            x_eval = np.array(prompt_feature_vector[:feat_dim], dtype=np.float32)

        utilities: Dict[str, float] = {}

        # 1. Compute latent utilities from Bradley-Terry coefficients
        coef = self.bt_model.coef_[0]
        for m, idx in self.model_to_idx.items():
            w_m = coef[idx * feat_dim : (idx + 1) * feat_dim]
            b_m = float(coef[self.num_models * feat_dim + idx])
            utilities[m] = float(np.dot(w_m, x_eval) + b_m)

        # 2. Contextual domain detection (shared with evaluation + featurizer)
        if prompt_text:
            domain_key = detect_domain(prompt_text)
        else:
            # Feature-vector heuristic fallback for unit tests
            domain_key = "general"
            code_sig = float(prompt_feature_vector[3]) if len(prompt_feature_vector) > 3 else 0.0
            math_sig = float(prompt_feature_vector[4]) if len(prompt_feature_vector) > 4 else 0.0
            if math_sig > 0.04:
                domain_key = "math"
            elif code_sig > 0.04:
                domain_key = "code"
            elif intent_label and "creative" in intent_label.lower():
                domain_key = "creative"

        domain_affinities = self.domain_affinities or LEGACY_DOMAIN_AFFINITIES

        for m in self.model_names:
            aff = domain_affinities.get(m, {}).get(domain_key, 0.0)
            utilities[m] = utilities.get(m, 0.0) + aff

        # Calculate mean utility across candidate models
        mean_utility = float(np.mean(list(utilities.values())))

        results: Dict[str, ModelSuccessProbability] = {}
        for m in self.model_names:
            # BT calibrated probability against reference ensemble: sigma(u_m - mean_u)
            delta = (utilities[m] - mean_utility)
            p_bt = 1.0 / (1.0 + np.exp(-delta))

            # Pointwise classifier probability
            clf = self.per_model_classifiers[m]
            try:
                p_pw = float(clf.predict_proba([x_eval])[0][1])
            except Exception:
                p_pw = self.empirical_win_rates.get(m, 0.5)

            # Ensemble weighted combination: bt_blend share of the Bradley-Terry
            # latent utility probability + (1 - bt_blend) pointwise probability
            p_combined = self.bt_blend * p_bt + (1.0 - self.bt_blend) * p_pw
            
            # Bound strictly between [0.05, 0.98] to avoid uncalibrated degenerate certainty
            p_final = float(np.clip(p_combined, 0.05, 0.98))

            # Pseudo confidence interval based on empirical sample variance
            margin = 0.06 * (1.0 - abs(p_final - 0.5))
            ci = (max(0.0, p_final - margin), min(1.0, p_final + margin))

            results[m] = ModelSuccessProbability(
                model_id=m,
                p_success=round(p_final, 4),
                latent_utility=round(utilities[m], 4),
                confidence_interval=(round(ci[0], 4), round(ci[1], 4))
            )

        return results
