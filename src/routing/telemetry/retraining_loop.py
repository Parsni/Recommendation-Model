"""Layer 6: Ground Truth Retraining Loop.

Closes the feedback loop by extracting recent user feedback and execution logs,
synthesizing new preference pairs, and incrementally updating the
Logistic Regression Ensemble and Resource Regressors.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from src.routing.telemetry.db_repository import TelemetryRepository
from src.routing.inference.classifier_ensemble import LogisticRegressionEnsemble


@dataclass
class RetrainingResult:
    """Summary metrics from a retraining run."""
    feedback_samples_incorporated: int
    total_comparisons_trained: int
    updated_win_rates: Dict[str, float]
    status: str


class GroundTruthRetrainingLoop:
    """Orchestrates continuous model re-training using collected telemetry and user feedback."""

    def __init__(self, repo: TelemetryRepository, ensemble: LogisticRegressionEnsemble) -> None:
        self.repo = repo
        self.ensemble = ensemble

    def execute_retraining(
        self,
        base_comparisons_df: pd.DataFrame,
        context_bridge,
        featurizer,
        anchor_matrix
    ) -> RetrainingResult:
        """Fetch unconsumed feedback, construct augmented dataset, and retrain."""
        unprocessed = self.repo.get_unprocessed_feedback()
        new_rows = []
        feedback_ids = []

        for item in unprocessed:
            fb_id = item["feedback_id"]
            prompt = item["prompt"]
            selected_model = item["selected_model"]
            user_id = item["user_id"]
            feedback_ids.append(fb_id)

            # If user explicitly specified a preferred winner
            if item.get("preferred_winner") and item["preferred_winner"] in self.ensemble.model_names:
                pref = item["preferred_winner"]
                if pref != selected_model:
                    new_rows.append({
                        "prompt": prompt,
                        "model_a": selected_model,
                        "model_b": pref,
                        "winner": "model_b",
                        "user_id": user_id
                    })
            elif item.get("is_positive"):
                # Positive reinforcement: pick a lower-ranked competitor to form winning pair
                competitors = [m for m in self.ensemble.model_names if m != selected_model]
                if competitors:
                    comp = competitors[0]
                    new_rows.append({
                        "prompt": prompt,
                        "model_a": selected_model,
                        "model_b": comp,
                        "winner": "model_a",
                        "user_id": user_id
                    })

        # Combine base dataset with newly incorporated feedback pairs
        if new_rows:
            augmented_df = pd.concat([base_comparisons_df, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            augmented_df = base_comparisons_df.copy()

        # Build feature matrix for all prompts
        feature_list = []
        for _, row in augmented_df.iterrows():
            prompt_text = str(row["prompt"])
            u_id = str(row.get("user_id", "usr_feedback"))
            struct = featurizer.extract(prompt_text)
            anchor_res = anchor_matrix.compute_similarity(prompt_text)
            ctx = context_bridge.build_context(prompt_text, u_id, struct, anchor_res)
            feature_list.append(ctx.feature_vector)

        X_mat = np.array(feature_list)

        # Retrain ensemble
        self.ensemble.fit(augmented_df, X_mat)

        # Mark feedback as processed in database
        self.repo.mark_feedback_processed(feedback_ids)

        return RetrainingResult(
            feedback_samples_incorporated=len(feedback_ids),
            total_comparisons_trained=len(augmented_df),
            updated_win_rates={m: round(w, 4) for m, w in self.ensemble.empirical_win_rates.items()},
            status="SUCCESS"
        )
