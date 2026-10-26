"""Layer 2 Track C: Context Enrichment Bridge.

Combines raw metadata metrics with text feature outputs, flattening them
into a high-density tabular numpy block for the Cognitive Inference Layer.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.routing.featurizer import StructuralFeatures
from src.routing.clustering import IntentClusterer
from src.routing.anchor_similarity import AnchorMatchResult
from src.routing.domains import DOMAINS, detect_domain


@dataclass
class EnrichedContext:
    """Enriched multi-feature representation ready for ML inference."""
    prompt: str
    user_id: str
    structural: StructuralFeatures
    cluster_id: int
    cluster_label: str
    anchor_similarity: float
    nearest_anchor_id: str
    feature_vector: np.ndarray
    feature_names: list[str]
    domain: str = "general"


class ContextEnrichmentBridge:
    """Flattens parallel feature streams into a high-density tabular numpy block."""

    def __init__(self, clusterer: IntentClusterer, num_anchors: int = 8) -> None:
        self.clusterer = clusterer
        self.num_anchors = num_anchors

    def build_context(
        self,
        sanitized_prompt: str,
        user_id: str,
        structural: StructuralFeatures,
        anchor_match: AnchorMatchResult,
        user_meta: Optional[Dict[str, Any]] = None
    ) -> EnrichedContext:
        """Flatten structural, spatial cluster, and anchor metrics into a numpy block."""
        # Predict intent cluster
        cluster_id, cluster_label, cluster_soft_probs = self.clusterer.predict(sanitized_prompt)

        # Build feature vector
        feats = []
        names = []

        # 1. Structural normalized features
        feats.extend([
            structural.char_length / 1000.0,       # normalized char length
            structural.word_count / 200.0,         # normalized word count
            structural.estimated_tokens / 250.0,   # normalized token count
            structural.code_indicator,             # [0, 1]
            structural.math_indicator,             # [0, 1]
            structural.question_indicator,         # {0, 1}
            structural.uppercase_ratio,            # [0, 1]
            structural.digit_ratio,                # [0, 1]
            structural.avg_word_length / 10.0,     # normalized avg word len
        ])
        names.extend([
            "norm_char_len", "norm_word_count", "norm_tokens",
            "code_indicator", "math_indicator", "question_indicator",
            "uppercase_ratio", "digit_ratio", "norm_avg_word_len"
        ])

        # 2. Intent cluster soft distribution
        for i, prob in enumerate(cluster_soft_probs):
            feats.append(float(prob))
            names.append(f"cluster_prob_{i}")

        # 3. Anchor similarity metrics
        feats.append(anchor_match.max_similarity)
        names.append("max_anchor_sim")

        # Full anchor similarity vector (padded/truncated to fixed dimension)
        anchor_vec = anchor_match.similarity_vector
        for i in range(self.num_anchors):
            val = float(anchor_vec[i]) if i < len(anchor_vec) else 0.0
            feats.append(val)
            names.append(f"anchor_sim_{i}")

        # 4. Prompt domain one-hot (appended last so legacy feature indices stay stable)
        domain = detect_domain(sanitized_prompt)
        for d in DOMAINS:
            feats.append(1.0 if d == domain else 0.0)
            names.append(f"domain_{d}")

        feature_block = np.array(feats, dtype=np.float32)

        return EnrichedContext(
            prompt=sanitized_prompt,
            user_id=user_id,
            structural=structural,
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            anchor_similarity=anchor_match.max_similarity,
            nearest_anchor_id=anchor_match.nearest_anchor_id,
            feature_vector=feature_block,
            feature_names=names,
            domain=domain
        )
