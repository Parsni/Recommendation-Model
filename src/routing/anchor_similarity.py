"""Layer 2 Track B: Visual/Semantic Similarity Matrix (Anchor Base).

Measures spatial proximity of incoming prompt against an index of pre-verified
Production Anchor Baseline Prompts via Cosine Similarity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class AnchorMatchResult:
    """Evaluation result against Production Anchor Baselines."""
    max_similarity: float
    nearest_anchor_id: str
    nearest_category: str
    nearest_text: str
    similarity_vector: np.ndarray


class AnchorSimilarityMatrix:
    """Calculates Cosine Similarity against Production Anchor Baselines."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.anchor_df: Optional[pd.DataFrame] = None
        self.anchor_matrix: Optional[np.ndarray] = None
        self.is_indexed = False

    def build_index(self, anchors_df: pd.DataFrame) -> "AnchorSimilarityMatrix":
        """Index pre-verified enterprise anchor prompts."""
        if anchors_df.empty or "prompt_text" not in anchors_df.columns:
            raise ValueError("Anchor dataframe must contain 'prompt_text' column.")

        self.anchor_df = anchors_df.copy().reset_index(drop=True)
        texts = self.anchor_df["prompt_text"].tolist()
        self.anchor_matrix = self.vectorizer.fit_transform(texts)
        self.is_indexed = True
        return self

    def compute_similarity(self, prompt: str) -> AnchorMatchResult:
        """Compute cosine similarity of prompt against indexed anchor baselines."""
        if not self.is_indexed or self.anchor_matrix is None or self.anchor_df is None:
            # Empty fallback
            return AnchorMatchResult(
                max_similarity=0.0,
                nearest_anchor_id="NONE",
                nearest_category="general",
                nearest_text="",
                similarity_vector=np.zeros(1)
            )

        query_vec = self.vectorizer.transform([prompt])
        sims = cosine_similarity(query_vec, self.anchor_matrix)[0]

        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])
        
        nearest_row = self.anchor_df.iloc[max_idx]
        anchor_id = str(nearest_row.get("anchor_id", f"ANC_{max_idx}"))
        category = str(nearest_row.get("category", "general"))
        text = str(nearest_row.get("prompt_text", ""))

        return AnchorMatchResult(
            max_similarity=round(max(0.0, min(1.0, max_sim)), 4),
            nearest_anchor_id=anchor_id,
            nearest_category=category,
            nearest_text=text,
            similarity_vector=sims
        )
