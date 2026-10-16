"""Layer 2 Track A: Unsupervised Clustering Module (K-Means Intent Segment).

Transforms text into spatial coordinates using TF-IDF and partitions the latent space
into categorical Intent Clusters (e.g. Coding, Creative, Reasoning/Math, Communication/Factual).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Tuple, Optional


DEFAULT_CLUSTER_LABELS = {
    0: "Coding & Technical Execution",
    1: "Creative & Narrative Synthesis",
    2: "Reasoning, Math & Logic",
    3: "Communication, General QA & Analysis",
}


class IntentClusterer:
    """Unsupervised K-Means Intent Segmentation Module."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=256,
            stop_words="english",
            sublinear_tf=True
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.is_fitted = False
        self.cluster_labels: Dict[int, str] = dict(DEFAULT_CLUSTER_LABELS)

    def fit(self, texts: List[str]) -> "IntentClusterer":
        """Fit TF-IDF vectorizer and KMeans on prompt corpus."""
        if not texts:
            raise ValueError("Cannot fit IntentClusterer on empty text list.")
        
        # Ensure we don't have fewer samples than clusters
        effective_k = min(self.n_clusters, len(texts))
        if effective_k != self.n_clusters:
            self.n_clusters = max(2, effective_k)
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.kmeans.fit(tfidf_matrix)
        self.is_fitted = True

        # Map cluster IDs to human-readable intent labels using top keywords
        self._name_clusters()
        return self

    def _name_clusters(self) -> None:
        """Inspect centroid top terms to assign human-readable intent names."""
        terms = np.array(self.vectorizer.get_feature_names_out())
        for idx, center in enumerate(self.kmeans.cluster_centers_):
            top_term_indices = center.argsort()[::-1][:3]
            top_terms = [terms[i] for i in top_term_indices if center[i] > 0]
            label_suffix = f" ({', '.join(top_terms)})" if top_terms else ""
            
            # Match keywords
            terms_joined = " ".join(top_terms).lower()
            if any(k in terms_joined for k in ["python", "function", "code", "sort", "script", "react"]):
                base = "Coding & Technical"
            elif any(k in terms_joined for k in ["story", "haiku", "poem", "scifi", "creative"]):
                base = "Creative & Narrative"
            elif any(k in terms_joined for k in ["integral", "math", "quantum", "solve", "probability"]):
                base = "Reasoning & Math"
            else:
                base = "General QA & Communication"
            self.cluster_labels[idx] = f"{base}{label_suffix}"

    def predict(self, text: str) -> Tuple[int, str, np.ndarray]:
        """Predict cluster ID, human label, and soft distance distribution.
        
        Returns:
            Tuple of (cluster_id, cluster_label, soft_cluster_probabilities)
        """
        if not self.is_fitted:
            # Fallback if invoked before explicit fit
            return 0, "Unfitted General Context", np.ones(self.n_clusters) / self.n_clusters

        vec = self.vectorizer.transform([text])
        cluster_id = int(self.kmeans.predict(vec)[0])
        cluster_name = self.cluster_labels.get(cluster_id, f"Intent Cluster #{cluster_id}")

        # Compute euclidean distance to each centroid and convert to soft distribution
        dists = self.kmeans.transform(vec)[0]  # distances to each center
        # Softmax of negative distances
        exp_neg = np.exp(-dists)
        soft_probs = exp_neg / np.maximum(np.sum(exp_neg), 1e-9)

        return cluster_id, cluster_name, soft_probs
