"""Recommendation evaluation metrics. All functions are pure and tested
against deterministic examples (see tests/test_evaluation.py).

Definitions:
- ranked_items : list of recommended item ids, best first.
- relevant     : set of held-out item ids (binary relevance) OR dict
                 {item: gain} for graded relevance (NDCG only).
"""
from __future__ import annotations

import math


def precision_at_k(ranked_items: list, relevant: set, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked_items[:k]
    if not top:
        return 0.0
    return len([i for i in top if i in relevant]) / k


def recall_at_k(ranked_items: list, relevant: set, k: int) -> float:
    if not relevant:
        raise ValueError("recall needs at least one relevant item")
    top = ranked_items[:k]
    return len([i for i in top if i in relevant]) / len(relevant)


def hit_rate_at_k(ranked_items: list, relevant: set, k: int) -> float:
    return 1.0 if any(i in relevant for i in ranked_items[:k]) else 0.0


def ndcg_at_k(ranked_items: list, relevant, k: int) -> float:
    """Graded or binary NDCG@K. relevant: set or {item: gain>=1} dict."""
    if isinstance(relevant, dict):
        gains = relevant
    else:
        gains = {i: 1 for i in relevant}
    top = ranked_items[:k]
    dcg = sum(gains.get(i, 0) / math.log2(pos + 2) for pos, i in enumerate(top))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(pos + 2) for pos, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recommended: list[list], catalog_size: int) -> float:
    """Share of the catalog that appears in at least one recommendation list."""
    if catalog_size <= 0:
        return 0.0
    distinct = {i for lst in all_recommended for i in lst}
    return len(distinct) / catalog_size


def intra_list_diversity(all_recommended: list[list], catalog_size: int) -> float:
    """Mean 1 - (pairwise same-item similarity proxy). Here: distinct share
    within lists averaged - simple, honest proxy for how varied lists are."""
    if not all_recommended or catalog_size <= 0:
        return 0.0
    scores = []
    for lst in all_recommended:
        if len(lst) > 1:
            scores.append(len(set(lst)) / len(lst))
        elif len(lst) == 1:
            scores.append(1.0)
    return sum(scores) / len(scores) if scores else 0.0


def shannon_entropy(items: list) -> float:
    """Entropy of item frequencies across recommendation lists (coverage of
    exposure). Higher = more even distribution."""
    if not items:
        return 0.0
    from collections import Counter
    counts = Counter(items)
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values())
