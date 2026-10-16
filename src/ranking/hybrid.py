"""Hybrid ranking engine: aggregate, normalize and rank algorithm outputs.

Scores produced here are NORMALIZED RANKING SCORES in [0, 1] per source.
They are comparable across sources only for ordering purposes and are
explicitly NOT probabilities of engagement.
"""
from __future__ import annotations

from src.recommenders.base import BaseRecommender, Recommendation


def minmax_normalize(scores: dict) -> dict:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}  # single-candidate or flat source: neutral max
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRanker:
    def __init__(self, recommenders: list[BaseRecommender], config: dict):
        self.recommenders = [r for r in recommenders if r.is_fitted]
        h = config.get("hybrid", {})
        self.weights = h.get("weights", {})
        self.normalize = h.get("normalize", "minmax")
        self.diversify_default = bool(h.get("diversify", False))
        self.diversity_field = None
        self.diversity_lambda = float(h.get("diversity_lambda", 0.35))
        data = self.recommenders[0].data if self.recommenders else None
        if data is not None and data.item_categorical_fields:
            self.diversity_field = data.item_categorical_fields[0]

    # ------------------------------------------------------------------
    def source_scores(self, user_id: str, candidate_ids=None, pool: int = 24) -> dict:
        """Per-algorithm raw recommendation lists for one user."""
        out = {}
        for rec in self.recommenders:
            try:
                rs = rec.recommend(user_id, top_k=pool, exclude_seen=False,
                                   candidate_ids=candidate_ids)
            except Exception:
                rs = []  # a failing source contributes nothing; engine logs separately
            out[rec.name] = rs
        return out

    def rank(self, user_id: str, top_k: int = 10, exclude_seen: bool = True,
             candidate_ids: list | None = None, diversify: bool | None = None,
             sources_map: dict | None = None) -> list[Recommendation]:
        sources_map = sources_map or self.source_scores(user_id, candidate_ids)
        valid = set()
        for rec in self.recommenders:
            if hasattr(rec, "data"):
                valid = set(rec.data.item_ids)
                break
        seen = self.recommenders[0].data.seen_items(str(user_id)) if self.recommenders else set()
        contrib = {}   # item -> {algo: (normalized_score, Recommendation)}
        for name, rs in sources_map.items():
            # candidate-relative normalization: filter ineligible items BEFORE
            # normalizing so excluded items' scores cannot compress candidates
            eligible_rs = [r for r in rs
                           if (r.item_id in valid or not valid)
                           and (not exclude_seen or r.item_id not in seen)]
            raw = {r.item_id: r.score for r in eligible_rs}
            norm = minmax_normalize(raw) if self.normalize == "minmax" else raw
            w = float(self.weights.get(name, 1.0))
            for r in eligible_rs:
                if r.item_id not in norm:
                    continue
                entry = contrib.setdefault(r.item_id, {})
                entry[name] = (norm[r.item_id], r, w)
        finals = {i: sum(n * w for n, _, w in e.values()) for i, e in contrib.items()}
        ordered = sorted(finals, key=lambda i: (-finals[i], i))
        diversify = self.diversify_default if diversify is None else diversify
        if diversify and self.diversity_field and self.recommenders:
            ordered = self._diversify(ordered, finals, contrib)
        out = []
        for rank, item in enumerate(ordered[:top_k], start=1):
            parts = contrib[item]
            total_w = sum(w for _, _, w in parts.values())
            display = finals[item] / total_w if total_w > 1.009 else finals[item]
            src_names = sorted(parts, key=lambda n: -parts[n][0])
            notes = [f"{n}: {r.explanation}" for n, (_, r, _) in sorted(
                parts.items(), key=lambda kv: -kv[1][0])]
            limits = sorted({r.limitations for _, r, _ in parts.values() if r.limitations})
            out.append(Recommendation(
                item_id=item, score=float(display),
                score_kind="normalized_ranking_score", rank=rank, sources=src_names,
                explanation=" | ".join(notes),
                limitations=("Weighted blend of per-source normalized ranking scores "
                             "[0,1]; not a probability. " + " ".join(limits))
            ))
        return out

    def _diversify(self, ordered, finals, contrib) -> list:
        """Greedy re-ranking: penalize repeating the same categorical value."""
        data = self.recommenders[0].data
        cat = {i: str(data.item_metadata(i).get(self.diversity_field, "")) for i in ordered}
        chosen, remaining = [], list(ordered)
        while remaining:
            best = max(remaining, key=lambda i: finals[i] * (
                1 - self.diversity_lambda *
                sum(1 for c in chosen if cat[c] == cat[i]) / max(len(chosen), 1)))
            chosen.append(best)
            remaining.remove(best)
        return chosen
