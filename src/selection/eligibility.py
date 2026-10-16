"""Automatic algorithm eligibility selection.

Checks data availability, volumes, sparsity, metadata presence, basket
suitability and computational feasibility, then instantiates only the
algorithms that can operate. Popularity is always retained as the baseline
wherever any recommendation candidate exists.
"""
from __future__ import annotations

from src.features.preprocess import PreparedData
from src.recommenders import RECOMMENDER_CLASSES


def select_eligible(data: PreparedData, algorithm_names: list | None = None) -> dict:
    """Return {"eligible": [recommender,...], "excluded": {name: reason}}."""
    names = algorithm_names or list(RECOMMENDER_CLASSES)
    eligible, excluded = [], {}
    for name in names:
        cls = RECOMMENDER_CLASSES.get(name)
        if cls is None:
            excluded[name] = "unknown algorithm name (not in registry)"
            continue
        rec = cls(data.config)
        try:
            suit = rec.check_suitability(data)
        except Exception as exc:  # a failing check excludes, never crashes the run
            excluded[name] = f"suitability check raised {type(exc).__name__}: {exc}"
            continue
        if suit.suitable:
            eligible.append(rec)
        else:
            excluded[name] = suit.reason
    if data.n_interactions > 0 and "popularity" not in {r.name for r in eligible}:
        # baseline must survive if candidates exist at all
        from src.recommenders.popularity import PopularityRecommender
        eligible.insert(0, PopularityRecommender(data.config))
        excluded.pop("popularity", None)
    return {"eligible": eligible, "excluded": excluded}
