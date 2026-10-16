"""Data-quality reporting and basic descriptive statistics on the interaction log."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocess import PreparedData


def quality_report(data: PreparedData) -> dict:
    inter = data.interactions
    report = {
        "n_users": data.n_users,
        "n_items_in_interactions": int(inter["item_id"].nunique()),
        "n_items_in_catalog": data.n_items,
        "n_interactions": data.n_interactions,
        "matrix_density": round(data.density, 4),
        "sparsity": round(1.0 - data.density, 4),
        "has_interaction_values": data.has_values,
        "has_timestamps": data.has_timestamps,
        "has_item_metadata": data.items is not None,
        "has_user_metadata": data.users is not None,
    }
    per_user = inter.groupby("user_id").size()
    report["interactions_per_user"] = {
        "min": int(per_user.min()) if len(per_user) else 0,
        "mean": round(float(per_user.mean()), 2) if len(per_user) else 0.0,
        "median": float(per_user.median()) if len(per_user) else 0.0,
        "max": int(per_user.max()) if len(per_user) else 0,
    }
    report["users_with_few_interactions"] = int((per_user < 3).sum())
    if data.has_values:
        vals = inter["interaction_value"].dropna()
        report["interaction_value"] = {
            "count": int(vals.size),
            "missing": int(inter["interaction_value"].isna().sum()),
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "mean": round(float(vals.mean()), 4) if vals.size else None,
            "std": round(float(vals.std(ddof=1)), 4) if vals.size > 1 else None,
        }
    if data.items is not None:
        seen = set(inter["item_id"].astype(str))
        catalog = set(data.items["item_id"].astype(str))
        report["items_without_interactions"] = sorted(catalog - seen)[:50]
        report["interacted_items_missing_metadata"] = sorted(seen - catalog)[:50]
    if data.users is not None:
        seen_u = set(inter["user_id"].astype(str))
        report["users_without_interactions"] = sorted(set(data.users["user_id"].astype(str)) - seen_u)[:50]
    return report


def print_quality(report: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in report.items())


def value_distribution(data: PreparedData, bins: int = 10) -> pd.DataFrame:
    """Simple frequency table of interaction values (or per-user counts)."""
    if data.has_values:
        vals = data.interactions["interaction_value"].dropna()
        if vals.nunique() <= bins:
            counts = vals.value_counts().sort_index()
            return pd.DataFrame({"value": counts.index, "count": counts.to_numpy()})
        hist, edges = np.histogram(vals, bins=bins)
        return pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist})
    per_user = data.interactions.groupby("user_id").size()
    hist, edges = np.histogram(per_user, bins=min(bins, max(1, int(per_user.max()) - per_user.min() + 1)))
    return pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist})
