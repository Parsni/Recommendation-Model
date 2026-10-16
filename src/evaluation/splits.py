"""Evaluation splits with leakage prevention.

- chronological_last_out: per user sorted by timestamp, last interaction is
  test, second-last is validation, the rest is train. Users need at least
  min_train + 2 interactions. Training data therefore never contains any
  event later than the user's validation/test event.
- user_holdout: for datasets without usable timestamps - hold out entire
  users (their interactions removed from train) and evaluate cold-start
  behaviour against their recorded items.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import RawData
from src.features.preprocess import PreparedData, prepare


def make_data(base: PreparedData, interactions: pd.DataFrame) -> PreparedData:
    """Rebuild a PreparedData with the same metadata but a given interaction set."""
    raw = RawData(interactions=interactions.copy(), users=base.users, items=base.items)
    return prepare(raw, base.config)


def chronological_last_out(data: PreparedData, min_train: int = 3) -> dict:
    if not data.has_timestamps:
        raise ValueError("dataset has no usable timestamps for a chronological split")
    train_rows, val, test = [], {}, {}
    for uid, hist in data.interactions.groupby("user_id"):
        hist = hist.sort_values("timestamp")
        if len(hist) < min_train + 2:
            continue
        train_rows.extend(hist.iloc[:-2].to_dict("records"))
        val[str(uid)] = str(hist.iloc[-2]["item_id"])
        test[str(uid)] = str(hist.iloc[-1]["item_id"])
    train_df = pd.DataFrame(train_rows, columns=data.interactions.columns)
    return {"train": make_data(data, train_df), "val": val, "test": test}


def user_holdout(data: PreparedData, hold_users: list, min_train: int = 3) -> dict:
    """Cold-start evaluation: entire users removed from training."""
    hold = {str(u) for u in hold_users}
    train_df = data.interactions[~data.interactions["user_id"].astype(str).isin(hold)]
    val, test = {}, {}
    for uid in hold:
        hist = data.interactions[data.interactions["user_id"].astype(str) == uid]
        if len(hist) >= min_train:
            test[uid] = set(hist["item_id"].astype(str))
    return {"train": make_data(data, train_df), "val": val, "test": test}
