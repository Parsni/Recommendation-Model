"""Data preparation: cleaning, feature typing, sparse interaction matrix.

Domain-specific transformations must NOT live here; this module only builds
the generic structures that recommenders and the analytics layer consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.loader import RawData
from src.utils.helpers import load_config


@dataclass
class PreparedData:
    """Immutable snapshot of a prepared dataset shared by all algorithms."""
    interactions: pd.DataFrame          # canonical columns, cleaned
    users: pd.DataFrame | None
    items: pd.DataFrame | None
    config: dict
    user_ids: list = field(default_factory=list)
    item_ids: list = field(default_factory=list)
    matrix: sparse.csr_matrix | None = None      # users x items, values
    has_values: bool = False
    has_timestamps: bool = False
    item_text_fields: list = field(default_factory=list)
    item_categorical_fields: list = field(default_factory=list)
    item_numerical_fields: list = field(default_factory=list)
    user_categorical_fields: list = field(default_factory=list)
    user_numerical_fields: list = field(default_factory=list)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)

    @property
    def density(self) -> float:
        if self.matrix is None or self.matrix.nnz == 0 or not self.item_ids:
            return 0.0
        return self.matrix.nnz / (self.n_users * self.n_items)

    def index_of(self, user_id=None, item_id=None):
        if user_id is not None:
            return self._uidx.get(str(user_id))
        if item_id is not None:
            return self._iidx.get(str(item_id))
        return None

    def item_metadata(self, item_id: str) -> dict:
        if self.items is None:
            return {}
        row = self.items[self.items["item_id"] == str(item_id)]
        return {} if row.empty else row.iloc[0].to_dict()

    def seen_items(self, user_id: str) -> set:
        return self._seen.get(str(user_id), set())

    def user_history(self, user_id: str) -> pd.DataFrame:
        hist = self.interactions[self.interactions["user_id"] == str(user_id)]
        if self.has_timestamps and not hist.empty:
            hist = hist.sort_values("timestamp")
        return hist

    def _build_indices(self):
        self._uidx = {u: i for i, u in enumerate(self.user_ids)}
        self._iidx = {it: i for i, it in enumerate(self.item_ids)}
        self._seen = {}
        for uid, grp in self.interactions.groupby("user_id"):
            self._seen[str(uid)] = set(grp["item_id"].astype(str))


def _infer_field_types(df: pd.DataFrame, id_col: str, config_map: dict) -> tuple[list, list, list]:
    """Classify metadata columns into categorical / numerical / text."""
    cat = [c for c in config_map.get("categorical_fields", []) if c in df.columns]
    num = [c for c in config_map.get("numerical_fields", []) if c in df.columns]
    txt = [c for c in config_map.get("text_fields", []) if c in df.columns]
    known = {id_col, *cat, *num, *txt}
    for c in df.columns:
        if c in known:
            continue
        series = df[c]
        nonnull = series.notna().sum()
        if nonnull == 0:
            continue
        as_num = pd.to_numeric(series, errors="coerce")
        nunique = series.dropna().nunique()
        if as_num.notna().mean() > 0.9 and nunique > 12:
            num.append(c)
        elif nunique <= max(20, 0.05 * nonnull):
            cat.append(c)
        elif series.astype(str).str.len().mean() > 40:
            txt.append(c)
        else:
            cat.append(c)
    return cat, num, txt


def prepare(raw: RawData, config: dict | None = None) -> PreparedData:
    """Clean interactions further and build the shared user-item structures."""
    config = config or load_config()
    prep_cfg = config.get("preprocessing", {})

    inter = raw.interactions.copy()
    dedup_key = [k for k in prep_cfg.get("deduplicate_key", ["user_id", "item_id", "timestamp"])
                 if k in inter.columns] or ["user_id", "item_id", "timestamp"]
    if dedup_key and inter.duplicated(subset=dedup_key).any():
        agg = {"interaction_value": "first"} if "interaction_value" in inter.columns else {}
        inter = inter.groupby([c for c in dedup_key if c in inter.columns], as_index=False,
                               sort=False).agg(agg) if agg else inter.drop_duplicates(subset=dedup_key)

    inter = inter.reset_index(drop=True)
    has_values = "interaction_value" in inter.columns and inter["interaction_value"].notna().any()
    has_ts = "timestamp" in inter.columns and inter["timestamp"].notna().any()

    user_ids = sorted(inter["user_id"].astype(str).unique())
    item_ids = []
    if raw.items is not None:
        item_ids = sorted(set(item_ids) | set(raw.items["item_id"].astype(str)))
    interaction_items = sorted(inter["item_id"].astype(str).unique())
    item_ids = sorted(set(item_ids) | set(interaction_items))

    data = PreparedData(interactions=inter, users=raw.users, items=raw.items, config=config,
                       user_ids=user_ids, item_ids=item_ids, has_values=has_values,
                       has_timestamps=has_ts)

    if raw.items is not None:
        cat, num, txt = _infer_field_types(raw.items, "item_id", config["column_mapping"].get("items", {}))
        data.item_categorical_fields, data.item_numerical_fields, data.item_text_fields = cat, num, txt
    if raw.users is not None:
        cat, num, _ = _infer_field_types(raw.users, "user_id", config["column_mapping"].get("users", {}))
        data.user_categorical_fields, data.user_numerical_fields = cat, num

    # Sparse user-item matrix (only interactions with usable values / presence)
    if data.n_users and data.n_items:
        uidx = _index_map(user_ids)
        iidx = _index_map(item_ids)
        rows = inter["user_id"].map(uidx).to_numpy(dtype=float)
        cols = inter["item_id"].map(iidx).to_numpy(dtype=float)
        vals = (inter["interaction_value"].fillna(1.0).to_numpy(dtype=float) if has_values
                else np.ones(len(inter)))
        mask = ~(np.isnan(rows) | np.isnan(cols))
        matrix = sparse.csr_matrix(
            (vals[mask], (rows[mask].astype(int), cols[mask].astype(int))),
            shape=(len(user_ids), len(item_ids)))
        agg_fn = prep_cfg.get("value_aggregation", "mean")
        data.matrix = _aggregate_duplicates(matrix, agg_fn)
    data._build_indices()
    return data


def _index_map(ids: list) -> dict:
    return {i: n for n, i in enumerate(ids)}


def _aggregate_duplicates(matrix: sparse.csr_matrix, how: str) -> sparse.csr_matrix:
    """Apply 'mean' or 'sum' to repeated (user,item) entries."""
    coo = matrix.tocoo()
    if how == "sum" or coo.data.size == 0:
        return sparse.csr_matrix((coo.data, (coo.row, coo.col)), shape=matrix.shape)
    # count entries per cell to divide for mean
    keys = coo.row.astype(np.int64) * matrix.shape[1] + coo.col
    counts = pd.Series(1, index=keys).groupby(level=0).transform("size").to_numpy()
    data = coo.data / counts
    return sparse.csr_matrix((data, (coo.row, coo.col)), shape=matrix.shape)
