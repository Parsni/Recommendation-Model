"""CSV ingestion with configurable column mappings. Originals stay untouched."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.schema import ValidationReport, validate_interactions, validate_metadata
from src.utils.helpers import load_config, PROJECT_ROOT


@dataclass
class RawData:
    interactions: pd.DataFrame
    users: pd.DataFrame | None
    items: pd.DataFrame | None
    reports: list[ValidationReport] = field(default_factory=list)


def read_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported input format '{path.suffix}' - only CSV is supported.")
    return pd.read_csv(path, dtype=object)


def load_dataset(interactions_path=None, users_path=None, items_path=None,
                 config: dict | None = None) -> RawData:
    """Load CSVs and validate them against the configured column mapping."""
    config = config or load_config()
    mapping = config["column_mapping"]
    reports: list[ValidationReport] = []

    if interactions_path is None:
        interactions_path = PROJECT_ROOT / "data" / "synthetic_demo" / "interactions.csv"
    inter_df = read_csv(interactions_path)
    inter_clean, rep = validate_interactions(inter_df, mapping["interactions"])
    reports.append(rep)

    users_df = None
    if users_path is not None and Path(users_path).exists():
        m = dict(mapping.get("users", {})); m["_dataset"] = "users"
        users_df, urep = validate_metadata(read_csv(users_path), m, "user_id")
        reports.append(urep)

    items_df = None
    if items_path is not None and Path(items_path).exists():
        m = dict(mapping.get("items", {})); m["_dataset"] = "items"
        items_df, irep = validate_metadata(read_csv(items_path), m, "item_id")
        reports.append(irep)

    return RawData(interactions=inter_clean, users=users_df, items=items_df, reports=reports)
