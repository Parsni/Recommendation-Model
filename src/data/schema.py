"""Generic data contract: column mapping, schema validation, validation reports.

The contract expects (after mapping) at minimum user_id and item_id in the
interaction table. All other fields are optional. Original data is never
mutated: validation works on copies and reports issues instead of silently
dropping rows unless a row is unusable (missing id), in which case it is
recorded in the validation report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CANONICAL_INTERACTION_COLS = ["user_id", "item_id", "interaction_value", "timestamp"]


@dataclass
class ValidationReport:
    dataset: str
    rows_in: int = 0
    rows_usable: int = 0
    unknown_columns: list = field(default_factory=list)
    missing_required_columns: list = field(default_factory=list)
    malformed_rows: list = field(default_factory=list)  # row indices dropped as unusable
    missing_value_counts: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    dtype_issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required_columns

    def summary_lines(self) -> list[str]:
        lines = [f"[{self.dataset}] rows read: {self.rows_in}, usable: {self.rows_usable}"]
        if self.missing_required_columns:
            lines.append(f"  ERROR missing required columns: {self.missing_required_columns}")
        if self.unknown_columns:
            lines.append(f"  extra columns kept as metadata candidates: {self.unknown_columns}")
        if self.malformed_rows:
            lines.append(f"  dropped {len(self.malformed_rows)} malformed rows "
                         f"(missing/blank identifiers): {self.malformed_rows[:10]}")
        if self.missing_value_counts:
            lines.append(f"  missing values: {self.missing_value_counts}")
        if self.duplicate_rows:
            lines.append(f"  exact duplicate rows: {self.duplicate_rows}")
        if self.dtype_issues:
            lines.append(f"  dtype issues: {self.dtype_issues}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return lines


def apply_mapping(df: pd.DataFrame, mapping: dict, dataset: str) -> pd.DataFrame:
    """Rename source columns to canonical names where a mapping is provided."""
    rename = {src: canon for canon, src in mapping.items()
              if canon in CANONICAL_INTERACTION_COLS and isinstance(src, str) and src in df.columns}
    if dataset == "interactions":
        for canon in CANONICAL_INTERACTION_COLS:
            if canon in mapping and mapping[canon] not in df.columns and mapping[canon] is not None:
                pass  # reported later as unknown/absent; no silent fill
    return df.rename(columns=rename)


def validate_interactions(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean the interaction table on a copy; return (clean_df, report)."""
    work = df.copy()
    work = apply_mapping(work, mapping, "interactions")
    report = ValidationReport(dataset="interactions", rows_in=len(work))

    required = [c for c in ("user_id", "item_id") if mapping.get(c, c) is not None]
    missing = [c for c in required if c not in work.columns]
    report.missing_required_columns = missing
    if missing:
        return work, report

    report.unknown_columns = [c for c in work.columns if c not in CANONICAL_INTERACTION_COLS]

    # IDs: drop rows with missing identifiers (recorded, not silent)
    id_mask = work["user_id"].isna() | work["item_id"].isna()
    blank_mask = work["user_id"].astype(str).str.strip().isin(("", "nan")) | \
        work["item_id"].astype(str).str.strip().isin(("", "nan"))
    bad = id_mask | blank_mask
    report.malformed_rows = [int(i) for i in work.index[bad][:1000]]
    work = work.loc[~bad].copy()
    for c in ("user_id", "item_id"):
        work[c] = work[c].astype(str).str.strip()

    # interaction value: coerce numeric, count unparseable as missing
    if "interaction_value" in work.columns:
        work["interaction_value"] = pd.to_numeric(work["interaction_value"], errors="coerce")
        n_bad = int(work["interaction_value"].isna().sum())
        if n_bad:
            report.dtype_issues.append(f"interaction_value: {n_bad} non-numeric values coerced to NaN")
    else:
        report.warnings.append("no interaction_value column; algorithms will treat "
                               "every recorded interaction as equally positive")

    # timestamp: parse to datetime when present
    if "timestamp" in work.columns:
        parsed = pd.to_datetime(work["timestamp"], errors="coerce")
        n_bad = int(parsed.isna().sum() - work["timestamp"].isna().sum())
        work["timestamp"] = parsed
        if n_bad > 0:
            report.dtype_issues.append(f"timestamp: {n_bad} unparseable values set to NaT")
        if parsed.notna().sum() == 0:
            report.warnings.append("timestamp column carries no usable times; "
                                   "chronological splitting will be unavailable")
    else:
        report.warnings.append("no timestamp column; chronological evaluation splits "
                               "fall back to user/group splits")

    report.duplicate_rows = int(work.duplicated().sum())
    report.missing_value_counts = {c: int(n) for c, n in work.isna().sum().items() if n > 0}
    report.rows_usable = len(work)
    return work, report


def validate_metadata(df: pd.DataFrame, mapping: dict, id_canon: str) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate a users or items metadata table."""
    work = df.copy()
    id_src = mapping.get(id_canon, id_canon)
    if id_src in work.columns and id_src != id_canon:
        work = work.rename(columns={id_src: id_canon})
    report = ValidationReport(dataset=mapping.get("_dataset", id_canon + "s"), rows_in=len(work))
    if id_canon not in work.columns:
        report.missing_required_columns = [id_canon]
        return work, report
    bad = work[id_canon].isna() | work[id_canon].astype(str).str.strip().isin(("", "nan"))
    report.malformed_rows = [int(i) for i in work.index[bad][:1000]]
    work = work.loc[~bad].copy()
    work[id_canon] = work[id_canon].astype(str).str.strip()
    report.duplicate_rows = int(work.duplicated(subset=[id_canon]).sum())
    if report.duplicate_rows:
        report.warnings.append(f"duplicate {id_canon} rows present; first occurrence will be used")
        work = work.drop_duplicates(subset=[id_canon], keep="first")
    report.missing_value_counts = {c: int(n) for c, n in work.isna().sum().items() if n > 0}
    report.rows_usable = len(work)
    return work, report
