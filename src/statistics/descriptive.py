"""Reusable descriptive-statistics utilities (domain-agnostic).

These operate on plain DataFrames so they can later serve any chosen
Business Analytics problem. Functions report what the data actually shows
and refuse to over-claim (e.g. normality is flagged, not assumed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def describe_numeric(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "note": "no numeric values"}
    n = len(s)
    mean = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    return {
        "count": n, "missing": int(series.isna().sum()),
        "mean": round(mean, 4), "std": round(sd, 4), "sem": round(se, 4),
        "min": round(float(s.min()), 4), "25%": round(float(s.quantile(.25)), 4),
        "50%": round(float(s.median()), 4), "75%": round(float(s.quantile(.75)), 4),
        "max": round(float(s.max()), 4),
        "skew": round(float(s.skew()), 4) if n > 2 else None,
        "kurtosis": round(float(s.kurtosis()), 4) if n > 3 else None,
    }


def describe_categorical(series: pd.Series) -> pd.DataFrame:
    s = series.dropna().astype(str)
    if s.empty:
        return pd.DataFrame(columns=["value", "count", "share"])
    counts = s.value_counts()
    return pd.DataFrame({"value": counts.index, "count": counts.to_numpy(),
                         "share": (counts / counts.sum()).round(4).to_numpy()})


def missing_data_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = [{"column": c, "missing": int(df[c].isna().sum()),
             "missing_pct": round(100.0 * df[c].isna().sum() / max(n, 1), 2)}
            for c in df.columns]
    return pd.DataFrame(rows)


def outlier_diagnostics(series: pd.Series, method: str = "iqr") -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return {"method": method, "n_outliers": 0, "note": "series too short"}
    if method == "iqr":
        q1, q3 = s.quantile(.25), s.quantile(.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    elif method == "zscore":
        m, sd = s.mean(), s.std(ddof=1)
        lo, hi = m - 3 * sd, m + 3 * sd
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
    out = s[(s < lo) | (s > hi)]
    return {"method": method, "bounds": [round(float(lo), 4), round(float(hi), 4)],
            "n_outliers": int(len(out)), "outlier_pct": round(100 * len(out) / len(s), 2)}


def normality_note(series: pd.Series, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk used only as a diagnostic flag, never a gate decision."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return {"test": "skipped", "reason": f"n={len(s)} < 8; normality not assessed"}
    stat, p = sps.shapiro(s.sample(min(len(s), 5000), random_state=42))
    return {"test": "shapiro_wilk", "W": round(float(stat), 4), "p": round(float(p), 4),
            "flag": ("possibly non-normal (p<%.2f); prefer rank-based or "
                     "non-parametric methods" % alpha) if p < alpha else "no strong "
            "evidence against normality (not proof of normality)"}


def correlation_table(df: pd.DataFrame, columns: list, method: str = "auto") -> pd.DataFrame:
    """Pearson for numeric-numeric, Spearman when monotonic-but-nonlinear or
    ordinal suspected; pairs of categorical variables should use
    categorical_association instead."""
    data = df[columns].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            pair = data[[a, b]].dropna()
            if len(pair) < 3:
                continue
            m = method
            if m == "auto":
                m = "spearman" if pair.nunique().max() <= 10 else "pearson"
            r, p = sps.spearmanr(pair[a], pair[b]) if m == "spearman" \
                else sps.pearsonr(pair[a], pair[b])
            rows.append({"var_a": a, "var_b": b, "method": m,
                         "r": round(float(r), 4), "p": round(float(p), 6),
                         "n": len(pair)})
    return pd.DataFrame(rows).sort_values("r", ascending=False, key=abs) \
        if rows else pd.DataFrame(columns=["var_a", "var_b", "method", "r", "p", "n"])


def categorical_association(df: pd.DataFrame, a: str, b: str) -> dict:
    """Chi-square test of independence + Cramer's V effect size."""
    ct = pd.crosstab(df[a], df[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"error": "need at least two levels in each variable"}
    chi2, p, dof, exp = sps.chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    small = (exp < 5).mean()
    note = ("expected counts < 5 in {:.0%} of cells: chi-square may be "
            "unreliable; consider Fisher-type exact tests or merging levels"
            .format(small)) if small > 0.2 else None
    return {"chi2": round(float(chi2), 4), "dof": int(dof),
            "p": float(p), "cramers_v": round(float(v), 4), "n": int(n), "caveat": note}
