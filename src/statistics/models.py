"""Optional modelling utilities: linear regression, clustering/segmentation
and simple time-series forecasting. Each returns honest diagnostics and
degrades transparently when assumptions or data volume are insufficient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, KFold


def linear_regression(df: pd.DataFrame, y_col: str, x_cols: list) -> dict:
    data = df[[y_col, *x_cols]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(data)
    if n < max(10, 5 * len(x_cols)):
        return {"error": f"insufficient rows for regression (n={n}); need at least "
                         f"max(10, 5*num predictors)"}
    X, y = data[x_cols].to_numpy(), data[y_col].to_numpy()
    model = LinearRegression().fit(X, y)
    r2 = float(model.score(X, y))
    kf = KFold(n_splits=min(5, max(2, n // 4)), shuffle=True, random_state=42)
    cv = cross_val_score(model, X, y, scoring="r2", cv=kf)
    resid = y - model.predict(X)
    return {
        "coefficients": dict(zip(x_cols, np.round(model.coef_, 4))),
        "intercept": round(float(model.intercept_), 4),
        "r2_in_sample": round(r2, 4),
        "cv_r2_mean": round(float(cv.mean()), 4),
        "cv_r2_min": round(float(cv.min()), 4),
        "n": n,
        "residual_std": round(float(resid.std(ddof=len(x_cols) + 1)) if n > len(x_cols) + 2
                              else float(resid.std(ddof=1)), 4),
        "warning": ("in-sample R2 can be optimistic; compare with cross-validated R2; "
                    "linearity/independence/homoscedasticity not formally verified "
                    "against residual diagnostics"),
    }


def cluster_segments(df: pd.DataFrame, feature_cols: list, k_range: list | None = None,
                     seed: int = 42) -> dict:
    """KMeans segmentation with silhouette-based stability check.

    If the best silhouette is weak, the caller is told clustering is not
    defensible on this data instead of being handed unstable clusters.
    """
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if hasattr(X, "drop"):
        X = X.loc[:, ~X.columns.duplicated()] if X.columns.duplicated().any() else X
    n = len(X)
    k_range = k_range or [2, 3, 4]
    if n < max(20, 10 * max(k_range)):
        return {"error": f"n={n} too small for clustering over k={k_range}"}
    best = {"k": None, "labels": None, "silhouette": -2.0}
    for k in k_range:
        if n <= k:
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_) if len(set(km.labels_)) > 1 else -1.0
        if sil > best["silhouette"]:
            best = {"k": k, "labels": km.labels_, "silhouette": float(sil)}
    if best["k"] is None:
        return {"error": "no valid k for the data volume"}
    if best["silhouette"] < 0.25:
        return {
            "stable": False,
            "best_k": best["k"], "silhouette": round(best["silhouette"], 4),
            "fallback": ("silhouette below 0.25: clusters weak/unstable - prefer "
                         "transparent descriptive groups (e.g. by a single business "
                         "attribute) rather than presenting these as real segments"),
            "labels": None,
        }
    X = X.copy()
    X["cluster"] = best["labels"]
    profile = X.groupby("cluster").mean(numeric_only=True).round(3)
    return {"stable": True, "best_k": best["k"], "silhouette": round(best["silhouette"], 4),
            "cluster_sizes": X["cluster"].value_counts().sort_index().to_dict(),
            "centroid_profile": profile}


def forecast_series(values, periods_ahead: int = 3, window: int = 3) -> dict:
    """Moving-average and Holt (linear-trend exponential smoothing) baseline
    forecasts. Deliberately simple: no fitted seasonal model without more
    data, and the result says so."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(s) < max(6, 2 * window):
        return {"error": f"need >= {max(6, 2 * window)} observations, got {len(s)}"}
    ma = np.convolve(s, np.ones(window) / window, mode="valid")
    ma_fc = float(ma[-1])
    # Holt with fixed, defensible defaults (alpha=0.4, beta=0.2) - no tuning
    alpha, beta = 0.4, 0.2
    level, trend = s[0], 0.0
    for v in s[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    holt_fc = [level + h * trend for h in range(1, periods_ahead + 1)]
    resid = s[window:] - ma[:len(s) - window]
    rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else None
    return {
        "moving_average": {"window": window, "next_values": [round(ma_fc, 4)] * periods_ahead},
        "holt_linear_trend": {"alpha": alpha, "beta": beta,
                              "next_values": [round(v, 4) for v in holt_fc]},
        "in_sample_rmse_moving_avg": round(rmse, 4) if rmse is not None else None,
        "caveat": ("fixed-smoothing baselines; no seasonality modelled and no "
                   "out-of-sample backtest performed - treat as directional, "
                   "not certified forecasts"),
    }
