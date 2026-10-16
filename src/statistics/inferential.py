"""Inferential-statistics utilities: confidence intervals, hypothesis tests,
effect sizes. Test choice must follow variable type, sampling unit and
assumptions - these helpers state which test they ran and why, and flag
violations instead of hiding them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

Z95 = 1.959963984540054


def mean_ci(sample, confidence: float = 0.95) -> dict:
    s = pd.to_numeric(pd.Series(sample), errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return {"error": f"need >= 2 observations, got {n}"}
    mean, se = float(s.mean()), float(s.std(ddof=1) / np.sqrt(n))
    t = sps.t.ppf((1 + confidence) / 2, df=n - 1)
    return {"n": n, "mean": round(mean, 4), "se": round(se, 4),
            "ci_low": round(mean - t * se, 4), "ci_high": round(mean + t * se, 4),
            "confidence": confidence, "distribution": "t (n small) / normal (n large)"}


def proportion_ci(successes: int, n: int, confidence: float = 0.95) -> dict:
    if n == 0:
        return {"error": "n must be positive"}
    p = successes / n
    z = sps.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / n)
    return {"n": n, "p": round(p, 4),
            "ci_low": round(max(0.0, p - z * se), 4),
            "ci_high": round(min(1.0, p + z * se), 4),
            "method": "normal approximation (Wald); check n*p>=5 and n*(1-p)>=5",
            "assumption_met": bool(n * p >= 5 and n * (1 - p) >= 5)}


def two_sample_ttest(a, b, equal_var: bool | None = None) -> dict:
    sa = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    sb = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(sa) < 2 or len(sb) < 2:
        return {"error": "both groups need >= 2 observations"}
    if equal_var is None:
        lev_p = sps.levene(sa, sb).pvalue
        equal_var = lev_p > 0.05
    t, p = sps.ttest_ind(sa, sb, equal_var=equal_var)
    n1, n2 = len(sa), len(sb)
    pooled = np.sqrt(((n1 - 1) * sa.var(ddof=1) + (n2 - 1) * sb.var(ddof=1)) / (n1 + n2 - 2))
    d = (sa.mean() - sb.mean()) / pooled if pooled > 0 else 0.0
    return {"test": "student t" if equal_var else "welch t",
            "equal_variance_assumption": "levene p={:.3f}".format(lev_p),
            "t": round(float(t), 4), "p": float(p),
            "cohens_d": round(float(d), 4),
            "n1": n1, "n2": n2,
            "interpretation": ("p<0.05: mean difference unlikely under equal means"
                               if p < 0.05 else
                               "no significant mean difference detected at 0.05 "
                               "(not proof of equality)")}


def mannwhitney(a, b) -> dict:
    sa = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    sb = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(sa) < 3 or len(sb) < 3:
        return {"error": "both groups need >= 3 observations"}
    u, p = sps.mannwhitneyu(sa, sb, alternative="two-sided")
    n1, n2 = len(sa), len(sb)
    r = 1 - 2 * u / (n1 * n2)
    return {"test": "mann-whitney u", "U": round(float(u), 4), "p": float(p),
            "rank_biserial_r": round(float(abs(r)), 4),
            "use_case": "ordinal/non-normal/small samples; compares distributions, not means"}


def paired_ttest(a, b) -> dict:
    sa = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    sb = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    n = min(len(sa), len(sb))
    if n < 3:
        return {"error": "need >= 3 matched pairs"}
    t, p = sps.ttest_rel(sa.iloc[:n], sb.iloc[:n])
    diff = sa.iloc[:n].to_numpy() - sb.iloc[:n].to_numpy()
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    return {"test": "paired t", "t": round(float(t), 4), "p": float(p),
            "mean_diff": round(float(diff.mean()), 4), "cohens_dz": round(float(d), 4), "n": int(n)}


def one_sample_ttest(sample, popmean: float) -> dict:
    s = pd.to_numeric(pd.Series(sample), errors="coerce").dropna()
    if len(s) < 3:
        return {"error": "need >= 3 observations"}
    t, p = sps.ttest_1samp(s, popmean)
    return {"test": "one-sample t", "t": round(float(t), 4), "p": float(p),
            "n": len(s), "reference_value": popmean}


def chi_square_gof(observed: dict) -> dict:
    labels = list(observed)
    obs = np.array([observed[l] for l in labels], dtype=float)
    if obs.sum() <= 0 or len(obs) < 2:
        return {"error": "need >= 2 categories with positive total count"}
    exp = np.full_like(obs, obs.mean())
    chi2, p = sps.chisquare(obs, exp)
    return {"test": "chi-square goodness of fit (uniform expectation)",
            "chi2": round(float(chi2), 4), "p": float(p), "dof": len(obs) - 1,
            "expected_each": round(float(exp[0]), 2)}
