"""Statistical utility arithmetic vs hand-computed / scipy-reference values."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.statistics.descriptive import (describe_numeric, correlation_table,
                                        categorical_association, outlier_diagnostics)
from src.statistics.descriptive import missing_data_diagnostics
from src.statistics.inferential import (mean_ci, proportion_ci, two_sample_ttest,
                                        mannwhitney, paired_ttest, chi_square_gof)
from src.statistics.models import linear_regression, forecast_series, cluster_segments


def test_describe_numeric_hand_checked():
    d = describe_numeric(pd.Series([1, 2, 3, 4, None, 10]))
    assert d["count"] == 5 and d["missing"] == 1
    assert d["mean"] == pytest.approx(4.0)
    assert d["std"] == pytest.approx(float(pd.Series([1, 2, 3, 4, 10]).std(ddof=1)), 4)
    assert d["50%"] == 3.0


def test_mean_ci_contains_mean_and_known_case():
    r = mean_ci([4.0, 5.0, 6.0, 7.0, 8.0])
    assert r["mean"] == 6.0 and r["ci_low"] < 6.0 < r["ci_high"]
    # n=5, sd=sqrt(2.5), t(0.975,4)=2.776 -> CI halfwidth 2.776*sqrt(2.5/5)=1.964
    assert r["ci_low"] == pytest.approx(4.036, abs=0.01)


def test_proportion_ci_wald():
    r = proportion_ci(40, 100)
    assert r["p"] == 0.4 and r["ci_low"] == pytest.approx(0.4 - 1.96 * np.sqrt(0.4 * 0.6 / 100), 3)


def test_ttest_matches_scipy():
    a, b = [1, 2, 3, 4, 5], [3, 4, 5, 6, 7]
    r = two_sample_ttest(a, b)
    t_ref, p_ref = sps.ttest_ind(a, b, equal_var=r["test"] == "student t")
    assert r["t"] == pytest.approx(t_ref, 4) and r["p"] == pytest.approx(p_ref, 6)


def test_mannwhitney_matches_scipy():
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]
    r = mannwhitney(a, b)
    u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    assert r["U"] == pytest.approx(u, 4) and r["p"] == pytest.approx(p, 6)
    assert r["rank_biserial_r"] == pytest.approx(1.0)


def test_paired_and_gof():
    r = paired_ttest([1, 2, 3, 4], [2, 4, 6, 8])
    assert r["mean_diff"] == pytest.approx(-2.5)
    g = chi_square_gof({"a": 10, "b": 20, "c": 30})
    assert g["chi2"] == pytest.approx(10.0) and g["dof"] == 2


def test_correlation_pearson_vs_spearman():
    df = pd.DataFrame({"x": range(20), "y": [v + (0 if v % 2 else 1) for v in range(30, 50)]})
    t = correlation_table(df, ["x", "y"])
    assert len(t) == 1 and abs(t["r"].iloc[0]) > 0.9


def test_categorical_association_caveat():
    df = pd.DataFrame({"a": list("xxxyyyzzz"), "b": list("ababababab")[:9]})
    r = categorical_association(df, "a", "b")
    assert "cramers_v" in r and r["caveat"] is not None


def test_outlier_iqr():
    r = outlier_diagnostics(pd.Series([1, 2, 2, 3, 100]))
    assert r["n_outliers"] == 1


def test_linear_regression_slope():
    xs = np.arange(1, 11)
    df = pd.DataFrame({"x": xs, "y": 2 * xs + 1 + (xs % 2) * 0.1})
    r = linear_regression(df, "y", ["x"])
    assert r["coefficients"]["x"] == pytest.approx(2.05, abs=0.15)
    assert r["r2_in_sample"] > 0.99


def test_forecast_errors_or_returns():
    assert "error" in forecast_series([1, 2, 3])
    r = forecast_series(list(range(1, 9)), 3, 3)
    assert len(r["holt_linear_trend"]["next_values"]) == 3
    assert r["holt_linear_trend"]["next_values"][0] > 7  # rising trend


def test_cluster_small_data_refuses():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=10), "b": rng.normal(size=10)})
    assert "error" in cluster_segments(df, ["a", "b"])
