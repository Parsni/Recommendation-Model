# Statistical methodology documentation

The statistics layer (`src/statistics/`) is deliberately **separate from the
recommendation code**: when the real DA problem is chosen, the relevant
analyses can be run on any DataFrame without touching the engine.

Every function states which test it ran, its assumptions, and refuses to
over-claim (e.g. "no strong evidence against normality (not proof of
normality)", "p<0.05: mean difference unlikely under equal means", "not proof
of equality").

## Implemented utilities and when to use them

### Descriptive (`descriptive.py`)
- `describe_numeric` — mean/median/quartiles/skew/kurtosis/SEM.
- `describe_categorical` — frequency/share table.
- `missing_data_diagnostics` — per-column missing counts and %.
- `outlier_diagnostics` — IQR or z-score bounds with outlier share.
- `normality_note` — Shapiro-Wilk *flag only* (n≥8, capped 5000-sample
  subsample, seed fixed); used to *choose* tests, never as a hard gate.
- `correlation_table` — Pearson (continuous, many distinct values) vs Spearman
  (ordinal/few levels) chosen automatically per pair; reports n and p.
- `categorical_association` — chi-square of independence + Cramer's V effect
  size; warns when >20% of cells have expected counts < 5.

### Inferential (`inferential.py`)
- `mean_ci` — t-based CI for a mean.
- `proportion_ci` — Wald CI with the n·p ≥ 5 assumption check reported.
- `two_sample_ttest` — Levene-tested equal/unequal variance choice, Cohen's d.
- `mannwhitney` — rank-based alternative with rank-biserial effect size;
  documented for ordinal/non-normal/small samples.
- `paired_ttest` — matched pairs, Cohen's d_z.
- `one_sample_ttest`, `chi_square_gof` — against reference mean / uniform.

### Models (`models.py`)
- `linear_regression` — OLS with cross-validated R² alongside in-sample R² and
  an explicit optimism warning.
- `cluster_segments` — KMeans over a k-range with silhouette stability gate:
  best silhouette < 0.25 ⇒ declared unstable with the "use transparent
  descriptive groups instead" fallback rather than presenting fake segments.
- `forecast_series` — moving-average + Holt linear-trend baselines with fixed
  smoothing constants; states no seasonality and no backtest were done.

## Test-selection guidance for the final DA report

1. Identify the sampling unit (user, session, item) - CI widths and test
   sample sizes are computed on that unit, not on rows.
2. Pick method by variable type: numeric↔numeric → correlation choice;
   categorical↔categorical → chi-square/Cramér's V (exact test if sparse);
   group mean comparisons → t/Welch or Mann-Whitney by distribution;
   prediction → regression with CV;
   segmentation → clustering only if silhouette supports it.
3. Report effect sizes and CIs next to p-values; a p-value alone is not the
   finding.
4. State the limitation the function already returns (assumption caveats are
   part of the returned dict on purpose).

All utilities are exercised by `tests/test_statistics.py` against
hand-computed or scipy-reference values (t, U, chi², slope, CI bounds).
