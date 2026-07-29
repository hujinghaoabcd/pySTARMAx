# Original-scale fitted values and forecast intervals

pySTARMAx 0.0.5 adds two outputs that remain separate from transformed-scale
parameter inference:

- aligned one-step fitted values on the original observation scale;
- simulation-based conditional innovation forecast intervals.

## Original-scale one-step fitted values

`STARIMA.fit()` and `SeasonalSTARIMA.fit()` still return `STARMAResult` on the
highest differenced scale. The new `fitted_original()` method reconstructs an
array with the same shape as the original observations.

For the combined differencing polynomial

\[
C(B)=(1-B)^d(1-B^s)^D=1+\sum_{k=1}^{K}c_kB^k,
\]

and a fitted transformed value \(\widehat w_t\), the aligned original-scale fit
is

\[
\widehat z_t=\widehat w_t-\sum_{k=1}^{K}c_k z_{t-k}.
\]

The historical terms are the **observed** values, not earlier fitted values.
This makes the output a one-step conditional fit and prevents accumulated
integration drift. Rows lacking differencing history or model conditional lags
remain `NaN`.

```python
model.fit(observations, weights)
fitted = model.fitted_original()
original_residuals = observations - fitted
```

`differencing_coefficients()` and `restore_fitted_values()` expose the same
calculation independently for audited workflows.

## Conditional innovation intervals

All model classes expose:

```python
interval = model.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=2000,
    random_state=42,
)
```

The returned immutable `ForecastInterval` contains:

- `mean`: the existing recursive conditional-mean forecast;
- `lower` and `upper`: central empirical quantiles;
- `level` and `n_simulations`;
- `method`: a description of the uncertainty calculation.

For each simulation path, pySTARMAx:

1. draws future innovations from the fitted contemporaneous location covariance;
2. propagates the draws recursively through all AR and MA terms;
3. retains the dependence between forecast horizons;
4. for integrated models, reverses seasonal and ordinary differencing on the
   complete path;
5. computes central empirical quantiles only after original-scale reconstruction.

Quantile endpoints are therefore not naively integrated independently. The full
path dependence created by differencing is retained.

## Covariance handling

The fitted innovation covariance is symmetrized before simulation. Tiny negative
eigenvalues attributable to floating-point error are clipped to zero, allowing
singular covariance estimates while preserving estimated cross-location
dependence. Materially indefinite covariance matrices are rejected.

## Interpretation

The current interval is conditional on:

- the fitted model coefficients;
- the observed history;
- the estimated innovation covariance;
- the package's zero pre-sample innovation convention.

It includes future innovation uncertainty but does **not** yet include parameter
estimation uncertainty, model-order uncertainty, or uncertainty in the spatial
weights. It is therefore a conditional predictive interval, not a full Bayesian
or bootstrap parameter-uncertainty interval.

## Reproducibility

Passing an integer `random_state` produces repeatable interval endpoints.
Passing a NumPy `Generator` advances that generator normally.
