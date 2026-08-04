# Step 18 handoff: Gaussian Kalman forecast intervals

## Repository position

- development version: `0.0.18`;
- branch: `agent/kalman-original-scale-intervals`;
- pull request: PR #18, `Add original-scale Kalman forecast intervals`;
- base: version 0.0.17 on `main` at merge commit
  `d62cadf210c223d3da62e87dbc9ea82c2b1a7803`;
- authoritative implementation/documentation validation: GitHub Actions CI
  #403, run ID `30878184542`;
- validation result: 155 tests passed and 87.09% total branch coverage;
- `src/pystarmax/kalman_forecasting.py` coverage: 80.2%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and Ubuntu/Windows/
  macOS Python 3.11–3.14 all passed;
- a validation-record-only merge-gate CI is required before merge.

## Delivered API

Stationary scale:

```python
interval = fitted_starma.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

Ordinary integrated scales:

```python
transformed = fitted_starima.predict_differenced_interval(...)
original = fitted_starima.predict_interval(...)
```

Combined ordinary-seasonal scales:

```python
transformed = fitted_seasonal.predict_differenced_interval(...)
original = fitted_seasonal.predict_interval(...)
```

Low-level functions:

- `simulate_kalman_forecast_paths()`;
- `kalman_forecast_interval()`;
- `integrated_kalman_forecast_interval()`;
- `inverse_forecast_paths()`.

All public interval methods return the existing immutable `ForecastInterval`.

## Conditional forecast law

For the fitted state-space model

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_{t+1},
\qquad
\eta_{t+1}\sim\mathcal N(0,Q),
\]

\[
x_{t+1}=Z\alpha_{t+1},
\]

future paths begin with

\[
\alpha_T\mid y_{1:T}
\sim
\mathcal N(a_{T\mid T},P_{T\mid T}).
\]

For each simulation `b`:

1. draw the final latent state from the final filtered posterior;
2. draw independent future location-level innovations from `Q`;
3. propagate the fitted transition and state intercept;
4. apply the design matrix at every horizon;
5. retain the complete transformed path.

The paths therefore include final filtered-state uncertainty and future process
innovation uncertainty. Fitted parameters remain fixed.

## State covariance factorization

The final filtered covariance is symmetrized and factorized through an
eigendecomposition. Small negative eigenvalues within a floating-point
tolerance are clipped to zero. Material negative eigenvalues raise.

This supports singular positive-semidefinite posteriors without adding arbitrary
jitter or forcing an invalid Cholesky factor.

## Point forecast convention

The returned `ForecastInterval.mean` is the deterministic recursive Kalman point
forecast. It is not the finite-simulation sample mean. This keeps the reported
center identical to the existing `predict()` or `predict_differenced()` result
and independent of the random seed.

## Pathwise inverse differencing

Original-scale quantiles are never obtained by inverse-transforming transformed
marginal bounds.

For one ordinary difference,

\[
y_{T+h}=y_T+\sum_{j=1}^{h}x_{T+j},
\]

so uncertainty couples and accumulates across horizons. Higher ordinary orders
add nested recursions. Seasonal differencing uses rolling histories,

\[
y_{T+h}=x_{T+h}+y_{T+h-s},
\]

and combined differencing requires seasonal inversion followed by ordinary
inversion.

The implementation applies the stored `DifferencingState` or
`CombinedDifferencingState` to every simulated path independently and takes
quantiles only after reconstruction.

## Scale contract

### `KalmanSTARMA`

- `predict_interval()` returns the stationary observation scale.

### `KalmanSTARIMA`

- `predict_differenced_interval()` returns the highest ordinary-difference
  scale;
- `predict_interval()` returns the original level scale.

### `SeasonalKalmanSTARIMA`

- `predict_differenced_interval()` returns the combined
  `(1-B)^d(1-B^s)^D` scale;
- `predict_interval()` returns the original scale after pathwise seasonal and
  ordinary inversion.

## Terminal history policy

Original-scale intervals require finite terminal information:

- all ordinary anchors below order `d`;
- all rolling seasonal histories below order `D`.

When these are unavailable, transformed intervals remain valid but original
intervals raise. No interpolation, forward fill, or invented anchor is used.

## Random-number policy

`random_state` accepts an integer, NumPy `Generator`, or `None`. The same fitted
model, arguments, and integer seed reproduce identical paths and bounds.

The default uses 2,000 paths. Larger values reduce Monte Carlo quantile error at
higher computational and memory cost.

## Validation references

Tests include:

1. an analytic scalar state process whose first two forecast means are `1.5`
   and `2.0` and variances are `5.0` and `6.0`;
2. direct evidence that final filtered covariance and future innovations both
   enter uncertainty;
3. stationary interval reproducibility and exact point-forecast centering;
4. exact per-path first-difference inversion;
5. ordinary random-walk first-horizon shifts and later widening;
6. seasonal random-walk rolling-cycle shifts and widening after one full
   period;
7. transformed interval availability with original interval refusal when the
   terminal anchor is missing;
8. steps, level, simulation count, covariance, shape, and type validation;
9. the complete inherited package test suite.

Authoritative CI #403 reported:

- 155 tests passed in 36.77 seconds in the coverage job;
- total branch coverage: 87.09%;
- Kalman forecast module coverage: 80.2%;
- ordinary integrated core coverage: 84.3%;
- seasonal forecast facade coverage: 89.5%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and diagnostic fixture
  regeneration passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

The validation-record-only head changes only this handoff and
`PROJECT_STATUS.md`. It receives one final merge-gate CI before PR #18 is
marked ready and merged.

## Files introduced or changed

Core:

- `src/pystarmax/kalman_forecasting.py`;
- `src/pystarmax/integrated_forecasting.py`;
- `src/pystarmax/seasonal_forecasting.py`;
- `src/pystarmax/maximum_likelihood.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_kalman_forecast_intervals.py`.

Documentation and example:

- `docs/kalman_forecast_intervals.md`;
- `examples/kalman_forecast_intervals.py`;
- README, docs home, navigation, ordinary/seasonal Kalman guides, roadmap, and
  project status;
- Step 18 handoff.

## Current limitations

- fitted parameters are fixed;
- observed-information or bootstrap parameter draws are not propagated;
- bounds are Monte Carlo quantiles rather than closed-form Gaussian bounds;
- the current state-space observation equation has no separate observation
  noise beyond the process innovation convention;
- ordinary and seasonal likelihoods remain conditional on transformation
  histories;
- original-scale filtered and smoothed level-state distributions remain
  unavailable;
- dense state matrices and large path arrays can be expensive;
- simulation paths are not yet streamed or parallelized.

## Next recommended stage

The next method stage should design exact diffuse integrated level-state
likelihood and smoothing as a separate API. A forecast-focused alternative is
parameter-aware Kalman paths using constrained observed-information draws or
model-refitting bootstrap paths.
