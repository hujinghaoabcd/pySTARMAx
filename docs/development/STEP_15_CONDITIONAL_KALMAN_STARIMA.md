# Step 15 handoff: conditional integrated Kalman STARIMA

## Repository position

- development version: `0.0.15`;
- branch: `agent/integrated-kalman-starima`;
- pull request: PR #15, `Add conditional integrated Kalman STARIMA`;
- base: version 0.0.14 on `main`;
- authoritative implementation/documentation validation: GitHub Actions CI
  #339, run ID `30856825182`;
- validation result: 134 tests passed and 86.96% total branch coverage;
- `src/pystarmax/integrated_maximum_likelihood.py` coverage: 84.3%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and Ubuntu/Windows/
  macOS Python 3.11--3.14 all passed.

## Delivered API

### Model

```python
model = KalmanSTARIMA(
    ar_order=p,
    integration_order=d,
    ma_order=q,
    covariance_type="full",
)
result = model.fit(level_data, weights)
```

### Differenced-scale operations

```python
model.predict_differenced(steps)
model.fitted_differenced()
model.filter()
model.smooth()
model.smooth_innovation_disturbances()
model.infer()
model.admissibility()
model.to_state_space()
```

### Original-scale operations

```python
model.predict(steps)
model.fitted_original()
```

### Result object

`KalmanSTARIMAResult` stores or delegates:

- `core_result`;
- `integration_order` and `order == (p, d, q)`;
- `n_original_rows` and `n_differenced_rows`;
- `original_missing_cells` and `differenced_missing_cells`;
- `original_scale_forecast_available`;
- dynamic parameters and names;
- coefficient and innovation covariance tables;
- training filter result;
- log likelihood, AIC, BIC, and convergence.

## Likelihood convention

For

\[
x_t=(1-B)^d y_t=\Delta^d y_t,
\]

the transformed series follows the stationary STARMA equation

\[
x_t=c+\sum_{i=1}^{p}A_i x_{t-i}
+\eta_t+\sum_{j=1}^{q}B_j\eta_{t-j}.
\]

The wrapper evaluates

\[
\ell_c(\vartheta;y_{1:T})
=
\ell\left(
\vartheta;\Delta^d y_{d+1:T}\mid y_{1:d}
\right).
\]

This is a conditional likelihood on the ordinary-difference process. The first
`d` level rows provide the differencing history and are not assigned a separate
probability model. The method must not be labelled as an exact diffuse
integrated likelihood.

`initialization="diffuse"` remains the existing large-variance approximation for
the stationary transformed STARMA state. It does not convert this wrapper into
an exact diffuse level-state estimator.

## Scale contract

The stationary core is defined entirely on `Delta^d y`. Therefore filtering,
state smoothing, original innovation smoothing, admissibility, state-space
conversion, and likelihood-Hessian inference all remain on the transformed
scale.

Original-scale forecast restoration is deterministic conditional on the
transformed forecast and terminal differencing anchors. For `d=1`,

\[
\widehat y_{T+h}=y_T+\sum_{r=1}^{h}\widehat x_{T+r}.
\]

For `d=2`, the predicted second differences update the terminal first
difference, and those updated first differences update the level. The reusable
`DifferencingState.inverse_forecast()` implements the general recursion.

## Missing-observation policy

Original missing cells are not imputed. Ordinary finite differences propagate
`NaN` through the complete stencil. With first differences, a missing `y_t`
usually removes both `y_t-y_(t-1)` and `y_(t+1)-y_t` for that location.

The stationary Kalman core then handles the transformed data:

- finite locations enter the measurement update;
- missing locations are omitted;
- a fully missing transformed row performs prediction only;
- only finite transformed observations contribute to the likelihood.

The result reports both original and transformed missing-cell counts so the
expansion is auditable.

## Terminal-anchor policy

Original-scale forecasting requires finite values of

\[
y_T,\Delta y_T,\ldots,\Delta^{d-1}y_T.
\]

If any terminal anchor is non-finite:

- fitting remains allowed;
- differenced-scale filtering, smoothing, inference, and prediction remain
  available;
- `original_scale_forecast_available` is false;
- `predict()` raises with a clear terminal-anchor message.

No last-observation carry-forward, interpolation, or hidden imputation is used.

## Fitted-value alignment

`fitted_differenced()` returns stationary-core one-step predicted observations
with `T-d` rows.

`fitted_original()` returns `T` rows aligned to the original sample. The first
`d` rows are unavailable. Later rows are restored only when the corresponding
observed lag history is finite. This intentionally avoids using smoothed or
imputed history to manufacture an original-scale one-step fit.

## Validation references

Tests include:

1. exact `d=0` equivalence with `KalmanSTARMA` under identical dynamic and
   covariance starting values;
2. Gaussian random walk with drift, with the estimated differenced mean
   cumulated on the original scale;
3. second-order integration using terminal level and slope;
4. missing-value propagation and transformed finite-observation counts;
5. differenced forecast availability but original forecast refusal when the
   terminal anchor is missing;
6. new-data filter, RTS smoother, and original innovation smoother lengths after
   differencing;
7. aligned original fitted values and incomplete-history behavior;
8. negative order, not-fitted, dimensionality, infinity, and insufficient
   transformed-sample validation;
9. the full inherited package suite on all supported Python versions.

Authoritative CI #339 reported:

- 134 tests passed in 50.04 seconds;
- total branch coverage: 86.96%;
- integrated module coverage: 84.3%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and diagnostic fixture
  regeneration passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

The validation-record-only head changes only this handoff and
`PROJECT_STATUS.md`. It receives a final merge-gate CI before PR #15 is marked
ready and merged.

## Files introduced or changed

Core:

- `src/pystarmax/integrated_maximum_likelihood.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_integrated_maximum_likelihood.py`.

Documentation and example:

- `docs/integrated_maximum_likelihood.md`;
- `examples/integrated_kalman_starima.py`;
- README, documentation home, navigation, ordinary STARIMA guide, roadmap, and
  project status;
- Step 15 handoff.

## Important implementation boundary

The wrapper delegates all stationary numerical work to one `KalmanSTARMA`
instance. Do not fork or copy the optimizer, covariance parameterization,
admissibility checks, Kalman filtering, Hessian inference, RTS smoothing, or
innovation smoothing into an integrated-specific implementation unless an exact
level-state likelihood is being introduced as a deliberately separate model.

## Current limitations

- no exact diffuse integrated level-state likelihood;
- no original-scale filtered or smoothed level-state distributions;
- no original-scale Kalman STARIMA forecast intervals;
- no multiplicative seasonal Kalman MLE;
- parameters are fixed in filtering and smoothing;
- no cross-time original innovation covariance or simulation smoother;
- matrices are dense;
- exogenous regressors and interventions are unsupported.

## Next recommended stage

The next major stage is multiplicative seasonal Kalman STARIMA with ordinary and
seasonal differencing, constrained factor expansion, transformed-scale Gaussian
likelihood, and pathwise original-scale reconstruction. Exact diffuse
integration should remain a separate subsequent API rather than being silently
mixed into this conditional wrapper.
