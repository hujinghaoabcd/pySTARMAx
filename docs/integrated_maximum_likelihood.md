# Conditional integrated Kalman STARIMA

Version 0.0.15 adds `KalmanSTARIMA(p, d, q)`, an ordinary-difference wrapper
around the stationary Gaussian `KalmanSTARMA` estimator.

The method is designed for integrated space-time processes when the desired
likelihood is conditional on the observed initial level history. It does **not**
claim to be an exact diffuse likelihood for a non-stationary level-state model.

## Model

Let

\[
x_t=(1-B)^d y_t=\Delta^d y_t,
\]

where \(y_t\) is the original `(time, location)` series and \(d\) is the
ordinary integration order. The transformed process follows a stationary
STARMA model,

\[
x_t=c+\sum_{i=1}^{p}A_i x_{t-i}
+\eta_t+\sum_{j=1}^{q}B_j\eta_{t-j},
\qquad
\eta_t\sim\mathcal N(0,\Sigma).
\]

The spatial lag operators retain the package convention

\[
A_i=\sum_k\phi_{ik}W_k,
\qquad
B_j=\sum_k\theta_{jk}W_k.
\]

## Conditional likelihood

For an original sample \(y_1,\ldots,y_T\), ordinary differencing produces
\(T-d\) transformed rows. `KalmanSTARIMA` maximizes

\[
\ell_c(\vartheta;y_{1:T})
=
\ell\left(\vartheta;
\Delta^d y_{d+1:T}\mid y_{1:d}
\right),
\]

using the existing stationary Gaussian Kalman likelihood for the transformed
process.

The first \(d\) level rows are conditioned on through the differencing
transformation. They do not receive a separate probability model. Therefore:

- the reported log likelihood, AIC, and BIC describe the transformed conditional
  likelihood;
- the effective likelihood time axis has length \(T-d\);
- `initialization="diffuse"` still refers only to the stationary STARMA state of
  the transformed process and remains a large-variance approximation;
- this API is not an exact diffuse ARIMA/STARIMA likelihood on the level state.

This distinction is exposed in the result summary rather than hidden behind the
word “integrated.”

Version 0.0.19 added the separate fixed-parameter exact diffuse
level-state route. Version 0.0.20 adds `ExactDiffuseKalmanSTARIMA`, which
optimizes STARMA parameters against the original-level exact diffuse
likelihood. It remains a separate estimator because this page's
`KalmanSTARIMA` intentionally reports a conditional differenced likelihood.
See [Exact diffuse filtering](exact_diffuse.md) and
[Exact diffuse STARIMA maximum likelihood](exact_diffuse_mle.md).

## Fitting

```python
from pystarmax import KalmanSTARIMA

model = KalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=True,
    enforce_stationarity=True,
    enforce_invertibility=True,
)
result = model.fit(level_series, weights)

print(result.summary())
print(result.order)
print(result.log_likelihood)
print(result.aic, result.bic)
```

`include_intercept=True` adds an intercept to the highest ordinary-difference
process. For \(d=1\), this acts as a drift term in the level process. For larger
\(d\), a constant highest difference generates a higher-order deterministic
trend after repeated integration.

## Explicit scale boundaries

The wrapper deliberately keeps transformed and original scales separate.

### Differenced scale

The following methods operate on \(x_t=\Delta^d y_t\):

```python
filtered = model.filter()
smoothed = model.smooth()
innovation_result = model.smooth_innovation_disturbances()
inference = model.infer()
state_space = model.to_state_space()
admissibility = model.admissibility()
stationary_forecast = model.predict_differenced(steps=12)
stationary_interval = model.predict_differenced_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=42,
)
stationary_fitted = model.fitted_differenced()
```

The state-space object is the stationary STARMA state for the differenced
process. AR stationarity and MA invertibility diagnostics apply to that
stationary polynomial.

### Original scale

```python
level_forecast = model.predict(steps=12)
level_interval = model.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=42,
)
level_fitted = model.fitted_original()
```

`predict()` recursively applies the stored terminal differencing state. For
\(d=1\),

\[
\widehat y_{T+h}
=
y_T+\sum_{r=1}^{h}\widehat x_{T+r}.
\]

For \(d=2\), the predicted second differences first update the terminal first
difference, and the updated first differences then update the level. The same
recursive construction extends to arbitrary non-negative \(d\).

`fitted_original()` returns an array aligned with the original sample. Its first
\(d\) rows are `NaN` because no transformed one-step prediction exists there.
Each later fitted value is reconstructed from the transformed prediction and
the corresponding **observed** lag history. It remains `NaN` when that history
is incomplete.

## Missing observations

Missing cells are not imputed before differencing. NumPy finite differences
naturally propagate `NaN` through the differencing stencil. For example, with
\(d=1\), a missing \(y_t\) makes both

\[
y_t-y_{t-1}
\quad\text{and}\quad
y_{t+1}-y_t
\]

missing for that location. In general, one missing level cell can affect up to
\(d+1\) transformed rows.

The stationary Kalman filter then handles the transformed matrix using its
existing rules:

- finite locations participate in the measurement update;
- missing locations are omitted at that time;
- a fully missing transformed row performs prediction only;
- no missing cell contributes to the Gaussian likelihood.

The result records both `original_missing_cells` and
`differenced_missing_cells` so this expansion remains visible.

## New data

Supplying new level data to filtering or smoothing applies the same ordinary
difference operator first:

```python
filtered_new = model.filter(new_levels)
smoothed_new = model.smooth(new_levels)
innovations_new = model.smooth_innovation_disturbances(new_levels)
```

If `new_levels` has \(T_{new}\) rows, the returned filter and smoother have
\(T_{new}-d\) state times. The new matrix must include its own first \(d\) level
rows because those rows provide the history required to form its first
transformed observation.

## Terminal anchors

Original-scale forecasting requires finite terminal values at every lower
difference level:

\[
y_T,
\Delta y_T,
\ldots,
\Delta^{d-1}y_T.
\]

If the trailing original sample contains missing values that make any anchor
non-finite, fitting can still succeed and all differenced-scale methods remain
available. However, `predict()` raises with a clear message because no unique
original-scale continuation can be constructed. Use
`predict_differenced()` or provide finite trailing observations.

The result exposes this state through
`original_scale_forecast_available`.

## Result object

`KalmanSTARIMAResult` wraps the immutable stationary
`KalmanSTARMAResult` and adds integrated-scale metadata:

- `integration_order`;
- `order == (p, d, q)`;
- `n_original_rows` and `n_differenced_rows`;
- `original_missing_cells` and `differenced_missing_cells`;
- `original_scale_forecast_available`;
- delegated coefficient, covariance, filter, likelihood, AIC, BIC, and
  convergence properties.

The underlying complete stationary result remains available as `core_result`.

## Inference and smoothing

```python
inference = model.infer(relative_step=1e-4)
state_result = model.smooth()
innovation_result = model.smooth_innovation_disturbances()
```

These methods inherit the tested stationary implementation without duplicating
the optimizer, observed-information Hessian, RTS recursion, or conditional
Gaussian innovation mapping. Their interpretation is always conditional on the
ordinary differencing transformation and the initial level history.

Version 0.0.18 adds fixed-parameter Gaussian intervals. Every path starts
from the final filtered state posterior, receives future innovations from the
fitted covariance, and is inverse-differenced independently before original-
scale quantiles. This propagates filtered-state and process uncertainty but not
parameter-estimation uncertainty. See
[Gaussian Kalman forecast intervals](kalman_forecast_intervals.md).

## Validation references

The implementation is checked through:

- exact \(d=0\) equivalence with `KalmanSTARMA` under identical starts;
- a random walk with drift, where first differences reduce to Gaussian white
  noise and forecasts cumulate the estimated drift;
- second-order inverse differencing using the terminal level and slope;
- explicit missing-value propagation and transformed likelihood counts;
- finite differenced forecasts but rejected level forecasts when the terminal
  anchor is missing;
- new-data filter, smoother, and innovation-smoother time-axis lengths;
- aligned original-scale fitted values with incomplete lag history;
- constructor, sample-length, dimensionality, infinity, and fitted-state
  validation.

## Current limitations

- likelihood is conditional on initial levels, not exact diffuse on the level
  process;
- this wrapper covers ordinary integration; multiplicative seasonal integration is provided by `SeasonalKalmanSTARIMA`;
- forecast intervals condition on fitted parameters and use Monte Carlo quantiles;
- original-scale filtered or smoothed level-state distributions are not
  returned;
- cross-time innovation covariance and simulation smoothing are unavailable;
- state and spatial matrices remain dense;
- exogenous regressors and interventions are unsupported.
