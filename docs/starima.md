# Ordinary STARIMA

pySTARMAx provides two ordinary-integration routes with the same differencing
and original-scale forecast convention but different estimators:

1. `STARIMA` uses iterative conditional least squares and supports conditional
   innovation and bootstrap forecast intervals.
2. `KalmanSTARIMA` uses a Gaussian Kalman likelihood for the differenced process,
   supports missing observations, filtering, smoothing, original innovation
   smoothing, and observed-information inference.

Both fit the highest ordinary difference

\[
w_t=(1-B)^d z_t,
\]

where `B` is the one-step temporal backshift operator. Differencing is applied
independently along time for every location; spatial weights retain their
supplied orientation.

## Conditional least-squares route

```python
import numpy as np
from pystarmax import STARIMA, SpatialWeights, lattice_weights

weights = SpatialWeights.from_adjacency(lattice_weights(2, 2))
model = STARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
)
result = model.fit(observations, weights)

print(result.summary())
forecast = model.predict(steps=6)
fitted = model.fitted_original()
interval = model.predict_interval(steps=6, random_state=42)
difference_forecast = model.predict_differenced(steps=6)
```

`STARIMA.fit()` returns the existing `STARMAResult`. Its coefficients, fitted
values, residuals, likelihood approximation, AIC, and BIC describe the
differenced process. `predict()` recursively integrates future differences;
`fitted_original()` reconstructs aligned one-step fitted values; and
`predict_interval()` inverts complete simulated paths before quantiles are
computed.

This route also supports parameter-aware residual and parametric bootstrap
intervals through `predict_bootstrap_interval()`.

## Conditional Gaussian Kalman route

```python
from pystarmax import KalmanSTARIMA

kalman = KalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=True,
)
kalman_result = kalman.fit(incomplete_levels, weights)

print(kalman_result.summary())
print(kalman.predict_differenced(steps=6))
print(kalman.predict(steps=6))
print(kalman.filter().log_likelihood)
print(kalman.smooth().smoothed_observations)
```

`KalmanSTARIMA` applies ordinary differencing and delegates estimation to the
stationary `KalmanSTARMA` core. Its likelihood is

\[
\ell_c(\vartheta;z_{1:T})
=
\ell\left(
\vartheta;\Delta^d z_{d+1:T}\mid z_{1:d}
\right).
\]

The first `d` original rows are conditioned on through the differencing
transformation. This is not an exact diffuse integrated level-state likelihood.
The distinction is recorded in `KalmanSTARIMAResult.summary()`.

Missing original cells are never imputed. They propagate through the finite
difference stencil, after which the stationary Kalman filter omits missing
locations from each transformed measurement update. A fully missing transformed
row performs prediction only.

See [Conditional integrated Kalman STARIMA](integrated_maximum_likelihood.md)
for likelihood scope, scale boundaries, missing propagation, terminal anchor
requirements, smoothing, and inference.

## Differencing state

`ordinary_difference()` returns both transformed observations and an immutable
end-of-sample state:

```python
from pystarmax import ordinary_difference

transformed, state = ordinary_difference(observations, order=2)
restored_forecast = state.inverse_forecast(predicted_second_differences)
```

For order `d`, the state stores the final observed value at each lower-order
level: the original series, first difference, and so on through order `d-1`.
Forecast inversion updates those levels from highest to lowest for every future
time step. The state is copied internally, so repeated inversion calls are
deterministic.

`KalmanSTARIMA` uses the same state concept. When trailing missing values make a
required terminal level or lower-order difference non-finite, transformed-scale
estimation and forecasting remain available, but original-scale `predict()`
raises instead of inventing an anchor.

## Intercept interpretation

The underlying STARMA intercept is estimated on the highest-difference scale.
With `d=1`, a non-zero intercept acts as drift on the original scale. For larger
`d`, a constant highest difference produces a higher-order deterministic trend
after recursive integration. pySTARMAx does not silently remove or reinterpret
this term.

## Simulation

`simulate_starima()` first simulates the stationary highest-difference process
and then integrates it:

```python
from pystarmax import simulate_starima

series = simulate_starima(
    phi=np.array([[0.4, 0.1]]),
    theta=np.zeros((0, 2)),
    weights=weights,
    n_steps=300,
    integration_order=1,
    random_state=42,
)
```

By default, all lower-order levels are anchored at zero immediately before the
returned sample. A custom `DifferencingState` can supply alternative starting
levels.

## Current limits

- both estimators report dynamic coefficients on the differenced scale;
- `STARIMA` forecast intervals do not automatically include parameter
  uncertainty unless the bootstrap route is selected;
- `KalmanSTARIMA` does not yet expose original-scale forecast intervals;
- the Kalman likelihood is conditional on initial levels, not exact diffuse;
- original-scale filtered and smoothed level-state distributions are not
  returned;
- seasonal integration remains in `SeasonalSTARIMA`; seasonal Kalman maximum
  likelihood is future work.
