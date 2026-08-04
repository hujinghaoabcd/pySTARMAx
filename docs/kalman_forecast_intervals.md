# Gaussian Kalman forecast intervals

Version 0.0.18 adds fixed-parameter Gaussian forecast intervals for stationary,
ordinary-integrated, and multiplicative seasonal Kalman models.

## Public API

A stationary model returns intervals on its observation scale:

```python
interval = fitted_starma.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

Ordinary and seasonal integrated models expose both transformed and original
scales:

```python
transformed = fitted_starima.predict_differenced_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)

original = fitted_starima.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

The returned immutable `ForecastInterval` contains `mean`, `lower`, `upper`,
`level`, `method`, `n_simulations`, and `refit_parameters`.

## Conditional Gaussian forecast law

For the fitted state-space system

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_{t+1},
\qquad
\eta_{t+1}\sim\mathcal N(0,Q),
\]

\[
x_{t+1}=Z\alpha_{t+1},
\]

the final filtered state is treated as

\[
\alpha_T\mid y_{1:T}
\sim
\mathcal N(a_{T\mid T},P_{T\mid T}).
\]

For every Monte Carlo path, pySTARMAx draws

\[
\alpha_T^{(b)}
=
a_{T\mid T}+L_P z_0^{(b)},
\qquad z_0^{(b)}\sim\mathcal N(0,I),
\]

and then recursively draws

\[
\eta_{T+h}^{(b)}\sim\mathcal N(0,Q),
\]

\[
\alpha_{T+h}^{(b)}
=
c+T\alpha_{T+h-1}^{(b)}+R\eta_{T+h}^{(b)},
\]

\[
x_{T+h}^{(b)}=Z\alpha_{T+h}^{(b)}.
\]

Thus the intervals include both:

1. uncertainty in the final filtered state; and
2. future process-innovation uncertainty.

They condition on fitted parameters and do not include parameter-estimation
uncertainty.

## Positive-semidefinite state covariance

The final filtered covariance can be numerically singular, especially when some
state coordinates are deterministically linked. Paths use an eigendecomposition
of the symmetrized covariance:

\[
P_{T\mid T}=V\Lambda V^\top,
\qquad
L_P=V\operatorname{diag}(\sqrt{\max(\lambda_i,0)}).
\]

Small negative eigenvalues within a floating-point tolerance are clipped to
zero. Materially negative eigenvalues raise instead of silently producing
invalid Gaussian draws.

## Why original-scale bounds are pathwise

For ordinary or seasonal integration, it is generally incorrect to
inverse-difference marginal lower and upper bounds separately. Inverse
differencing couples forecast horizons.

For a first ordinary difference,

\[
y_{T+h}=y_T+\sum_{j=1}^{h}x_{T+j}.
\]

Even if each transformed marginal interval has constant width, uncertainty on
the original level accumulates with the full path. For higher ordinary orders,
the recursion introduces additional horizon dependence.

For a seasonal difference,

\[
y_{T+h}=x_{T+h}+y_{T+h-s},
\]

so each future value depends on a rolling seasonal cycle. With combined
ordinary-seasonal differencing, both recursions interact.

pySTARMAx therefore computes

\[
y_{T+1:T+H}^{(b)}
=
\mathcal D^{-1}
\left(x_{T+1:T+H}^{(b)};\mathcal H_T\right)
\]

for every simulated path and only then takes horizon-location quantiles:

\[
L_{h,i}=Q_{(1-\gamma)/2}
\{y_{T+h,i}^{(b)}\}_{b=1}^{B},
\]

\[
U_{h,i}=Q_{(1+\gamma)/2}
\{y_{T+h,i}^{(b)}\}_{b=1}^{B}.
\]

This preserves ordinary accumulation, rolling seasonal histories, and their
joint temporal dependence.

## Point forecast convention

The reported interval mean is the deterministic recursive Kalman point
forecast. For integrated models it is inverse-differenced with the same stored
terminal anchors and seasonal histories used by `predict()`.

Monte Carlo sample means are not substituted for the point forecast, avoiding
small seed-dependent deviations in the reported center.

## Scale-specific methods

### `KalmanSTARMA`

`predict_interval()` returns the model observation scale, which is stationary by
construction.

### `KalmanSTARIMA`

- `predict_differenced_interval()` returns the highest ordinary-difference
  scale;
- `predict_interval()` returns the original level scale using pathwise ordinary
  inverse differencing.

### `SeasonalKalmanSTARIMA`

- `predict_differenced_interval()` returns the combined
  `(1-B)^d(1-B^s)^D` scale;
- `predict_interval()` reverses seasonal histories first and ordinary anchors
  second for every path.

## Missing terminal histories

Transformed forecasts remain available when the training sample ends with
missing levels, provided the transformed Kalman model was fitted successfully.
Original-scale intervals require all terminal integration information:

- ordinary lower-order anchors for `d > 0`;
- every required rolling seasonal history for `D > 0`.

When these histories are incomplete, `predict_interval()` raises and directs the
caller to `predict_differenced_interval()` rather than inventing level anchors.

## Reproducibility and simulation size

`random_state` accepts an integer, a NumPy `Generator`, or `None`. Reusing the
same integer seed with the same fitted model and arguments produces identical
interval bounds.

Monte Carlo quantiles have simulation error. Increase `n_simulations` when
accurate tail estimates matter. The default is 2,000 paths; research reporting
will often benefit from 5,000 or more paths and a documented seed.

## Low-level functions

Advanced users can call:

```python
paths = simulate_kalman_forecast_paths(
    filter_result,
    steps=12,
    n_simulations=5000,
    random_state=2026,
)
```

```python
transformed_interval = kalman_forecast_interval(
    filter_result,
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

```python
original_interval = integrated_kalman_forecast_interval(
    filter_result,
    differencing_state,
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

`inverse_forecast_paths()` is also public for applying an existing ordinary or
combined differencing state to a three-dimensional path array.

## Validation references

The test suite verifies:

- an analytic one-dimensional state model with future means `1.5` and `2.0`
  and future variances `5.0` and `6.0`;
- inclusion of both final filtered-state covariance and future innovation
  covariance;
- deterministic reproducibility under a fixed seed;
- equality of interval means and existing recursive point forecasts;
- exact pathwise first-difference reconstruction;
- ordinary random-walk uncertainty widening across horizons;
- rolling seasonal-cycle reconstruction and widening after one full period;
- transformed interval availability when original terminal anchors are missing;
- argument, dimensionality, covariance, and fitted-state validation.

## Limitations

- fitted parameters are fixed;
- parameter covariance is not propagated into forecast paths;
- intervals are Monte Carlo quantiles rather than closed-form Gaussian bounds;
- no observation-noise term exists beyond the STARMA process innovation in the
  current state-space convention;
- the integrated likelihood remains conditional on removed transformation
  history, not exact diffuse on the original level process;
- original-scale filtered or smoothed level-state distributions are not
  returned;
- large seasonal state dimensions and many simulation paths can be expensive.
