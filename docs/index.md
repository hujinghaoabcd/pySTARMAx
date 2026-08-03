# pySTARMAx

pySTARMAx provides transparent, typed building blocks for classical STARMA,
ordinary STARIMA, and multiplicative seasonal STARIMA modelling in Python. The
current workflow covers spatial-weight construction, simulation, conditional
and Gaussian Kalman maximum-likelihood estimation, conditional ordinary and
multiplicative seasonal Kalman STARIMA, AR stationarity and positive-sign MA
invertibility diagnostics, observed-likelihood Hessian inference, natural-scale
innovation covariance inference, missing-observation filtering, fixed-interval
state smoothing, original location-level innovation disturbance smoothing,
combined differencing, original-scale forecast reconstruction, conditional and
bootstrap intervals, and rolling-origin calibration diagnostics.

```python
import numpy as np

from pystarmax import (
    KalmanSTARIMA,
    KalmanSTARMA,
    STARMA,
    SeasonalKalmanSTARIMA,
    rolling_origin_evaluate,
)

conditional = STARMA(ar_order=1, ma_order=1)
conditional_result = conditional.fit(y, weights)
print(conditional_result.summary())

incomplete = y.copy()
incomplete[20:24, 1] = np.nan
incomplete[80, :] = np.nan

mle = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    enforce_stationarity=True,
    enforce_invertibility=True,
)
mle_result = mle.fit(incomplete, weights)
likelihood_inference = mle.infer(relative_step=1e-4)
smoothed = mle.smooth(incomplete)
innovation_disturbances = mle.smooth_innovation_disturbances(incomplete)

print(mle_result.summary())
print(likelihood_inference.coefficient_table)
print(smoothed.smoothed_observations[20:24, 1])
print(innovation_disturbances.innovation_mean)

integrated_mle = KalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    covariance_type="full",
)
integrated_result = integrated_mle.fit(level_series, weights)
print(integrated_result.summary())
print(integrated_mle.predict(steps=6))

seasonal_mle = SeasonalKalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=1,
    seasonal_period=24,
    covariance_type="full",
)
seasonal_result = seasonal_mle.fit(seasonal_levels, weights)
print(seasonal_result.summary())
print(seasonal_mle.admissibility().summary())
print(seasonal_mle.predict(steps=24))

evaluation = rolling_origin_evaluate(
    lambda: STARMA(ar_order=1, ma_order=1),
    y,
    weights,
    initial_window=120,
    horizon=6,
    step=6,
    interval_kwargs={"n_simulations": 500},
    random_state=42,
)
print(evaluation.metrics())
```

See [Model convention](model.md) for the STARMA equation and data orientation,
[Stationarity and invertibility](admissibility.md) for AR and inverse-MA
companion diagnostics, [State-space filtering](state_space.md) for missing-data
filtering, and [Fixed-interval smoothing](smoothing.md) for RTS state and
state-disturbance moments.

[Original innovation smoothing](innovation_smoothing.md) documents the exact
conditional-Gaussian transformation from `R @ eta_t` back to the original
location-level innovation, including posterior covariance, rank-deficient
process support, and selection-matrix null-space uncertainty.

[Maximum likelihood](maximum_likelihood.md),
[Likelihood inference](likelihood_inference.md), and
[Innovation covariance inference](covariance_inference.md) document stationary
Gaussian estimation, observed-information curvature, and analytic delta-method
transformation to natural variance and covariance elements.

[Conditional integrated Kalman STARIMA](integrated_maximum_likelihood.md)
documents ordinary differencing before the stationary likelihood, conditioning
on the first `d` level rows, missing-value propagation, and the separation
between transformed and recursively restored original scales.

[Seasonal likelihood inference](seasonal_likelihood_inference.md) reuses the
observed-information result contract for ordinary and seasonal factor parameters,
rejects expanded-boundary stencil points, and enables natural innovation covariance
delta-method inference with factor/covariance cross uncertainty.

[Seasonal STARIMA](seasonal.md) compares the conditional and Gaussian seasonal
routes. [Multiplicative seasonal Kalman STARIMA](seasonal_maximum_likelihood.md)
documents ordered matrix-polynomial expansion, arbitrary-lag companion states,
factor parameter counting, combined ordinary-seasonal differencing, expanded
stationarity/invertibility checks, and missing-data handling. Both likelihood
wrappers remain conditional on the transformation history and are distinct from
an exact diffuse integrated level-state likelihood.

[Forecasting](forecasting.md), [Bootstrap intervals](bootstrap.md), and
[Rolling evaluation](evaluation.md) cover predictive uncertainty, parameter
refitting, calibration, sharpness, and point-error assessment.
