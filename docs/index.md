# pySTARMAx

pySTARMAx is a typed Python toolkit for classical and extended space-time
autoregressive moving-average modelling.

Version 0.0.29 supports stationary, integrated, multiplicative seasonal, and
original-level exact-diffuse workflows, including optimizer-facing seasonal
exact-diffuse maximum likelihood, fixed-interval state and disturbance
smoothing, observed-information inference, and natural innovation-covariance
inference for

\[
(1-B)^d(1-B^s)^D y_t.
\]

## Core conventions

- observations use `(time, location)`;
- spatial lag zero is the identity matrix;
- supplied orientation is preserved for non-symmetric weights;
- moving-average operators use the positive-sign convention;
- missing values are never silently imputed;
- public result arrays are immutable;
- conditional and exact-diffuse likelihood scopes remain separate.

See [Model convention](model.md) and
[Stationarity and invertibility](admissibility.md).

## Main estimators

```python
from pystarmax import (
    ExactDiffuseKalmanSTARIMA,
    KalmanSTARIMA,
    KalmanSTARMA,
    SeasonalExactDiffuseKalmanSTARIMA,
    SeasonalKalmanSTARIMA,
)
```

### Stationary Gaussian STARMA

Use `KalmanSTARMA` for stationary Gaussian state-space maximum likelihood with
scalar, diagonal, or full-Cholesky innovation covariance.

See [Maximum likelihood](maximum_likelihood.md).

### Conditional integrated STARIMA

Use `KalmanSTARIMA` when ordinary differencing history is conditioned out before
evaluating the stationary transformed-data likelihood.

See [Conditional Kalman STARIMA](integrated_maximum_likelihood.md).

### Ordinary exact-diffuse STARIMA

Use `ExactDiffuseKalmanSTARIMA` when integration directions must be represented
on the original scale with exact diffuse initialization.

The ordinary fitted model includes smoothing, primitive disturbance smoothing,
likelihood inference, forecast intervals, and conditional simulation smoothing.

See:

- [Exact diffuse filtering](exact_diffuse.md)
- [Exact diffuse STARIMA MLE](exact_diffuse_mle.md)
- [Exact diffuse smoothing](exact_diffuse_smoothing.md)
- [Exact diffuse likelihood inference](exact_diffuse_inference.md)
- [Exact diffuse forecast intervals](exact_diffuse_forecast_intervals.md)
- [Exact diffuse simulation smoothing](exact_diffuse_simulation_smoothing.md)

### Conditional multiplicative seasonal STARIMA

Use `SeasonalKalmanSTARIMA` for the conditional transformed-data seasonal
likelihood.

See [Seasonal Kalman STARIMA](seasonal_maximum_likelihood.md).

### Seasonal exact-diffuse STARIMA

Use `SeasonalExactDiffuseKalmanSTARIMA` for optimizer-facing multiplicative
seasonal estimation on original levels:

```python
model = SeasonalExactDiffuseKalmanSTARIMA(
    ar_order=1,
    integration_order=0,
    ma_order=0,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=0,
    seasonal_period=12,
    covariance_type="scalar",
)
result = model.fit(data, weights)
```

At each optimizer candidate the estimator expands the ordered ordinary and
seasonal AR/MA factors, builds the stationary transformed state, constructs the
seasonal original-level exact-diffuse state, and evaluates the exact-diffuse
likelihood. Expanded cross-lag matrices are deterministic and do not add free
AIC/BIC parameters.

The fitted posterior and uncertainty routes are:

```python
smoothed = model.smooth()
disturbances = model.smooth_innovation_disturbances()
inference = model.likelihood_inference()
natural_covariance = inference.innovation_covariance_inference()
```

The smoother routes use the generic exact-diffuse backward information
recursions and the complete seasonal augmented-state selection matrix.

The inference route rebuilds the factor expansion, transformed state, seasonal
exact-diffuse state, and original-level exact filter at every finite-difference
candidate. Its Hessian therefore uses the same likelihood scope as fitting.
Expanded cross lags remain deterministic rather than becoming extra inference
parameters.

See:

- [Seasonal exact diffuse integration](exact_seasonal_integrated.md)
- [Seasonal exact diffuse MLE](seasonal_exact_diffuse_mle.md)
- [Seasonal exact diffuse smoothing](seasonal_exact_diffuse_smoothing.md)
- [Seasonal exact diffuse inference](seasonal_exact_diffuse_inference.md)

## Diagnostics and uncertainty

The package includes:

- STACF and STPACF diagnostics;
- space-time portmanteau testing;
- AR stationarity and inverse-MA invertibility diagnostics;
- stationary, conditional-integrated, conditional-seasonal, ordinary exact-
  diffuse, and seasonal exact-diffuse observed-information inference;
- natural innovation-covariance delta-method inference;
- Gaussian, bootstrap, and ordinary exact-diffuse interval workflows;
- rolling-origin calibration evaluation.

See [Diagnostics](diagnostics.md), [Likelihood inference](likelihood_inference.md),
[Innovation covariance inference](covariance_inference.md), and
[Rolling evaluation](evaluation.md).

## Likelihood scope

Do not combine likelihoods, AIC, BIC, Hessians, or covariance estimates from
different likelihood conventions:

- conditional transformed-data estimators remove or condition on differencing
  history;
- exact-diffuse estimators evaluate original observations while diffuse
  directions are identified by the data.

This boundary is part of the public API and test suite.

## Validation status

Version 0.0.29 adds closed-form seasonal-random-walk information, natural scalar-
variance delta-method inference, ordinary exact-diffuse reduction, missing-data
curvature, full-Cholesky covariance transformation, singular-Hessian policy, and
public API tests.

The final synchronized test count and coverage are recorded in
`PROJECT_STATUS.md` and the Step 29 handoff. CI covers Ubuntu, Windows, and macOS
on Python 3.11–3.14, strict MkDocs, formatting, linting, typing, reference
regeneration, and package builds.

## Development

The active roadmap is documented in [Roadmap](roadmap.md). Detailed delivery
inventory and immutable handoffs are maintained under `docs/development/`.
