# pySTARMAx

pySTARMAx is a typed Python toolkit for classical and extended space-time
autoregressive moving-average modelling.

Version 0.0.27 supports stationary, integrated, multiplicative seasonal, and
original-level exact-diffuse workflows, including optimizer-facing seasonal
exact-diffuse maximum likelihood for

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
likelihood.

Expanded cross-lag matrices are deterministic and do not add free AIC/BIC
parameters.

See:

- [Seasonal exact diffuse integration](exact_seasonal_integrated.md)
- [Seasonal exact diffuse MLE](seasonal_exact_diffuse_mle.md)

## Diagnostics and uncertainty

The package includes:

- STACF and STPACF diagnostics;
- space-time portmanteau testing;
- AR stationarity and inverse-MA invertibility diagnostics;
- finite-difference observed-information inference;
- natural innovation-covariance delta-method inference;
- Gaussian, bootstrap, and exact-diffuse interval workflows;
- rolling-origin calibration evaluation.

See [Diagnostics](diagnostics.md), [Likelihood inference](likelihood_inference.md),
[Innovation covariance inference](covariance_inference.md), and
[Rolling evaluation](evaluation.md).

## Likelihood scope

Do not combine AIC or BIC from different likelihood conventions:

- conditional transformed-data estimators remove or condition on differencing
  history;
- exact-diffuse estimators evaluate original observations while diffuse
  directions are identified by the data.

This boundary is part of the public API and test suite.

## Validation status

The 0.0.27 implementation is validated by 231 tests with 87.10% total branch
coverage and 82.9% branch coverage for the new seasonal exact-diffuse MLE
module. CI covers Ubuntu, Windows, and macOS on Python 3.11–3.14, strict MkDocs,
formatting, linting, typing, reference regeneration, and package builds.

## Development

The active roadmap is documented in [Roadmap](roadmap.md). Detailed delivery
inventory and immutable handoffs are maintained under `docs/development/`.
