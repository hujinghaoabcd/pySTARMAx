# pySTARMAx

pySTARMAx is a typed Python toolkit for classical and extended space-time
autoregressive moving-average modelling.

Version 0.0.32 supports stationary, integrated, multiplicative seasonal, and
original-level exact-diffuse workflows. Ordinary and seasonal exact-diffuse
models now include estimation, smoothing, adjacent-time covariance, likelihood
inference, forecasting, and conditional path simulation.

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

### Stationary and conditional models

- `KalmanSTARMA` provides stationary Gaussian state-space maximum likelihood.
- `KalmanSTARIMA` provides an ordinary-difference conditional likelihood.
- `SeasonalKalmanSTARIMA` provides a multiplicative seasonal conditional
  transformed-data likelihood.

See [Maximum likelihood](maximum_likelihood.md),
[Conditional Kalman STARIMA](integrated_maximum_likelihood.md), and
[Seasonal Kalman STARIMA](seasonal_maximum_likelihood.md).

### Ordinary exact-diffuse STARIMA

Use `ExactDiffuseKalmanSTARIMA` when integration directions must be represented
on the original scale with exact diffuse initialization. Its fitted facade
includes:

```python
model.smooth()
model.smooth_lag_one_covariance()
model.smooth_innovation_disturbances()
model.likelihood_inference()
model.simulate_smoothing_paths(
    n_simulations=1000,
    random_state=7,
)
model.predict_interval(steps=12, n_simulations=5000, random_state=7)
```

See:

- [Exact diffuse filtering](exact_diffuse.md)
- [Exact diffuse STARIMA MLE](exact_diffuse_mle.md)
- [Exact diffuse smoothing](exact_diffuse_smoothing.md)
- [Exact diffuse disturbance smoothing](exact_diffuse_disturbance_smoothing.md)
- [Exact diffuse lag-one covariance](exact_diffuse_lag_one_covariance.md)
- [Exact diffuse likelihood inference](exact_diffuse_inference.md)
- [Exact diffuse forecast intervals](exact_diffuse_forecast_intervals.md)
- [Exact diffuse simulation smoothing](exact_diffuse_simulation_smoothing.md)

### Seasonal exact-diffuse STARIMA

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
seasonal factor polynomials, builds the transformed state, constructs the
original-level exact-diffuse augmentation, and filters the original
observations. Expanded cross lags remain deterministic and do not add free
parameters.

The seasonal fitted facade exposes the same posterior operations on the complete
augmented state:

```python
model.smooth()
model.smooth_lag_one_covariance()
model.smooth_innovation_disturbances()
model.likelihood_inference()
model.simulate_smoothing_paths(
    n_simulations=1000,
    random_state=7,
)
model.predict_interval(steps=12, n_simulations=5000, random_state=7)
model.predict_differenced_interval(
    steps=12,
    n_simulations=5000,
    random_state=7,
)
```

See:

- [Seasonal exact diffuse integration](exact_seasonal_integrated.md)
- [Seasonal exact diffuse MLE](seasonal_exact_diffuse_mle.md)
- [Seasonal exact diffuse smoothing](seasonal_exact_diffuse_smoothing.md)
- [Seasonal exact diffuse inference](seasonal_exact_diffuse_inference.md)
- [Seasonal exact diffuse forecasting](seasonal_exact_diffuse_forecasting.md)
- [Seasonal exact diffuse simulation smoothing](seasonal_exact_diffuse_simulation_smoothing.md)

## Exact adjacent-time covariance

Version 0.0.32 returns

\[
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y_{1:T})
\]

for genuine diffuse phases. The dense exact construction analytically eliminates
identified diffuse coordinates and conditions the remaining proper Gaussian
source vector. It does not use a large finite diffuse variance or a Monte Carlo
covariance estimate.

The output uses the ordinary RTS orientation

```python
result.lag_one_covariance[t]
```

for left state `alpha_t` and right state `alpha_(t+1)`. State-disturbance
covariance reconstructed from these moments is checked against the independent
exact information-form disturbance smoother.

The same routine covers ordinary states and complete seasonal augmented states.
It is a transparent moderate-sample reference for the future memory-linear
exact diffuse `L2` recursion.

See [Exact diffuse lag-one covariance](exact_diffuse_lag_one_covariance.md).

## Diagnostics and uncertainty

The package includes:

- STACF and STPACF diagnostics;
- space-time portmanteau testing;
- AR stationarity and inverse-MA invertibility diagnostics;
- observed-information and natural innovation-covariance inference;
- Gaussian, bootstrap, ordinary exact-diffuse, and seasonal exact-diffuse
  interval workflows;
- exact-diffuse conditional path simulation and adjacent-time covariance;
- rolling-origin calibration evaluation.

See [Diagnostics](diagnostics.md), [Likelihood inference](likelihood_inference.md),
[Innovation covariance inference](covariance_inference.md), and
[Rolling evaluation](evaluation.md).

## Likelihood and uncertainty boundaries

Do not combine likelihoods, AIC, BIC, Hessians, or covariance estimates from
different likelihood conventions. Conditional estimators remove or condition on
differencing history; exact-diffuse estimators evaluate original observations.

Forecast intervals and conditional paths currently condition on fitted
parameters. Parameter uncertainty remains a separate future contract.

## Validation status

Authoritative implementation CI #660 passed 266 tests with 87.40% total branch
coverage. The new exact-diffuse lag-one covariance module has 89.7% branch
coverage. Independent references include a diffuse local-level closed form,
exact ordinary RTS reduction, non-symmetric transition orientation,
conditional-path Monte Carlo cross-covariance, seasonal augmented states, and
state-disturbance reconstruction.

CI covers Ubuntu, Windows, and macOS on Python 3.11–3.14, strict MkDocs,
formatting, linting, typing, reference regeneration, and package builds.

## Development

The active roadmap is documented in [Roadmap](roadmap.md). Detailed delivery
inventory and immutable handoffs are maintained under `docs/development/`.