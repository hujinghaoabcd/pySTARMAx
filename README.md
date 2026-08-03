# pySTARMAx

**pySTARMAx** is a modern, extensible Python toolkit for classical
space-time autoregressive moving-average modelling.

The project starts from the STARMA framework of Pfeifer and Deutsch and follows
the engineering conventions used in **pyGWRx** and **pyKDEX**: a `src/` layout,
strict validation, typed public APIs, structured result objects, independent
numerical implementation, reproducible tests, and explicit research references.

> Status: version 0.0.9 implements spatial-weight handling, conditional and
> Gaussian Kalman maximum-likelihood STARMA estimation, ordinary
> `STARIMA(p, d, q)` and multiplicative seasonal `(p,d,q)x(P,D,Q)_s` modelling,
> reversible ordinary-seasonal differencing, original-scale fitted values and
> forecasts, conditional and bootstrap intervals, rolling-origin evaluation,
> and state-space filtering with partial missing observations. Likelihood-Hessian
> inference, smooth stability constraints, sparse computation, and time-varying
> extensions remain planned.

## Installation

```bash
python -m pip install -e ".[test]"
```

## Conditional-estimation quick start

```python
import numpy as np
from pystarmax import STARMA, SpatialWeights, lattice_weights, simulate_starma

weights = SpatialWeights.from_adjacency(
    lattice_weights(2, 3),
    max_order=1,
)

series = simulate_starma(
    phi=np.array([[0.45, 0.20]]),
    theta=np.array([[0.15, 0.05]]),
    weights=weights,
    n_steps=300,
    random_state=42,
)

model = STARMA(ar_order=1, ma_order=1, max_iter=100)
result = model.fit(series, weights)

print(result.summary())
print(model.predict(steps=6))
```

The observation matrix uses the convention `(time, location)`. Spatial lag zero
is the identity matrix; higher spatial lags are stored in `SpatialWeights`.

## Kalman maximum likelihood

```python
from pystarmax import KalmanSTARMA

incomplete = series.copy()
incomplete[20, 2] = np.nan
incomplete[80, :] = np.nan

mle = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=False,
)
mle_result = mle.fit(incomplete, weights)

print(mle_result.summary())
print(mle.predict(steps=6))
```

`KalmanSTARMA` directly maximizes the Gaussian likelihood evaluated by the
state-space filter. Missing locations are omitted from the corresponding
measurement update; fully missing rows perform state prediction only. Covariance
options are `"scalar"`, `"diagonal"`, and positive-definite `"full"` Cholesky
parameterization.

The optimizer reports convergence, iterations, function evaluations, spectral
radius, log likelihood, AIC, and BIC. The current stationarity control is an
explicit spectral-radius feasibility penalty, not a smooth reparameterization.
Likelihood-Hessian standard errors and MA invertibility constraints are not yet
included. See [`docs/maximum_likelihood.md`](docs/maximum_likelihood.md).

## Fixed-parameter state-space filtering

```python
state_space = model.to_state_space()
filtered = model.filter_state_space(incomplete)

print(filtered.log_likelihood)
print(filtered.n_observations)
print(filtered.filtered_observations[-1])
```

`to_state_space()` maps fitted conditional `STAR` or `STARMA` coefficients to an
explicit companion representation. `filter_state_space()` evaluates those fixed
parameters without replacing the original estimator or its result object. The
default stationary initialization uses the unconditional state mean and a
discrete Lyapunov covariance. Known initialization and an explicit approximate
diffuse option are also available. See
[`docs/state_space.md`](docs/state_space.md).

## Ordinary STARIMA

```python
from pystarmax import STARIMA

model = STARIMA(ar_order=1, integration_order=1, ma_order=1)
result = model.fit(integrated_series, weights)

# Inference remains on the differenced scale.
print(result.summary())

# Forecasts are reconstructed on the original scale.
print(model.predict(steps=6))
```

`ordinary_difference()` exposes the same reversible transformation independently,
and `predict_differenced()` returns forecasts before inverse differencing. See
[`docs/starima.md`](docs/starima.md) for state and scale conventions.

## Seasonal STARIMA

```python
from pystarmax import SeasonalSTARIMA

model = SeasonalSTARIMA(
    ar_order=1,
    integration_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_period=24,
    include_intercept=False,
)
result = model.fit(seasonal_series, weights)
print(model.predict(steps=24))
```

Seasonal AR and MA factors are estimated under true multiplicative constraints.
Cross-lag matrices are generated as ordered matrix products rather than fitted as
independent coefficients. See [`docs/seasonal.md`](docs/seasonal.md).

## Fitted values and conditional intervals

```python
# One-step fitted values aligned to the original observation matrix.
fitted = model.fitted_original()

interval = model.predict_interval(
    steps=24,
    level=0.95,
    n_simulations=2000,
    random_state=42,
)
print(interval.mean)
print(interval.lower)
print(interval.upper)
```

`fitted_original()` uses observed historical values when reversing differencing,
so it represents aligned one-step conditional fits rather than a recursively
integrated pseudo-series. `predict_interval()` simulates future innovations from
the fitted location covariance, propagates them through AR and MA dynamics, and
then inverts each ordinary-seasonal path before computing quantiles. These
intervals condition on estimated parameters. See
[`docs/forecasting.md`](docs/forecasting.md).

## Bootstrap parameter uncertainty

```python
bootstrap = model.predict_bootstrap_interval(
    steps=24,
    level=0.95,
    n_bootstrap=500,
    bootstrap_method="residual",
    include_future_innovations=True,
    random_state=42,
)
```

Each accepted replication generates a same-length pseudo-series, refits the same
model specification, and contributes either a refitted conditional mean or one
future path. Residual bootstrap samples complete innovation vectors, preserving
contemporaneous location dependence. Parametric bootstrap draws from the fitted
innovation covariance.

Set `include_future_innovations=False` for a parameter-only interval. The default
combines parameter-estimation and future-innovation uncertainty. Failed refits
are retried up to `max_attempts`; the method never silently returns fewer paths
than requested. See [`docs/bootstrap.md`](docs/bootstrap.md).

## Diagnostics

```python
from pystarmax import space_time_portmanteau, stacf, stcov, stpacf

gamma_10_1 = stcov(
    series,
    weights,
    past_spatial_lag=1,
    future_spatial_lag=0,
    temporal_lag=1,
)
acf = stacf(result.residuals, weights, max_tlag=8)
pacf = stpacf(series, weights, max_tlag=4)  # classical Yule-Walker default
test = space_time_portmanteau(
    result.residuals,
    weights,
    max_tlag=8,
    fit_params=result.n_params,
)

print(gamma_10_1)
print(acf)
print(pacf)
print(test)
```

## Design commitments

- independent NumPy/SciPy implementation rather than runtime delegation to R;
- explicit `(time, location)` data conventions and immutable weight collections;
- identity, adjacency, distance, and higher-order spatial-weight construction;
- separate conditional and Kalman maximum-likelihood estimation routes;
- scalar, diagonal, and Cholesky full innovation covariance models;
- structured fit results with coefficients, covariance, likelihood, information
  criteria, convergence state, optimizer diagnostics, and readable summaries;
- deterministic stationary and integrated simulation with static numerical tests;
- ordinary and seasonal differencing with immutable forecast-inversion state;
- aligned original-scale one-step fitted values;
- conditional innovation intervals with full pathwise inverse differencing;
- residual and parametric direct-bootstrap intervals with model refitting;
- factorized multiplicative seasonal operators with explicit matrix order;
- explicit state-space matrices and missing-observation Kalman updates;
- Cholesky likelihood solves with recorded numerical jitter;
- exact-rational reference fixtures generated without importing pySTARMAx;
- explicit covariance orientation for non-symmetric row-standardized weights;
- one public numerical route first, with sparse and compiled acceleration hidden
  behind stable interfaces later;
- research references and implementation limitations documented in the repository.

The classical STPACF is computed from nested leading-principal Yule-Walker
systems in temporal-major, spatial-minor order. `stpacf(..., method="regression")`
retains the projection-based diagnostic shipped in 0.0.1 for reproducibility.

## Estimation and likelihood scope

`STAR` uses ordinary least squares, while `STARMA` uses iterative conditional
least squares. Seasonal factor models use nonlinear conditional least squares so
multiplicative cross terms remain parameter products rather than independent
coefficients. These estimators remain the basis of existing bootstrap intervals.

`KalmanSTARMA` is a distinct stationary Gaussian maximum-likelihood estimator.
It supports complete or incomplete observations and scalar, diagonal, or full
contemporaneous innovation covariance. It currently uses L-BFGS-B with an
explicit stationarity feasibility boundary. Likelihood-Hessian uncertainty,
smooth stationarity and invertibility parameterization, exact diffuse likelihood,
smoothing, and integrated-seasonal maximum-likelihood wrappers remain future
work.

## References

The architecture is grounded in the classical STARMA identification, estimation,
seasonal modelling, residual-diagnostic, bootstrap predictive-inference, and
linear Gaussian state-space literature. See
[`docs/references.md`](docs/references.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Licence

MIT. See `THIRD_PARTY_NOTICES.md` for research references and implementation
independence notes.
