# pySTARMAx

**pySTARMAx** is a modern, extensible Python toolkit for classical
space-time autoregressive moving-average modelling.

The project starts from the STARMA framework of Pfeifer and Deutsch and follows
the engineering conventions used in **pyGWRx** and **pyKDEX**: a `src/` layout,
strict validation, typed public APIs, structured result objects, independent
numerical implementation, reproducible tests, and explicit research references.

> Status: version 0.0.5 implements spatial-weight handling, STAR and iterative
> conditional STARMA estimation, ordinary `STARIMA(p, d, q)` and multiplicative seasonal
> `(p,d,q)x(P,D,Q)_s` modelling, reversible ordinary-seasonal differencing,
> original-scale forecast inversion, stationary and integrated simulation, the
> classical Pfeifer–Deutsch STACF and nested Yule–Walker STPACF, the earlier
> regression diagnostic as an explicit alternative, and a residual portmanteau
> test. Missing-data state-space estimation, correlated-innovation likelihoods,
> parameter-uncertainty intervals, missing-data state-space estimation, and
time-varying extensions are planned.

## Installation

```bash
python -m pip install -e ".[test]"
```

## Quick start

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


## Fitted values and forecast intervals

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
- structured fit results with coefficients, uncertainty, residual covariance,
  log likelihood, AIC, BIC, convergence state, and readable summaries;
- deterministic stationary and integrated simulation with static numerical tests;
- ordinary and seasonal differencing with immutable forecast-inversion state;
- aligned original-scale one-step fitted values;
- conditional innovation intervals with full pathwise inverse differencing;
- factorized multiplicative seasonal operators with explicit matrix order;
- exact-rational reference fixtures generated without importing pySTARMAx;
- explicit covariance orientation for non-symmetric row-standardized weights;
- one public numerical route first, with sparse and compiled acceleration hidden
  behind stable interfaces later;
- research references and implementation limitations documented in the repository.

The classical STPACF is computed from nested leading-principal Yule–Walker
systems in temporal-major, spatial-minor order. ``stpacf(..., method="regression")``
retains the projection-based diagnostic shipped in 0.0.1 for reproducibility.

## Scope of the conditional estimator

The current estimator uses ordinary least squares for pure STAR models and an
iterative conditional least-squares procedure for STARMA models. Seasonal factor
models use nonlinear conditional least squares so multiplicative cross terms remain
parameter products rather than independent coefficients. The package is intended
as a transparent, testable baseline. It is not yet a replacement for a fully
specified state-space maximum-likelihood implementation when innovations are
contemporaneously correlated, observations are missing, or uncertainty from
estimated innovations must be propagated exactly.

## References

The initial architecture is grounded in the classical STARMA identification,
estimation, seasonal modelling, and residual-diagnostic literature. See
[`docs/references.md`](docs/references.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Licence

MIT. See `THIRD_PARTY_NOTICES.md` for research references and implementation
independence notes.
