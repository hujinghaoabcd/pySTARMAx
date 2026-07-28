# pySTARMAx

**pySTARMAx** is a modern, extensible Python toolkit for classical
space-time autoregressive moving-average modelling.

The project starts from the STARMA framework of Pfeifer and Deutsch and follows
the engineering conventions used in **pyGWRx** and **pyKDEX**: a `src/` layout,
strict validation, typed public APIs, structured result objects, independent
numerical implementation, reproducible tests, and explicit research references.

> Status: version 0.0.3 implements spatial-weight handling, STAR and iterative
> conditional STARMA estimation, ordinary `STARIMA(p, d, q)` differencing,
> original-scale forecast inversion, stationary and integrated simulation, the
> classical Pfeifer–Deutsch STACF and nested Yule–Walker STPACF, the earlier
> regression diagnostic as an explicit alternative, and a residual portmanteau
> test. Seasonal operators, missing-data state-space estimation,
> correlated-innovation likelihoods, and time-varying extensions are planned.

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
- ordinary differencing with immutable forecast-inversion state;
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
iterative conditional least-squares procedure for STARMA models. It is intended
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
