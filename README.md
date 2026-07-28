# pySTARMAx

**pySTARMAx** is a modern, extensible Python toolkit for classical
space-time autoregressive moving-average modelling.

The project starts from the STARMA framework of Pfeifer and Deutsch and follows
the engineering conventions used in **pyGWRx** and **pyKDEX**: a `src/` layout,
strict validation, typed public APIs, structured result objects, independent
numerical implementation, reproducible tests, and explicit research references.

> Status: the first development baseline implements spatial-weight handling,
> STAR and iterative conditional STARMA estimation, recursive forecasting,
> simulation, STACF/STPACF diagnostics, and a residual portmanteau test.
> STARIMA differencing, seasonal operators, missing-data state-space estimation,
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

## Diagnostics

```python
from pystarmax import space_time_portmanteau, stacf, stpacf

acf = stacf(result.residuals, weights, max_tlag=8)
pacf = stpacf(series, weights, max_tlag=4)
test = space_time_portmanteau(
    result.residuals,
    weights,
    max_tlag=8,
    fit_params=result.n_params,
)

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
- deterministic simulation and static numerical tests;
- one public numerical route first, with sparse and compiled acceleration hidden
  behind stable interfaces later;
- research references and implementation limitations documented in the repository.

## Scope of the first baseline

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
