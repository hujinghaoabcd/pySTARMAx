# pySTARMAx

**pySTARMAx** is a modern, extensible Python toolkit for classical
space-time autoregressive moving-average modelling.

The project starts from the STARMA framework of Pfeifer and Deutsch and follows
the engineering conventions used in **pyGWRx** and **pyKDEX**: a `src/` layout,
strict validation, typed public APIs, structured immutable result objects,
independent numerical implementation, reproducible tests, and explicit research
references.

> Status: version 0.0.11 implements spatial-weight handling, conditional and
> Gaussian Kalman maximum-likelihood STARMA estimation, AR stationarity and MA
> invertibility diagnostics, dual admissibility enforcement, observed-likelihood
> Hessian inference, ordinary `STARIMA(p, d, q)` and multiplicative seasonal
> `(p,d,q)x(P,D,Q)_s` modelling, reversible ordinary-seasonal differencing,
> original-scale fitted values and forecasts, conditional and bootstrap
> intervals, rolling-origin evaluation, and state-space filtering with partial
> missing observations. Smooth admissibility parameterization, sparse
> computation, smoothing, and time-varying extensions remain planned.

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

## Stationarity and invertibility

For the stationary model

\[
z_t = c + \sum_i A_i z_{t-i} + \varepsilon_t
      + \sum_j B_j\varepsilon_{t-j},
\]

pySTARMAx evaluates AR stationarity with the block companion top row
`[A_1, ..., A_p]`.

Because the package uses a positive MA sign, the innovation inverse recursion is

\[
\varepsilon_t = r_t - \sum_j B_j\varepsilon_{t-j},
\]

so MA invertibility uses the inverse-recursion companion top row
`[-B_1, ..., -B_q]`.

```python
from pystarmax import (
    autoregressive_diagnostics,
    moving_average_diagnostics,
    starma_admissibility,
)

ar = autoregressive_diagnostics(phi, weights, margin=1e-6)
ma = moving_average_diagnostics(theta, weights, margin=1e-6)
joint = starma_admissibility(phi, theta, weights)

print(ar.spectral_radius)
print(ma.eigenvalues)
print(joint.summary())
```

The immutable diagnostics retain composed lag matrices, block companions,
complex eigenvalues, spectral radii, configured limits, signed boundary
distances, and admissibility decisions. Non-symmetric spatial-matrix orientation
is preserved exactly. Zero-order AR or MA polynomials have radius zero.

See [`docs/admissibility.md`](docs/admissibility.md).

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
    enforce_stationarity=True,
    stability_margin=1e-6,
    enforce_invertibility=True,
    invertibility_margin=1e-6,
)
mle_result = mle.fit(incomplete, weights)
fitted_admissibility = mle.admissibility()
inference = mle.infer(relative_step=1e-4)

print(mle_result.summary())
print(fitted_admissibility.summary())
print(inference.summary())
print(mle.predict(steps=6))
```

`KalmanSTARMA` directly maximizes the Gaussian likelihood evaluated by the
state-space filter. Missing locations are omitted from the corresponding
measurement update; fully missing rows perform state prediction only. Covariance
options are `"scalar"`, `"diagonal"`, and positive-definite `"full"` Cholesky
parameterization.

Automatic conditional-estimator starts are independently shrunk into the
configured AR and inverse-MA regions. Likelihood candidates outside either
enabled region receive a large feasibility penalty, and the final optimizer
candidate is hard-checked again.

The optimizer reports convergence, iterations, function evaluations, both
spectral radii and limits, enforcement flags, signed boundary distances, joint
admissibility, log likelihood, AIC, and BIC. The current controls are explicit
spectral-radius feasibility penalties, not a smooth reparameterization.

Setting either enforcement flag to `False` is an explicit research control. The
corresponding diagnostics are still computed, so a returned non-stationary or
non-invertible result is visible rather than hidden.

See [`docs/maximum_likelihood.md`](docs/maximum_likelihood.md).

## Likelihood-curvature inference

```python
inference = mle.infer(
    relative_step=1e-4,
    absolute_step=1e-6,
    rcond=1e-10,
)

print(inference.coefficient_table)
print(inference.optimizer_table)
print(inference.confidence_intervals(level=0.95))
print(inference.eigenvalues)
print(inference.condition_number)
print(inference.stability_boundary_distance)
print(inference.invertibility_boundary_distance)
print(inference.minimum_admissibility_distance)
```

`KalmanSTARMA.infer()` computes a central finite-difference score and Hessian of
the negative Gaussian Kalman log likelihood at the fitted optimizer parameters.
The inverse observed-information matrix provides covariance, standard errors,
z statistics, normal-approximation p values, parameter correlations, and
confidence intervals for the dynamic coefficients.

Inference is reported on the raw optimizer scale. Intercept, AR, and MA entries
are already on their natural coefficient scale. Covariance entries remain
log-standard-deviation or Cholesky-factor parameters; their standard errors are
not automatically transformed into covariance-element standard errors.

A non-positive-definite or rank-deficient Hessian raises by default. Setting
`allow_singular=True` explicitly uses only the positive-curvature eigenspace and
marks `used_pseudoinverse=True`; this route is diagnostic and should not be
silently treated as regular publication-quality inference.

Finite-difference stencils use the same AR and MA constraints as fitting. A
perturbed point crossing either enabled feasibility boundary is rejected rather
than differentiated through the artificial penalty.

See [`docs/likelihood_inference.md`](docs/likelihood_inference.md).

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
diffuse option are also available.

See [`docs/state_space.md`](docs/state_space.md).

## Ordinary STARIMA

```python
from pystarmax import STARIMA

model = STARIMA(ar_order=1, integration_order=1, ma_order=1)
result = model.fit(integrated_series, weights)

print(result.summary())
print(model.predict(steps=6))
```

`ordinary_difference()` exposes the same reversible transformation independently,
and `predict_differenced()` returns forecasts before inverse differencing. The
fit result remains on the highest-difference scale; forecasts are reconstructed
on the original scale.

See [`docs/starima.md`](docs/starima.md).

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
independent coefficients. The 0.0.11 admissibility diagnostics currently target
the stationary non-seasonal STARMA polynomial; multiplicative seasonal factor
admissibility remains future work.

See [`docs/seasonal.md`](docs/seasonal.md).

## Fitted values and conditional intervals

```python
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
intervals condition on estimated parameters.

See [`docs/forecasting.md`](docs/forecasting.md).

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
than requested.

See [`docs/bootstrap.md`](docs/bootstrap.md).

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
pacf = stpacf(series, weights, max_tlag=4)
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

The classical STPACF is computed from nested leading-principal Yule-Walker
systems in temporal-major, spatial-minor order. `stpacf(..., method="regression")`
retains the projection-based diagnostic shipped in 0.0.1 for reproducibility.

## Design commitments

- independent NumPy/SciPy implementation rather than runtime delegation to R;
- explicit `(time, location)` convention and immutable spatial weights;
- preservation of non-symmetric matrix orientation;
- separate conditional and Kalman maximum-likelihood estimation routes;
- scalar, diagonal, and Cholesky full innovation covariance models;
- explicit AR and positive-sign inverse-MA companion definitions;
- complete eigensystems, spectral radii, limits, and signed boundary distances;
- dual feasibility enforcement with visible disabled-constraint behavior;
- central finite-difference likelihood score and observed-information Hessian;
- explicit Hessian rank, eigenvalue, condition-number, score, and boundary checks;
- structured fit, admissibility, and inference results with immutable arrays;
- deterministic stationary and integrated simulation with static numerical tests;
- ordinary and seasonal differencing with immutable forecast-inversion state;
- aligned original-scale one-step fitted values;
- conditional innovation intervals with pathwise inverse differencing;
- residual and parametric direct-bootstrap intervals with model refitting;
- factorized multiplicative seasonal operators with explicit matrix order;
- explicit state-space matrices and missing-observation Kalman updates;
- Cholesky likelihood solves with recorded numerical jitter;
- exact-rational reference fixtures generated without importing pySTARMAx;
- one public numerical route first, with sparse and compiled acceleration hidden
  behind stable interfaces later;
- explicit implementation limitations in repository documentation.

## Estimation and inference scope

`STAR` uses ordinary least squares, while `STARMA` uses iterative conditional
least squares. Seasonal factor models use nonlinear conditional least squares so
multiplicative cross terms remain parameter products rather than independent
coefficients. These estimators remain the basis of existing bootstrap intervals.

`KalmanSTARMA` is a distinct stationary Gaussian maximum-likelihood estimator.
It supports complete or incomplete observations and scalar, diagonal, or full
contemporaneous innovation covariance. It uses L-BFGS-B with explicit AR
stationarity and MA invertibility feasibility boundaries.

The current general matrix-polynomial constraints are not smooth bijections.
Coefficient-wise clipping or `tanh` transforms are not used because they do not
guarantee a multivariate companion radius below one.

Likelihood-curvature inference uses the observed Hessian at the fitted optimizer
point. It does not currently include sandwich covariance, delta-method covariance
transforms, profile likelihood, exact diffuse likelihood, or automatic weak-
identification correction. Smooth admissibility parameterization, smoothing,
integrated-seasonal maximum-likelihood wrappers, and sparse computation remain
future work.

## References

The architecture is grounded in the classical STARMA identification, estimation,
seasonal modelling, residual-diagnostic, bootstrap predictive-inference, and
linear Gaussian state-space literature. See
[`docs/references.md`](docs/references.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Licence

MIT. See `THIRD_PARTY_NOTICES.md` for research references and implementation
independence notes.
