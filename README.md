# pySTARMAx

**pySTARMAx** is a typed, research-oriented Python toolkit for classical and
extended space-time autoregressive moving-average modelling.

The package follows the STARMA framework of Pfeifer and Deutsch while keeping
its numerical conventions explicit: observations use `(time, location)`,
spatial lag zero is the identity matrix, non-symmetric spatial weights retain
their supplied orientation, missing observations are never silently imputed,
and public numerical result arrays are immutable.

> **Status — 0.0.13:** conditional and Gaussian Kalman maximum-likelihood
> STARMA estimation, AR stationarity and MA invertibility diagnostics,
> observed-information inference, natural-scale innovation covariance
> delta-method inference, missing-observation filtering, Rauch--Tung--Striebel
> fixed-interval state smoothing, ordinary and multiplicative seasonal STARIMA,
> conditional and bootstrap forecast intervals, and rolling-origin evaluation.
> Smooth admissibility parameterization, original-innovation disturbance
> smoothing, sparse computation, integrated/seasonal Kalman MLE, exogenous
> regressors, and time-varying extensions remain planned.

## Installation

```bash
python -m pip install -e ".[test]"
```

Development and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

## Conditional STARMA quick start

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

`SpatialWeights` stores an ordered collection `(W0, W1, ...)`. `W0` is the
identity matrix. A parameter row therefore contains one coefficient per stored
spatial lag.

## Model convention

For a stationary STARMA model,

\[
z_t = c + \sum_{i=1}^{p} A_i z_{t-i}
      + \varepsilon_t
      + \sum_{j=1}^{q} B_j\varepsilon_{t-j},
\qquad
\varepsilon_t\sim\mathcal N(0,\Sigma),
\]

with

\[
A_i=\sum_k\phi_{ik}W_k,
\qquad
B_j=\sum_k\theta_{jk}W_k.
\]

The package uses a **positive moving-average sign**. Consequently the inverse
innovation recursion is

\[
\varepsilon_t=r_t-\sum_jB_j\varepsilon_{t-j},
\]

and MA invertibility uses the companion top row `[-B1, ..., -Bq]`.

See [`docs/model.md`](docs/model.md).

## Stationarity and invertibility

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

The immutable diagnostic results retain composed lag operators, block companion
matrices, complex eigenvalues, spectral radii, configured limits, signed
boundary distances, and admissibility decisions. Zero-order AR or MA
polynomials have spectral radius zero.

See [`docs/admissibility.md`](docs/admissibility.md).

## Gaussian Kalman maximum likelihood

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
fit = mle.fit(incomplete, weights)

print(fit.summary())
print(mle.admissibility().summary())
print(mle.predict(steps=6))
```

`KalmanSTARMA` maximizes the Gaussian likelihood evaluated by the state-space
filter. Missing locations are removed from that time step's measurement update;
a fully missing row performs prediction only and contributes zero to the
likelihood.

Innovation covariance options are:

- `"scalar"`: one shared variance;
- `"diagonal"`: one variance per location;
- `"full"`: positive-definite Cholesky covariance.

Automatic conditional-estimator starts are independently shrunk into the
configured AR and inverse-MA regions. Candidates outside an enabled region
receive a feasibility penalty, and the final optimizer candidate is checked
again. These are explicit feasibility controls, not a smooth bijective
parameterization.

See [`docs/maximum_likelihood.md`](docs/maximum_likelihood.md).

## Observed-information inference

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
print(inference.minimum_admissibility_distance)
```

The score and Hessian are evaluated with central finite differences of the
negative Gaussian Kalman log likelihood. The inverse observed-information
matrix supplies covariance, standard errors, normal tests, coefficient
intervals, correlations, Hessian rank, eigenvalues, condition number, and score
diagnostics.

A non-positive-definite or rank-deficient Hessian raises by default.
`allow_singular=True` explicitly retains only the positive-curvature eigenspace
and marks the result as a diagnostic pseudoinverse.

See [`docs/likelihood_inference.md`](docs/likelihood_inference.md).

## Natural innovation covariance inference

The optimizer covariance parameters are not covariance elements. Scalar and
diagonal models use log standard deviations; full covariance uses a Cholesky
factor. Version 0.0.12 provides an analytic delta-method transformation:

```python
natural = inference.innovation_covariance_inference()

print(natural.element_table)
print(natural.covariance_matrix)
print(natural.standard_error_matrix)
print(natural.confidence_intervals(level=0.95))
print(natural.dynamic_cross_covariance)
```

For \(\Sigma=LL^\top\), the full-covariance Jacobian uses

\[
d\Sigma=dL\,L^\top+L\,dL^\top.
\]

The natural result reports one shared scalar variance, location variances for a
diagonal model, or unique lower-triangle covariance elements for a full model.
It preserves cross covariance with the dynamic coefficients.

Intervals are first-order unbounded normal approximations. Negative lower
endpoints are not silently clipped, and an ordinary `variance = 0` Wald test is
not reported because zero is a boundary null.

See [`docs/covariance_inference.md`](docs/covariance_inference.md).

## State-space filtering

```python
state_space = mle.to_state_space()
filtered = mle.filter(incomplete)

print(filtered.log_likelihood)
print(filtered.n_observations)
print(filtered.filtered_observations[-1])
```

The fixed-parameter API also supports direct construction:

```python
from pystarmax import build_starma_state_space, kalman_filter

state_space = build_starma_state_space(
    ar_parameters=phi,
    ma_parameters=theta,
    weights=weights,
    innovation_covariance=np.eye(weights.n_locations),
)
filtered = kalman_filter(incomplete, state_space)
```

Initialization options are stationary, user-supplied known moments, and an
explicit approximate diffuse initialization. The diffuse option is a
large-variance approximation, not an exact diffuse likelihood.

See [`docs/state_space.md`](docs/state_space.md).

## Fixed-interval state smoothing

```python
smoothed = mle.smooth(incomplete)

print(smoothed.smoothed_observations)
print(smoothed.smoothed_observation_covariance)
print(smoothed.smoothing_gain)
print(smoothed.lag_one_covariance)
print(smoothed.state_disturbance_mean)
print(smoothed.prediction_rank)
print(smoothed.used_pseudoinverse)
```

`KalmanSTARMA.smooth()` runs a Rauch--Tung--Striebel backward pass over the
filter output. Future observations can therefore refine earlier latent states
and missing blocks. A rank-deficient predicted covariance uses a documented
positive-eigenspace pseudoinverse, with rank and usage recorded per transition.

The returned disturbance arrays describe

```text
alpha_(t+1) - state_intercept - transition @ alpha_t
```

They are **state-equation disturbances**, not automatically the original
location-level innovations when the state selection matrix is not one-to-one.

Low-level use:

```python
from pystarmax import kalman_smoother

smoothed = kalman_smoother(filtered, rcond=1e-10)
```

See [`docs/smoothing.md`](docs/smoothing.md).

## Ordinary STARIMA

```python
from pystarmax import STARIMA

integrated = STARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
)
integrated.fit(integrated_series, weights)

print(integrated.predict_differenced(steps=6))
print(integrated.predict(steps=6))
```

Ordinary differencing is reversible. Forecasts can be returned on the highest
difference scale or reconstructed on the original scale.

See [`docs/starima.md`](docs/starima.md).

## Multiplicative seasonal STARIMA

```python
from pystarmax import SeasonalSTARIMA

seasonal = SeasonalSTARIMA(
    ar_order=1,
    integration_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_period=24,
    include_intercept=False,
)
seasonal.fit(seasonal_series, weights)
print(seasonal.predict(steps=24))
```

Seasonal AR and MA factors are estimated under multiplicative constraints.
Cross-lag matrices are ordered products rather than independent coefficients.
The current direct Kalman maximum-likelihood estimator remains stationary and
non-seasonal.

See [`docs/seasonal.md`](docs/seasonal.md).

## Forecast intervals

Conditional future-innovation intervals:

```python
interval = model.predict_interval(
    steps=24,
    level=0.95,
    n_simulations=2000,
    random_state=42,
)
```

Parameter-aware bootstrap intervals:

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

Each accepted bootstrap replication generates a same-length pseudo-series,
refits the same model specification, and contributes a refitted conditional
mean or a future path. Residual bootstrap samples complete innovation vectors;
parametric bootstrap draws from the fitted contemporaneous covariance.

See [`docs/forecasting.md`](docs/forecasting.md) and
[`docs/bootstrap.md`](docs/bootstrap.md).

## Rolling-origin interval evaluation

```python
from pystarmax import rolling_origin_evaluate

rolling = rolling_origin_evaluate(
    lambda: STARMA(ar_order=1, ma_order=1),
    series,
    weights,
    initial_window=160,
    horizon=6,
    step=6,
    interval_kwargs={"n_simulations": 500},
    random_state=42,
)

print(rolling.metrics())
print(rolling.metrics_by_horizon())
```

Metrics include empirical coverage, signed and absolute coverage error, mean
width, Winkler interval score, MAE, and RMSE.

See [`docs/evaluation.md`](docs/evaluation.md).

## Diagnostics

```python
from pystarmax import space_time_portmanteau, stacf, stcov, stpacf

covariance = stcov(
    series,
    weights,
    past_spatial_lag=1,
    future_spatial_lag=0,
    temporal_lag=1,
)
acf = stacf(result.residuals, weights, max_tlag=8)
pacf = stpacf(series, weights, max_tlag=8)
portmanteau = space_time_portmanteau(
    result.residuals,
    weights,
    max_tlag=8,
)
```

The default STPACF follows the classical nested Yule--Walker construction. A
regression analogue remains available behind an explicit option.

See [`docs/diagnostics.md`](docs/diagnostics.md).

## Validation and supported Python versions

The CI workflow checks:

- Black, isort, Ruff, and mypy;
- independent diagnostic reference regeneration;
- strict MkDocs construction;
- source distribution, wheel, and Twine metadata;
- Ubuntu, Windows, and macOS;
- Python 3.11, 3.12, 3.13, and 3.14;
- branch coverage with an 80% minimum.

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the current authoritative test
count, coverage, pull request, and next-stage handoff.

## Current limitations

- Kalman MLE is stationary and non-seasonal;
- exact diffuse likelihood and smoothing are unavailable;
- smoothing is fixed-parameter and does not propagate parameter uncertainty;
- original location-level innovation disturbance smoothing is not yet exposed;
- AR/MA admissibility uses explicit feasibility penalties rather than a smooth
  parameterization;
- state and spatial matrices are dense;
- natural covariance intervals are first-order normal approximations;
- robust, profile-likelihood, likelihood-ratio, and Kalman-MLE bootstrap
  inference remain future work;
- exogenous regressors and interventions are unsupported;
- bootstrap and rolling refits execute serially.

## Licence and citation

pySTARMAx is released under the MIT licence. Citation metadata is provided in
[`CITATION.cff`](CITATION.cff).
