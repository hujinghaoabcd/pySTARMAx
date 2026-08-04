# pySTARMAx

**pySTARMAx** is a typed, research-oriented Python toolkit for classical and
extended space-time autoregressive moving-average modelling.

The package keeps its numerical conventions explicit:

- observations use `(time, location)`;
- spatial lag zero is the identity matrix;
- non-symmetric spatial weights retain their supplied orientation;
- moving-average operators use the positive-sign convention;
- missing observations are never silently imputed;
- public numerical result arrays are immutable;
- conditional and exact-diffuse likelihoods are separate contracts.

> **Status — 0.0.29:** stationary, ordinary-integrated, multiplicative seasonal,
> and original-level exact-diffuse STARMA/STARIMA workflows are available.
> Version 0.0.29 adds seasonal exact-diffuse observed-information inference and
> natural innovation-covariance delta-method inference for
> `(1-B)^d(1-B^s)^D`. Seasonal exact-diffuse forecast intervals, conditional
> simulation smoothing, diffuse lag-one covariance, parameter-aware paths, and
> sparse execution remain planned.

## Installation

```bash
python -m pip install -e ".[test]"
```

Development and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

## Spatial weights

```python
import numpy as np
from pystarmax import SpatialWeights, lattice_weights

weights = SpatialWeights.from_adjacency(
    lattice_weights(2, 3),
    max_order=1,
)
```

`SpatialWeights` stores an ordered collection `(W0, W1, ...)`, where `W0` is
the identity. A coefficient row contains one parameter per stored spatial lag.

## Stationary STARMA

```python
from pystarmax import STARMA, simulate_starma

series = simulate_starma(
    phi=np.array([[0.45, 0.20]]),
    theta=np.array([[0.15, 0.05]]),
    weights=weights,
    n_steps=300,
    random_state=42,
)

model = STARMA(ar_order=1, ma_order=1)
result = model.fit(series, weights)
print(result.summary())
print(model.predict(steps=6))
```

For

\[
z_t=c+\sum_i A_i z_{t-i}+\varepsilon_t+\sum_j B_j\varepsilon_{t-j},
\]

pySTARMAx uses

\[
A_i=\sum_k\phi_{ik}W_k,
\qquad
B_j=\sum_k\theta_{jk}W_k.
\]

The positive moving-average sign implies the inverse recursion

\[
\varepsilon_t=r_t-\sum_jB_j\varepsilon_{t-j}.
\]

## Gaussian Kalman maximum likelihood

```python
from pystarmax import KalmanSTARMA

mle = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=False,
)
fit = mle.fit(series, weights)

print(fit.summary())
print(mle.admissibility().summary())
print(mle.predict(steps=6))
```

Innovation covariance options are `"scalar"`, `"diagonal"`, and positive-
definite `"full"` Cholesky covariance. Missing locations are removed from the
measurement update; a fully missing row performs prediction only.

## Conditional integrated STARIMA

```python
from pystarmax import KalmanSTARIMA

integrated = KalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    covariance_type="full",
)
result = integrated.fit(level_series, weights)

print(integrated.predict_differenced(steps=6))
print(integrated.predict(steps=6))
```

`KalmanSTARIMA` evaluates a conditional likelihood after ordinary differencing.
Its likelihood, AIC, and BIC must not be mixed with original-level exact-diffuse
results.

## Ordinary exact-diffuse STARIMA

```python
from pystarmax import ExactDiffuseKalmanSTARIMA

exact = ExactDiffuseKalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    covariance_type="full",
)
result = exact.fit(level_series, weights)

print(result.summary())
print(result.filter_result.filtered_diffuse_rank)
print(exact.predict(steps=6))
```

The exact filter stores finite covariance `P_*` and diffuse covariance `P_inf`
separately. No arbitrary finite large variance substitutes for unidentified
integration directions.

The ordinary exact-diffuse fitted model also exposes:

```python
exact.smooth()
exact.smooth_innovation_disturbances()
exact.likelihood_inference()
exact.predict_interval(steps=6, n_simulations=5000, random_state=42)
exact.simulation_smoother(n_simulations=1000, random_state=42)
```

## Seasonal exact-diffuse state construction

For

\[
x_t=(1-B)^d(1-B^s)^D y_t,
\]

version 0.0.26 introduced a reusable original-level state specification:

```python
from pystarmax import build_exact_seasonal_integrated_state_space

specification = build_exact_seasonal_integrated_state_space(
    transformed_state_space,
    ordinary_integration_order=d,
    seasonal_integration_order=D,
    seasonal_period=s,
)
filtered = specification.filter(level_series)
```

The positive-seasonal-order state stores the required original-level lag
companion together with the stationary transformed state. All integration lag
coordinates receive exact diffuse covariance.

## Seasonal exact-diffuse maximum likelihood

Version 0.0.27 added the optimizer-facing estimator:

```python
from pystarmax import SeasonalExactDiffuseKalmanSTARIMA

seasonal_exact = SeasonalExactDiffuseKalmanSTARIMA(
    ar_order=1,
    integration_order=0,
    ma_order=0,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=0,
    seasonal_period=12,
    covariance_type="scalar",
)
result = seasonal_exact.fit(level_series, weights)

print(result.summary())
print(seasonal_exact.admissibility().summary())
print(seasonal_exact.predict_differenced(steps=12))
print(seasonal_exact.predict(steps=12))
```

The free dynamic parameters remain ordinary and seasonal AR/MA **factor
coefficients**. At every optimizer candidate pySTARMAx:

1. expands the ordered multiplicative matrix polynomials;
2. builds the stationary transformed state-space model;
3. rebuilds the original-level seasonal exact-diffuse state;
4. evaluates the exact-diffuse likelihood on the original observations.

Expanded cross-lag matrices are deterministic functions of factor parameters.
They are not optimized independently and do not add AIC/BIC parameters.

The estimator retains both factor parameters and expanded operators:

```python
result.ar_parameters
result.seasonal_ar_parameters
result.ma_parameters
result.seasonal_ma_parameters
result.ar_lags
result.ar_matrices
result.ma_lags
result.ma_matrices
result.transformed_state_space
result.integrated_state_space
result.filter_result
```

## Seasonal exact-diffuse smoothing

Version 0.0.28 added fitted posterior operations:

```python
smoothed = seasonal_exact.smooth()
disturbances = seasonal_exact.smooth_innovation_disturbances()

print(smoothed.smoothed_observations)
print(smoothed.smoothed_observation_covariance)
print(disturbances.innovation_mean)
print(disturbances.innovation_covariance)
```

The state route returns exact-diffuse fixed-interval state and original-level
observation marginals. The primitive disturbance route uses the complete
seasonal augmented-state selection matrix, so each innovation is mapped into
both the original-level lag companion and the stationary transformed state.

The seasonal facade deliberately reuses the validated generic exact-diffuse
backward information recursion. It does not create a second seasonal smoother
and does not replace diffuse coordinates with a finite large variance.

Passing new observations starts a fresh exact-diffuse initialization under the
fitted parameters:

```python
new_smoothed = seasonal_exact.smooth(new_level_series)
new_disturbances = seasonal_exact.smooth_innovation_disturbances(
    new_level_series
)
```

This is not a continuation from the terminal training posterior.

## Seasonal exact-diffuse likelihood inference

Version 0.0.29 adds observed-information inference on the same original-level
exact-diffuse likelihood used during fitting:

```python
inference = seasonal_exact.likelihood_inference()
print(inference.summary())

natural_covariance = inference.innovation_covariance_inference()
print(natural_covariance.summary())
```

At every finite-difference stencil point pySTARMAx re-expands the ordinary and
seasonal factors, decodes the innovation covariance, rebuilds the transformed
state, rebuilds the seasonal exact-diffuse state, and filters the original
observations. The Hessian therefore includes the complete candidate-dependent
model construction.

The inference vector contains only free factor, intercept, and covariance
coordinates. Expanded multiplicative cross lags are deterministic and do not
become extra Hessian parameters.

By default the observed-information Hessian must be positive definite and full
rank. Generalized-inverse inference requires an explicit opt-in:

```python
inference = seasonal_exact.likelihood_inference(allow_singular=True)
```

The result records rank, eigenvalues, condition number, pseudoinverse use,
finite-difference steps, score, evaluation count, and distances from the fitted
AR and inverse-MA admissibility boundaries.

## Multiplicative seasonal convention

Seasonal factors multiply ordinary factors on the left. For the AR side,

\[
(I-S(B^s))(I-A(B)),
\]

cross terms have the ordered form `-S_j @ A_i`. Moving-average cross terms have
positive sign. Equal temporal lags are aggregated without projecting the result
back onto the supplied spatial-weight basis.

## Likelihood scope

The following likelihoods are deliberately distinct:

- `SeasonalKalmanSTARIMA`: conditional transformed-data likelihood;
- `SeasonalExactDiffuseKalmanSTARIMA`: original-level exact-diffuse likelihood;
- `KalmanSTARIMA`: conditional ordinary-difference likelihood;
- `ExactDiffuseKalmanSTARIMA`: original-level ordinary exact-diffuse likelihood.

Only compare likelihoods, AIC, BIC, Hessians, or parameter covariances when
candidates use the same likelihood scope.

## Diagnostics, inference, and evaluation

The package includes:

- STACF, STPACF, and space-time portmanteau diagnostics;
- AR stationarity and positive-sign MA invertibility diagnostics;
- ordinary, conditional-seasonal, and seasonal exact-diffuse finite-difference
  observed-information inference;
- natural innovation-covariance delta-method inference;
- Gaussian and bootstrap forecast intervals;
- rolling-origin evaluation and interval scoring;
- ordinary and seasonal exact-diffuse state/disturbance smoothing.

Seasonal exact-diffuse interval forecasting remains a separate future stage.

## Documentation

The MkDocs site includes dedicated guides for:

- model conventions and admissibility;
- stationary, integrated, and seasonal estimation;
- exact-diffuse filtering, MLE, smoothing, inference, intervals, and simulation;
- seasonal exact-diffuse integration, MLE, smoothing, and likelihood inference;
- diagnostics, forecasting, covariance inference, and evaluation;
- the roadmap and development handoffs.

Build it locally with:

```bash
python -m mkdocs serve
```

## Validation

Version 0.0.29 adds independent references for:

- closed-form intercept and log-scale information in a seasonal random walk;
- the natural scalar-variance standard error
  `q * sqrt(2 / n)`;
- exact reduction to ordinary exact-diffuse inference when seasonal orders are
  zero;
- missing-data curvature evaluation;
- scalar, diagonal, and full-Cholesky covariance coordinate contracts;
- explicit singular-Hessian and generalized-inverse behavior;
- immutable inference outputs and public API exports.

The final 0.0.29 test count and coverage are recorded in `PROJECT_STATUS.md` and
the Step 29 handoff after synchronized CI. CI covers Ubuntu, Windows, and macOS
on Python 3.11–3.14, strict MkDocs, formatting, linting, typing,
diagnostic-reference regeneration, source and wheel builds, and Twine checks.

## License and citation

pySTARMAx is released under the MIT License. Citation metadata is provided in
`CITATION.cff`.
