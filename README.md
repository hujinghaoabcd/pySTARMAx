# pySTARMAx

**pySTARMAx** is a typed, research-oriented Python toolkit for classical and
extended space-time autoregressive moving-average modelling.

The package keeps its numerical contracts explicit:

- observations use `(time, location)`;
- spatial lag zero is the identity matrix;
- non-symmetric spatial weights retain their supplied orientation;
- moving-average operators use the positive-sign convention;
- missing observations are skipped, never silently imputed;
- public numerical result arrays are immutable;
- conditional and original-level exact-diffuse likelihoods remain separate.

> **Status — 0.0.30:** stationary, ordinary-integrated, multiplicative seasonal,
> and original-level exact-diffuse STARMA/STARIMA workflows are available.
> Version 0.0.30 adds seasonal exact-diffuse original-level and transformed-scale
> forecast paths and intervals with pathwise ordinary-seasonal level restoration
> and explicit terminal diffuse-rank protection.

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
the identity. Supplied matrix orientation is preserved.

## Classical and Gaussian models

```python
from pystarmax import KalmanSTARMA, STARMA, simulate_starma

series = simulate_starma(
    phi=np.array([[0.45, 0.20]]),
    theta=np.array([[0.15, 0.05]]),
    weights=weights,
    n_steps=300,
    random_state=42,
)

least_squares = STARMA(ar_order=1, ma_order=1)
least_squares.fit(series, weights)

mle = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
)
mle.fit(series, weights)
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

The positive moving-average sign implies

\[
\varepsilon_t=r_t-\sum_jB_j\varepsilon_{t-j}.
\]

## Conditional and exact-diffuse integration

`KalmanSTARIMA` evaluates a conditional likelihood after ordinary
differencing. `ExactDiffuseKalmanSTARIMA` evaluates original observations while
representing unidentified integration directions with a separate diffuse
covariance component.

```python
from pystarmax import ExactDiffuseKalmanSTARIMA, KalmanSTARIMA

conditional = KalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
)
conditional.fit(level_series, weights)

exact = ExactDiffuseKalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    covariance_type="full",
)
exact.fit(level_series, weights)
```

The exact filter never replaces diffuse directions with an arbitrary finite
large variance. Ordinary exact-diffuse posterior operations include:

```python
exact.smooth()
exact.smooth_innovation_disturbances()
exact.likelihood_inference()
exact.predict_interval(steps=6, n_simulations=5000, random_state=42)
exact.simulation_smoother(n_simulations=1000, random_state=42)
```

## Seasonal exact-diffuse STARIMA

For

\[
x_t=(1-B)^d(1-B^s)^D y_t,
\]

`SeasonalExactDiffuseKalmanSTARIMA` fits multiplicative ordinary and seasonal
AR/MA factors on the original observation scale:

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
```

At every optimizer candidate pySTARMAx:

1. decodes ordinary and seasonal AR/MA factor coordinates;
2. expands the ordered multiplicative matrix polynomials;
3. builds the stationary transformed state;
4. builds the original-level seasonal exact-diffuse augmentation;
5. filters the original observations with their original missing-data mask.

Expanded cross-lag matrices are deterministic functions of the factor
parameters. They are not independent optimization parameters and do not add
AIC/BIC degrees of freedom.

The fitted model exposes point forecasting, smoothing, disturbance smoothing,
and likelihood inference:

```python
seasonal_exact.predict_differenced(steps=12)
seasonal_exact.predict(steps=12)
seasonal_exact.smooth()
seasonal_exact.smooth_innovation_disturbances()
seasonal_exact.likelihood_inference()
```

## Seasonal exact-diffuse forecast paths and intervals

Version 0.0.30 adds fixed-parameter paths and intervals on both scales:

```python
original_paths = seasonal_exact.simulate_forecast_paths(
    steps=12,
    n_simulations=5000,
    random_state=42,
)
transformed_paths = seasonal_exact.simulate_differenced_forecast_paths(
    steps=12,
    n_simulations=5000,
    random_state=42,
)

original_interval = seasonal_exact.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=42,
)
transformed_interval = seasonal_exact.predict_differenced_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=42,
)
```

The complete augmented seasonal state is simulated from the terminal filtered
posterior. Each replication restores ordinary and seasonal levels through the
state transition before original-level quantiles are computed. The code does
not transform marginal quantiles after the fact.

Forecasting requires

```python
result.filter_result.final_diffuse_rank == 0
```

because an unresolved diffuse direction does not define a proper finite
terminal Gaussian posterior. Such cases raise an error rather than substituting
a large finite covariance.

The intervals include terminal state uncertainty and future fitted innovation
uncertainty. They do not yet include fitted-parameter uncertainty.

## Multiplicative seasonal convention

Seasonal factors multiply ordinary factors on the left. For the AR side,

\[
(I-S(B^s))(I-A(B)),
\]

cross terms have ordered form `-S_j @ A_i`. Moving-average cross terms have
positive sign. Equal temporal lags are aggregated without projecting the
result back onto the supplied spatial-weight basis.

## Likelihood scope

The following contracts are deliberately distinct:

- `SeasonalKalmanSTARIMA`: conditional transformed-data seasonal likelihood;
- `SeasonalExactDiffuseKalmanSTARIMA`: original-level seasonal exact-diffuse
  likelihood;
- `KalmanSTARIMA`: conditional ordinary-difference likelihood;
- `ExactDiffuseKalmanSTARIMA`: original-level ordinary exact-diffuse likelihood.

Only compare likelihoods, AIC, BIC, Hessians, or parameter covariances when
candidate models use the same likelihood scope.

## Diagnostics and evaluation

The package includes:

- STACF, STPACF, and space-time portmanteau diagnostics;
- AR stationarity and positive-sign inverse-MA invertibility diagnostics;
- observed-information inference across stationary, integrated, seasonal, and
  exact-diffuse fitted models;
- natural innovation-covariance delta-method inference;
- Gaussian, bootstrap, ordinary exact-diffuse, and seasonal exact-diffuse
  interval workflows;
- rolling-origin evaluation and interval scoring.

## Validation

The 0.0.30 suite independently checks:

- pathwise seasonal inverse-differencing identities;
- period-two seasonal-random-walk forecast means and variance growth;
- transformed-scale constant innovation variance;
- exact empirical-quantile agreement between path and interval APIs;
- exact reduction to ordinary exact-diffuse forecasting when seasonal orders
  are zero;
- fitted facade consistency with deterministic point forecasts;
- invalid arguments, mismatched state specifications, and unresolved terminal
  diffuse rank;
- immutable interval outputs and public exports.

The synchronized CI record is maintained in `PROJECT_STATUS.md` and the Step 30
handoff. CI covers Ubuntu, Windows, and macOS on Python 3.11–3.14, formatting,
linting, typing, strict MkDocs, reference regeneration, package builds, and
Twine checks.

## Documentation

Build the MkDocs site locally with:

```bash
python -m mkdocs serve
```

Dedicated guides cover filtering, estimation, smoothing, inference,
forecasting, diagnostics, evaluation, and development handoffs.

## License and citation

pySTARMAx is released under the MIT License. Citation metadata is provided in
`CITATION.cff`.
