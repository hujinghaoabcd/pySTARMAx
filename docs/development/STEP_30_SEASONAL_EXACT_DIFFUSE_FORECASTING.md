# Step 30 handoff: seasonal exact-diffuse forecasting

## Development position

- development version: `0.0.30`;
- branch: `agent/seasonal-exact-diffuse-forecasting`;
- pull request: PR #30, `Add seasonal exact diffuse forecasting`;
- base: version 0.0.29 on `main` at merge commit
  `1f92083fcd51fa788095d753ed293137f5886204`.

## Scope

Step 30 adds fixed-parameter forecast paths and central simulation intervals for
seasonal original-level exact-diffuse models. It covers both:

- original observations `y_t`;
- the combined transformed process
  `(1-B)^d (1-B^s)^D y_t`.

The complete augmented state is simulated from the terminal exact-diffuse
posterior. Original-level values are therefore restored separately for every
simulated path before empirical quantiles are computed.

## Public fitted API

```python
original_paths = model.simulate_forecast_paths(
    steps=12,
    n_simulations=5000,
    random_state=7,
)
transformed_paths = model.simulate_differenced_forecast_paths(
    steps=12,
    n_simulations=5000,
    random_state=7,
)
original_interval = model.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=7,
)
transformed_interval = model.predict_differenced_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=7,
)
```

Equivalent functional entry points accept the fitted exact filter result and
its matching `ExactSeasonalIntegratedStateSpace`:

```python
from pystarmax import (
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)
```

## State and uncertainty contract

Let the combined differencing polynomial have degree

\[
m=d+Ds.
\]

For positive seasonal integration the augmented state is ordered as

\[
\alpha_t=
(y_t',y_{t-1}',\ldots,y_{t-m+1}',\beta_t')',
\]

where `beta_t` is the stationary transformed STARMA state. The terminal draw is

\[
\alpha_T^{(r)}\sim
\mathcal N(a_{T|T},P_{T|T}),
\]

followed by

\[
\alpha_{T+h}^{(r)}
=c+T\alpha_{T+h-1}^{(r)}+R\eta_{T+h}^{(r)},
\qquad
\eta_{T+h}^{(r)}\sim\mathcal N(0,Q).
\]

Original paths use the augmented observation design. Transformed paths use a
zero-padded design that selects the stationary block at offset
`m * n_locations`. Therefore both products are projections of the same exact
terminal-posterior simulation contract.

The returned `ForecastInterval.mean` is the deterministic propagated posterior
mean. Lower and upper bounds are empirical central quantiles of the simulated
paths. Only fitted state and innovation uncertainty is included; parameter
uncertainty is not.

## Safety contract

Forecast simulation requires `final_diffuse_rank == 0`. An unresolved terminal
diffuse direction does not define a proper finite Gaussian posterior and is
rejected explicitly. No large finite covariance is substituted for unresolved
diffuse directions.

The supplied seasonal integrated specification must be the exact object whose
state-space model was used by the filter result. This prevents accidental
projection with a different state layout.

## Independent validation targets

The test suite requires:

1. a period-two seasonal random walk to satisfy the pathwise identity
   `x_t = y_t - y_(t-2)` for every simulated path;
2. original-level variance to grow as `q, q, 2q, 2q, ...` while transformed
   variance remains `q`;
3. interval bounds to equal direct empirical quantiles from same-seed paths;
4. zero seasonal order to reduce exactly to ordinary exact-diffuse forecasting;
5. fitted interval means to equal `predict()` and `predict_differenced()`;
6. unresolved terminal diffuse rank, invalid specifications, and invalid
   interval arguments to fail explicitly;
7. interval arrays and public fitted outputs to remain immutable.

## Formatting cleanup

The first CI identified only Black formatting changes in the new forecasting
module and the expanded public-API test. A temporary branch-only workflow pinned
Black 26.5.1, applied the formatter, committed the exact output, and deleted
itself. The formal PR diff contains no workflow file.

## Deliberate boundaries

This stage does not add parameter-uncertainty propagation, profile likelihood,
bootstrap refitting, seasonal conditional simulation smoothing, sparse forecast
execution, or analytic forecast covariance recursions.

The final CI record, documentation inventory, remaining-work count, and
next-stage handoff will be completed after implementation validation.
