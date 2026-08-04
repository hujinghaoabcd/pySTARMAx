# Seasonal exact-diffuse forecasting

Version 0.0.30 extends `SeasonalExactDiffuseKalmanSTARIMA` with fixed-parameter
forecast paths and simulation intervals on both the original and combined
ordinary-seasonal transformed scales.

## Fitted-model interface

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

Each path array has shape

```text
(n_simulations, steps, n_locations)
```

Each interval is an immutable `ForecastInterval` with `mean`, `lower`, `upper`,
`level`, and a method description.

## Functional interface

The same operations are available for an already filtered state specification:

```python
from pystarmax import (
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)

original_paths = simulate_seasonal_exact_diffuse_forecast_paths(
    filter_result,
    integrated_state_space,
    steps=12,
    n_simulations=5000,
    random_state=7,
)
```

The supplied `ExactSeasonalIntegratedStateSpace` must be the exact specification
used to produce the filter result. A state-space object with merely compatible
shapes is rejected because its lag ordering or transformed-state offset could
be different.

## Mathematical contract

The transformed process is

\[
x_t=(1-B)^d(1-B^s)^D y_t.
\]

Let

\[
\delta(B)=(1-B)^d(1-B^s)^D
          =1+\delta_1B+\cdots+\delta_mB^m,
\qquad m=d+Ds.
\]

For positive seasonal integration the augmented state stores

\[
\alpha_t=
\begin{bmatrix}
 y_t\\
 y_{t-1}\\
 \vdots\\
 y_{t-m+1}\\
 \beta_t
\end{bmatrix},
\]

where `beta_t` is the stationary transformed STARMA state. The state transition
contains the exact inverse-differencing companion, so no separate post-hoc
history approximation is required.

After filtering through time `T`, a forecast replication draws

\[
\alpha_T^{(r)}\sim
\mathcal N(a_{T|T},P_{T|T})
\]

and propagates

\[
\alpha_{T+h}^{(r)}
=c+T\alpha_{T+h-1}^{(r)}+R\eta_{T+h}^{(r)},
\qquad
\eta_{T+h}^{(r)}\sim\mathcal N(0,Q).
\]

Original-level paths use

\[
y_{T+h}^{(r)}=Z\alpha_{T+h}^{(r)}.
\]

Transformed paths use a zero-padded design whose nonzero block is the
transformed-state observation design at offset

\[
(d+Ds)n_{\text{locations}}.
\]

Thus original and transformed products use the same terminal posterior and the
same future innovation convention.

## Pathwise restoration and quantiles

Original-level intervals are not obtained by transforming marginal quantiles.
The complete augmented state is propagated separately for every replication,
which restores ordinary and seasonal levels path by path. Only then are the
central empirical quantiles computed:

\[
L_h=\operatorname{quantile}_{\alpha/2}
    \{y_{T+h}^{(r)}\}_{r=1}^R,
\qquad
U_h=\operatorname{quantile}_{1-\alpha/2}
    \{y_{T+h}^{(r)}\}_{r=1}^R.
\]

This matters because inverse differencing creates cross-horizon dependence and
horizon-dependent original-level variance.

## Posterior mean

`ForecastInterval.mean` is obtained by deterministic state propagation from the
terminal filtered mean. It is not the finite-simulation sample mean. Therefore
it is reproducible independently of `random_state` and agrees with:

```python
model.predict(steps=steps)
model.predict_differenced(steps=steps)
```

for the original and transformed interval methods respectively.

## Uncertainty scope

The implementation includes:

- terminal fitted-state uncertainty `P_(T|T)`;
- future innovation uncertainty under fitted `Q`;
- all cross-horizon dependence induced by state propagation and inverse
  differencing.

It does not include fitted-parameter uncertainty. Changing `n_simulations`
changes Monte Carlo quantile precision, not the model or fitted posterior.

## Proper terminal posterior requirement

Forecasting requires

```python
filter_result.final_diffuse_rank == 0
```

A nonzero final diffuse rank means at least one state direction has no proper
finite Gaussian posterior. In that case path and interval methods raise
`RuntimeError`. The implementation never replaces the unresolved diffuse
component with an arbitrary large covariance.

Long missing-data endings are allowed when earlier observations have already
resolved all diffuse directions. Missing observations themselves are not filled
or conditioned on fabricated values.

## Random-number contract

`random_state` accepts:

- an integer seed;
- a `numpy.random.Generator`;
- `None` for a newly created generator.

The same seed and arguments produce identical projected paths. Original and
transformed calls with the same seed use corresponding state draws, which makes
pathwise differencing identities directly testable.

## Independent references

### Period-two seasonal random walk

For

\[
y_t=y_{t-2}+\mu+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,q),
\]

the tests verify every simulated path satisfies

\[
x_t=y_t-y_{t-2}.
\]

They also verify the original-level forecast variance sequence

\[
q,q,2q,2q,3q,3q,\ldots
\]

while the transformed-scale variance remains `q` at every horizon.

### Empirical interval identity

For a fixed seed, interval lower and upper arrays must equal direct NumPy
quantiles of the corresponding public path function. This checks that the
interval route does not use a different restoration or random-number contract.

### Ordinary reduction

When seasonal integration and seasonal AR/MA orders are zero, the seasonal
state specification delegates to the ordinary exact-integrated state. Tests
require exact equality with ordinary exact-diffuse forecast paths and intervals
under a common seed.

### Fitted facade

Tests require fitted path methods, functional projections, deterministic point
forecasts, and interval quantiles to agree. Invalid types, mismatched state
specifications, invalid steps, insufficient simulations, invalid levels, and
unresolved diffuse rank are tested explicitly.

## Performance boundary

The implementation allocates the full array

```text
(n_simulations, steps, state_dim)
```

before projection. It is a transparent dense reference for moderate state
sizes and horizons. Sparse propagation, chunked quantiles, deterministic
parallel seed partitioning, and streaming storage remain future work.

## Related guides

- [Seasonal exact diffuse integration](exact_seasonal_integrated.md)
- [Seasonal exact diffuse MLE](seasonal_exact_diffuse_mle.md)
- [Seasonal exact diffuse smoothing](seasonal_exact_diffuse_smoothing.md)
- [Seasonal exact diffuse inference](seasonal_exact_diffuse_inference.md)
- [Ordinary exact diffuse forecast intervals](exact_diffuse_forecast_intervals.md)
