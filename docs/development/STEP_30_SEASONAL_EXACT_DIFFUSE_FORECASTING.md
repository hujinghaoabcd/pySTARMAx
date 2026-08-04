# Step 30 handoff: seasonal exact-diffuse forecasting

## Development position

- development version: `0.0.30`;
- branch: `agent/seasonal-exact-diffuse-forecasting`;
- pull request: PR #30, `Add seasonal exact diffuse forecasting`;
- base: version 0.0.29 on `main` at merge commit
  `1f92083fcd51fa788095d753ed293137f5886204`.

## Delivered API

Fitted methods:

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

Functional entry points:

```python
from pystarmax import (
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)
```

The functional route accepts a matching `ExactDiffuseFilterResult` and
`ExactSeasonalIntegratedStateSpace`. The state specification must be the exact
object used by the filter result.

## Mathematical and scale contract

The transformed process is

\[
x_t=(1-B)^d(1-B^s)^D y_t.
\]

Let the combined differencing degree be

\[
m=d+Ds.
\]

For positive seasonal integration the augmented state is ordered as

\[
\alpha_t=
(y_t',y_{t-1}',\ldots,y_{t-m+1}',\beta_t')',
\]

where `beta_t` is the stationary transformed STARMA state. Each replication
draws

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

Original-level paths use the augmented observation design. Transformed paths
use a zero-padded design selecting the stationary state block at offset

\[
m n_{\text{locations}}.
\]

Both products therefore share the terminal posterior, transition, innovation,
and random-number contract.

## Pathwise level restoration

Original-level intervals are formed by:

1. drawing the terminal augmented state;
2. propagating the complete state at every horizon;
3. projecting every replication to original observations;
4. computing empirical central quantiles across restored paths.

The implementation does not transform marginal transformed-scale quantiles.
This preserves the cross-horizon dependence and variance accumulation created by
ordinary and seasonal inverse differencing.

`ForecastInterval.mean` is the deterministic propagated posterior mean and
agrees with `predict()` or `predict_differenced()` on the corresponding scale.
The bounds include terminal state uncertainty and future fitted innovation
uncertainty, but not fitted-parameter uncertainty.

## Proper-posterior policy

Forecast simulation requires

```python
filter_result.final_diffuse_rank == 0
```

A nonzero rank means the terminal state lacks a proper finite Gaussian posterior.
Path and interval functions raise `RuntimeError`; no arbitrary large covariance
is substituted for unresolved diffuse directions.

## Independent validation

### Period-two seasonal random walk

For

\[
y_t=y_{t-2}+\mu+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,q),
\]

the suite verifies every same-seed path satisfies

\[
x_t=y_t-y_{t-2}.
\]

It also verifies original-level forecast variance

\[
q,q,2q,2q,3q,3q,\ldots
\]

and transformed variance `q` at every horizon.

### Quantile identity

Original and transformed interval bounds are compared directly with NumPy
quantiles from their corresponding public path functions under the same seed.

### Ordinary reduction

When seasonal integration and seasonal AR/MA orders are zero, path and interval
outputs must equal ordinary exact-diffuse forecasting exactly under a common
seed.

### Fitted facade and failures

Tests verify:

- interval means equal existing point forecasts;
- fitted and functional path/interval routes agree;
- unresolved diffuse rank fails explicitly;
- mismatched state specifications fail explicitly;
- invalid steps, simulation counts, levels, and argument types fail explicitly;
- returned interval arrays are immutable;
- all public exports and fitted methods are present.

## Formatting incident

Initial CI #626 identified only Black formatting in the new forecasting module
and expanded public-API test before later quality steps ran. A temporary
branch-only workflow pinned Black 26.5.1, applied the exact formatter output,
committed it, and deleted itself. The formal PR diff contains no workflow file.

## Authoritative implementation validation

CI #630, run ID `30959749037`, passed on exact head
`652a1c514b96c267f6aeaabfa6090e28b9249436`:

- 252 tests passed;
- total branch coverage: 87.28%;
- `src/pystarmax/seasonal_exact_diffuse_forecasting.py`: complete coverage;
- Black, isort, Ruff, and mypy passed;
- diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A final synchronized documentation and validation-record merge gate is required
on the exact final PR head.

## Documentation inventory

Step 30 adds or updates:

- `src/pystarmax/seasonal_exact_diffuse_forecasting.py`;
- `tests/test_seasonal_exact_diffuse_forecasting.py`;
- `docs/seasonal_exact_diffuse_forecasting.md`;
- `examples/seasonal_exact_diffuse_forecasting.py`;
- this handoff;
- top-level exports, version, and citation metadata;
- fitted seasonal exact-diffuse facade;
- README, documentation home, MkDocs navigation, roadmap, remaining-work
  inventory, and project status.

## Deliberate boundaries

Step 30 does not provide:

- fitted-parameter uncertainty propagation;
- bootstrap refitting or profile-likelihood intervals;
- seasonal exact-diffuse conditional simulation smoothing;
- diffuse lag-one state covariance;
- cross-time disturbance covariance;
- sparse, chunked, streaming, or parallel path execution;
- analytic forecast covariance recursions.

The dense implementation allocates the complete simulation-state tensor and is
a transparent moderate-size reference.

## Remaining roadmap

After Step 30 the inventory contains six major technical workstreams, six
numbered core milestones, and approximately 11–15 independently reviewable
projects.

The next implementation candidate is seasonal exact-diffuse conditional
simulation smoothing. The diffuse `L2` lag-one covariance derivation remains a
separate subsequent theory-and-validation project.

## Merge checklist

Before merging PR #30:

1. run complete synchronized CI on the exact final head;
2. record the final run and head in the PR body;
3. confirm the formal diff contains no temporary workflow or generated artifact;
4. confirm no submitted review or unresolved review thread remains;
5. mark PR #30 ready and squash-merge version 0.0.30;
6. create `agent/seasonal-exact-diffuse-simulation-smoothing` from new `main`.
