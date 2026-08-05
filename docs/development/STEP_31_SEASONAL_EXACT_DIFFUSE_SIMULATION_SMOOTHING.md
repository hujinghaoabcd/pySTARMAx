# Step 31 handoff: seasonal exact-diffuse simulation smoothing

## Development position

- development version: `0.0.31`;
- branch: `agent/seasonal-exact-diffuse-simulation-smoothing`;
- pull request: PR #31, `Add seasonal exact diffuse simulation smoothing`;
- base: version 0.0.30 on `main` at merge commit
  `02bb1a14409e3b679cacb90970c5f44b6314694d`.

## Scope

Step 31 adds dense conditional simulation smoothing for the complete seasonal
original-level exact-diffuse state associated with

\[
x_t=(1-B)^d(1-B^s)^D y_t.
\]

It reuses the generic exact-diffuse source-conditioning algorithm and projects
the stationary transformed-state block from the same conditional draws.

## Public API

The fitted entry point is:

```python
paths = model.simulate_smoothing_paths(
    n_simulations=1000,
    random_state=42,
)
```

Passing explicit data starts a fresh exact-diffuse filter under fitted
parameters:

```python
new_paths = model.simulate_smoothing_paths(
    new_level_data,
    n_simulations=1000,
    random_state=42,
)
```

The functional API is:

```python
from pystarmax import seasonal_exact_diffuse_simulation_smoother

paths = seasonal_exact_diffuse_simulation_smoother(
    filter_result,
    integrated_state_space,
    n_simulations=1000,
    random_state=42,
)
```

The public immutable result is
`SeasonalExactDiffuseSimulationSmootherResult`.

## Conditional construction

The generic source-conditioning algorithm writes all latent states and observed
cells as linear functions of:

1. initial diffuse coordinates;
2. initial proper finite coordinates;
3. primitive state innovations through time.

Observed original-level cells are exact linear constraints. Diffuse coordinates
are eliminated analytically. The remaining proper Gaussian source vector is
conditioned and sampled. This preserves the existing exact-diffuse convention
without a finite large-variance approximation.

The seasonal layer does not duplicate this algorithm. It validates the matching
state specification, delegates complete-state conditioning to the generic
routine, and projects transformed blocks.

## State layout and projections

Let

\[
m=d+Ds.
\]

The transformed state starts after `m` original-level lag blocks, at offset

\[
m n_{\mathrm{locations}}.
\]

The result contains:

- `state_paths` and `observation_paths` for the complete augmented state and
  original levels;
- `transformed_state_paths` and `transformed_observation_paths` from the
  stationary block;
- complete and transformed posterior marginal means and covariances;
- diffuse rank, conditioning rank, proper/posterior source dimensions;
- maximum observed-cell residual and marginal verification discrepancies.

The wrapper verifies that every transformed array equals its complete-state
projection. All public arrays are immutable.

## Safety contract

- every observed original-level cell is imposed as an exact linear constraint;
- every initial diffuse direction must be identified by the observed sample;
- unresolved terminal diffuse rank is rejected explicitly;
- the supplied seasonal state specification must be the exact object used by
  the filter result;
- no finite large-variance approximation is used;
- transformed paths are projections of the complete conditional state paths,
  not draws from a second conditioning algorithm.

## Independent validation

The test suite requires:

1. a period-two seasonal-random-walk bridge with missing values at times 2 and 3
   to have means 2 and 12 and variances `q/2`;
2. every observed original-level cell to be reproduced by every path;
3. the pathwise identity `x_t = y_t - y_(t-2)`;
4. transformed paths and posterior marginals to equal exact block projections;
5. partially observed multivariate locations to retain their observation mask;
6. seasonal integration order zero to reduce exactly to the ordinary generic
   simulation smoother under a common seed;
7. fitted training paths to reuse the training filter and new data to start a
   fresh exact-diffuse initialization;
8. unresolved diffuse rank, mismatched specifications, invalid arguments, and
   mutation attempts to fail explicitly.

## Formatting cleanup

Initial implementation CI #641 found only Black formatting differences in the
new simulation module and two tests. A temporary branch-only workflow pinned
Black 26.5.1, applied the formatter output, committed it, and deleted itself.
The formal PR diff contains no workflow file.

## Authoritative implementation validation

CI #644, run ID `30962744283`, passed on exact implementation head
`91ef1f95aab8371af14899e87c0c4ce9f1813c6d`:

- 259 tests passed;
- total branch coverage: 87.31%;
- new module branch coverage: 88.8%;
- Black, isort, Ruff, mypy, strict MkDocs, and diagnostic-reference
  regeneration passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11–3.14.

A final synchronized merge-gate CI is required on the exact final PR head after
documentation synchronization.

## Deliberate boundaries

This stage does not add diffuse lag-one covariance, cross-time disturbance
covariance, parameter-uncertainty propagation, bootstrap refitting, sparse
conditioning, or chunked/parallel execution.

## Next stage

The recommended next branch is
`agent/exact-diffuse-lag-one-covariance`. It should isolate the diffuse `L2`
recursion derivation, closed-form and source-simulation references, missing-data
behavior, ordinary finite-phase reduction, and explicit theory limitations from
unrelated uncertainty or performance work.
