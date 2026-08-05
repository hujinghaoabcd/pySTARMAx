# Step 31 handoff: seasonal exact-diffuse simulation smoothing

## Development position

- development version: `0.0.31`;
- branch: `agent/seasonal-exact-diffuse-simulation-smoothing`;
- pull request: PR #31, `Add seasonal exact diffuse simulation smoothing`;
- base: version 0.0.30 on `main` at merge commit
  `02bb1a14409e3b679cacb90970c5f44b6314694d`.

## Scope

Step 31 adds dense conditional simulation smoothing for the complete seasonal
original-level exact-diffuse state. It reuses the generic exact-diffuse
source-conditioning algorithm and projects the stationary transformed-state
block from the same conditional draws.

The public fitted entry point is:

```python
paths = model.simulate_smoothing_paths(
    n_simulations=1000,
    random_state=42,
)
```

The functional entry point accepts the exact filter result and the exact
`ExactSeasonalIntegratedStateSpace` object used to construct that filter.

## Safety contract

- every observed original-level cell is imposed as an exact linear constraint;
- every initial diffuse direction must be identified by the observed sample;
- unresolved terminal diffuse rank is rejected explicitly;
- the supplied seasonal state specification must be the exact object used by
  the filter result;
- no finite large-variance approximation is used;
- transformed paths are projections of the complete conditional state paths,
  not draws from a second conditioning algorithm.

## Deliberate boundaries

This stage does not add diffuse lag-one covariance, cross-time disturbance
covariance, parameter-uncertainty propagation, bootstrap refitting, sparse
conditioning, or chunked/parallel execution.

The final API inventory, validation record, documentation synchronization, and
next-stage handoff will be completed after implementation CI.
