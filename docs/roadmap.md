# Roadmap

## Current release line

Version 0.0.31 adds dense seasonal exact-diffuse conditional simulation
smoothing for complete augmented paths and transformed-state projections.

The project follows a common progression:

1. stationary transformed-state construction;
2. ordinary and seasonal differencing contracts;
3. original-level exact-diffuse state augmentation;
4. optimizer-facing fitted models;
5. posterior operations and uncertainty;
6. scalable execution and ecosystem integration.

## Delivered foundations

### Classical and conditional models

- `STARMA`, `STARIMA`, and `SeasonalSTARIMA`;
- stationary Gaussian `KalmanSTARMA`;
- conditional ordinary-integrated `KalmanSTARIMA`;
- conditional multiplicative seasonal `SeasonalKalmanSTARIMA`.

### Ordinary exact-diffuse integration

- exact diffuse filtering and original-level likelihood;
- optimizer-facing ordinary STARIMA MLE;
- state and primitive disturbance smoothing;
- likelihood and natural covariance inference;
- original-level and transformed-scale forecast intervals;
- dense conditional simulation smoothing.

### Seasonal exact-diffuse workflow

Versions 0.0.26 through 0.0.31 deliver:

- complete `(1-B)^d(1-B^s)^D` differencing polynomial;
- auditable original-level lag companion and exact diffuse initialization;
- optimizer-facing multiplicative seasonal MLE;
- deterministic factor expansion and factor-based parameter counting;
- expanded AR stationarity and positive-sign inverse-MA invertibility;
- original-level and transformed point forecasts;
- fixed-interval state and observation smoothing;
- primitive innovation and state-disturbance smoothing;
- observed-information and natural innovation-covariance inference;
- original-level and transformed forecast paths and intervals;
- complete conditional simulation smoothing;
- transformed-state and transformed-observation projections from the same
  conditional draws;
- pathwise ordinary-seasonal identities and observed-cell consistency;
- explicit unresolved-diffuse-rank protection;
- ordinary reduction, closed-form random-walk references, missing-data behavior,
  and immutable public results.

## Implemented in 0.0.31

The fitted estimator now exposes:

```python
paths = model.simulate_smoothing_paths(
    n_simulations=1000,
    random_state=42,
)
```

The functional API accepts a matching exact filter result and
`ExactSeasonalIntegratedStateSpace`.

The implementation reuses the generic dense exact-diffuse source-conditioning
algorithm. Observed original-level cells are exact linear constraints. Initial
diffuse coordinates are eliminated analytically and the remaining proper
Gaussian sources are sampled. The transformed state begins at offset
`(d + D*s) * n_locations` and is projected from the same complete draws.

The method refuses simulation when `final_diffuse_rank` is nonzero. It does not
replace unresolved diffuse directions with a large finite variance. When
seasonal integration is zero, complete paths reduce exactly to the ordinary
simulation smoother under the same seed.

## Next priorities

### 1. Diffuse cross-time covariance theory

Derive and independently validate the diffuse `L2` recursion required for
lag-one smoothed state covariance during a nontrivial diffuse phase. Use the
validated recursion to expose cross-time state-disturbance and primitive
innovation covariance. Do not replace exact diffuse initialization with a
finite large-variance approximation.

### 2. Stronger uncertainty

Add robust/sandwich covariance, profile likelihood, boundary-aware inference,
and parameter-uncertainty propagation into forecast and selected posterior
summaries. Analytic derivatives require independent checks against current
finite-difference objectives.

### 3. Scalable numerical execution

Introduce sparse state matrices, memory-aware filtering/smoothing, chunked
forecast quantiles, and reproducible parallel seed partitioning. Dense
implementations remain the numerical reference path.

### 4. Model specification automation

Split into independently reviewable stages:

- smooth admissibility parameterization;
- automatic ordinary and seasonal order selection;
- automatic spatial-weight and spatial-lag selection;
- comparable reporting without mixing likelihood scopes.

### 5. Exogenous inputs and ecosystem work

- exogenous regressors and intervention variables;
- GIS-oriented data and spatial-weight adapters;
- cross-language numerical reference fixtures;
- public release and signed automation;
- explicitly time-varying state-space extensions.

## Delivery policy

Each milestone should provide:

- an explicit mathematical, scale, and likelihood contract;
- immutable typed public results;
- independently constructed numerical references;
- missing-data and rank-deficiency behavior;
- cross-platform tests;
- formatting, linting, typing, strict documentation, and package-build success;
- synchronized README, status, roadmap, method guide, example, and handoff;
- no temporary workflows, generated artifacts, or unresolved review threads.
