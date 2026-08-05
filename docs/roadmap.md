# Roadmap

## Current release line

Version 0.0.32 adds an exact dense reference for adjacent-time smoothed state
covariance during genuine diffuse phases. It covers ordinary and complete
seasonal augmented states and provides the numerical oracle for a future
memory-linear diffuse `L2` recursion.

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
- dense conditional simulation smoothing;
- exact dense adjacent-time state and observation covariance.

### Seasonal exact-diffuse workflow

Versions 0.0.26 through 0.0.32 deliver:

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
- exact dense adjacent-time covariance for the complete augmented state;
- pathwise ordinary-seasonal identities and observed-cell consistency;
- explicit unresolved-diffuse-rank protection;
- ordinary reduction, closed-form random-walk references, missing-data behavior,
  and immutable public results.

## Implemented in 0.0.32

The functional API is:

```python
moments = exact_diffuse_lag_one_covariance(filter_result)
```

The ordinary and seasonal fitted estimators expose:

```python
moments = model.smooth_lag_one_covariance()
```

The output convention is

\[
\texttt{lag\_one\_covariance}[t]
=
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y_{1:T}).
\]

The implementation writes the complete state path as a linear function of flat
diffuse coordinates and proper Gaussian source coordinates. Observed cells are
exact equality constraints. Diffuse coordinates are eliminated analytically,
and adjacent covariance is formed from the resulting posterior path loading.

The result also projects adjacent covariance to observation space and
reconstructs state-disturbance covariance. The latter is compared with the
independent exact information-form disturbance smoother, providing a strong
orientation and indexing check.

The method refuses computation when `final_diffuse_rank` is nonzero. It does not
replace unresolved diffuse directions with a large finite variance and does not
estimate covariance from Monte Carlo paths.

## Next priorities

### 1. Memory-linear exact diffuse `L2` recursion

Derive the recursive lag-one covariance correction for a nontrivial diffuse
phase and validate it against the 0.0.32 dense oracle. Required validation
includes:

- genuine diffuse closed forms;
- finite-phase reduction to the ordinary RTS recursion;
- non-symmetric transitions;
- missing and partially observed measurements;
- ordinary and seasonal augmented states;
- equality with the dense adjacent covariance and reconstructed disturbance
  moments.

Only after this equivalence is established should the recursive implementation
become the default scalable path.

### 2. Cross-time disturbance covariance

Use the validated recursive theory to expose non-adjacent state covariance,
cross-time state-disturbance covariance, and primitive-innovation covariance.
Keep orientation, indexing, and covariance-image contracts explicit.

### 3. Stronger uncertainty

Add robust/sandwich covariance, profile likelihood, boundary-aware inference,
and parameter-uncertainty propagation into forecast and selected posterior
summaries. Analytic derivatives require independent checks against current
finite-difference objectives.

### 4. Scalable numerical execution

Introduce sparse state matrices, memory-aware filtering/smoothing, chunked
forecast quantiles, and reproducible parallel seed partitioning. Dense
implementations remain the numerical reference path.

### 5. Model specification automation

Split into independently reviewable stages:

- smooth admissibility parameterization;
- automatic ordinary and seasonal order selection;
- automatic spatial-weight and spatial-lag selection;
- comparable reporting without mixing likelihood scopes.

### 6. Exogenous inputs and ecosystem work

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