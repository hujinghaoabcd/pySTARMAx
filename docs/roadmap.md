# Roadmap

## Current release line

Version 0.0.27 adds optimizer-facing multiplicative seasonal exact-diffuse
maximum likelihood on original observations.

The project now has a common progression:

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

### Exact-diffuse ordinary integration

- exact diffuse filtering and original-level likelihood;
- optimizer-facing ordinary STARIMA MLE;
- fixed-interval state smoothing;
- primitive innovation and state-disturbance smoothing;
- likelihood and natural covariance inference;
- original-level and transformed-scale forecast intervals;
- dense conditional simulation smoothing.

### Seasonal exact-diffuse foundation

Version 0.0.26 delivered:

- complete `(1-B)^d(1-B^s)^D` differencing polynomial;
- auditable original-level lag companion;
- exact diffuse initialization of all integration coordinates;
- stationary finite initialization of the transformed subsystem;
- fixed-parameter filtering and original-level likelihood;
- exact `D=0` reduction to the ordinary state contract.

### Implemented in 0.0.27

Version 0.0.27 delivers:

- `SeasonalExactDiffuseKalmanSTARIMA`;
- ordinary and seasonal AR/MA factor optimization;
- deterministic ordered cross-lag expansion for every candidate;
- expanded AR stationarity and positive-sign inverse-MA admissibility;
- scalar, diagonal, and full-Cholesky innovation covariance;
- original-level exact-diffuse likelihood;
- factor-based AIC/BIC parameter counting;
- immutable fitted factor, operator, covariance, state-space, filter, diffuse,
  and optimizer metadata;
- original-level and highest-difference point forecasts;
- closed-form, reduction, missing-data, public-API, and cross-platform tests;
- synchronized method guide, example, status, and handoff documentation.

## Next priorities

### 1. Seasonal exact-diffuse smoothing

Extend fixed-interval state smoothing and primitive disturbance smoothing to the
seasonal augmented state. The implementation must reuse the 0.0.26 state and
0.0.27 fitted-model contracts.

### 2. Seasonal exact-diffuse likelihood inference

Add finite-difference observed-information inference and natural innovation-
covariance inference around the seasonal exact-diffuse objective. Boundary and
rank-deficiency policies must remain explicit.

### 3. Seasonal exact-diffuse forecasting uncertainty

Add original-level and transformed-scale forecast paths and intervals using the
terminal exact-diffuse posterior after diffuse resolution.

### 4. Diffuse cross-time covariance theory

Derive and independently validate the diffuse `L2` recursion required for
lag-one smoothed state covariance and cross-time disturbance covariance. Do not
replace exact diffuse initialization with a finite large-variance approximation.

### 5. Stronger uncertainty

Add robust/sandwich covariance, profile likelihood, and parameter-uncertainty
propagation into forecasts and selected posterior summaries.

### 6. Scalable numerical execution

Introduce sparse state matrices, memory-aware filtering/smoothing, and
reproducible chunked or parallel execution. Dense implementations remain the
reference path for numerical equivalence.

### 7. Model specification automation

Split into independently reviewable stages:

- smooth admissibility parameterization;
- automatic ordinary and seasonal order selection;
- automatic spatial-weight and spatial-lag selection;
- comparable model-selection reporting without mixing likelihood scopes.

### 8. Exogenous inputs and ecosystem work

- exogenous regressors and intervention variables;
- GIS-oriented data and spatial-weight adapters;
- cross-language numerical reference fixtures;
- public release and signed automation;
- explicitly time-varying state-space extensions.

## Delivery policy

Each milestone should provide:

- an explicit mathematical and likelihood-scale contract;
- immutable typed public results;
- independently constructed numerical references;
- missing-data and rank-deficiency behavior;
- cross-platform tests;
- formatting, linting, typing, strict documentation, and package-build success;
- synchronized README, status, roadmap, method guide, example, and handoff;
- no temporary workflows, generated artifacts, or unresolved review threads.
