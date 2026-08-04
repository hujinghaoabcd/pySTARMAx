# Roadmap

## Current release line

Version 0.0.29 adds seasonal exact-diffuse observed-information inference and
natural innovation-covariance delta-method inference on original observations.

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

### Exact-diffuse ordinary integration

- exact diffuse filtering and original-level likelihood;
- optimizer-facing ordinary STARIMA MLE;
- fixed-interval state smoothing;
- primitive innovation and state-disturbance smoothing;
- likelihood and natural covariance inference;
- original-level and transformed-scale forecast intervals;
- dense conditional simulation smoothing.

### Seasonal exact-diffuse state, estimation, and smoothing

Versions 0.0.26 through 0.0.28 delivered:

- complete `(1-B)^d(1-B^s)^D` differencing polynomial;
- auditable original-level lag companion;
- exact diffuse initialization of all integration coordinates;
- stationary finite initialization of the transformed subsystem;
- fixed-parameter filtering and original-level likelihood;
- optimizer-facing multiplicative seasonal MLE;
- deterministic factor expansion and factor-based parameter counting;
- expanded AR stationarity and positive-sign inverse-MA invertibility;
- original-level and highest-difference point forecasts;
- fixed-interval state and observation smoothing;
- primitive innovation and state-disturbance smoothing;
- exact ordinary reduction, bridge references, and missing-data validation.

### Implemented in 0.0.29

Version 0.0.29 delivers:

- `SeasonalExactDiffuseKalmanSTARIMA.likelihood_inference()`;
- public functional inference API;
- finite-difference score and observed-information Hessian on raw optimizer
  coordinates;
- complete factor expansion and exact-diffuse model reconstruction at every
  stencil point;
- scalar, diagonal, and full-Cholesky covariance coordinate support;
- natural innovation-covariance delta-method inference;
- immutable estimates, Hessian, covariance, standard errors, Wald statistics,
  correlations, eigenvalues, score, and step diagnostics;
- explicit full-rank/positive-definite Hessian policy;
- explicit generalized-inverse opt-in with retained-rank reporting;
- fitted AR and inverse-MA boundary-distance diagnostics;
- closed-form seasonal-random-walk information references;
- ordinary exact-diffuse reduction and missing-data curvature validation;
- synchronized method guide, example, navigation, status, inventory, and Step
  29 handoff.

## Next priorities

### 1. Seasonal exact-diffuse forecasting uncertainty

Add original-level and transformed-scale forecast paths and intervals using the
resolved terminal exact-diffuse posterior. Require explicit behavior when the
final diffuse rank is nonzero. Restore levels pathwise before computing original-
scale quantiles.

### 2. Seasonal exact-diffuse conditional simulation smoothing

Extend complete latent path simulation to the seasonal augmented state only
after the fitted posterior and interval contracts are stable. Preserve exact
conditioning and deterministic observation consistency without an arbitrary
finite diffuse variance.

### 3. Diffuse cross-time covariance theory

Derive and independently validate the diffuse `L2` recursion required for
lag-one smoothed state covariance and cross-time disturbance covariance. Do not
replace exact diffuse initialization with a finite large-variance approximation.

### 4. Stronger uncertainty

Add robust/sandwich covariance, profile likelihood, boundary-aware inference,
and parameter-uncertainty propagation into forecasts and selected posterior
summaries. Analytic derivatives may be added only with independent checks
against the finite-difference objectives.

### 5. Scalable numerical execution

Introduce sparse state matrices, memory-aware filtering/smoothing, and
reproducible chunked or parallel execution. Dense implementations remain the
reference path for numerical equivalence.

### 6. Model specification automation

Split into independently reviewable stages:

- smooth admissibility parameterization;
- automatic ordinary and seasonal order selection;
- automatic spatial-weight and spatial-lag selection;
- comparable model-selection reporting without mixing likelihood scopes.

### 7. Exogenous inputs and ecosystem work

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
