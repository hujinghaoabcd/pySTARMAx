# pySTARMAx project status

Updated: 2026-07-29

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 was squash-merged into `main` as commit `795c7e45`.
- PR #2 was squash-merged into `main` as commit `689f9927`.
- PR #3 was squash-merged into `main` as commit `d89fa7ce`.
- PR #4 was squash-merged into `main` as commit `6bd29a63`.
- PR #5 was squash-merged into `main` as commit `cc1e8ac7` after GitHub Actions
  passed quality, coverage, distributions, and the Ubuntu/Windows/macOS matrix
  for Python 3.11 through 3.14.
- Current branch: `agent/bootstrap-forecast-intervals`.
- Current draft pull request: PR #6, `Add parameter-aware bootstrap forecast intervals`.
- Current development target: version `0.0.6`.
- Current stage: residual and parametric direct-bootstrap forecast intervals.
- Authoritative validation: GitHub Actions CI run #137 completed successfully.

## Completed baseline through 0.0.5

- repository metadata, MIT licence, citation file, contribution and security policies;
- `src/` package layout and typed public API;
- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA conditional least squares;
- coefficient uncertainty, innovation covariance, likelihood, AIC, and BIC;
- recursive point forecasts and deterministic simulation;
- classical STACF, nested Yule–Walker STPACF, and residual portmanteau tests;
- exact-rational diagnostic reference fixture;
- ordinary STARIMA differencing and forecast inversion;
- multiplicative seasonal STARIMA with constrained matrix-polynomial factors;
- aligned original-scale one-step fitted values;
- conditional future-innovation intervals with pathwise inverse differencing;
- packaging, strict documentation, and multi-platform CI.

## Completed in the current 0.0.6 step

- added `pystarmax.bootstrap` with bootstrap-method and replication validation;
- added complete finite residual-row extraction and location-wise centering;
- added joint residual-vector resampling that retains contemporaneous location dependence;
- added Gaussian parametric bootstrap draws from fitted covariance;
- added recursive same-length pseudo-series generation for STAR and STARMA;
- added combined-differencing pseudo-series reconstruction for ordinary and seasonal models;
- added estimator cloning and refitting for every accepted replication;
- added `predict_bootstrap_interval()` to public `STAR`, `STARMA`, `STARIMA`, and
  `SeasonalSTARIMA` classes;
- added parameter-only intervals through `include_future_innovations=False`;
- added full predictive intervals that combine parameter and future-innovation uncertainty;
- added residual and parametric future-path simulation after every refit;
- added linear seasonal reuse of the STARMA core and a separate multiplicative
  seasonal `LagOperator` recursion;
- added bounded retry through `max_attempts` and optional convergence enforcement;
- added deterministic random-state handling through one shared NumPy generator;
- added nine focused tests covering utilities, stationary models, ordinary
  integration, linear seasonal integration, and nonlinear multiplicative seasonality;
- increased the complete suite from 52 to 61 tests;
- updated README, MkDocs, research references, third-party notices, roadmap, and
  citation metadata to version 0.0.6.

## Statistical interpretation

`predict_interval()` remains a conditional future-innovation interval. It holds
estimated coefficients fixed.

`predict_bootstrap_interval()` generates and refits same-length pseudo-samples.
With `include_future_innovations=False`, its empirical spread represents
parameter-estimation variation under the selected bootstrap method. With the
default `True`, every refitted model contributes one new future path and the
interval combines parameter-estimation and future-innovation uncertainty.

The model order and spatial weights remain fixed across replications. The
bootstrap therefore does not include order-selection or weight-construction
uncertainty.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep numerical methods independent and auditable.
3. Separate transforms, weights, estimation, diagnostics, forecasting, bootstrap,
   simulation, and results.
4. Validate dimensions and assumptions before numerical work.
5. Treat conditional estimation as a baseline, not exact likelihood.
6. Add reference fixtures before claiming cross-language equivalence.
7. Make matrix orientation explicit whenever non-symmetric weights matter.
8. Keep transformed-scale inference distinct from original-scale reconstruction.
9. Use observed history only for the initial conditions of bootstrap inversion;
   reconstruct later pseudo-observations from their own pseudo-history.
10. Resample complete innovation vectors rather than independent location cells.
11. Propagate complete stochastic paths before inverse-differencing quantiles.
12. Label innovation-only and parameter-aware intervals separately.
13. Never silently return fewer successful bootstrap replications than requested.
14. Keep future sparse, parallel, and state-space acceleration behind stable interfaces.

## Final validation for 0.0.6

GitHub Actions CI run #137 completed successfully on the documented branch head:

- 61 tests passed;
- branch coverage: 89.57%, above the configured 80% threshold;
- Black passed;
- isort passed;
- Ruff passed;
- mypy passed;
- exact diagnostic fixture regeneration produced a clean diff;
- strict MkDocs build passed;
- source distribution and wheel built successfully;
- Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

## Immediate next tasks

1. Keep PR #6 as draft unless the user explicitly requests review or merge.
2. After merge, begin rolling-origin interval scoring and empirical coverage examples.
3. Add optional parallel bootstrap execution behind the existing API.
4. Investigate block, wild, predictive-residual, studentized, and bias-corrected methods.
5. Implement exact state-space/Kalman maximum likelihood and missing observations.
6. Add full innovation likelihoods and stationarity/invertibility checks.
7. Add sparse matrices and NetworkX/libpysal/GeoPandas/OSMnx adapters.
8. Add automatic order selection and exogenous regressors/interventions.
9. Add cross-language estimator fixtures.
10. Prepare the first PyPI pre-release after the next release review.

## Known limitations

- MA estimation is conditional and uses recursively estimated innovations;
- seasonal nonlinear estimation uses zero pre-sample innovations;
- stationarity and invertibility constraints are not imposed during optimization;
- seasonal standard errors use a local nonlinear least-squares Jacobian;
- information criteria use a scalar innovation-variance approximation;
- residual bootstrap assumes complete innovation vectors are exchangeable over time;
- no block, wild, robust, predictive-residual, studentized, or bias-corrected
  bootstrap is included yet;
- model order and spatial weights are fixed across bootstrap replications;
- bootstrap replications execute serially;
- `STARMAResult` for integrated models remains on the transformed scale;
- dense matrices are used throughout;
- no missing-value handling, exogenous regressors, or interventions yet.

## Handoff instruction

Before every substantial development step, read this file, `docs/model.md`,
`docs/forecasting.md`, `docs/bootstrap.md`, and
`docs/development/STEP_06_BOOTSTRAP.md`. After completing a step, update the
completed, validation, next-task, and limitation sections so another conversation
can continue without reconstructing project history.
