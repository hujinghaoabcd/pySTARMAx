# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 through PR #7 were squash-merged into `main`.
- PR #7, rolling-origin interval evaluation, was merged as `884634b0`.
- `main` is version `0.0.7`.
- Current branch: `agent/state-space-kalman-core`.
- Current draft pull request: PR #8, `Add Kalman state-space filtering and missing observations`.
- Current development target: version `0.0.8`.
- Current stage: fixed-parameter Gaussian state-space filtering and missing observations.

## Completed baseline through 0.0.7

- typed `src/` package, MIT licence, citation metadata, strict docs, and CI;
- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA iterative conditional least squares;
- coefficient uncertainty, innovation covariance, approximate likelihood, AIC, and BIC;
- deterministic simulation and recursive point prediction;
- classical STACF, Yule-Walker STPACF, and residual portmanteau diagnostics;
- independent exact-rational diagnostic reference fixture;
- ordinary STARIMA and multiplicative seasonal STARIMA;
- reversible ordinary-seasonal differencing and original-scale reconstruction;
- conditional innovation forecast intervals;
- residual and Gaussian parametric bootstrap intervals with model refitting;
- expanding and fixed-window rolling-origin interval evaluation;
- empirical coverage, width, Winkler score, MAE, and RMSE summaries.

## Completed in the current 0.0.8 step

- added immutable `StateSpaceModel` and `KalmanFilterResult`;
- added companion-form construction from temporal-by-spatial AR and MA parameters;
- preserved explicit matrix orientation for non-symmetric spatial weights;
- added the state disturbance covariance `R Q R'`;
- added stationary initialization using the unconditional mean and discrete Lyapunov covariance;
- added user-supplied known initialization;
- added explicit approximate diffuse initialization;
- added Gaussian Kalman filtering and fixed-parameter log likelihood;
- added partial-location missing-observation updates using reduced measurement matrices;
- added prediction-only handling for fully missing time rows;
- retained `NaN` in unobserved innovation and innovation-covariance entries;
- added scale-aware Cholesky jitter and filtered-covariance stability checks;
- exposed `to_state_space()` and `filter_state_space()` on stationary `STAR` and `STARMA`;
- updated public exports and package metadata to 0.0.8;
- added a scalar AR(1) closed-form likelihood check;
- added tests for non-symmetric operators, missing patterns, initialization, validation, and fitted-model integration;
- added state-space documentation and a runnable missing-observation example.

## Statistical interpretation

The new Kalman layer evaluates a supplied fixed STARMA parameter set. When used
through `STAR.to_state_space()` or `STARMA.to_state_space()`, those parameters
come from the existing conditional estimator. The Kalman likelihood is reported
separately and does not overwrite the historical `STARMAResult` fields.

For `initialization="stationary"`, the transition spectral radius must be below
one. `initialization="diffuse"` is a large-variance approximation and is not
claimed to be exact diffuse likelihood.

Missing cells are omitted from each measurement update. They are not interpolated,
mean-filled, or treated as zero. A fully missing row advances the latent state
without adding a likelihood contribution.

## Final validation for 0.0.8

GitHub Actions CI run #179 completed successfully on the clean implementation
head:

- 77 tests passed;
- total branch coverage was 88.25%, above the configured 80% threshold;
- Black passed;
- isort passed;
- Ruff passed;
- mypy passed with no type errors;
- exact diagnostic reference regeneration produced a clean diff;
- strict MkDocs construction passed;
- source distribution and wheel built successfully;
- Twine metadata checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14;
- the final PR surface contains 14 formal files and no temporary workflow files.

The final handoff edit changes documentation only; the numerical implementation,
tests, public API, metadata, and workflow configuration are unchanged from the
validated implementation head.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep conditional estimation and state-space likelihood conceptually separate.
3. Make every non-symmetric spatial-matrix orientation explicit.
4. Never impute missing observations inside the Kalman filter.
5. Use stable linear solves rather than explicit covariance inverses.
6. Record numerical jitter instead of silently hiding it.
7. Distinguish stationary, known, approximate diffuse, and future exact diffuse initialization.
8. Do not call fixed-parameter filtering maximum-likelihood estimation.
9. Keep transformed-scale inference distinct from original-scale reconstruction.
10. Add independent numerical references before claiming external equivalence.
11. Keep future sparse and compiled acceleration behind stable public APIs.

## Immediate next tasks

1. Add direct optimization of the Kalman likelihood.
2. Add scalar, diagonal, and full innovation-covariance parameterizations.
3. Add likelihood-Hessian standard errors and optimizer diagnostics.
4. Add stationarity and invertibility checks with optional constrained fitting.
5. Add integrated and multiplicative seasonal state-space wrappers.
6. Add sparse spatial matrices and large-network computation.
7. Add automatic order selection and exogenous regressors/interventions.
8. Add NetworkX, libpysal, GeoPandas, and OSMnx adapters.
9. Add cross-language estimator fixtures and prepare the first PyPI pre-release.
10. Add optional smoothing, advanced bootstrap, and interval calibration extensions.

## Known limitations

- direct Kalman maximum-likelihood parameter estimation is not yet implemented;
- exact diffuse initialization and smoothing are not yet implemented;
- current state-space model integration is limited to stationary `STAR` and `STARMA`;
- stationarity and invertibility constraints are not imposed during conditional fitting;
- the conditional estimator still uses recursively estimated innovations;
- seasonal nonlinear estimation uses zero pre-sample innovations;
- historical information criteria still use a scalar innovation-variance approximation;
- bootstrap and rolling-origin refits execute serially;
- no block, wild, studentized, or bias-corrected bootstrap is included;
- model order and spatial weights remain fixed across bootstrap replications;
- `STARMAResult` for integrated models remains on the transformed scale;
- dense matrices are used throughout;
- exogenous regressors and interventions are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/state_space.md`, `docs/forecasting.md`, `docs/bootstrap.md`,
`docs/evaluation.md`, and the latest file under `docs/development/`. After a
stage is completed, update repository state, validation, next tasks, and known
limitations so development can continue without reconstructing history.
