# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 through PR #8 were squash-merged into `main`.
- PR #8, state-space filtering and missing observations, was merged as `258ceac7`.
- `main` is version `0.0.8`.
- Current branch: `agent/kalman-maximum-likelihood`.
- Current draft pull request: PR #9, `Add Kalman maximum-likelihood STARMA estimation`.
- Current development target: version `0.0.9`.
- Current stage: direct stationary Gaussian Kalman maximum likelihood.

## Completed baseline through 0.0.8

- typed package, MIT licence, citation metadata, strict documentation, and CI;
- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA iterative conditional least squares;
- deterministic simulation, recursive prediction, diagnostics, and result summaries;
- ordinary STARIMA and constrained multiplicative seasonal STARIMA;
- reversible ordinary-seasonal differencing and original-scale reconstruction;
- conditional and bootstrap forecast intervals;
- rolling-origin calibration, sharpness, and point-error evaluation;
- explicit stationary STARMA state-space construction;
- Gaussian fixed-parameter Kalman likelihood;
- stationary, known, and approximate diffuse initialization;
- partial-location and fully missing-row filtering.

## Completed in the current 0.0.9 step

- added independent `KalmanSTARMA` without changing the conditional estimator;
- added direct Gaussian likelihood optimization with L-BFGS-B;
- added scalar shared innovation variance;
- added diagonal location-specific innovation variances;
- added full positive-definite Cholesky innovation covariance;
- counted all dynamic and covariance parameters in AIC and BIC;
- supported complete and partially missing observations directly in the objective;
- restricted location-mean filling to automatic starting-value construction;
- used the conditional estimator for dynamic starting values;
- added spectral-radius feasibility penalties and final stability validation;
- added immutable `KalmanSTARMAResult` with optimizer, covariance, likelihood,
  information-criterion, state-space, and filtering diagnostics;
- added filtering of new incomplete matrices and recursive conditional-mean prediction;
- added scalar white-noise exact MLE, AR recovery, diagonal/full covariance,
  missing-data, prediction, and validation tests;
- added complete maximum-likelihood documentation, example, roadmap, and handoff;
- updated public exports, package version, and citation metadata to 0.0.9.

## Statistical interpretation

`KalmanSTARMA` is a distinct stationary Gaussian maximum-likelihood estimator.
It does not replace `STARMA.fit()`, which remains the transparent conditional
baseline used by existing bootstrap and seasonal wrappers.

Missing cells are omitted from Kalman measurement updates. Automatic mean filling
is used only to obtain starting values. Full innovation covariance is represented
as `L @ L.T`, with exponentiated Cholesky diagonal elements.

The current stationarity control is an explicit spectral-radius feasibility
boundary. It is not a smooth stability reparameterization. MA invertibility is
not yet constrained.

## Final validation for 0.0.9

GitHub Actions CI run #217 completed successfully on the fully documented branch
head:

- 85 tests passed;
- total branch coverage was 87.84%, above the configured 80% threshold;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic reference regeneration produced a clean diff;
- strict MkDocs construction passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14;
- the final PR surface contains only formal source, test, example, documentation,
  metadata, and navigation files, with no temporary workflow files.

The final validation-record edit changes documentation only. Numerical code,
tests, public exports, metadata, and CI configuration are unchanged from the
validated head.

## Design principles

1. Keep conditional and maximum-likelihood estimators separate and explicit.
2. Preserve one `(time, location)` convention.
3. Never impute missing observations inside a likelihood evaluation.
4. Parameterize full covariance so positive definiteness is structural.
5. Count covariance parameters honestly in information criteria.
6. Report optimizer convergence and feasibility diagnostics without hiding failures.
7. Treat spectral-radius penalties as feasibility controls, not smooth constraints.
8. Keep non-symmetric spatial-matrix orientation explicit.
9. Use stable linear solves and immutable public result arrays.
10. Add independent numerical references before claiming external equivalence.

## Immediate next tasks

1. Merge PR #9 after the documentation-only final check repeats successfully.
2. Add likelihood-Hessian covariance and standard errors.
3. Add curvature rank, condition-number, and weak-identification diagnostics.
4. Add explicit stationarity and MA invertibility checks.
5. Add constrained fitting or stable reparameterization.
6. Add integrated and multiplicative seasonal maximum-likelihood wrappers.
7. Add state and disturbance smoothing.
8. Add sparse matrices, ecosystem adapters, order selection, and exogenous inputs.
9. Add cross-language estimator fixtures and prepare the first PyPI pre-release.

## Known limitations

- no likelihood-Hessian standard errors or confidence intervals;
- no smooth stationarity parameterization or MA invertibility constraint;
- approximate diffuse initialization is not exact diffuse likelihood;
- no state or disturbance smoothing;
- maximum likelihood currently covers stationary STARMA only;
- integrated and seasonal wrappers still use conditional estimation;
- dense matrices are used throughout;
- conditional/bootstrap rolling refits execute serially;
- model order and spatial weights remain fixed across bootstrap replications;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/state_space.md`, `docs/maximum_likelihood.md`, `docs/forecasting.md`, and
the latest file under `docs/development/`. Update repository state, validation,
next tasks, and limitations after every completed stage.
