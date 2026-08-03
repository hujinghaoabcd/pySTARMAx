# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with transparent statistical conventions, typed public APIs, immutable result
objects, independent numerical tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #14 have been squash-merged into `main`.
- `main` is version `0.0.14` at merge commit
  `97e8e202c49c9c613d06a578f71e2945e62d01bd`.
- Current branch: `agent/integrated-kalman-starima`.
- Current draft pull request: PR #15, `Add conditional integrated Kalman
  STARIMA`.
- Current development version: `0.0.15`.
- The current implementation estimates ordinary-integrated models through a
  stationary Gaussian Kalman likelihood on `Delta^d y`, conditional on the
  first `d` original level rows.
- It is explicitly not presented as an exact diffuse integrated level-state
  likelihood.

## Completed baseline through 0.0.14

- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA iterative conditional least squares;
- ordinary and multiplicative seasonal conditional STARIMA;
- reversible ordinary-seasonal differencing and original-scale reconstruction;
- simulation, recursive prediction, STACF/STPACF, and residual diagnostics;
- conditional and bootstrap forecast intervals;
- rolling-origin calibration, sharpness, and point-error evaluation;
- stationary STARMA state-space construction;
- stationary, known, and approximate diffuse Kalman initialization;
- Gaussian filtering with partial-location and fully missing rows;
- independent `KalmanSTARMA` Gaussian maximum-likelihood estimation;
- scalar, diagonal, and Cholesky full innovation covariance;
- AIC, BIC, optimizer, covariance, filtering, and prediction diagnostics;
- finite-difference likelihood score and observed-information Hessian;
- natural-scale innovation covariance delta-method inference;
- AR stationarity and positive-sign MA invertibility diagnostics and enforcement;
- Rauch--Tung--Striebel fixed-interval state smoothing;
- lag-one state covariance and state-equation disturbance moments;
- conditional-Gaussian original location-level innovation smoothing;
- unresolved selection-nullspace covariance, rank, pseudoinverse, and support
  diagnostics.

## Completed in 0.0.15

### Conditional integrated likelihood

For original observations `y_t`, define

\[
x_t=(1-B)^d y_t=\Delta^d y_t.
\]

The transformed process is fitted with the existing stationary Gaussian
`KalmanSTARMA` core. The likelihood scope is

\[
\ell_c(\vartheta;y_{1:T})
=
\ell\left(
\vartheta;\Delta^d y_{d+1:T}\mid y_{1:d}
\right).
\]

Consequences are explicit:

- the first `d` original level rows are conditioned on through differencing;
- the likelihood time axis has `T - d` rows;
- log likelihood, AIC, and BIC describe the transformed conditional model;
- `initialization="diffuse"` concerns only the stationary transformed STARMA
  state and remains a large-variance approximation;
- no claim of exact diffuse integration on the level process is made.

### Public API

- added `KalmanSTARIMA(p,d,q)`;
- added immutable `KalmanSTARIMAResult` wrapping the full stationary
  `KalmanSTARMAResult`;
- exposed original and differenced row counts and missing-cell counts;
- exposed `order`, coefficient and covariance tables, convergence, likelihood,
  AIC, and BIC;
- inherited stationary-core admissibility diagnostics, state-space conversion,
  filtering, smoothing, original innovation smoothing, and likelihood-Hessian
  inference without duplicating optimizer code.

### Scale boundaries

Differenced-scale methods:

- `filter()`;
- `smooth()`;
- `smooth_innovation_disturbances()`;
- `infer()`;
- `to_state_space()`;
- `predict_differenced()`;
- `fitted_differenced()`.

Original-scale methods:

- `predict()` recursively inverts ordinary differences using the stored
  terminal `DifferencingState`;
- `fitted_original()` aligns transformed one-step means to the original sample
  and keeps unavailable rows as `NaN`.

For `d=1`, original forecasts cumulate predicted first differences. For `d=2`,
predicted second differences update the terminal slope before updating the
level. The same recursion extends to arbitrary non-negative `d`.

### Missing observations and anchors

- original missing cells are not imputed before differencing;
- `NaN` propagates through the finite-difference stencil;
- the transformed Kalman filter then applies its existing partial-location and
  fully missing-row rules;
- the result records both original and transformed missing-cell counts;
- new level data are transformed before filtering and smoothing;
- original-scale prediction requires finite terminal values for the level and
  every lower-order difference;
- when terminal anchors are unavailable, transformed-scale fitting and
  forecasting remain valid but `predict()` raises rather than inventing a
  level continuation.

### Validation references

Tests cover:

1. exact `d=0` equivalence with `KalmanSTARMA` under identical starting values;
2. a random walk with drift and cumulative original-scale forecast means;
3. second-order inverse differencing from terminal level and slope;
4. explicit missing-value expansion through first differences;
5. available differenced forecasts but refused original forecasts with a
   missing terminal anchor;
6. new-data filtering, state smoothing, and innovation smoothing on the
   transformed time axis;
7. aligned original-scale fitted values with incomplete observed history;
8. constructor, sample-length, dimensionality, infinity, and fitted-state
   validation.

## Core validation for 0.0.15

GitHub Actions CI #330, run ID `30856284894`, validated the formatted numerical
core before the final documentation expansion:

- 134 tests passed in the coverage job;
- total branch coverage was 86.96%, above the required 80%;
- `src/pystarmax/integrated_maximum_likelihood.py` coverage was 84.3%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu and macOS passed on Python 3.11 through 3.14;
- Windows test steps passed on Python 3.11 through 3.14; one superseded Windows
  3.12 job was marked cancelled only after its pytest and cleanup steps had
  already succeeded because later documentation commits started newer runs.

A complete final CI matrix is required on the documentation head before PR #15
is marked ready and merged. Its run identifier and final result must be recorded
in this file and the Step 15 handoff.

## Design principles

1. Keep conditional least-squares and Gaussian maximum-likelihood estimators
   separate and explicit.
2. Distinguish a conditional differenced likelihood from an exact diffuse
   integrated likelihood.
3. Preserve one `(time, location)` observation convention.
4. Never impute missing observations inside differencing, likelihood, filtering,
   or smoothing.
5. Keep transformed-scale and original-scale outputs explicitly labelled.
6. Never restore original-scale forecasts without finite terminal anchors.
7. Preserve supplied non-symmetric spatial-matrix orientation.
8. Use the positive MA sign consistently in inverse-recursion diagnostics.
9. Treat spectral-radius penalties as feasibility controls, not smooth
   parameterizations.
10. Refuse indefensible observed information by default.
11. Use stable solves before explicit inverses and record pseudoinverse use.
12. Distinguish state disturbances from original location innovations.
13. Preserve innovation uncertainty not identified by the state selection map.
14. Require analytic or independent Gaussian references for numerical claims.
15. Keep public numerical arrays immutable.

## Immediate next tasks

1. Complete final documentation synchronization and Step 15 review.
2. Run the complete CI matrix on the documentation head.
3. Record the authoritative final run, test count, and coverage.
4. Update PR #15, mark it ready, and squash-merge it into `main`.
5. Begin multiplicative seasonal Kalman STARIMA support.
6. Design an exact diffuse integrated level-state likelihood as a separate API.
7. Add original-scale Kalman STARIMA forecast intervals.
8. Add cross-time innovation covariance, simulation smoothing, sparse matrices,
   order selection, exogenous inputs, adapters, and cross-language fixtures.

## Known limitations

- `KalmanSTARIMA` uses a conditional differenced likelihood, not exact diffuse
  integration on the original level process;
- seasonal Kalman maximum likelihood is unavailable;
- original-scale Kalman STARIMA forecast intervals are unavailable;
- original-scale filtered and smoothed level-state distributions are not
  returned;
- state and innovation smoothing treat parameters as fixed;
- innovation smoothing exposes marginal covariance by transition, not cross-time
  covariance;
- the initialization disturbance before the first stored state is unavailable;
- exact diffuse filtering and smoothing are unavailable;
- simulation smoothing is unavailable;
- feasibility is enforced by penalties rather than a smooth bijection;
- dense eigendecomposition and dense state matrices limit large networks;
- natural covariance intervals are first-order unbounded normal approximations;
- robust, sandwich, profile-likelihood, likelihood-ratio, and Kalman-MLE
  bootstrap inference are unavailable;
- bootstrap and rolling refits execute serially;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/starima.md`, `docs/integrated_maximum_likelihood.md`,
`docs/admissibility.md`, `docs/state_space.md`, `docs/smoothing.md`,
`docs/innovation_smoothing.md`, `docs/maximum_likelihood.md`,
`docs/likelihood_inference.md`, `docs/covariance_inference.md`, and the latest
development handoff. Update repository state, validation, next tasks, and
limitations after every completed stage.
