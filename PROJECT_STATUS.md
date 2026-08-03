# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with transparent statistical conventions, typed public APIs, immutable result
objects, independent numerical tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #15 have been squash-merged into `main`.
- `main` is version `0.0.15` at merge commit
  `6306da985ec760142f15cb113760ab58d758afae`.
- Current branch: `agent/seasonal-kalman-starima`.
- Current draft pull request: PR #16, `Add multiplicative seasonal Kalman
  STARIMA`.
- Current development version: `0.0.16`.
- The current estimator fits ordinary and seasonal factor parameters by a
  Gaussian Kalman likelihood on the combined transformed process.
- Multiplicative cross-lag matrices are ordered products of factor matrices,
  applied directly without projection onto the spatial-weight basis.
- The likelihood remains conditional on the ordinary-seasonal transformation
  history and is not presented as an exact diffuse level-state likelihood.

## Completed baseline through 0.0.15

- immutable spatial weights, simulation, diagnostics, and conditional STARMA;
- ordinary and multiplicative seasonal conditional STARIMA;
- reversible ordinary-seasonal differencing and original-scale reconstruction;
- conditional and parameter-refitting bootstrap forecast intervals;
- rolling-origin calibration, sharpness, and point-error evaluation;
- stationary STARMA state-space construction and Gaussian filtering;
- partial-location and fully missing-row likelihood handling;
- stationary `KalmanSTARMA` with scalar, diagonal, and full Cholesky covariance;
- AR stationarity and positive-sign inverse-MA admissibility enforcement;
- observed-information Hessian inference and natural covariance delta method;
- RTS fixed-interval state smoothing and lag-one covariance;
- state-equation and original location-level innovation disturbance smoothing;
- unresolved selection-nullspace covariance and support diagnostics;
- conditional ordinary-integrated `KalmanSTARIMA(p,d,q)`;
- transformed/original-scale contracts, missing propagation, terminal-anchor
  validation, and arbitrary ordinary inverse differencing.

## Completed in 0.0.16

### Multiplicative factor model

For

\[
x_t=(1-B)^d(1-B^s)^D y_t,
\]

`SeasonalKalmanSTARIMA` fits

\[
\Phi_s(B^s)\Phi(B)x_t
=
c+\Theta_s(B^s)\Theta(B)\eta_t,
\qquad
\eta_t\sim\mathcal N(0,\Sigma).
\]

The package convention is

\[
\Phi(B)=I-\sum_iA_iB^i,
\qquad
\Phi_s(B^s)=I-\sum_rS_rB^{rs},
\]

\[
\Theta(B)=I+\sum_jM_jB^j,
\qquad
\Theta_s(B^s)=I+\sum_uN_uB^{us}.
\]

Consequently:

- ordinary AR terms are `+A_i`;
- seasonal AR terms are `+S_r`;
- AR cross terms are `-S_r @ A_i`;
- ordinary and seasonal positive-sign MA terms are `+M_j` and `+N_u`;
- MA cross terms are `+N_u @ M_j`.

Matrix order is preserved for non-commuting spatial operators.

### Arbitrary-lag state-space construction

- equal temporal lags from factor expansion are aggregated;
- expanded matrices remain arbitrary location-by-location matrices;
- no cross term is projected back onto `(W0, W1, ...)`;
- dense zero blocks represent absent temporal lags up to the maximum lag;
- the companion state includes observation histories and MA innovation
  histories;
- the current innovation enters the current observation block and, when needed,
  the first innovation-history block;
- transformed predictions, filtering, RTS smoothing, and original innovation
  smoothing use this single state-space representation.

### Parameterization and information criteria

For `K` spatial weights, dynamic factor count is

\[
\mathbf 1_c+K(p+P+q+Q).
\]

Cross-lag matrices are deterministic functions of factor parameters and are not
counted as independent AIC/BIC parameters. Covariance parameter counts follow
the scalar, diagonal, or full Cholesky codec.

The result exposes:

- optional intercept;
- ordinary and seasonal AR/MA factor arrays;
- raw optimizer parameters and names;
- complete expanded AR/MA lags and matrices;
- innovation covariance;
- likelihood, AIC, BIC, finite observation count, and optimizer diagnostics;
- expanded AR and inverse-MA spectral radii and configured limits;
- original/transformed row and missing-cell counts;
- original-scale forecast availability and the training filter result.

### Expanded admissibility

- AR stationarity uses the companion of the complete expanded AR recursion;
- MA invertibility uses the companion whose top row contains the negatives of
  the complete expanded positive-sign MA matrices;
- ordinary and seasonal AR factor starts are shrunk together when necessary;
- ordinary and seasonal MA factor starts are shrunk together independently;
- infeasible optimizer candidates receive explicit penalties;
- final candidates are checked again before a result is accepted;
- `SeasonalKalmanAdmissibility` preserves expanded lags, matrices, radii,
  limits, and joint status.

### Conditional transformation likelihood

The combined offset is

\[
o=d+Ds.
\]

The evaluated likelihood is

\[
\ell_c(\vartheta;y_{1:T})
=
\ell\left(
\vartheta;(1-B)^d(1-B^s)^D y_{o+1:T}
\mid\mathcal H_o
\right).
\]

The removed ordinary and seasonal history is conditioned on. Approximate
diffuse initialization applies only to the stationary expanded state and does
not convert the method into an exact diffuse level-state likelihood.

### Missing observations and scale boundaries

- level observations are never imputed before transformation;
- `NaN` propagates through ordinary and seasonal difference stencils;
- transformed finite locations participate in measurement updates;
- fully missing transformed rows perform prediction only;
- both original and transformed missing-cell counts are retained;
- new level matrices are transformed before filtering and smoothing;
- `filter()`, `smooth()`, `smooth_innovation_disturbances()`,
  `to_state_space()`, `predict_differenced()`, and `fitted_differenced()` remain
  on the combined transformed scale;
- `predict()` reverses seasonal histories first and ordinary anchors second;
- `fitted_original()` uses the complete combined differencing polynomial and
  observed lag history;
- original-scale prediction is refused when any required ordinary anchor or
  seasonal history is non-finite.

### Validation references

Tests cover:

1. exact zero-seasonal-order equivalence with `KalmanSTARMA` under identical
   starts;
2. direct AR and MA multiplicative cross-lag signs;
3. pure seasonal AR estimation and complete companion admissibility;
4. seasonal random-walk cycle reconstruction;
5. seasonal missing-value propagation and finite likelihood counts;
6. transformed forecasts with refusal of original forecasts when terminal
   seasonal history is incomplete;
7. combined ordinary-seasonal new-data filter, smoother, and innovation lengths;
8. original fitted alignment using the complete combined lag polynomial;
9. constructor, sample-offset, dimensionality, infinity, and fitted-state
   validation.

## Core validation for 0.0.16

GitHub Actions CI #346, run ID `30858019154`, validated the formatted numerical
core before the final documentation expansion:

- 143 tests passed in the coverage job;
- total branch coverage was 87.04%, above the required 80%;
- `src/pystarmax/seasonal_maximum_likelihood.py` coverage was 87.6%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu and macOS passed on Python 3.11 through 3.14;
- Windows jobs were superseded by later documentation commits after their
  numerical test steps had completed successfully.

A complete final CI matrix is required on the documentation head before PR #16
is marked ready and merged. The authoritative final run must be recorded here
and in the Step 16 handoff.

## Design principles

1. Keep conditional least-squares and Gaussian maximum-likelihood estimators
   separate and explicit.
2. Distinguish conditional transformed likelihoods from exact diffuse level-state
   likelihoods.
3. Optimize and count multiplicative factor parameters, not deterministic
   expanded cross terms.
4. Preserve ordered matrix products and non-symmetric spatial orientation.
5. Never project cross-lag matrices onto a weight basis without declaring an
   approximation model.
6. Never impute missing levels before differencing or likelihood evaluation.
7. Keep transformed and original scales explicitly labelled.
8. Never reconstruct original forecasts without finite ordinary anchors and
   seasonal histories.
9. Use the positive MA sign consistently in inverse-recursion diagnostics.
10. Treat spectral-radius penalties as feasibility controls, not smooth
    parameterizations.
11. Use stable solves before explicit inverses and expose pseudoinverse use.
12. Distinguish state disturbances from original location innovations.
13. Preserve innovation uncertainty not identified by the state selection map.
14. Require analytic or independent Gaussian references for numerical claims.
15. Keep public numerical arrays immutable.

## Immediate next tasks

1. Run the complete final CI matrix on the documentation head.
2. Record the final run identifier, test count, and coverage.
3. Update PR #16, mark it ready, and squash-merge it into `main`.
4. Add observed-information and natural covariance inference for seasonal factor
   parameters.
5. Add original-scale Gaussian forecast intervals for ordinary and seasonal
   Kalman STARIMA.
6. Design exact diffuse integrated level-state likelihood and smoothing as a
   separate API.
7. Add sparse arbitrary-lag state matrices, cross-time innovation covariance,
   simulation smoothing, order selection, exogenous inputs, adapters, and
   cross-language fixtures.

## Known limitations

- ordinary and seasonal Kalman STARIMA likelihoods are conditional on
  transformation history, not exact diffuse on the level process;
- seasonal-factor observed-information and natural covariance inference are not
  yet exposed;
- original-scale Kalman STARIMA forecast intervals are unavailable;
- original-scale filtered and smoothed level-state distributions are not
  returned;
- state and innovation smoothing treat parameters as fixed;
- innovation smoothing exposes marginal covariance by transition, not cross-time
  covariance;
- exact diffuse filtering and smoothing are unavailable;
- simulation smoothing is unavailable;
- feasibility uses penalties rather than a smooth bijection;
- dense arbitrary-lag companions can grow rapidly with seasonal period/order and
  spatial dimension;
- robust, sandwich, profile-likelihood, likelihood-ratio, and Kalman bootstrap
  inference are unavailable;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/starima.md`, `docs/integrated_maximum_likelihood.md`, `docs/seasonal.md`,
`docs/seasonal_maximum_likelihood.md`, `docs/admissibility.md`,
`docs/state_space.md`, `docs/smoothing.md`, `docs/innovation_smoothing.md`,
`docs/maximum_likelihood.md`, `docs/likelihood_inference.md`,
`docs/covariance_inference.md`, and the latest development handoff. Update
repository state, validation, next tasks, and limitations after every completed
stage.
