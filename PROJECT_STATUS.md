# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with transparent statistical conventions, typed public APIs, immutable result
objects, independent numerical tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #16 have been squash-merged into `main`.
- `main` is version `0.0.16` at merge commit
  `7b1fe194abf710e8f929066eb2669dd6b5dc777e`.
- Current branch: `agent/seasonal-likelihood-inference`.
- Current draft pull request: PR #17, `Add seasonal likelihood-Hessian
  inference`.
- Current development version: `0.0.17`.
- The branch adds observed-information inference for multiplicative ordinary and
  seasonal factor parameters and covariance optimizer coordinates.
- Every finite-difference stencil point reconstructs the complete expanded
  recursion and rejects enabled stationarity/invertibility penalty regions.

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

### Multiplicative seasonal Gaussian Kalman model

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

Matrix order is preserved for non-commuting spatial operators. Multiplicative
cross-lag matrices are applied directly and are not projected onto the supplied
spatial-weight basis.

### Arbitrary-lag state space and admissibility

- equal temporal lags from factor expansion are aggregated;
- dense zero blocks represent absent temporal lags through the maximum lag;
- the companion state contains observation and MA-innovation histories;
- one state-space representation is reused for likelihood, filtering,
  prediction, RTS smoothing, and original innovation smoothing;
- AR stationarity is checked on the complete expanded AR companion;
- positive-sign MA invertibility is checked on the inverse-recursion companion;
- ordinary and seasonal factor starts are shrunk toward feasible regions;
- infeasible optimizer candidates receive explicit penalties and final hard
  checks.

For `K` spatial weights, the dynamic factor count is

\[
\mathbf 1_c+K(p+P+q+Q).
\]

Deterministic multiplicative cross matrices are not counted as independent
AIC/BIC parameters.

### Conditional likelihood and scale contract

The combined transformation offset is

\[
o=d+Ds.
\]

The evaluated likelihood is conditional on the removed ordinary and seasonal
history. Approximate diffuse initialization applies to the stationary expanded
transformed state; it is not an exact diffuse likelihood for the original level
process.

Missing level cells are never imputed. `NaN` propagates through ordinary and
seasonal differencing stencils. Filtering and smoothing operate on the combined
transformed scale. Original-scale forecasts reverse seasonal histories first
and ordinary anchors second, and are refused when required terminal histories
are non-finite.

## Completed in 0.0.17

### Seasonal observed-information inference

- added `infer_seasonal_kalman_starima()`;
- added `SeasonalKalmanSTARIMA.infer()`;
- reused the central finite-difference curvature engine and immutable
  `LikelihoodInferenceResult` contract;
- evaluates central score differences, diagonal second differences, and
  four-corner mixed partials on the raw optimizer scale;
- reconstructs the optional intercept, ordinary/seasonal AR and MA factors,
  innovation covariance, multiplicative matrix expansion, arbitrary-lag state
  space, and Gaussian likelihood at every objective point;
- rejects stencil points that enter enabled expanded AR-stationarity or
  MA-invertibility penalty regions rather than differentiating the penalty;
- reports factor-only and complete optimizer tables, standard errors, normal
  statistics, p-values, confidence intervals, correlation, score, finite-
  difference steps, Hessian eigenvalues, rank, condition number, objective
  value, evaluation count, and admissibility-boundary distances;
- requires positive-definite full-rank observed information by default;
- permits an explicit positive-eigenspace generalized inverse only through
  `allow_singular=True` and records `used_pseudoinverse=True`;
- reuses scalar, diagonal, and full-Cholesky natural innovation-covariance delta
  inference;
- preserves ordinary/seasonal factor-to-natural-covariance cross uncertainty.

### Public result contract

The seasonal estimator returns the same `LikelihoodInferenceResult` used by
stationary Kalman STARMA. Natural covariance inference is obtained with

```python
natural = inference.innovation_covariance_inference()
```

The public natural-result accessors are `parameter_names`, `estimates`, `table`,
`confidence_intervals()`, covariance/standard-error arrays, and
`dynamic_cross_covariance`. Lower-level element metadata remains on
`natural.transform`.

### Validation coverage

Tests cover:

1. exact zero-seasonal-order inference equivalence with `KalmanSTARMA` under
   identical observations, starts, and finite-difference settings;
2. full-rank pure seasonal AR curvature;
3. equality between `.infer()` and `infer_seasonal_kalman_starima()`;
4. natural scalar innovation-variance inference and factor/covariance cross
   covariance;
5. direct expanded non-stationary objective penalty verification;
6. strict singular-Hessian rejection and explicit finite positive-eigenspace
   generalized inverse;
7. fitted-state, rank-threshold, and finite-difference-step validation;
8. the complete inherited package test suite.

## Authoritative validation for 0.0.17

GitHub Actions CI #386, run ID `30876026103`, validated the complete
implementation, tests, example, metadata, and documentation head:

- 148 tests passed in the coverage job;
- total branch coverage was 87.14%, above the required 80%;
- `src/pystarmax/seasonal_likelihood_inference.py` coverage was 93.5%;
- `src/pystarmax/seasonal_maximum_likelihood.py` coverage remained 87.6%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A validation-record-only merge-gate CI is required after this status and the
Step 17 handoff are updated. No implementation, test, API, example, or method
changes are made after CI #386.

## Prior validation for 0.0.16

GitHub Actions CI #360, run ID `30858747303`, reported 143 passing tests,
87.04% total branch coverage, 87.6% coverage for the seasonal Kalman module, and
successful quality, strict documentation, packaging, and Ubuntu/Windows/macOS
Python 3.11–3.14 checks. Validation-record-only CI #363 also passed before PR
#16 was squash-merged.

## Design principles

1. Keep conditional least-squares and Gaussian maximum-likelihood estimators
   separate and explicit.
2. Distinguish conditional transformed likelihoods from exact diffuse
   level-state likelihoods.
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
11. Do not interpret finite-difference penalty curvature as likelihood
    information.
12. Require positive-definite full-rank observed information by default and
    label diagnostic pseudoinverses explicitly.
13. Use stable solves before explicit inverses and expose pseudoinverse use.
14. Distinguish state disturbances from original location innovations.
15. Preserve innovation uncertainty not identified by the state selection map.
16. Require analytic or independent Gaussian references for numerical claims.
17. Keep public numerical arrays immutable.

## Immediate next tasks

1. Run the validation-record-only merge-gate CI.
2. Update PR #17, mark it ready, and squash-merge it into `main`.
3. Add original-scale Gaussian forecast intervals for ordinary and seasonal
   Kalman STARIMA.
4. Design exact diffuse integrated level-state likelihood and smoothing as a
   separate API.
5. Add sparse arbitrary-lag state matrices, cross-time innovation covariance,
   simulation smoothing, order selection, exogenous inputs, adapters, and
   cross-language fixtures.

## Known limitations

- ordinary and seasonal Kalman STARIMA likelihoods are conditional on
  transformation history, not exact diffuse on the level process;
- central finite-difference inference requires `1 + 2*k**2` objective
  evaluations for `k` optimizer coordinates;
- inference can be sensitive to step size in flat, highly curved, or
  near-boundary likelihood directions;
- feasibility uses penalties rather than a smooth constrained
  parameterization;
- robust, sandwich, profile-likelihood, likelihood-ratio, and Kalman bootstrap
  inference are unavailable;
- original-scale Kalman STARIMA forecast intervals are unavailable;
- original-scale filtered and smoothed level-state distributions are not
  returned;
- state and innovation smoothing treat parameters as fixed;
- innovation smoothing exposes marginal covariance by transition, not
  cross-time covariance;
- exact diffuse filtering and smoothing are unavailable;
- simulation smoothing is unavailable;
- dense arbitrary-lag companions can grow rapidly with seasonal period/order and
  spatial dimension;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/starima.md`, `docs/integrated_maximum_likelihood.md`, `docs/seasonal.md`,
`docs/seasonal_maximum_likelihood.md`, `docs/seasonal_likelihood_inference.md`,
`docs/admissibility.md`, `docs/state_space.md`, `docs/smoothing.md`,
`docs/innovation_smoothing.md`, `docs/maximum_likelihood.md`,
`docs/likelihood_inference.md`, `docs/covariance_inference.md`, and the latest
development handoff. Update repository state, validation, next tasks, and
limitations after every completed stage.
