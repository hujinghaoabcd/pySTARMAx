# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with transparent statistical conventions, typed public APIs, immutable result
objects, independent numerical tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #12 have been squash-merged into `main`.
- `main` is version `0.0.12` plus the restored complete CI workflow.
- Current branch: `agent/kalman-state-smoothing`.
- Current draft pull request: PR #13, `Add Kalman fixed-interval state smoothing`.
- Current development version: `0.0.13`.
- PR #13 contains 20 formal code, test, example, metadata, and documentation
  files; no temporary workflow or diagnostic files remain.
- The implementation and documentation head passed the complete CI matrix in
  GitHub Actions run #304, run ID `30852463900`.

## Completed baseline through 0.0.11

- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA iterative conditional least squares;
- simulation, recursive prediction, STACF/STPACF, and residual diagnostics;
- ordinary STARIMA and constrained multiplicative seasonal STARIMA;
- reversible ordinary-seasonal differencing and original-scale reconstruction;
- conditional and bootstrap forecast intervals;
- rolling-origin calibration, sharpness, and point-error evaluation;
- stationary STARMA state-space construction;
- stationary, known, and approximate diffuse Kalman initialization;
- Gaussian filtering with partial-location and fully missing rows;
- independent `KalmanSTARMA` Gaussian maximum-likelihood estimation;
- scalar, diagonal, and Cholesky full innovation covariance;
- AIC, BIC, optimizer, covariance, filtering, and prediction diagnostics;
- finite-difference likelihood score and observed-information Hessian;
- coefficient standard errors, normal tests, intervals, rank, condition, score,
  and boundary diagnostics;
- reusable AR and inverse-MA companion eigensystem diagnostics;
- dual stationarity/invertibility start shrinkage, penalties, and final checks.

## Completed in 0.0.12

### Natural covariance transformations

- scalar shared variance from one log standard deviation;
- diagonal location variances from location log standard deviations;
- full covariance lower triangle from log-Cholesky diagonal and unrestricted
  lower-factor entries;
- analytic Jacobian for `Sigma = L L.T`;
- first-order delta covariance `J V J.T`;
- retained dynamic-coefficient/covariance-element cross covariance;
- immutable natural estimates, standard errors, correlations, intervals, and
  symmetric matrix-shaped standard errors;
- no duplicated scalar variance parameters;
- no ordinary variance-equals-zero Wald test for a boundary null;
- no silent clipping of negative lower endpoints from unbounded normal
  approximations.

### Integration corrections completed during 0.0.13

- `infer_kalman_starma()` now obtains `n_locations` from the fitted innovation
  covariance instead of an undefined local variable;
- covariance-element indices have an explicit variable-length tuple type for
  current mypy versions;
- the affected covariance and likelihood inference regression tests pass.

## Completed in 0.0.13

### Rauch--Tung--Striebel smoothing

- added immutable `KalmanSmootherResult`;
- added public `kalman_smoother()`;
- added fitted `KalmanSTARMA.smooth()` for training or new incomplete data;
- implemented RTS state mean and covariance recursion;
- retained every smoothing gain;
- retained lag-one covariance `Cov(alpha_t, alpha_(t+1) | y_1:T)`;
- exposed smoothed observation means and covariances;
- derived state-equation disturbance means and conditional covariances;
- preserved partial-location and fully missing-row semantics from filtering.

### Rank-deficient prediction policy

- full-rank predicted covariance uses a stable linear solve;
- rank-deficient prediction uses a positive-eigenspace pseudoinverse;
- `rcond` defines the retained eigenspace;
- every transition records numerical prediction rank and pseudoinverse use;
- covariance matrices are symmetrized and only tiny negative eigenvalues are
  projected to zero;
- materially indefinite covariance raises.

### Validation references

- scalar AR(1) one-gap Gaussian bridge with analytic mean and variance;
- contiguous missing block compared with direct joint-Gaussian conditioning;
- exact state-disturbance recovery for fully observed scalar AR(1);
- final smoothed state/covariance equality with the final filtered values;
- smoothed covariance reduction relative to filtering;
- explicit rank-deficient prediction path;
- fitted-model smoothing of a new incomplete matrix;
- one-time-point edge case, immutability, and validation tests.

## Mathematical convention

The state equation is

\[
\alpha_{t+1}=d+T\alpha_t+w_{t+1}.
\]

The RTS gain is

\[
J_t=P_{t|t}T^\top P_{t+1|t}^{+}.
\]

The smoothed moments are

\[
a_{t|T}=a_{t|t}+J_t(a_{t+1|T}-a_{t+1|t}),
\]

\[
P_{t|T}=P_{t|t}+J_t(P_{t+1|T}-P_{t+1|t})J_t^\top.
\]

The stored lag-one covariance is

\[
C_{t,t+1|T}=J_tP_{t+1|T}.
\]

`state_disturbance_mean` describes
`alpha_(t+1) - d - T @ alpha_t`. It is not automatically the original
location-level innovation because the state selection matrix may not be
one-to-one.

## Authoritative 0.0.13 validation

GitHub Actions CI run #304, run ID `30852463900`, validated the complete
implementation and documentation head:

- 119 tests passed;
- total branch coverage was 87.32%, above the required 80%;
- `src/pystarmax/smoothing.py` branch coverage was 87.1%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14;
- no temporary workflows or diagnostic files were present in the validated
  branch head.

A final documentation-only commit records these results. Its CI run is used as
the merge gate and is reported in the PR description without rewriting this
section again.

## Design principles

1. Keep conditional and maximum-likelihood estimators separate and explicit.
2. Preserve one `(time, location)` observation convention.
3. Never impute missing observations inside likelihood, filtering, or smoothing.
4. Preserve supplied non-symmetric spatial-matrix orientation.
5. Use the positive MA sign consistently in inverse-recursion diagnostics.
6. Treat spectral-radius penalties as feasibility controls, not smooth
   parameterizations.
7. Refuse indefensible observed information by default.
8. Keep optimizer-scale and natural-scale uncertainty separate and auditable.
9. Use stable solves before explicit inverses.
10. Record rank-deficient pseudoinverse use rather than hiding it.
11. Distinguish state disturbances from original location innovations.
12. Require analytic or independent Gaussian references for new numerical claims.
13. Keep public numerical arrays immutable.

## Immediate next tasks

1. Run the final complete CI matrix on this validation-record-only head.
2. Update PR #13 with the final merge-gate CI identifier.
3. Mark PR #13 ready and squash-merge it into `main`.
4. Create `agent/innovation-disturbance-smoothing` from the merge commit.
5. Derive original location-level innovation disturbance smoothing without using
   a naive selection-matrix pseudoinverse.
6. Add integrated and multiplicative seasonal state-space/MLE wrappers.
7. Add sparse spatial/state matrices, order selection, exogenous inputs,
   ecosystem adapters, and cross-language fixtures.

## Known limitations

- state smoothing is fixed-parameter and does not propagate parameter
  uncertainty;
- state disturbances are not yet mapped to original location innovations;
- exact diffuse filtering and smoothing are unavailable;
- maximum likelihood covers stationary non-seasonal STARMA only;
- integrated and seasonal wrappers still use conditional estimation;
- feasibility is enforced by penalties rather than a smooth bijection;
- dense eigendecomposition and dense state matrices limit large networks;
- natural covariance intervals are first-order unbounded normal approximations;
- robust, sandwich, profile-likelihood, likelihood-ratio, and Kalman-MLE
  bootstrap inference are unavailable;
- bootstrap and rolling refits execute serially;
- model order and spatial weights remain fixed across bootstrap replications;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/admissibility.md`, `docs/state_space.md`, `docs/smoothing.md`,
`docs/maximum_likelihood.md`, `docs/likelihood_inference.md`,
`docs/covariance_inference.md`, and the latest development handoff. Update
repository state, validation, next tasks, and limitations after every completed
stage.
