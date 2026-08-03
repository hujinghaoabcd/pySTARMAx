# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with transparent statistical conventions, typed public APIs, immutable result
objects, independent numerical tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #13 have been squash-merged into `main`.
- `main` is version `0.0.13`.
- Current branch: `agent/innovation-disturbance-smoothing`.
- Current draft pull request: PR #14, `Add original innovation disturbance
  smoothing`.
- Current development version: `0.0.14`.
- The implementation maps RTS state disturbances back to original
  location-level innovations with an exact conditional-Gaussian derivation.
- No naive selection-matrix inverse is used, and unidentified innovation
  covariance is retained explicitly.

## Completed baseline through 0.0.13

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
- natural-scale innovation covariance delta-method inference;
- reusable AR and inverse-MA companion eigensystem diagnostics;
- dual stationarity/invertibility start shrinkage, penalties, and final checks;
- Rauch--Tung--Striebel fixed-interval state smoothing;
- lag-one state covariance and state-equation disturbance moments;
- explicit rank and pseudoinverse diagnostics for singular state prediction.

## Completed in 0.0.14

### Conditional-Gaussian innovation recovery

For

\[
w_t=R\eta_t,
\qquad
\eta_t\sim\mathcal N(0,Q),
\]

define

\[
S=RQR^\top,
\qquad
A=QR^\top S^+.
\]

Given the RTS posterior moments

\[
\mu_{w,t}=E(w_t\mid y_{1:T}),
\qquad
V_{w,t}=\operatorname{Var}(w_t\mid y_{1:T}),
\]

the implementation returns

\[
E(\eta_t\mid y_{1:T})=A\mu_{w,t},
\]

\[
\operatorname{Var}(\eta_t\mid y_{1:T})
=Q-AS A^\top+A V_{w,t}A^\top.
\]

### Identifiability and rank policy

- `unresolved_covariance = Q - A S A.T` is retained rather than discarded;
- a full-rank process covariance uses a stable linear solve;
- a rank-deficient process covariance uses a positive-eigenspace pseudoinverse;
- `rcond` controls the retained eigenspace;
- process rank and pseudoinverse use are explicit result fields;
- materially indefinite process or posterior covariance raises;
- only numerically tiny negative covariance eigenvalues are projected to zero.

### Support diagnostics

Exact state-disturbance moments lie in the range of `S`. The result reports:

- `mean_support_residual[t] = ||mu_w - Pi_S mu_w||_2`;
- `covariance_support_residual[t] = ||V_w - Pi_S V_w Pi_S||_F`.

These diagnostics expose numerical leakage or inconsistent externally
constructed state-smoother results. Unsupported state directions are not used to
fabricate original innovation information.

### Public API

- added immutable `InnovationDisturbanceResult`;
- added public `innovation_disturbance_smoother()`;
- added fitted `KalmanSTARMA.smooth_innovation_disturbances()`;
- supports the training sample or a new incomplete observation matrix;
- exposes innovation posterior means and marginal covariances, conditioning map,
  unresolved covariance, process support, rank, and support residuals;
- all public numerical arrays are defensive read-only copies.

### Time convention

State transition index `t` describes

```text
alpha_(t+1) - state_intercept - transition @ alpha_t
```

and therefore innovation result index `t` corresponds to `eta_(t+1)`. A sample
with `T` stored state times produces `T - 1` innovation disturbances. The
initialization disturbance before the first stored state is not reconstructed.

## Validation status

The initial implementation head passed GitHub Actions CI #308 numerically on
all supported operating-system/Python combinations. The only failure was Black
formatting of the new module, which was subsequently corrected using the exact
CI Black version.

The initial coverage job reported:

- 126 tests passed;
- total branch coverage 87.06%, above the required 80%;
- `src/pystarmax/innovation_smoothing.py` coverage 80.1%;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS tests passed on Python 3.11 through 3.14.

On the formatted core head, CI #311 passed Black, isort, Ruff, mypy, independent
reference regeneration, strict MkDocs, and distribution checks. A final complete
CI matrix is required after the documentation expansion. Its run identifier and
final result must be recorded here before PR #14 is marked ready and merged.

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
12. Preserve innovation uncertainty not identified by the state selection map.
13. Require analytic or independent Gaussian references for numerical claims.
14. Keep public numerical arrays immutable.

## Immediate next tasks

1. Finish documentation synchronization and Step 14 handoff review.
2. Run the complete final CI matrix on the documentation head.
3. Record the authoritative run identifier, test count, and coverage.
4. Update PR #14, mark it ready, and squash-merge it into `main`.
5. Begin integrated and multiplicative seasonal state-space/Kalman MLE support.
6. Add cross-time innovation covariance and conditional simulation smoothing.
7. Add sparse spatial/state matrices, order selection, exogenous inputs,
   ecosystem adapters, and cross-language fixtures.

## Known limitations

- state and innovation smoothing are fixed-parameter and do not propagate
  estimator uncertainty;
- innovation smoothing exposes marginal covariance by transition, not cross-time
  innovation covariance;
- the initialization disturbance before the first stored state is unavailable;
- exact diffuse filtering and smoothing are unavailable;
- simulation smoothing is unavailable;
- maximum likelihood covers stationary non-seasonal STARMA only;
- integrated and seasonal wrappers still use conditional estimation;
- feasibility is enforced by penalties rather than a smooth bijection;
- dense eigendecomposition and dense state matrices limit large networks;
- natural covariance intervals are first-order unbounded normal approximations;
- robust, sandwich, profile-likelihood, likelihood-ratio, and Kalman-MLE
  bootstrap inference are unavailable;
- bootstrap and rolling refits execute serially;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/admissibility.md`, `docs/state_space.md`, `docs/smoothing.md`,
`docs/innovation_smoothing.md`, `docs/maximum_likelihood.md`,
`docs/likelihood_inference.md`, `docs/covariance_inference.md`, and the latest
development handoff. Update repository state, validation, next tasks, and
limitations after every completed stage.
