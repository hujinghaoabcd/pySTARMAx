# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with explicit statistical conventions, typed public APIs, immutable numerical
results, independent reference tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #18 have been squash-merged into `main`.
- `main` is version `0.0.18` at merge commit
  `407ced8fec73fd63a4a5357da3b422f62b650e95`.
- Current branch: `agent/exact-diffuse-kalman`.
- Current draft pull request: PR #19, `Add exact diffuse Kalman filtering`.
- Current development version: `0.0.19`.
- The branch adds a separate exact diffuse filtering kernel and ordinary
  integrated level-state constructor.
- Existing `kalman_filter(..., initialization="diffuse")` behavior remains an
  approximate finite large-variance initialization.
- Version 0.0.19 provides filtering and fixed-parameter exact diffuse
  likelihoods; it does not yet provide optimizer-facing exact diffuse MLE or an
  exact diffuse smoother.

## Completed baseline through 0.0.18

### Conditional and Gaussian model families

- immutable `SpatialWeights` with explicit identity lag and preserved matrix
  orientation;
- conditional STAR, STARMA, ordinary STARIMA, and multiplicative seasonal
  STARIMA;
- stationary Gaussian `KalmanSTARMA` with scalar, diagonal, and full Cholesky
  innovation covariance;
- conditional ordinary-integrated `KalmanSTARIMA(p,d,q)`;
- multiplicative seasonal
  `SeasonalKalmanSTARIMA(p,d,q)x(P,D,Q)_s`;
- ordered seasonal matrix products without projection of cross terms onto the
  spatial-weight basis;
- transformed and original scale contracts with reversible ordinary-seasonal
  differencing.

### Diagnostics, admissibility, and inference

- ST covariance, STACF, nested Yule--Walker and regression STPACF, and residual
  portmanteau diagnostics;
- AR stationarity and positive-sign inverse-MA companion diagnostics;
- start shrinkage, optimizer penalties, final feasibility checks, eigenvalues,
  spectral radii, limits, and signed boundary distances;
- finite-difference observed-information Hessians for stationary and seasonal
  factor models;
- coefficient covariance, standard errors, normal tests, intervals,
  correlations, rank, eigenvalues, condition number, and score diagnostics;
- strict full-rank positive-definite Hessian policy with explicit diagnostic
  positive-eigenspace generalized inverse;
- scalar, diagonal, and full-Cholesky natural innovation covariance delta-method
  inference with dynamic/covariance cross uncertainty.

### Filtering, smoothing, and missing observations

- auditable stationary and arbitrary-lag state-space construction;
- stationary, known, and approximate diffuse initialization;
- partial-location measurement updates and prediction-only fully missing rows;
- immutable filtering states, covariance, innovations, masks, likelihood
  contributions, and numerical stabilization diagnostics;
- Rauch--Tung--Striebel fixed-interval state smoothing;
- smoothed observation moments, lag-one state covariance, smoothing gains,
  state-disturbance moments, and rank diagnostics;
- conditional-Gaussian original location-level innovation smoothing;
- retained uncertainty in selection-matrix null spaces and support residual
  diagnostics.

### Forecasting and evaluation

- deterministic recursive forecasts and aligned original-scale fitted values;
- conditional future-innovation and parameter-refitting bootstrap intervals;
- rolling-origin coverage, width, interval score, MAE, RMSE, and horizon
  summaries;
- fixed-parameter Gaussian Kalman forecast paths initialized from the final
  filtered posterior;
- future process-innovation uncertainty;
- pathwise ordinary and seasonal inverse differencing before original-scale
  quantiles;
- reproducible integer-seed and NumPy-generator behavior;
- explicit refusal of original-scale forecasts and intervals when required
  terminal anchors or seasonal histories are missing.

## Completed in 0.0.19

### Exact diffuse covariance decomposition

The initial covariance is represented as

\[
P_1(\kappa)=P_{\ast,1}+\kappa P_{\infty,1},
\qquad \kappa\rightarrow\infty.
\]

- `P_*` stores finite initial uncertainty;
- `P_inf` identifies diffuse state directions;
- both components are predicted and updated separately;
- the filtered diffuse rank records how many unresolved directions remain;
- no arbitrary diffuse scale is used in the exact likelihood.

### Sequential diffuse and Gaussian updates

For an observed scalar design row `z`, the implementation computes

\[
F_\infty=zP_\infty z^\top,
\qquad
F_\ast=zP_\ast z^\top,
\]

\[
M_\infty=P_\infty z^\top,
\qquad
M_\ast=P_\ast z^\top.
\]

When `F_inf` is positive,

\[
K_0=M_\infty/F_\infty,
\]

\[
K_1=M_\ast/F_\infty-K_0F_\ast/F_\infty,
\]

and

\[
a^+=a+K_0v,
\]

\[
P_\ast^+=P_\ast-M_\ast K_0^\top-M_\infty K_1^\top,
\]

\[
P_\infty^+=P_\infty-M_\infty K_0^\top.
\]

The exact diffuse likelihood contribution is

\[
-\tfrac12\log(2\pi F_\infty).
\]

When diffuse variance is numerically zero and finite variance is positive, the
ordinary scalar Gaussian update and likelihood contribution are used.

### Missing and deterministic observations

- observed locations are processed sequentially;
- missing cells are skipped;
- fully missing rows perform prediction only and contribute zero likelihood;
- missing observations do not reduce diffuse rank;
- rank-deficient `P_inf` is supported directly;
- deterministic zero-variance agreement contributes zero;
- deterministic contradiction raises rather than adding arbitrary jitter.

### Exact diffuse result contract

`ExactDiffuseFilterResult` retains immutable arrays for:

- predicted and filtered state means;
- predicted and filtered finite covariance;
- predicted and filtered diffuse covariance;
- scalar innovations;
- finite and diffuse innovation variances;
- observed and diffuse-update masks;
- time likelihood contributions;
- predicted and filtered diffuse-rank paths.

It also reports total exact diffuse likelihood, finite and diffuse observation
counts, initial and final diffuse rank, `diffuse_end_time`, and observation-scale
predicted and filtered values.

### Ordinary integrated level-state construction

Given a stationary transformed state

\[
\beta_t=c+T\beta_{t-1}+R\eta_t,
\qquad x_t=Z\beta_t,
\]

and `x_t = Delta^d y_t`, the augmented state is

\[
[y_t,\Delta y_t,\ldots,\Delta^{d-1}y_t,\beta_t].
\]

Every integration block follows

\[
\Delta^r y_t
=\sum_{k=r}^{d-1}\Delta^k y_{t-1}+x_t.
\]

The constructor therefore inserts the ordered transformed effects `Z @ T`,
`Z @ c`, and `Z @ R` into each integration block.

Initialization uses:

- identity diffuse covariance over the `d * N` integration directions;
- zero finite covariance on integration directions;
- stationary finite mean and covariance for the transformed STARMA state;
- zero initial finite cross covariance between integration and transformed
  blocks.

A nonstationary transformed transition is rejected: nonstationarity must be
represented in the explicit integration chain rather than hidden inside the
finite covariance component.

### Public APIs

Generic exact diffuse filtering:

```python
result = exact_diffuse_filter(
    observations,
    state_space,
    initial_state=initial_state,
    initial_covariance=P_star,
    initial_diffuse_covariance=P_inf,
)
```

Ordinary integrated construction:

```python
specification = build_exact_integrated_state_space(
    transformed_state_space,
    integration_order=d,
)
result = specification.filter(level_observations)
```

Convenience functions:

- `exact_diffuse_loglikelihood()`;
- `exact_integrated_filter()`;
- `exact_integrated_loglikelihood()`.

Public immutable types:

- `ExactDiffuseFilterResult`;
- `ExactIntegratedStateSpace`.

## Authoritative validation for 0.0.19

GitHub Actions CI #422, run ID `30880321271`, validated the complete
implementation, tests, example, metadata, README, documentation home,
navigation, state-space guide, conditional-integrated guide, roadmap, project
status, and Step 19 documentation head:

- 169 tests passed in the coverage job;
- total branch coverage was 87.12%, above the required 80%;
- `src/pystarmax/exact_diffuse.py` coverage was 87.8%;
- `src/pystarmax/exact_integrated.py` coverage was 86.9%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A validation-record-only merge-gate CI is required after this status and the
Step 19 handoff are updated. No implementation, test, API, example, README, or
method-guide changes are made after CI #422.

## Validation references

Tests cover:

1. a local-level exact likelihood equal to a diffuse first-level term plus the
   Gaussian increment likelihood;
2. a local-linear-trend comparison with the adjusted large-variance limit;
3. delayed diffuse-rank reduction under missing observations;
4. sequential partial-location rank reduction;
5. deterministic zero-variance agreement and contradiction;
6. custom rank-deficient diffuse covariance and immutable output arrays;
7. first-order integrated likelihood equal to the drift-adjusted increment
   likelihood;
8. second-order integrated likelihood equal to the second-difference
   likelihood;
9. exact augmented transition, intercept, selection, and design orientation;
10. `d=0` equality with ordinary stationary initialization;
11. rejection of a nonstationary transformed finite-state transition;
12. the complete inherited package test suite.

## Design principles

1. Keep conditional transformed likelihoods and exact diffuse level-state
   likelihoods as separate named APIs.
2. Never reinterpret approximate `diffuse_scale` initialization as exact.
3. Store finite and diffuse covariance components separately.
4. Do not reduce diffuse rank for missing observations.
5. Do not hide deterministic contradictions behind jitter.
6. Represent integration directions explicitly in the state.
7. Require the transformed finite-state subsystem to be stationary.
8. Preserve ordered spatial-matrix orientation.
9. Keep transformed and original observation scales explicit.
10. Preserve positive-sign MA conventions and expanded admissibility rules.
11. Never project seasonal cross operators onto a spatial basis without an
    explicit approximation model.
12. Distinguish state, future innovation, and parameter uncertainty.
13. Keep public numerical arrays immutable.
14. Require analytic or independent large-variance references for exact diffuse
    claims.

## Immediate next tasks

1. Run the validation-record-only merge-gate CI.
2. Update PR #19, mark it ready, and squash-merge it into `main`.
3. Add optimizer-facing exact diffuse ordinary STARIMA maximum likelihood.
4. Add exact diffuse fixed-interval smoothing and disturbance smoothing.
5. Add ordinary-seasonal diffuse state augmentation.
6. Add parameter-aware Kalman paths, sparse state matrices, cross-time
   innovation covariance, simulation smoothing, order selection, exogenous
   inputs, adapters, and cross-language fixtures.

## Known limitations

- `KalmanSTARIMA.fit()` still maximizes a conditional transformed likelihood;
- exact diffuse filtering and fixed-parameter ordinary integrated likelihood are
  separate low-level APIs;
- optimizer-facing exact diffuse MLE is unavailable;
- exact diffuse smoothing is unavailable;
- seasonal integration-state augmentation is unavailable;
- the current state-space observation equation has no separate measurement
  noise covariance;
- sequential location order can change floating-point rounding;
- diffuse rank decisions depend on a numerical tolerance;
- state and covariance matrices are dense;
- Kalman forecast intervals condition on fitted parameters;
- robust, profile-likelihood, sandwich, likelihood-ratio, and Kalman bootstrap
  inference are unavailable;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial stage, read this file, `docs/exact_diffuse.md`,
`docs/state_space.md`, `docs/integrated_maximum_likelihood.md`,
`docs/model.md`, `docs/admissibility.md`, `docs/maximum_likelihood.md`,
`docs/likelihood_inference.md`, `docs/covariance_inference.md`,
`docs/smoothing.md`, `docs/innovation_smoothing.md`,
`docs/kalman_forecast_intervals.md`, `docs/seasonal_maximum_likelihood.md`, and
the latest development handoff. Update repository state, validation, next tasks,
and limitations after every completed stage.
