# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with transparent statistical conventions, typed public APIs, immutable result
objects, independent numerical tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #17 have been squash-merged into `main`.
- `main` is version `0.0.17` at merge commit
  `d62cadf210c223d3da62e87dbc9ea82c2b1a7803`.
- Current branch: `agent/kalman-original-scale-intervals`.
- Current draft pull request: PR #18, `Add original-scale Kalman forecast
  intervals`.
- Current development version: `0.0.18`.
- The branch adds fixed-parameter Gaussian forecast paths from the final
  filtered state posterior and future fitted process innovations.
- Ordinary and seasonal integrated intervals are reconstructed pathwise before
  original-scale quantiles.

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

- added `infer_seasonal_kalman_starima()` and
  `SeasonalKalmanSTARIMA.infer()`;
- reused the central finite-difference curvature engine and immutable
  `LikelihoodInferenceResult` contract;
- reconstructs optional intercept, ordinary/seasonal AR and MA factors,
  innovation covariance, multiplicative expansion, arbitrary-lag state space,
  and Gaussian likelihood at every objective point;
- rejects stencil points that enter enabled expanded AR-stationarity or
  MA-invertibility penalty regions;
- reports factor and optimizer tables, standard errors, normal tests,
  confidence intervals, correlation, score, steps, Hessian eigenvalues, rank,
  condition number, objective, evaluation count, and boundary distances;
- requires positive-definite full-rank observed information by default;
- permits an explicit positive-eigenspace generalized inverse only through
  `allow_singular=True`;
- reuses scalar, diagonal, and full-Cholesky natural innovation-covariance delta
  inference with factor-to-covariance cross uncertainty.

The public natural covariance accessors are `parameter_names`, `estimates`,
`table`, `confidence_intervals()`, covariance/standard-error arrays, and
`dynamic_cross_covariance`. Lower-level element metadata remains on
`natural.transform`.

## Completed in 0.0.18

### Conditional Gaussian forecast paths

For

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_{t+1},
\qquad
\eta_{t+1}\sim\mathcal N(0,Q),
\]

\[
x_{t+1}=Z\alpha_{t+1},
\]

paths begin from

\[
\alpha_T\mid y_{1:T}
\sim
\mathcal N(a_{T\mid T},P_{T\mid T}).
\]

For every simulation:

1. draw the terminal state from the final filtered Gaussian posterior;
2. draw future original location-level process innovations from the fitted
   innovation covariance;
3. propagate the fitted transition and state intercept;
4. apply the design matrix at every horizon;
5. retain the complete transformed path.

The intervals therefore include final filtered-state uncertainty and future
process-innovation uncertainty. Fitted parameters remain fixed.

### Positive-semidefinite covariance handling

- symmetrizes the final filtered covariance;
- uses an eigendecomposition rather than requiring a strictly positive-definite
  Cholesky factor;
- clips only floating-point-scale negative eigenvalues to zero;
- raises for materially negative covariance eigenvalues;
- supports singular Gaussian state posteriors without arbitrary jitter.

### Public API and scale contract

Stationary:

```python
interval = fitted_starma.predict_interval(...)
```

Ordinary integrated:

```python
transformed = fitted_starima.predict_differenced_interval(...)
original = fitted_starima.predict_interval(...)
```

Multiplicative seasonal integrated:

```python
transformed = fitted_seasonal.predict_differenced_interval(...)
original = fitted_seasonal.predict_interval(...)
```

Low-level functions are:

- `simulate_kalman_forecast_paths()`;
- `kalman_forecast_interval()`;
- `integrated_kalman_forecast_interval()`;
- `inverse_forecast_paths()`.

All methods return the existing immutable `ForecastInterval`. The reported mean
is the deterministic recursive point forecast rather than a seed-dependent
Monte Carlo sample mean.

### Pathwise original-scale reconstruction

For ordinary integration,

\[
y_{T+h}=y_T+\sum_{j=1}^{h}x_{T+j}
\]

at order one, with nested recursions for higher orders. Seasonal integration
uses rolling cycle histories,

\[
y_{T+h}=x_{T+h}+y_{T+h-s}.
\]

The implementation inverse-differences every complete simulated path and only
then takes horizon-location quantiles. It never inverse-transforms marginal
transformed lower and upper bounds as though forecast horizons were
independent.

Combined seasonal-ordinary paths reverse seasonal differencing first and
ordinary differencing second, matching the point forecast convention.

### Terminal history and reproducibility policy

- transformed intervals remain available when trailing missing levels prevent
  original-scale reconstruction;
- original intervals require every ordinary anchor and seasonal rolling
  history;
- missing histories raise rather than being imputed or carried forward;
- `random_state` accepts an integer, NumPy `Generator`, or `None`;
- repeated integer seeds reproduce the same paths and bounds;
- the default is 2,000 paths, with larger values recommended for more stable
  tail quantiles.

### Validation coverage

Tests cover:

1. an analytic scalar state model with first two forecast means `1.5` and `2.0`
   and variances `5.0` and `6.0`;
2. inclusion of both terminal filtered covariance and future innovation
   covariance;
3. stationary reproducibility and exact point-forecast centering;
4. exact per-path first-difference reconstruction;
5. ordinary random-walk first-horizon shifts and later widening;
6. seasonal rolling-cycle shifts and widening after one full period;
7. transformed interval availability and original interval refusal when the
   terminal anchor is missing;
8. steps, level, simulation count, covariance, shape, and type validation;
9. the complete inherited package test suite.

## Authoritative validation for 0.0.18

GitHub Actions CI #403, run ID `30878184542`, validated the complete
implementation, tests, example, metadata, README, method guides, navigation,
roadmap, project status, and Step 18 documentation head:

- 155 tests passed in the coverage job;
- total branch coverage was 87.09%, above the required 80%;
- `src/pystarmax/kalman_forecasting.py` coverage was 80.2%;
- `src/pystarmax/integrated_maximum_likelihood.py` coverage was 84.3%;
- `src/pystarmax/seasonal_forecasting.py` coverage was 89.5%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A validation-record-only merge-gate CI is required after this status and the
Step 18 handoff are updated. No implementation, test, API, example, README, or
method-guide changes are made after CI #403.

## Prior validation for 0.0.17

GitHub Actions CI #386, run ID `30876026103`, reported 148 passing tests,
87.14% total branch coverage, 93.5% seasonal inference module coverage, and
successful quality, strict documentation, packaging, and Ubuntu/Windows/macOS
Python 3.11–3.14 checks. Validation-record-only CI #388 also passed before PR
#17 was squash-merged.

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
9. Inverse-difference complete forecast paths before original-scale quantiles.
10. Distinguish filtered-state uncertainty, future innovation uncertainty, and
    parameter uncertainty.
11. Use the positive MA sign consistently in inverse-recursion diagnostics.
12. Treat spectral-radius penalties as feasibility controls, not smooth
    parameterizations.
13. Do not interpret finite-difference penalty curvature as likelihood
    information.
14. Require positive-definite full-rank observed information by default and
    label diagnostic pseudoinverses explicitly.
15. Use stable solves before explicit inverses and expose pseudoinverse use.
16. Distinguish state disturbances from original location innovations.
17. Preserve innovation uncertainty not identified by the state selection map.
18. Require analytic or independent Gaussian references for numerical claims.
19. Keep public numerical arrays immutable.

## Immediate next tasks

1. Run the validation-record-only merge-gate CI.
2. Update PR #18, mark it ready, and squash-merge it into `main`.
3. Design exact diffuse integrated level-state likelihood and smoothing as a
   separate API.
4. Add parameter-aware Kalman paths using constrained observed-information or
   model-refitting bootstrap draws.
5. Add sparse arbitrary-lag state matrices, cross-time innovation covariance,
   simulation smoothing, order selection, exogenous inputs, adapters, and
   cross-language fixtures.

## Known limitations

- ordinary and seasonal Kalman STARIMA likelihoods are conditional on
  transformation history, not exact diffuse on the level process;
- Kalman forecast intervals condition on fitted parameters;
- parameter covariance is not propagated into forecast paths;
- forecast bounds are Monte Carlo quantiles and have finite-simulation error;
- large dense states and path arrays can be computationally and memory
  intensive;
- central finite-difference inference requires `1 + 2*k**2` objective
  evaluations for `k` optimizer coordinates;
- inference can be sensitive to step size in flat, highly curved, or
  near-boundary likelihood directions;
- feasibility uses penalties rather than a smooth constrained
  parameterization;
- robust, sandwich, profile-likelihood, likelihood-ratio, and Kalman bootstrap
  inference are unavailable;
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
`docs/kalman_forecast_intervals.md`, `docs/admissibility.md`,
`docs/state_space.md`, `docs/smoothing.md`, `docs/innovation_smoothing.md`,
`docs/maximum_likelihood.md`, `docs/likelihood_inference.md`,
`docs/covariance_inference.md`, and the latest development handoff. Update
repository state, validation, next tasks, and limitations after every completed
stage.
