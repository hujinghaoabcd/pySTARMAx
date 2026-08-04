# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models
with explicit likelihood conventions, typed public APIs, immutable numerical
results, independent reference tests, strict documentation, and reproducible
cross-platform CI.

## Repository state

- PR #1 through PR #21 have been squash-merged into `main`.
- `main` is version `0.0.21` at merge commit
  `b6e9b167a012191590aae2abef0218f25482d00b`.
- Current branch: `agent/exact-diffuse-disturbance-smoothing`.
- Current draft pull request: PR #22, `Add exact diffuse disturbance smoothing`.
- Current development version: `0.0.22`.
- The branch adds exact diffuse primitive innovation and state-equation
  disturbance marginal posterior moments from backward information quantities.
- Lag-one state autocovariance, cross-time disturbance covariance, and
  simulation smoothing remain explicitly outside this stage.

## Completed baseline through 0.0.18

### Model families

- immutable spatial-weight collections with explicit identity lag and preserved
  non-symmetric orientation;
- conditional STAR, STARMA, ordinary STARIMA, and multiplicative seasonal
  STARIMA;
- stationary Gaussian `KalmanSTARMA` with scalar, diagonal, and full Cholesky
  innovation covariance;
- conditional ordinary-integrated `KalmanSTARIMA(p,d,q)`;
- multiplicative seasonal
  `SeasonalKalmanSTARIMA(p,d,q)x(P,D,Q)_s`;
- ordered seasonal matrix products without projecting deterministic cross terms
  back onto the spatial-weight basis;
- reversible ordinary and seasonal differencing with explicit transformed and
  original scale contracts.

### Diagnostics, admissibility, and inference

- space-time covariance, STACF, nested Yule--Walker and regression STPACF, and
  residual portmanteau diagnostics;
- AR stationarity and positive-sign inverse-MA companion diagnostics;
- start shrinkage, optimizer penalties, final feasibility checks, eigenvalues,
  spectral radii, limits, and signed boundary distances;
- finite-difference observed-information inference for stationary and seasonal
  factor models;
- coefficient covariance, standard errors, normal tests, confidence intervals,
  correlations, rank, eigenvalue, condition-number, and score diagnostics;
- strict full-rank positive-definite observed-information policy with an
  explicitly labelled diagnostic positive-eigenspace generalized inverse;
- scalar, diagonal, and full-Cholesky natural innovation covariance
  delta-method inference with dynamic/covariance cross uncertainty.

### Filtering, smoothing, forecasting, and missing data

- stationary and arbitrary-lag STARMA state-space construction;
- stationary, known, and approximate large-variance diffuse initialization;
- partial-location updates and prediction-only fully missing rows;
- immutable filtering states, covariance, innovations, masks, likelihood
  contributions, and numerical stabilization diagnostics;
- Rauch--Tung--Striebel state smoothing, lag-one covariance, smoothing gains,
  and state-disturbance moments;
- original location-level innovation disturbance smoothing with retained
  selection-nullspace uncertainty;
- deterministic recursive forecasts and aligned original-scale fitted values;
- conditional future-innovation and parameter-refitting bootstrap intervals;
- rolling-origin calibration, sharpness, interval score, MAE, and RMSE;
- fixed-parameter Gaussian Kalman forecast paths initialized from the terminal
  filtered posterior;
- future process-innovation uncertainty and pathwise ordinary/seasonal inverse
  differencing before original-scale quantiles;
- explicit refusal of original-scale forecasts when required terminal anchors
  or seasonal histories are unavailable.

## Completed in 0.0.19

### Exact diffuse filtering

The initial covariance is represented as

\[
P_1(\kappa)=P_{\ast,1}+\kappa P_{\infty,1},
\qquad \kappa\rightarrow\infty.
\]

- finite and diffuse covariance components are stored separately;
- scalar observed locations are processed sequentially;
- exact diffuse `K0` and `K1` recursions are used while diffuse variance is
  positive;
- ordinary Gaussian updates begin after the corresponding diffuse direction is
  resolved;
- missing observations do not reduce diffuse rank;
- fully missing rows perform prediction only;
- deterministic zero-variance contradictions raise instead of receiving
  arbitrary jitter;
- the result records finite/diffuse covariance paths, innovations, finite and
  diffuse innovation variances, masks, likelihood contributions, rank paths,
  diffuse observation count, and diffuse end time.

### Ordinary integrated state construction

For stationary transformed state

\[
\beta_t=c+T\beta_{t-1}+R\eta_t,
\qquad x_t=Z\beta_t,
\]

with `x_t = Delta^d y_t`, the augmented state is

\[
[y_t,\Delta y_t,\ldots,\Delta^{d-1}y_t,\beta_t].
\]

- integration directions receive exact diffuse covariance;
- the transformed STARMA state receives its stationary finite mean and
  covariance;
- ordered `Z @ T`, `Z @ c`, and `Z @ R` effects enter every integration block;
- first- and second-order integrated likelihoods have analytic random-walk
  references;
- `d=0` is equivalent to ordinary stationary initialization;
- a nonstationary transformed subsystem is rejected rather than hidden in the
  finite initial covariance.

Public APIs include `exact_diffuse_filter()`,
`exact_diffuse_loglikelihood()`, `build_exact_integrated_state_space()`,
`exact_integrated_filter()`, and `exact_integrated_loglikelihood()`.

## Completed in 0.0.20

### Optimizer-facing exact diffuse ordinary STARIMA

`ExactDiffuseKalmanSTARIMA(p,d,q)` maximizes the original-level exact diffuse
Gaussian likelihood. For every optimizer coordinate vector it:

1. decodes the optional intercept, transformed AR factors, transformed MA
   factors, and scalar/diagonal/full-Cholesky covariance coordinates;
2. constructs the stationary transformed STARMA state space;
3. computes transformed AR and positive-sign inverse-MA spectral radii;
4. applies enabled stationarity and invertibility feasibility penalties;
5. constructs the ordinary integrated original-level state space;
6. initializes only integration directions diffusely;
7. initializes the transformed subsystem from its stationary finite
   distribution;
8. exact-filters the original observations;
9. returns the negative exact diffuse log likelihood.

The optimizer never substitutes the conditional differenced recursion while
reporting an exact diffuse result.

### Likelihood separation

The existing conditional estimator reports

\[
L_c
=
L\left(\Delta^d y_{d+1:T}\mid y_{1:d}\right).
\]

The exact estimator reports

\[
L_D=L_D(y_{1:T}).
\]

These are separate named estimators with separate likelihood, AIC, and BIC
contracts. Integration order changes latent-state dimension but adds no free
coefficient. For `K` spatial weights,

\[
k=\mathbf 1_c+K(p+q)+k_Q.
\]

Information criteria use finite original-level cells:

\[
AIC=-2\ell_D+2k,
\qquad
BIC=-2\ell_D+k\log n_{obs}.
\]

### Starts and admissibility

- dynamic and covariance starts reuse the stationary Kalman STARMA machinery on
  differenced observations;
- user-supplied starts remain available;
- scalar, diagonal, and full-Cholesky covariance codecs are shared;
- L-BFGS-B and covariance bounds are reused;
- stationarity and invertibility apply only to the transformed STARMA subsystem;
- intentional integration unit roots are represented explicitly and are not
  subjected to the transformed stationarity test;
- final optimizer candidates receive hard transformed admissibility checks.

### Result and fitted API

`ExactDiffuseKalmanSTARIMAResult` stores immutable:

- dynamic and raw optimizer coordinates and names;
- intercept, transformed AR factors, transformed MA factors, and natural
  innovation covariance;
- exact diffuse likelihood, AIC, and BIC;
- optimizer convergence, iterations, function evaluations, method, and message;
- finite and diffuse observation counts;
- transformed AR and inverse-MA radii, limits, and enforcement flags;
- complete `ExactDiffuseFilterResult`;
- stationary transformed `StateSpaceModel`;
- exact integrated state-space specification.

Fitted methods include:

- `filter(data=None)` for retained training or newly initialized exact diffuse
  filtering;
- `to_state_space()` for the augmented original-level state;
- `to_transformed_state_space()` for the stationary highest-difference state;
- `admissibility()` for transformed dynamics;
- `predict()` for original-level recursive means;
- `predict_differenced()` for highest-difference recursive means.

### Validation references

Tests cover:

1. random-walk drift and variance against closed-form increment MLEs;
2. second-order integrated drift and variance against closed-form
   second-difference MLEs;
3. `d=0` optimizer coordinates and likelihood against stationary
   `KalmanSTARMA` under identical starts;
4. missing initial levels and delayed diffuse completion;
5. retained training filtering and new-data exact filtering;
6. transformed and original state-space identity contracts;
7. original-level and highest-difference forecast recursions;
8. transformed admissibility diagnostics;
9. immutable result arrays and explicit summary labels;
10. constructor, fitted-state, data-length, and start validation;
11. the complete inherited package test suite.

## Completed in 0.0.21

### Exact diffuse backward information smoothing

- retains ordinary estimator `r` and diffuse estimator `r_inf`;
- retains ordinary covariance `N`, cross covariance `N1`, and second diffuse
  covariance `N2`;
- reconstructs every sequential forward scalar covariance update;
- reports maximum discrepancy against retained filter covariance;
- computes smoothed means as `a + P_* r + P_inf r_inf`;
- computes finite posterior covariance with ordinary, diffuse, and both cross
  correction terms;
- returns smoothed observation means and covariance;
- propagates future information through fully missing and partially observed
  rows;
- stabilizes only floating-point-scale negative covariance eigenvalues and
  reports the largest correction;
- exposes immutable backward information arrays and diagnostics;
- integrates as `ExactDiffuseKalmanSTARIMA.smooth()` for training or newly
  initialized data.

### Validation scope

Tests cover closed-form random-walk bridges, leading missing levels,
zero-diffuse equality with stationary RTS smoothing, a local-linear-trend
large-variance limit, partial locations, observed-cell reconstruction, PSD
covariance, immutable arrays, forward covariance reconstruction, and the
fitted-model smoothing facade.

### Deliberate boundary

Version 0.0.21 does not fabricate lag-one exact diffuse state covariance. The
diffuse autocovariance recursion needs an additional higher-order transition
term beyond retained `L0` and `L1`. State-disturbance, original innovation, and
simulation smoothing remain future stages.

## Completed in 0.0.22

### Exact diffuse primitive innovation and state disturbance moments

- computes `E(eta_(t+1) | Y) = Q R.T r_t`;
- computes `Var(eta_(t+1) | Y) = Q - Q R.T N_t R Q`;
- maps primitive moments to state-equation disturbances through `R`;
- uses exact diffuse backward information directly rather than an ordinary RTS
  lag-one formula;
- preserves unresolved prior variance in selection-nullspace directions;
- returns immutable means, marginal covariance, tolerances, and maximum
  covariance corrections;
- integrates as
  `ExactDiffuseKalmanSTARIMA.smooth_innovation_disturbances()` for retained
  training data or a newly initialized data segment.

### Validation scope

Tests cover fully observed random-walk increments, a closed-form missing bridge,
leading missing diffuse levels, zero-diffuse equality with the ordinary RTS
innovation smoother, valid rank-deficient selection, PSD covariance, immutable
arrays, validation errors, and the fitted-model facade.

### Deliberate boundary

Exact diffuse lag-one state autocovariance remains unavailable because the
diffuse autocovariance recursion requires a nontrivial higher-order `L2` term.
Cross-time disturbance covariance and simulation smoothing are not claimed.

## Core validation for 0.0.22

GitHub Actions CI #472, run ID `30891197678`, validated the numerical core,
public exports, model facade, packaging, and full cross-platform matrix:

- 193 tests passed in the coverage job;
- total branch coverage was 87.19%, above the required 80%;
- `src/pystarmax/exact_diffuse_disturbance_smoothing.py` coverage was 88.1%;
- `src/pystarmax/exact_diffuse_smoothing.py` coverage was 91.6%;
- `src/pystarmax/exact_diffuse_mle.py` coverage was 84.3%;
- `src/pystarmax/exact_diffuse.py` coverage was 87.8%;
- `src/pystarmax/exact_integrated.py` coverage was 86.9%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A complete final CI is required on the synchronized documentation head. The
authoritative final run must be recorded before merge.

## Authoritative validation for 0.0.21

GitHub Actions CI #463, run ID `30887094839`, validated the complete
implementation, tests, example, metadata, README, documentation home,
navigation, exact diffuse filtering/MLE/smoothing guides, roadmap, project
status, and Step 21 documentation head:

- 184 tests passed in the coverage job;
- total branch coverage was 87.16%, above the required 80%;
- `src/pystarmax/exact_diffuse_smoothing.py` coverage was 91.6%;
- `src/pystarmax/exact_diffuse_mle.py` coverage was 84.3%;
- `src/pystarmax/exact_diffuse.py` coverage was 87.8%;
- `src/pystarmax/exact_integrated.py` coverage was 86.9%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A validation-record-only merge-gate CI is required after this status and the
Step 21 handoff are updated. No implementation, test, API, example, README, or
method-guide changes are made after CI #463.

## Authoritative validation for 0.0.20

GitHub Actions CI #440, run ID `30884139939`, validated the complete
implementation, tests, example, metadata, README, documentation home,
navigation, exact-diffuse guides, conditional-integrated guide, roadmap,
project status, and Step 20 documentation head:

- 176 tests passed in the coverage job;
- total branch coverage was 86.99%, above the required 80%;
- `src/pystarmax/exact_diffuse_mle.py` coverage was 84.3%;
- `src/pystarmax/exact_diffuse.py` coverage was 87.8%;
- `src/pystarmax/exact_integrated.py` coverage was 86.9%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A validation-record-only merge-gate CI is required after this status and the
Step 20 handoff are updated. No implementation, test, API, example, README, or
method-guide changes are made after CI #440.

## Design principles

1. Keep conditional transformed and exact diffuse original-level likelihoods as
   separate named APIs.
2. Never reinterpret approximate `diffuse_scale` initialization as exact.
3. Rebuild the complete objective recursion at every optimizer candidate.
4. Represent integration directions explicitly in the state.
5. Require the transformed finite-state subsystem to be stationary.
6. Apply the package positive MA sign consistently.
7. Do not reduce diffuse rank for missing observations.
8. Do not hide deterministic contradictions behind jitter.
9. Preserve ordered spatial-matrix orientation.
10. Count optimized factors rather than deterministic state augmentation terms.
11. Distinguish state, future innovation, and parameter uncertainty.
12. Keep public numerical arrays immutable.
13. Require analytic or independent equivalence references for likelihood and
    optimizer claims.

## Immediate next tasks

1. Run complete CI on the synchronized 0.0.22 documentation head.
2. Record the authoritative final run identifier, test count, and coverage.
3. Update PR #22, mark it ready, and squash-merge it into `main`.
4. Add exact diffuse observed-information and natural covariance inference.
5. Add exact diffuse simulation smoothing and seasonal diffuse augmentation.
6. Research the nontrivial diffuse `L2` autocovariance recursion separately.
7. Add forecast intervals, sparse state matrices, parameter-aware paths,
   cross-time innovation covariance, order selection, exogenous inputs, GIS
   adapters, and cross-language fixtures.

## Known limitations

- exact diffuse observed-information and natural covariance inference are not
  implemented;
- exact diffuse marginal state and primitive innovation/state-disturbance
  smoothing are implemented, but lag-one state autocovariance, cross-time
  disturbance covariance, and simulation smoothing are not;
- seasonal ordinary-seasonal diffuse augmentation and smoothing are not
  implemented;
- forecast intervals are not exposed from `ExactDiffuseKalmanSTARIMA`;
- filtering a new segment starts a new diffuse initialization and does not
  continue the training terminal posterior;
- the observation equation has no separate measurement-noise covariance;
- exact filtering processes locations sequentially, so floating-point rounding
  can depend on location order;
- diffuse rank decisions use a numerical tolerance;
- state and covariance matrices are dense;
- differenced start construction can be weak for highly incomplete data;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial stage, read this file,
`docs/exact_diffuse_disturbance_smoothing.md`,
`docs/exact_diffuse_smoothing.md`, `docs/exact_diffuse_mle.md`,
`docs/exact_diffuse.md`,
`docs/integrated_maximum_likelihood.md`, `docs/state_space.md`,
`docs/admissibility.md`, `docs/maximum_likelihood.md`,
`docs/likelihood_inference.md`, `docs/covariance_inference.md`,
`docs/smoothing.md`, `docs/innovation_smoothing.md`,
`docs/kalman_forecast_intervals.md`, `docs/seasonal_maximum_likelihood.md`, and
the latest development handoff. Update repository state, validation, next tasks,
and limitations after every completed stage.
