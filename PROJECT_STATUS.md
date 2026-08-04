# pySTARMAx project status

Updated: 2026-08-05

## Purpose

Build a modern, typed, research-oriented Python implementation of classical and
extended STARMA models with:

- explicit and non-interchangeable likelihood conventions;
- immutable numerical result objects;
- transparent state-space construction;
- missing-data support without silent imputation;
- independent analytic or equivalence references;
- strict documentation and reproducible cross-platform CI.

## Repository state

- PR #1 through PR #24 have been squash-merged into `main`.
- `main` is version `0.0.24` at merge commit
  `0639f4bd932aa86020215b50b6ef9867bb5ad566`.
- Current branch: `agent/exact-diffuse-simulation-smoothing`.
- Current pull request: PR #25, `Add exact diffuse simulation smoothing`.
- Current development version: `0.0.25`.
- PR #25 is a release candidate after authoritative implementation CI #536.
- This status update is validation-record-only. No implementation, test, public
  API, example, README, or method-guide change follows CI #536.
- No temporary workflow or generated artifact is intended to remain in the PR.

## Public model families

### Conditional models

- `STAR`;
- `STARMA(p,q)`;
- ordinary `STARIMA(p,d,q)`;
- multiplicative seasonal
  `SeasonalSTARIMA(p,d,q)x(P,D,Q)_s`.

These routes use conditional least-squares or conditional transformed
likelihood conventions and retain explicit differencing histories.

### Gaussian Kalman models

- stationary `KalmanSTARMA(p,q)`;
- conditional ordinary-integrated `KalmanSTARIMA(p,d,q)`;
- conditional multiplicative seasonal
  `SeasonalKalmanSTARIMA(p,d,q)x(P,D,Q)_s`;
- original-level exact diffuse `ExactDiffuseKalmanSTARIMA(p,d,q)`.

The conditional transformed likelihoods and original-level exact diffuse
likelihood are separate APIs. Their log likelihoods, AIC, and BIC are not
interchangeable.

## Completed milestones

### 0.0.1–0.0.7: conditional modelling and evaluation

- immutable ordered spatial-weight collections with identity `W0`;
- preserved non-symmetric matrix orientation;
- conditional STAR and iterative STARMA estimation;
- ordinary and seasonal differencing with reversible state;
- conditional ordinary and multiplicative seasonal STARIMA;
- ordered seasonal matrix-polynomial expansion without cross-term basis
  projection;
- simulation for stationary, integrated, and seasonal models;
- STCOV, STACF, nested Yule–Walker STPACF, regression STPACF, and residual
  portmanteau diagnostics;
- conditional future-innovation intervals;
- residual and parametric bootstrap intervals with refitting;
- rolling-origin calibration, width, Winkler score, MAE, and RMSE summaries.

### 0.0.8–0.0.9: state space and stationary Kalman MLE

- explicit STARMA companion state space;
- stationary, known, and approximate large-variance diffuse initialization;
- partial-location filtering and prediction-only fully missing rows;
- Gaussian log likelihood with scalar, diagonal, and full-Cholesky innovation
  covariance;
- independent `KalmanSTARMA` optimizer;
- immutable states, covariance, innovations, masks, likelihood contributions,
  optimizer metadata, AIC, and BIC;
- fitted filtering, state-space access, and deterministic recursive prediction.

### 0.0.10–0.0.12: admissibility and observed-information inference

- AR stationarity and positive-sign inverse-MA companion diagnostics;
- complex eigenvalues, spectral radii, feasibility limits, and signed boundary
  distances;
- start shrinkage, optimizer penalties, and hard final admissibility checks;
- central finite-difference score and observed-information Hessian;
- parameter-scaled diagonal and mixed-partial stencils;
- covariance, standard errors, Wald summaries, confidence intervals,
  correlations, rank, eigenvalues, condition number, and score diagnostics;
- strict rejection of indefinite or rank-deficient Hessians by default;
- explicitly labelled positive-eigenspace generalized inverse;
- analytic scalar, diagonal, and full-Cholesky natural innovation covariance
  delta-method inference;
- dynamic-parameter/covariance-element cross covariance.

### 0.0.13–0.0.14: ordinary smoothing and original innovations

- Rauch–Tung–Striebel fixed-interval state smoothing;
- state and observation posterior means and covariance;
- smoothing gains and lag-one state covariance;
- state-equation disturbance posterior moments;
- prediction-rank and pseudoinverse diagnostics;
- exact conditional-Gaussian mapping from `w_t = R eta_t` back to primitive
  location innovations;
- retained selection-nullspace uncertainty;
- training and newly filtered data facades.

### 0.0.15: conditional ordinary-integrated Kalman STARIMA

- public `KalmanSTARIMA(p,d,q)`;
- conditional likelihood on `Delta^d y` after finite history removal;
- arbitrary non-negative ordinary integration order;
- transformed and original-scale filtering, smoothing, inference, forecasting,
  and innovation smoothing contracts;
- pathwise recursive inverse differencing;
- refusal of original-level forecasts without finite terminal anchors;
- missing-value propagation through the complete differencing stencil;
- exact `d=0` equivalence and random-walk reconstruction tests.

### 0.0.16–0.0.17: multiplicative seasonal Kalman STARIMA

- Gaussian conditional seasonal likelihood;
- ordinary and seasonal factor parameters rather than independent expanded
  cross-lag parameters;
- ordered products `-S_r @ A_i` for AR cross terms and `+N_u @ M_j` for the
  package positive-sign MA convention;
- arbitrary-lag dense companion state construction;
- expanded stationarity and inverse-MA checks;
- factor-based information-criterion parameter counting;
- combined ordinary-seasonal missingness propagation;
- transformed filtering, RTS smoothing, innovation smoothing, point forecasts,
  and pathwise original-scale reconstruction;
- observed-information inference for ordinary and seasonal factors;
- natural covariance delta-method inference with factor/covariance cross
  uncertainty.

### 0.0.18: Gaussian Kalman forecast intervals

- draws from the final filtered Gaussian state posterior;
- future primitive innovation simulation;
- PSD covariance factorization without arbitrary jitter;
- stationary, ordinary-integrated, and seasonal transformed/original interval
  facades;
- pathwise inverse differencing before original-scale quantiles;
- deterministic recursive point forecasts retained as interval means;
- reproducible integer-seed and NumPy-generator behavior;
- explicit fixed-parameter uncertainty contract.

### 0.0.19: exact diffuse filtering and integrated level states

The exact filter represents initial covariance as

\[
P_1(\kappa)=P_{\ast,1}+\kappa P_{\infty,1},
\qquad \kappa\rightarrow\infty.
\]

Implemented capabilities:

- separate finite and diffuse covariance paths;
- exact sequential scalar `K0` and `K1` updates;
- ordinary Gaussian updates after diffuse directions are resolved;
- exact diffuse likelihood contributions and observation counts;
- missing cells that do not spuriously reduce diffuse rank;
- deterministic zero-variance agreement and contradiction handling;
- diffuse rank path and completion time;
- ordinary integrated augmentation
  `[y, Delta y, ..., Delta^(d-1)y, beta]`;
- stationary finite initialization for the transformed STARMA substate;
- diffuse initialization only for integration directions;
- first- and second-order analytic integrated likelihood references;
- exact `d=0` equivalence with stationary initialization.

Approximate `initialization="diffuse"` remains a separate large-finite-variance
API and is never renamed as exact.

### 0.0.20: exact diffuse ordinary STARIMA MLE

- optimizer-facing `ExactDiffuseKalmanSTARIMA(p,d,q)`;
- original-level exact diffuse likelihood at every optimizer candidate;
- reconstruction of transformed and integrated state spaces at every candidate;
- shared scalar, diagonal, and full-Cholesky covariance codecs;
- transformed-subsystem stationarity and positive-sign MA invertibility;
- immutable dynamic, optimizer, covariance, likelihood, information-criterion,
  admissibility, filter, and state-space metadata;
- retained training filtering and newly initialized data filtering;
- transformed and original state-space access;
- highest-difference and original-level deterministic forecasts;
- random-walk, second-order integration, and stationary `d=0` references.

### 0.0.21: exact diffuse fixed-interval state smoothing

- backward ordinary and diffuse information recursions for `r`, `r_inf`, `N`,
  `N1`, and `N2`;
- exact reconstruction of sequential forward covariance updates;
- state means `a + P_* r + P_inf r_inf`;
- finite posterior state covariance with ordinary, diffuse, and both cross
  correction terms;
- observation-scale posterior means and covariance;
- missing-row, leading-missing, and partial-location smoothing;
- PSD stabilization diagnostics;
- stationary zero-diffuse equivalence with ordinary RTS;
- random-walk bridge and large-variance-limit references;
- fitted `ExactDiffuseKalmanSTARIMA.smooth()`.

The exact diffuse lag-one autocovariance recursion is deliberately not
fabricated.

### 0.0.22: exact diffuse primitive disturbance smoothing

For

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_{t+1},
\qquad \eta_{t+1}\sim\mathcal N(0,Q),
\]

implemented posterior moments are

\[
E(\eta_{t+1}\mid Y)=QR^\top r_t,
\]

\[
\operatorname{Var}(\eta_{t+1}\mid Y)
=Q-QR^\top N_tRQ.
\]

Additional capabilities:

- exact state-disturbance moments through `R`;
- preserved prior uncertainty in selection-nullspace directions;
- immutable means, marginal covariance, tolerance, and correction diagnostics;
- training and newly initialized data facade;
- zero-diffuse equivalence with the ordinary innovation smoother.

This stage does not claim cross-time disturbance covariance.

### 0.0.23: exact diffuse observed-information inference

- central finite-difference score and Hessian of the original-level exact
  diffuse objective;
- complete optimizer-coordinate, transformed-state, integrated-state, and
  exact-filter reconstruction at every stencil point;
- fitted admissibility boundaries;
- optimizer-coordinate covariance, standard errors, tests, intervals,
  correlation, rank, eigenvalues, condition number, and score;
- strict full-rank positive-definite policy and explicit diagnostic generalized
  inverse;
- natural innovation covariance delta-method inference;
- random-walk closed-form Hessian and stationary `d=0` equivalence references;
- `ExactDiffuseKalmanSTARIMA.likelihood_inference()` and
  `infer_exact_diffuse_kalman_starima()`.

### 0.0.24: exact diffuse forecast intervals

- draws from the final finite exact diffuse filtered posterior;
- explicit refusal when final diffuse rank is nonzero;
- future primitive innovations propagated through the augmented integrated
  state;
- original-level `predict_interval()`;
- highest ordinary-difference `predict_differenced_interval()` through explicit
  transformed-state projection;
- deterministic recursive means retained as interval means;
- PSD terminal covariance handling without a finite diffuse scale;
- analytic state-variance, random-walk growth, stationary `d=0` path equality,
  second-order integration, reproducibility, and unresolved-rank tests.

CI #518 and merge-gate CI #519 passed 207 tests with 87.30% total branch
coverage.

### 0.0.25: exact diffuse conditional simulation smoothing

PR #25 adds a dense exact conditional simulator for complete state and
observation paths.

Algorithmic contract:

1. represent the first predicted state with flat diffuse coordinates and finite
   standard-normal source coordinates;
2. append later primitive process innovations as proper Gaussian source
   coordinates;
3. reconstruct every observed location-time value from exact filter innovations
   and sequential update conventions;
4. analytically eliminate identified diffuse coordinates;
5. use the left-null constraint system to condition the remaining proper source
   vector through SVD;
6. sample the posterior source nullspace;
7. map draws back to complete state and observation paths;
8. independently reconstruct posterior marginal means and covariance;
9. compare those marginals with the exact information smoother and raise on
   material disagreement.

Public API:

```python
paths = fitted_exact_model.simulate_smoothing_paths(
    n_simulations=2000,
    random_state=2026,
)
```

Low-level API:

```python
paths = exact_diffuse_simulation_smoother(
    exact_filter_result,
    n_simulations=2000,
    random_state=2026,
)
```

`ExactDiffuseSimulationSmootherResult` exposes immutable:

- `state_paths`;
- `observation_paths`;
- independently reconstructed `posterior_state_mean`;
- independently reconstructed marginal `posterior_state_covariance`;
- identified diffuse rank;
- conditioning rank;
- proper and posterior source dimensions;
- maximum observed-constraint residual;
- maximum mean discrepancy against the information smoother;
- maximum covariance discrepancy against the information smoother;
- `rcond` and numerical tolerance;
- the source exact filter and information smoother results.

Validation covers:

- a closed-form random-walk bridge;
- fully observed deterministic paths;
- stationary zero-diffuse incomplete data;
- exact marginal equivalence with the information smoother to floating-point
  precision;
- partial-location observations;
- retained training and newly initialized data;
- deterministic seeded reproducibility;
- unresolved diffuse-rank refusal;
- immutable arrays and argument validation.

Deliberate boundary:

- dense source-coordinate matrices grow with sample length, state dimension,
  initial finite rank, and innovation rank;
- this is a moderate-sample reference implementation, not a sparse large-system
  performance claim;
- it returns state and observation paths, not primitive innovation paths;
- it does not expose diffuse lag-one autocovariance or cross-time disturbance
  covariance;
- it conditions on fitted parameters;
- seasonal exact diffuse augmentation remains separate future work.

## Authoritative validation for 0.0.25

GitHub Actions CI #536, run ID `30944093465`, validated the synchronized
implementation, tests, exports, fitted facade, example, metadata, README,
documentation home, navigation, exact diffuse guides, roadmap, project status,
Step 25 handoff, and remaining-work inventory on head
`ca58bdee2e1c391e2915b4789e7421d536c4b7dc`.

Results:

- 214 tests passed in the coverage job;
- total branch coverage was 87.25%, above the required 80%;
- `src/pystarmax/exact_diffuse_simulation_smoothing.py` coverage was 86.2%;
- `src/pystarmax/exact_diffuse_smoothing.py` coverage was 91.6%;
- `src/pystarmax/exact_diffuse_disturbance_smoothing.py` coverage was 88.1%;
- `src/pystarmax/exact_diffuse_forecasting.py` coverage was 87.8%;
- `src/pystarmax/exact_diffuse_inference.py` coverage was 88.8%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

This status record is the only repository-content change after CI #536. A
validation-record-only merge-gate CI must pass before PR #25 is marked ready and
squash-merged.

## Numerical and research safeguards

1. Distinguish conditional transformed likelihoods from exact diffuse
   original-level likelihoods.
2. Never reinterpret a finite `diffuse_scale` as exact initialization.
3. Preserve non-symmetric spatial-matrix orientation.
4. Preserve the package positive MA sign.
5. Do not reduce diffuse rank for missing observations.
6. Do not hide deterministic contradictions behind jitter.
7. Count optimized factor parameters, not deterministic expanded cross terms or
   state augmentation coordinates.
8. Never project multiplicative cross-lag matrices back onto a supplied spatial
   basis without an explicit approximation model.
9. Rebuild complete state and likelihood recursions at optimizer and
   finite-difference candidates.
10. Reject infeasible curvature stencil points rather than treating penalty
    surfaces as likelihood curvature.
11. Distinguish state uncertainty, future innovation uncertainty, and parameter
    uncertainty.
12. Inverse-difference complete simulated paths before original-scale
    quantiles.
13. Distinguish state-equation disturbances from primitive innovations.
14. Preserve unresolved selection-nullspace innovation uncertainty.
15. Do not claim diffuse lag-one covariance without the full higher-order
    recursion.
16. Require every diffuse direction to be identified before exact forecast or
    conditional simulation draws.
17. Treat the dense simulation smoother as a reference implementation, not a
    scalability claim.
18. Keep public numerical arrays immutable.
19. Require analytic or independent-equivalence references for likelihood,
    inference, smoothing, and forecasting claims.
20. Keep limitations visible in public guides and result contracts.

## Known limitations

- exact diffuse lag-one state autocovariance is unavailable;
- cross-time state-disturbance and primitive-innovation covariance is
  unavailable;
- exact diffuse simulation smoothing does not yet return primitive innovation
  or state-disturbance paths;
- seasonal ordinary-seasonal exact diffuse augmentation, likelihood, smoothing,
  inference, forecasting, and simulation are unavailable;
- robust covariance, profile likelihood, analytic exact diffuse derivatives,
  and parameter-uncertainty propagation are unavailable;
- Gaussian forecast intervals and simulation smoothing condition on fitted
  parameters;
- the observation equation has no separate measurement-noise covariance;
- filtering a new segment starts a new exact diffuse initialization rather than
  continuing an external terminal posterior;
- exact filtering processes observed locations sequentially, so location order
  can affect floating-point rounding;
- diffuse rank, PSD, and SVD decisions use numerical tolerances;
- state, covariance, arbitrary-lag companion, and simulation-source matrices are
  dense;
- the exact simulation smoother is intended for moderate samples and can grow
  rapidly in memory and factorization cost;
- differenced optimizer starts can be weak for highly incomplete series;
- smooth admissibility parameterizations and automatic order selection are
  unavailable;
- exogenous regressors, intervention variables, GIS adapters, and time-varying
  dynamics are unavailable;
- bootstrap and rolling refits remain serial;
- PyPI release automation and cross-language numerical fixtures remain future
  work.

## Immediate next tasks

1. Run the validation-record-only merge-gate CI for PR #25.
2. Confirm no temporary workflow, generated artifact, unresolved review thread,
   or unaddressed comment remains.
3. Mark PR #25 ready and squash-merge it into `main`.
4. Begin seasonal exact diffuse state augmentation and original-level
   likelihood as the next independent implementation stage.
5. Add seasonal exact diffuse smoothing, inference, forecasting, and simulation
   only after the foundation is independently validated.
6. Research the nontrivial diffuse `L2` recursion for lag-one state
   autocovariance and cross-time disturbance covariance separately.
7. Add robust and parameter-aware uncertainty paths.
8. Add sparse and memory-aware numerical execution.
9. Continue model/ecosystem work listed in
   `docs/development/REMAINING_WORK.md`.

## Handoff instruction

Before the next substantial stage, read:

- this file;
- `docs/development/REMAINING_WORK.md`;
- `docs/development/STEP_25_EXACT_DIFFUSE_SIMULATION_SMOOTHING.md`;
- `docs/exact_diffuse_simulation_smoothing.md`;
- `docs/exact_diffuse_forecast_intervals.md`;
- `docs/exact_diffuse_inference.md`;
- `docs/exact_diffuse_disturbance_smoothing.md`;
- `docs/exact_diffuse_smoothing.md`;
- `docs/exact_diffuse_mle.md`;
- `docs/exact_diffuse.md`;
- `docs/integrated_maximum_likelihood.md`;
- `docs/seasonal_maximum_likelihood.md`;
- `docs/state_space.md`;
- `docs/admissibility.md`;
- `docs/maximum_likelihood.md`;
- `docs/likelihood_inference.md`;
- `docs/covariance_inference.md`;
- `docs/smoothing.md`;
- `docs/innovation_smoothing.md`;
- `docs/kalman_forecast_intervals.md`.

After every completed stage, update repository state, authoritative validation,
next tasks, known limitations, roadmap, method documentation, and the latest
handoff.
