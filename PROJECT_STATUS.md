# pySTARMAx project status

## Repository position

- `main` is version `0.0.31` at squash merge commit
  `6d4f0eda336f121315769a6910606381676cfe56`.
- Active branch: `agent/exact-diffuse-lag-one-covariance`.
- Active pull request: PR #32,
  `Add exact diffuse lag-one covariance reference`.
- Development version: `0.0.32`.
- PR #1 through PR #31 have been squash-merged.

## Public model families

The package exposes:

- conditional least-squares `STARMA`, `STARIMA`, and `SeasonalSTARIMA`;
- stationary Gaussian `KalmanSTARMA`;
- conditional ordinary-integrated `KalmanSTARIMA`;
- conditional multiplicative seasonal `SeasonalKalmanSTARIMA`;
- original-level ordinary exact-diffuse `ExactDiffuseKalmanSTARIMA`;
- seasonal exact-diffuse state construction and optimizer-facing MLE through
  `SeasonalExactDiffuseKalmanSTARIMA`;
- ordinary and seasonal exact-diffuse smoothing, disturbance smoothing,
  adjacent-time covariance, inference, forecasting, and conditional path
  simulation.

Conditional transformed-data likelihoods and original-level exact-diffuse
likelihoods are separate contracts. Their likelihoods, AIC, BIC, Hessians, and
parameter covariance estimates are not interchangeable.

## Completed numerical foundations

### Spatial and temporal conventions

- ordered spatial weights with identity `W0`;
- supplied orientation preserved for non-symmetric weights;
- positive moving-average sign and matching inverse-MA companion convention;
- explicit ordinary and seasonal differencing histories;
- multiplicative seasonal factors applied on the left of ordinary factors;
- deterministic cross-lag aggregation without spatial-basis projection.

### Estimation and inference

- stationary conditional and Gaussian maximum likelihood;
- ordinary conditional integrated Gaussian estimation;
- multiplicative seasonal conditional Gaussian estimation;
- ordinary and seasonal original-level exact-diffuse MLE;
- scalar, diagonal, and full-Cholesky innovation covariance;
- missing and partially observed measurement rows;
- expanded AR stationarity and positive-sign inverse-MA invertibility checks;
- observed-information inference across fitted Gaussian and exact-diffuse
  families;
- natural innovation-covariance delta-method inference.

### Posterior operations and forecasting

- ordinary Kalman fixed-interval and RTS lag-one smoothing;
- ordinary and seasonal exact-diffuse fixed-interval smoothing;
- ordinary and seasonal primitive innovation/state-disturbance smoothing;
- exact dense adjacent-time covariance for genuine diffuse phases;
- observation-scale adjacent covariance and state-disturbance reconstruction;
- ordinary Gaussian and exact-diffuse forecast intervals;
- seasonal exact-diffuse original-level and transformed paths and intervals;
- dense ordinary and seasonal exact-diffuse conditional simulation smoothing;
- pathwise ordinary-seasonal level restoration and transformed-state projection;
- explicit rejection of unresolved diffuse directions.

### Diagnostics and evaluation

- STACF and STPACF variants;
- space-time portmanteau tests;
- admissibility diagnostics with companion matrices and eigenvalues;
- rolling-origin evaluation;
- interval scoring and coverage diagnostics;
- bootstrap prediction workflows.

## Completed in 0.0.32

### Public adjacent-time covariance API

The functional API is:

```python
from pystarmax import exact_diffuse_lag_one_covariance

moments = exact_diffuse_lag_one_covariance(filter_result)
```

The ordinary and seasonal fitted estimators expose:

```python
moments = model.smooth_lag_one_covariance()
```

Passing explicit data to a fitted method starts a fresh exact-diffuse filter
under the fitted parameters rather than continuing from the training terminal
posterior.

### Orientation and result contract

The public orientation matches the existing ordinary RTS smoother:

\[
C_t=
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y_{1:T}).
\]

`ExactDiffuseLagOneCovarianceResult` exposes immutable:

- `lag_one_covariance`;
- `observation_lag_one_covariance`;
- reconstructed `state_disturbance_mean`;
- reconstructed `state_disturbance_covariance`;
- exact smoother result;
- diffuse and conditioning ranks;
- proper and posterior source dimensions;
- support, marginal, and disturbance verification discrepancies;
- numerical positive-semidefinite correction diagnostics.

For a single observation time, all transition-indexed arrays have length zero.

### Exact dense source construction

The complete state path is represented as

\[
\alpha=b+D\delta+G\xi,
\]

where `delta` contains flat diffuse coordinates and `xi` contains proper
standard-normal coordinates. Exact observations impose

\[
A\delta+B\xi=c.
\]

All diffuse directions must be identified. The diffuse coordinates are
eliminated analytically and the remaining proper source vector is conditioned
on its exact equality constraints. If `H` is the resulting posterior state-path
loading, then

\[
C_t=H_tH_{t+1}^{\mathsf T}.
\]

No finite large-variance initialization and no Monte Carlo approximation are
used by the public covariance routine.

### Independent cross-checks

The implementation reconstructs state-equation disturbance covariance as

\[
P_{t+1}+TP_tT^{\mathsf T}
-C_t^{\mathsf T}T^{\mathsf T}
-TC_t
\]

and compares it with the independent exact information-form disturbance
smoother. A material discrepancy raises `LinAlgError`.

The test suite also uses:

- a diffuse local-level final-anchor closed form;
- exact zero-diffuse reduction to ordinary RTS covariance;
- a non-symmetric transition with partial observations;
- conditional-path Monte Carlo cross-covariance;
- a period-two seasonal augmented-state comparison;
- ordinary and seasonal fitted training/new-data behavior;
- unresolved diffuse-rank, validation, immutability, and public API checks.

### Computational boundary

The implementation materializes complete path loadings and dense equality
constraints. It is a transparent moderate-sample exact reference and a numerical
oracle for a future memory-linear diffuse `L2` recursion. Version 0.0.32 does not
claim that recursive scaling result.

## Authoritative validation

Initial CI #656 found one invalid empty-array immutability assertion and Black
formatting in the new test file. The numerical covariance tests otherwise
passed. The assertion was moved to a nonempty immutable array without changing
formulas or tolerances. A temporary branch-only workflow pinned Black 26.5.1,
applied its exact one-expression output, committed it, and deleted itself. The
formal PR diff contains no workflow file.

Authoritative implementation CI #660, run ID `30964497447`, passed on exact head
`37ddfb4920ff1af37d7a25140efbb71c6aeb6550`:

- 266 tests passed;
- total branch coverage: 87.40%;
- `src/pystarmax/exact_diffuse_lag_one_covariance.py` branch coverage: 89.7%;
- Black, isort, Ruff, and mypy passed;
- diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11–3.14.

Synchronized documentation CI #669, run ID `30964956888`, passed on exact head
`d93f198f8abd0d08ea438b8c548b7349a8aa4b54` with the same 266-test and
87.40%-coverage results. The method guide, example, README, documentation home,
MkDocs navigation, roadmap, inventory, project status, and Step 32 handoff all
passed strict documentation and cross-platform gates.

A final record-only merge gate remains required on the exact candidate produced
by this validation-record update.

## Numerical and research safeguards

- no exact-diffuse method substitutes a finite large covariance for diffuse
  directions;
- missing observations are skipped, not silently imputed;
- observed cells are exact conditional constraints where relevant;
- non-symmetric spatial and state operators retain orientation;
- adjacent covariance orientation is explicitly left-time/right-time;
- state-disturbance identities independently check adjacent moments;
- expanded multiplicative cross terms are deterministic, not extra parameters;
- covariance optimizer coordinates and natural covariance elements remain
  distinct;
- singular-curvature inference is explicit rather than silent;
- result arrays are immutable;
- dense implementations are moderate-sample references, not large-data claims.

## Current limitations

Version 0.0.32 does not yet provide:

- a memory-linear diffuse `L2` lag-one covariance recursion;
- arbitrary non-adjacent state covariance;
- cross-time state-disturbance or primitive-innovation covariance;
- robust or sandwich covariance;
- profile-likelihood intervals;
- parameter-uncertainty propagation into forecasts or smoother summaries;
- analytic score or Hessian recursions;
- sparse, chunked, or parallel numerical execution.

## Remaining plan

After Step 32, the delivery inventory contains:

- 5 major technical workstreams;
- 6 numbered core milestones, with the recursive part of milestone 15 still
  outstanding;
- approximately 9–13 independently reviewable projects.

The next stage should derive and validate the memory-linear exact diffuse `L2`
recursion against the 0.0.32 dense oracle. Arbitrary cross-time disturbance and
primitive-innovation covariance should follow only after that equivalence is
established.

## Merge checklist for PR #32

1. Synchronize README, documentation home, method guide, example, navigation,
   roadmap, inventory, status, and Step 32 handoff.
2. Pass complete CI on the exact final head.
3. Record final run number, run ID, head, test count, and coverage.
4. Confirm the formal diff contains no temporary workflow or generated artifact.
5. Confirm no submitted review or unresolved review thread remains.
6. Mark PR #32 ready and squash-merge version 0.0.32.
7. Create the exact diffuse `L2` recursion branch from the new `main`.

## Handoff documents

Read these before the next stage:

- `docs/development/STEP_32_EXACT_DIFFUSE_LAG_ONE_COVARIANCE.md`;
- `docs/exact_diffuse_lag_one_covariance.md`;
- `docs/development/STEP_31_SEASONAL_EXACT_DIFFUSE_SIMULATION_SMOOTHING.md`;
- `docs/seasonal_exact_diffuse_simulation_smoothing.md`;
- `docs/development/REMAINING_WORK.md`;
- `docs/roadmap.md`.