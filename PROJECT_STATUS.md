# pySTARMAx project status

## Repository position

- `main` is version `0.0.26` at merge commit
  `e626ff330c2e6966d1474883656bf3543f56d84a`.
- Active branch: `agent/seasonal-exact-diffuse-mle`.
- Active pull request: PR #27, `Add seasonal exact diffuse maximum likelihood`.
- Development version: `0.0.27`.
- PR #1 through PR #26 have been squash-merged.

## Public model families

The package currently exposes:

- conditional least-squares `STARMA`, `STARIMA`, and `SeasonalSTARIMA`;
- stationary Gaussian `KalmanSTARMA`;
- conditional ordinary-integrated `KalmanSTARIMA`;
- conditional multiplicative seasonal `SeasonalKalmanSTARIMA`;
- original-level ordinary exact-diffuse `ExactDiffuseKalmanSTARIMA`;
- fixed-parameter seasonal exact-diffuse state construction through
  `ExactSeasonalIntegratedStateSpace`;
- optimizer-facing original-level seasonal exact-diffuse MLE through
  `SeasonalExactDiffuseKalmanSTARIMA`.

Conditional transformed-data likelihoods and original-level exact-diffuse
likelihoods are separate APIs. Their likelihoods, AIC, and BIC are not
interchangeable.

## Completed numerical foundations

### Spatial and temporal conventions

- ordered spatial weights with identity `W0`;
- supplied orientation preserved for non-symmetric weights;
- positive moving-average sign and matching inverse-MA companion convention;
- explicit ordinary and seasonal differencing histories;
- multiplicative seasonal factors applied on the left of ordinary factors;
- deterministic cross-lag aggregation without spatial-basis projection.

### Estimation

- stationary conditional and Gaussian maximum likelihood;
- ordinary conditional integrated Gaussian estimation;
- multiplicative seasonal conditional Gaussian estimation;
- ordinary original-level exact-diffuse MLE;
- seasonal original-level exact-diffuse MLE in version 0.0.27;
- scalar, diagonal, and full-Cholesky innovation covariance;
- missing and partially observed measurement rows;
- expanded AR stationarity and positive-sign inverse-MA invertibility checks.

### Posterior operations

- ordinary Kalman fixed-interval smoothing;
- ordinary exact-diffuse fixed-interval smoothing;
- primitive innovation and state-disturbance smoothing;
- ordinary Gaussian and exact-diffuse forecast intervals;
- dense ordinary exact-diffuse conditional simulation smoothing;
- observed-information inference for established fitted families;
- natural innovation-covariance delta-method inference.

### Diagnostics and evaluation

- STACF and STPACF variants;
- space-time portmanteau tests;
- admissibility diagnostics with companion matrices and eigenvalues;
- rolling-origin evaluation;
- interval scoring and coverage diagnostics;
- bootstrap prediction workflows.

## Completed in 0.0.26

Version 0.0.26 introduced the original-level seasonal exact-diffuse state for

\[
(1-B)^d(1-B^s)^D y_t.
\]

The state stores the required original-level lag companion and a stationary
transformed subsystem. All integration coordinates receive exact diffuse
covariance; no arbitrary finite large variance is used. When `D=0`, the
constructor reduces exactly to the ordinary integrated state contract.

## Completed in 0.0.27

### Seasonal exact-diffuse maximum likelihood

`SeasonalExactDiffuseKalmanSTARIMA` now:

- optimizes ordinary and seasonal AR/MA factor coefficients;
- expands ordered multiplicative cross lags for every optimizer candidate;
- rebuilds the stationary transformed and original-level integrated states;
- evaluates the exact-diffuse likelihood on original observations;
- enforces admissibility on fully expanded AR and inverse-MA recursions;
- counts only free factor, intercept, and covariance parameters in AIC/BIC;
- retains immutable factor parameters, expanded operators, covariance, exact
  filter, diffuse diagnostics, fitted state spaces, and optimizer metadata;
- exposes original-level and highest-difference point forecasts;
- supports missing observations without artificial diffuse-rank reduction.

### Independent references

The version 0.0.27 suite covers:

- closed-form seasonal-random-walk drift, variance, and likelihood;
- combined `(1-B)(1-B^2)` integration;
- deterministic multiplicative cross-lag signs and parameter counting;
- exact reduction to ordinary exact-diffuse MLE when seasonal orders are zero;
- missing initial observations delaying diffuse completion;
- public exports, immutable arrays, fitted state identities, and predictions.

## Authoritative implementation validation

GitHub Actions CI #576, run ID `30949430144`, validated implementation head
`246b2bcf38789adc4caf6b43b7db312d602991c7`.

Results:

- 231 tests passed;
- total branch coverage: 87.10%;
- `src/pystarmax/seasonal_exact_diffuse_mle.py` branch coverage: 82.9%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A final documentation-only merge-gate CI is required on the final PR head.

## Numerical and research safeguards

- no exact-diffuse method substitutes a finite large covariance for diffuse
  directions;
- missing observations are skipped, not silently imputed;
- non-symmetric spatial operators retain orientation;
- expanded multiplicative cross terms are deterministic, not extra parameters;
- covariance parameters and natural covariance elements remain distinct;
- result arrays are immutable;
- likelihood-scope boundaries are documented and tested;
- dense implementations are moderate-sample references, not large-data claims.

## Current limitations

Version 0.0.27 does not yet provide seasonal exact-diffuse:

- fixed-interval state smoothing;
- primitive disturbance smoothing;
- observed-information likelihood inference;
- forecast paths and intervals;
- conditional simulation smoothing;
- parameter-uncertainty propagation;
- sparse or parallel numerical execution.

The diffuse `L2` recursion for lag-one smoothed covariance also remains a
separate research task.

## Remaining plan

After Step 27, the delivery inventory contains:

- 6 major technical workstreams;
- 9 numbered core milestones;
- approximately 14–18 independently reviewable projects.

The next implementation stage is seasonal exact-diffuse fixed-interval state
and primitive disturbance smoothing, reusing the common 0.0.26 state and 0.0.27
fitted-model contracts.

## Merge checklist for PR #27

1. Pass complete CI on the final synchronized head.
2. Confirm the closed-form and ordinary-equivalence tests pass on all platforms.
3. Confirm no workflow helper, generated artifact, review submission, or
   unresolved review thread remains.
4. Mark PR #27 ready and squash-merge version 0.0.27.
5. Create the next branch for seasonal exact-diffuse smoothing.

## Handoff documents

Read these before the next stage:

- `docs/development/STEP_27_SEASONAL_EXACT_DIFFUSE_MLE.md`;
- `docs/seasonal_exact_diffuse_mle.md`;
- `docs/development/STEP_26_SEASONAL_EXACT_DIFFUSE_STATE_SPACE.md`;
- `docs/exact_seasonal_integrated.md`;
- `docs/development/REMAINING_WORK.md`;
- `docs/roadmap.md`.
