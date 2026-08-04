# pySTARMAx project status

## Repository position

- `main` is version `0.0.27` at merge commit
  `66fc0442b561ce257f2653e54bbecfd635c78f09`.
- Active branch: `agent/seasonal-exact-diffuse-smoothing`.
- Active pull request: PR #28, `Add seasonal exact diffuse smoothing`.
- Development version: `0.0.28`.
- PR #1 through PR #27 have been squash-merged.
- Synchronized merge-gate CI #599, run ID `30952280713`, passed on
  `e3d9ac6155847bc3dbb8289452953bf54e95189e`.

## Public model families

The package exposes:

- conditional least-squares `STARMA`, `STARIMA`, and `SeasonalSTARIMA`;
- stationary Gaussian `KalmanSTARMA`;
- conditional ordinary-integrated `KalmanSTARIMA`;
- conditional multiplicative seasonal `SeasonalKalmanSTARIMA`;
- original-level ordinary exact-diffuse `ExactDiffuseKalmanSTARIMA`;
- fixed-parameter seasonal exact-diffuse state construction through
  `ExactSeasonalIntegratedStateSpace`;
- optimizer-facing original-level seasonal exact-diffuse MLE through
  `SeasonalExactDiffuseKalmanSTARIMA`;
- seasonal exact-diffuse fixed-interval state and primitive/state-disturbance
  smoothing through the 0.0.28 fitted facade.

Conditional transformed-data likelihoods and original-level exact-diffuse
likelihoods are separate contracts. Their likelihoods, AIC, and BIC are not
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
- seasonal original-level exact-diffuse MLE;
- scalar, diagonal, and full-Cholesky innovation covariance;
- missing and partially observed measurement rows;
- expanded AR stationarity and positive-sign inverse-MA invertibility checks.

### Posterior operations

- ordinary Kalman fixed-interval smoothing;
- ordinary and seasonal exact-diffuse fixed-interval smoothing;
- ordinary and seasonal primitive innovation/state-disturbance smoothing;
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

## Completed in 0.0.28

### Seasonal exact-diffuse state smoothing

The fitted seasonal exact-diffuse estimator now exposes:

```python
model.smooth()
model.smooth(new_data)
```

The facade reuses the generic exact-diffuse backward information recursion. The
0.0.26 seasonal augmented state is already a general linear Gaussian state-space
model, so no seasonal copy of the recursion and no finite large-variance
initialization are required.

The immutable result retains:

- smoothed state means and covariances;
- smoothed original-level observations and marginal covariances;
- finite and diffuse backward information arrays;
- reconstruction, symmetry, rank, and pseudoinverse diagnostics;
- the exact filter result used by the posterior calculation.

Passing new data starts a fresh exact-diffuse initialization under the fitted
parameters. It does not continue from the terminal training posterior.

### Seasonal primitive innovation smoothing

The fitted facade also exposes:

```python
model.smooth_innovation_disturbances()
model.smooth_innovation_disturbances(new_data)
```

The primitive innovation posterior uses the full seasonal augmented-state
selection matrix. It therefore maps each innovation consistently into the
original-level lag companion and stationary transformed state. The result
includes primitive innovation moments, state-equation disturbance moments,
process-rank diagnostics, unresolved covariance, and support residuals.

### Independent references

The 0.0.28 tests cover:

- a period-two seasonal random walk decomposed into two exact Gaussian bridges;
- closed-form missing-level means and variances;
- closed-form primitive innovation means and variances, including the unresolved
  first diffuse transition;
- multivariate partial-location observation masks;
- training and fresh-data fitted facade behavior;
- immutable results and tolerance validation;
- exact reduction to ordinary exact-diffuse state and innovation smoothing when
  all seasonal orders are zero.

## Authoritative validation

Implementation CI #589, run ID `30951765063`, validated implementation head
`50a80b5cca2fbca188631bed5beffc8925deaf97`. Synchronized merge-gate CI #599,
run ID `30952280713`, validated final implementation and documentation head
`e3d9ac6155847bc3dbb8289452953bf54e95189e`.

Results:

- 238 tests passed;
- total branch coverage: 87.16%;
- the new fitted facade was fully covered;
- `src/pystarmax/seasonal_exact_diffuse_mle.py` branch coverage: 83.9%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

## Numerical and research safeguards

- no exact-diffuse method substitutes a finite large covariance for diffuse
  directions;
- missing observations are skipped, not silently imputed;
- non-symmetric spatial operators retain orientation;
- expanded multiplicative cross terms are deterministic, not extra parameters;
- covariance parameters and natural covariance elements remain distinct;
- result arrays are immutable;
- likelihood-scope boundaries are documented and tested;
- seasonal smoothing reuses the common state/filter contracts;
- dense implementations are moderate-sample references, not large-data claims.

## Current limitations

Version 0.0.28 does not yet provide seasonal exact-diffuse:

- lag-one state covariance during a nontrivial diffuse phase;
- cross-time disturbance covariance;
- observed-information likelihood inference;
- natural innovation-covariance inference;
- forecast paths and intervals;
- conditional simulation smoothing;
- parameter-uncertainty propagation;
- sparse or parallel numerical execution.

The diffuse `L2` recursion remains a separate research task.

## Remaining plan

After Step 28, the delivery inventory contains:

- 6 major technical workstreams;
- 8 numbered core milestones;
- approximately 13–17 independently reviewable projects.

The next implementation stage is seasonal exact-diffuse likelihood inference
and natural innovation-covariance inference. Forecast paths and intervals should
remain a later independent stage.

## PR #28 merge readiness

- implementation CI #589 passed;
- synchronized merge-gate CI #599 passed;
- bridge, disturbance, partial-observation, and ordinary-reduction tests passed
  on all supported platforms;
- the formal diff contains no workflow helper or generated artifact;
- no submitted review or unresolved review thread remains;
- PR #28 is ready for squash merge after the validation-record-only gate.

## Handoff documents

Read these before the next stage:

- `docs/development/STEP_28_SEASONAL_EXACT_DIFFUSE_SMOOTHING.md`;
- `docs/seasonal_exact_diffuse_smoothing.md`;
- `docs/development/STEP_27_SEASONAL_EXACT_DIFFUSE_MLE.md`;
- `docs/seasonal_exact_diffuse_mle.md`;
- `docs/development/STEP_26_SEASONAL_EXACT_DIFFUSE_STATE_SPACE.md`;
- `docs/exact_seasonal_integrated.md`;
- `docs/development/REMAINING_WORK.md`;
- `docs/roadmap.md`.
