# pySTARMAx project status

## Repository position

- `main` is version `0.0.29` at squash merge commit
  `1f92083fcd51fa788095d753ed293137f5886204`.
- Active branch: `agent/seasonal-exact-diffuse-forecasting`.
- Active pull request: PR #30, `Add seasonal exact diffuse forecasting`.
- Development version: `0.0.30`.
- PR #1 through PR #29 have been squash-merged.

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
- seasonal exact-diffuse smoothing, likelihood inference, and fixed-parameter
  forecast paths and intervals.

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

- ordinary Kalman fixed-interval smoothing;
- ordinary and seasonal exact-diffuse fixed-interval smoothing;
- ordinary and seasonal primitive innovation/state-disturbance smoothing;
- ordinary Gaussian and exact-diffuse forecast intervals;
- dense ordinary exact-diffuse conditional simulation smoothing;
- seasonal exact-diffuse original-level and transformed-scale paths and
  intervals;
- pathwise ordinary-seasonal level restoration;
- explicit rejection of unresolved terminal diffuse rank.

### Diagnostics and evaluation

- STACF and STPACF variants;
- space-time portmanteau tests;
- admissibility diagnostics with companion matrices and eigenvalues;
- rolling-origin evaluation;
- interval scoring and coverage diagnostics;
- bootstrap prediction workflows.

## Completed in 0.0.30

### Public forecast API

The fitted seasonal exact-diffuse estimator exposes:

```python
original_paths = model.simulate_forecast_paths(...)
transformed_paths = model.simulate_differenced_forecast_paths(...)
original_interval = model.predict_interval(...)
transformed_interval = model.predict_differenced_interval(...)
```

Equivalent functional APIs accept a matching `ExactDiffuseFilterResult` and
`ExactSeasonalIntegratedStateSpace`.

### State and uncertainty contract

Every replication draws the complete terminal augmented state from

\[
\mathcal N(a_{T|T},P_{T|T})
\]

and propagates it with future fitted innovations. Original observations are
projected from the complete augmented state. The combined ordinary-seasonal
transformed process is projected from the stationary state block at offset

\[
(d+Ds)n_{\text{locations}}.
\]

Original-level empirical quantiles are computed only after each path has been
restored by the augmented inverse-differencing companion. The interval mean is
the deterministic propagated posterior mean and agrees with existing point
forecast methods.

The intervals include terminal state and future fitted innovation uncertainty.
They do not include fitted-parameter uncertainty.

### Proper-posterior policy

Forecast paths require `final_diffuse_rank == 0`. If diffuse directions remain,
path and interval functions raise `RuntimeError`. No large finite covariance is
substituted for unresolved diffuse uncertainty.

The supplied seasonal integrated specification must be the exact state-space
object used by the filter result, preventing accidental projection under a
different state layout.

### Independent references

The 0.0.30 suite covers:

- pathwise `x_t = y_t - y_(t-2)` identity for a period-two seasonal random walk;
- original-level variance sequence `q, q, 2q, 2q, ...`;
- constant transformed-scale innovation variance `q`;
- exact empirical-quantile identity between public path and interval APIs;
- exact reduction to ordinary exact-diffuse forecasting when seasonal orders
  are zero;
- fitted facade consistency with `predict()` and `predict_differenced()`;
- terminal diffuse-rank rejection and input validation;
- immutable intervals and public exports.

## Validation status

Initial implementation CI #626 identified only Black formatting in the new
forecasting module and public-API test before later quality steps ran. A
temporary branch-only workflow pinned Black 26.5.1, applied the exact formatter
output, committed it, and deleted itself. The formal PR diff contains no
workflow file.

Authoritative implementation CI #630, run ID `30959749037`, passed on exact head
`652a1c514b96c267f6aeaabfa6090e28b9249436`:

- 252 tests passed;
- total branch coverage: 87.28%;
- `src/pystarmax/seasonal_exact_diffuse_forecasting.py`: complete coverage;
- Black, isort, Ruff, and mypy passed;
- diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11–3.14.

A synchronized documentation and merge-gate CI remains required on the final PR
head.

## Numerical and research safeguards

- no exact-diffuse method substitutes a finite large covariance for diffuse
  directions;
- missing observations are skipped, not silently imputed;
- non-symmetric spatial operators retain orientation;
- expanded multiplicative cross terms are deterministic, not extra parameters;
- covariance optimizer coordinates and natural covariance elements remain
  distinct;
- singular-curvature inference is explicit rather than silent;
- original-level seasonal quantiles use restored paths, not transformed marginal
  quantiles;
- result arrays are immutable;
- dense implementations are moderate-sample references, not large-data claims.

## Current limitations

Version 0.0.30 does not yet provide seasonal exact-diffuse:

- conditional simulation smoothing;
- diffuse lag-one state covariance during a nontrivial diffuse phase;
- cross-time disturbance covariance;
- robust or sandwich covariance;
- profile-likelihood intervals;
- parameter-uncertainty propagation into forecasts or smoother summaries;
- analytic score or Hessian recursions;
- sparse, chunked, or parallel numerical execution.

The diffuse `L2` recursion remains a separate research task.

## Remaining plan

After Step 30, the delivery inventory contains:

- 6 major technical workstreams;
- 6 numbered core milestones;
- approximately 11–15 independently reviewable projects.

The next implementation candidate is seasonal exact-diffuse conditional
simulation smoothing, followed by the diffuse `L2` theory project.

## Merge checklist for PR #30

1. Synchronize README, documentation home, method guide, example, navigation,
   roadmap, inventory, status, and Step 30 handoff.
2. Pass complete CI on the exact final head.
3. Record final run number, run ID, head, test count, and coverage.
4. Confirm the formal diff contains no temporary workflow or generated artifact.
5. Confirm no submitted review or unresolved review thread remains.
6. Mark PR #30 ready and squash-merge version 0.0.30.
7. Create the seasonal exact-diffuse simulation-smoothing branch from new
   `main`.

## Handoff documents

Read these before the next stage:

- `docs/development/STEP_30_SEASONAL_EXACT_DIFFUSE_FORECASTING.md`;
- `docs/seasonal_exact_diffuse_forecasting.md`;
- `docs/development/STEP_29_SEASONAL_EXACT_DIFFUSE_INFERENCE.md`;
- `docs/seasonal_exact_diffuse_inference.md`;
- `docs/development/REMAINING_WORK.md`;
- `docs/roadmap.md`.
