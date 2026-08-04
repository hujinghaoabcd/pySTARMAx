# pySTARMAx project status

## Repository position

- `main` is version `0.0.28` at merge commit
  `83656043b916395b95c7a1135625cb223e8a43ce`.
- Active branch: `agent/seasonal-exact-diffuse-inference`.
- Active pull request: PR #29, `Add seasonal exact diffuse likelihood inference`.
- Development version: `0.0.29`.
- PR #1 through PR #28 have been squash-merged.

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
  smoothing;
- seasonal exact-diffuse observed-information and natural innovation-covariance
  inference through the 0.0.29 fitted facade.

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

### Estimation

- stationary conditional and Gaussian maximum likelihood;
- ordinary conditional integrated Gaussian estimation;
- multiplicative seasonal conditional Gaussian estimation;
- ordinary original-level exact-diffuse MLE;
- seasonal original-level exact-diffuse MLE;
- scalar, diagonal, and full-Cholesky innovation covariance;
- missing and partially observed measurement rows;
- expanded AR stationarity and positive-sign inverse-MA invertibility checks.

### Posterior operations and uncertainty

- ordinary Kalman fixed-interval smoothing;
- ordinary and seasonal exact-diffuse fixed-interval smoothing;
- ordinary and seasonal primitive innovation/state-disturbance smoothing;
- ordinary Gaussian and exact-diffuse forecast intervals;
- dense ordinary exact-diffuse conditional simulation smoothing;
- observed-information inference for stationary, integrated, conditional-
  seasonal, ordinary exact-diffuse, and seasonal exact-diffuse fitted models;
- natural innovation-covariance delta-method inference.

### Diagnostics and evaluation

- STACF and STPACF variants;
- space-time portmanteau tests;
- admissibility diagnostics with companion matrices and eigenvalues;
- rolling-origin evaluation;
- interval scoring and coverage diagnostics;
- bootstrap prediction workflows.

## Completed in 0.0.29

### Seasonal original-level exact-diffuse curvature

The fitted estimator now exposes:

```python
inference = model.likelihood_inference()
```

The equivalent functional API is:

```python
from pystarmax import infer_seasonal_exact_diffuse_kalman_starima

inference = infer_seasonal_exact_diffuse_kalman_starima(model)
```

Every finite-difference candidate performs the complete fitted-model
construction again:

1. decode ordinary and seasonal AR/MA factor coordinates;
2. expand the ordered multiplicative matrix polynomials;
3. decode the fitted scalar, diagonal, or full-Cholesky covariance coordinates;
4. build the stationary transformed state-space model;
5. build the original-level `(1-B)^d(1-B^s)^D` exact-diffuse state;
6. evaluate the original observation mask with the exact-diffuse filter.

The resulting score and observed-information Hessian therefore use the same
original-level likelihood scope as fitting. Expanded cross-lag operators remain
deterministic and do not become additional inference parameters.

### Rank and boundary policy

By default the Hessian must be positive definite and full rank. Singular or
indefinite curvature raises `numpy.linalg.LinAlgError` instead of silently
reporting ordinary standard errors.

Generalized-inverse inference requires an explicit opt-in:

```python
inference = model.likelihood_inference(allow_singular=True)
```

Only positive eigenvalue directions above the `rcond` threshold contribute to
the covariance. The immutable result retains Hessian rank, eigenvalues,
condition number, positive-definite status, pseudoinverse use, score,
finite-difference steps, evaluation count, and fitted AR/inverse-MA boundary
distances.

### Natural innovation covariance

The existing delta-method contract converts optimizer covariance coordinates to
natural lower-triangular covariance elements:

```python
natural = inference.innovation_covariance_inference()
```

The result includes natural estimates, standard errors, Wald statistics,
correlations, and dynamic-parameter cross covariance for scalar, diagonal, and
full-Cholesky covariance models.

### Independent references

The 0.0.29 suite covers:

- closed-form intercept information `n/q` for a period-two seasonal random walk;
- closed-form raw log-standard-deviation information `2n`;
- natural scalar variance standard error `q * sqrt(2/n)`;
- exact reduction to ordinary exact-diffuse estimates, Hessian, and covariance
  when seasonal orders are zero;
- missing-data candidate filtering and immutable results;
- full-Cholesky natural covariance transformation and factor-based parameter
  counting;
- explicit singular-Hessian failure and generalized-inverse behavior;
- public functional and fitted-method exports.

## Validation status

Initial implementation CI #603, run ID `30957031201`, passed the complete
cross-platform numerical test matrix and coverage job. Its only failure was
Black formatting for the two newly added Python files.

A temporary read-only Black 26.5.1 diagnostic workflow printed the exact two
formatting changes. Those changes were applied and the diagnostic workflow was
removed. It is not part of the formal PR diff.

A fully synchronized authoritative CI run is required after the 0.0.29 public
API, documentation, status, roadmap, example, and handoff are complete. The
final test count, total coverage, and new-module coverage will be recorded here
before merge.

## Numerical and research safeguards

- no exact-diffuse method substitutes a finite large covariance for diffuse
  directions;
- missing observations are skipped, not silently imputed;
- non-symmetric spatial operators retain orientation;
- expanded multiplicative cross terms are deterministic, not extra parameters;
- every seasonal exact-diffuse curvature candidate rebuilds the complete state
  and filter;
- covariance optimizer coordinates and natural covariance elements remain
  distinct;
- singular-curvature inference is explicit rather than silent;
- result arrays are immutable;
- likelihood-scope boundaries are documented and tested;
- dense implementations are moderate-sample references, not large-data claims.

## Current limitations

Version 0.0.29 does not yet provide seasonal exact-diffuse:

- original-level or transformed-scale forecast intervals;
- conditional simulation smoothing;
- diffuse lag-one state covariance during a nontrivial diffuse phase;
- cross-time disturbance covariance;
- robust or sandwich covariance;
- profile-likelihood intervals;
- parameter-uncertainty propagation into forecasts or smoother summaries;
- analytic score or Hessian recursions;
- sparse or parallel numerical execution.

The diffuse `L2` recursion remains a separate research task.

## Remaining plan

After Step 29, the delivery inventory contains:

- 6 major technical workstreams;
- 7 numbered core milestones;
- approximately 12–16 independently reviewable projects.

The next implementation stage is seasonal exact-diffuse forecasting
uncertainty. Original-level and transformed-scale paths must use the terminal
exact-diffuse posterior, require resolved diffuse rank, and restore levels
pathwise before computing original-scale quantiles.

## Merge checklist for PR #29

1. Complete the synchronized README, documentation home, method guide, example,
   navigation, citation metadata, roadmap, remaining-work inventory, project
   status, and Step 29 handoff.
2. Pass complete CI on the exact final head.
3. Record final run number, run ID, test count, total coverage, and new-module
   coverage.
4. Confirm the formal diff contains no temporary workflow or generated artifact.
5. Confirm no submitted review or unresolved review thread remains.
6. Mark PR #29 ready and squash-merge version 0.0.29.
7. Create the seasonal exact-diffuse forecasting branch from the new `main`.

## Handoff documents

Read these before the next stage:

- `docs/development/STEP_29_SEASONAL_EXACT_DIFFUSE_INFERENCE.md`;
- `docs/seasonal_exact_diffuse_inference.md`;
- `docs/development/STEP_28_SEASONAL_EXACT_DIFFUSE_SMOOTHING.md`;
- `docs/seasonal_exact_diffuse_smoothing.md`;
- `docs/development/STEP_27_SEASONAL_EXACT_DIFFUSE_MLE.md`;
- `docs/seasonal_exact_diffuse_mle.md`;
- `docs/development/REMAINING_WORK.md`;
- `docs/roadmap.md`.
