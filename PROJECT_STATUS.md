# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 through PR #9 were squash-merged into `main`.
- PR #8, state-space filtering and missing observations, was merged as `258ceac7`.
- PR #9, Kalman maximum-likelihood estimation, was merged as `4c3c7146`.
- `main` is version `0.0.9`.
- Current branch: `agent/kalman-likelihood-inference`.
- Current draft pull request: PR #10, `Add likelihood-Hessian inference diagnostics`.
- Current development target: version `0.0.10`.
- Current stage: observed-likelihood curvature, coefficient uncertainty, and
  weak-curvature diagnostics for stationary Gaussian Kalman STARMA.

## Completed baseline through 0.0.9

- typed package, MIT licence, citation metadata, strict documentation, and CI;
- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA iterative conditional least squares;
- deterministic simulation, recursive prediction, diagnostics, and result summaries;
- ordinary STARIMA and constrained multiplicative seasonal STARIMA;
- reversible ordinary-seasonal differencing and original-scale reconstruction;
- conditional and bootstrap forecast intervals;
- rolling-origin calibration, sharpness, and point-error evaluation;
- explicit stationary STARMA state-space construction;
- Gaussian fixed-parameter Kalman likelihood;
- stationary, known, and approximate diffuse initialization;
- partial-location and fully missing-row filtering;
- independent `KalmanSTARMA` Gaussian maximum-likelihood estimator;
- scalar, diagonal, and Cholesky full innovation covariance;
- complete likelihood, covariance, optimizer, AIC, BIC, filtering, and prediction
  diagnostics.

## Completed in the current 0.0.10 step

- added central finite-difference score and Hessian evaluation;
- added parameter-scaled relative and absolute perturbations;
- added exact four-corner mixed-partial central stencils;
- added function-evaluation accounting and immutable curvature results;
- reconstructed the fitted Gaussian Kalman negative log likelihood on the raw
  optimizer scale;
- rejected stencil points that enter the stationarity feasibility penalty;
- added observed-information covariance and parameter correlation;
- added standard errors, z statistics, two-sided normal p values, and normal
  confidence intervals for intercept and AR/MA coefficients;
- added separate dynamic-coefficient and complete optimizer-parameter tables;
- added Hessian eigenvalues, numerical rank, condition number, maximum score, and
  stationarity-boundary distance;
- made non-positive-definite or rank-deficient Hessians fail by default;
- added an explicit positive-eigenspace pseudoinverse route for diagnosis;
- marked pseudoinverse results and kept zero-information directions finite;
- exposed inference through `KalmanSTARMA.infer()`;
- exported `FiniteDifferenceCurvature`, `LikelihoodInferenceResult`,
  `finite_difference_curvature()`, and `finite_difference_hessian()`;
- added analytic quadratic, scalar Gaussian white-noise, incomplete AR(1),
  validation, and singular-Hessian tests;
- added README, MkDocs, method documentation, example, roadmap, citation, and
  Step 10 handoff updates;
- updated package and citation metadata to 0.0.10.

## Statistical interpretation

`KalmanSTARMA.infer()` computes local observed-likelihood curvature at the fitted
raw optimizer point. The inverse Hessian is used only when the observed
information is positive definite and numerically full rank.

Intercept and AR/MA entries are already on their natural coefficient scale.
Scalar and diagonal covariance entries remain log standard deviations. Full
covariance entries remain log Cholesky diagonals and unconstrained lower
Cholesky elements. Version 0.0.10 does not transform covariance-factor
uncertainty into variance, covariance, or correlation uncertainty.

The default behavior refuses indefinite and rank-deficient curvature. The
explicit `allow_singular=True` route uses only positive eigen-directions above
the numerical threshold and records `used_pseudoinverse=True`. It is diagnostic,
not a silent substitute for regular maximum-likelihood inference.

The estimator still uses a spectral-radius feasibility boundary rather than a
smooth stability parameterization. Finite-difference points entering the large
penalty region are rejected. MA invertibility is not yet constrained.

## Final validation for 0.0.10

GitHub Actions CI run #237 completed successfully on the fully documented branch
head:

- 91 tests passed without Python test warnings;
- total branch coverage was 87.84%, above the configured 80% threshold;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic reference regeneration produced a clean diff;
- strict MkDocs construction passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14;
- the explicit singular-Hessian path used masked division and produced no
  divide-by-zero warning;
- the PR surface contains exactly 14 formal source, test, example, documentation,
  metadata, and navigation files, with no temporary workflow or diagnostic files.

The validation-record edits after run #237 are documentation only. Numerical
code, tests, public exports, package metadata, and CI configuration are unchanged
from the fully validated head. A final documentation-head CI repeat is required
before marking PR #10 ready and merging it.

## Design principles

1. Keep conditional and maximum-likelihood estimators separate and explicit.
2. Preserve one `(time, location)` convention.
3. Never impute missing observations inside a likelihood evaluation.
4. Parameterize full covariance so positive definiteness is structural.
5. Count covariance parameters honestly in information criteria.
6. Report optimizer convergence and feasibility diagnostics without hiding failures.
7. Treat spectral-radius penalties as feasibility controls, not smooth constraints.
8. Do not differentiate through invalid feasibility-penalty regions.
9. Refuse indefinite or rank-deficient observed information by default.
10. Keep optimizer-scale and natural-scale uncertainty explicitly distinguished.
11. Keep non-symmetric spatial-matrix orientation explicit.
12. Use stable linear solves and immutable public result arrays.
13. Add independent numerical references before claiming external equivalence.

## Immediate next tasks

1. Complete the documentation-head CI repeat for PR #10.
2. Mark PR #10 ready and squash-merge it into `main`.
3. Add reusable stationarity diagnostics independent of fitting.
4. Add an explicit MA invertibility definition and checks for spatial STARMA.
5. Add constrained fitting or a smooth stability/invertibility parameterization.
6. Add delta-method transforms for innovation covariance elements.
7. Add integrated and multiplicative seasonal maximum-likelihood wrappers.
8. Add state and disturbance smoothing.
9. Add sparse matrices, ecosystem adapters, order selection, and exogenous inputs.
10. Add cross-language estimator fixtures and prepare the first PyPI pre-release.

## Known limitations

- covariance-factor standard errors are not transformed to covariance elements;
- no robust, sandwich, profile-likelihood, or likelihood-ratio inference;
- no smooth stationarity parameterization or MA invertibility constraint;
- approximate diffuse initialization is not exact diffuse likelihood;
- no state or disturbance smoothing;
- maximum likelihood currently covers stationary STARMA only;
- integrated and seasonal wrappers still use conditional estimation;
- dense matrices are used throughout;
- conditional/bootstrap rolling refits execute serially;
- model order and spatial weights remain fixed across bootstrap replications;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/state_space.md`, `docs/maximum_likelihood.md`,
`docs/likelihood_inference.md`, and
`docs/development/STEP_10_LIKELIHOOD_INFERENCE.md`. Update repository state,
validation, next tasks, and limitations after every completed stage.
