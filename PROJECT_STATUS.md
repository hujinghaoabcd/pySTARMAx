# pySTARMAx project status

Updated: 2026-08-04

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 through PR #10 were squash-merged into `main`.
- PR #8, state-space filtering and missing observations, was merged as `258ceac7`.
- PR #9, Kalman maximum-likelihood estimation, was merged as `4c3c7146`.
- PR #10, likelihood-Hessian inference, was merged as `288a32b6`.
- `main` is version `0.0.10`.
- Current branch: `agent/stability-invertibility-diagnostics`.
- Current draft pull request: PR #11, `Add STARMA stability and invertibility diagnostics`.
- Current development target: version `0.0.11`.
- Current stage: reusable AR stationarity and MA invertibility diagnostics plus
  dual admissibility enforcement in stationary Gaussian Kalman STARMA.

## Completed baseline through 0.0.10

- typed package, MIT licence, citation metadata, strict documentation, and CI;
- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA iterative conditional least squares;
- deterministic simulation, recursive prediction, diagnostics, and summaries;
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
- likelihood, optimizer, AIC, BIC, filtering, and prediction diagnostics;
- central finite-difference likelihood score and Hessian;
- observed-information covariance, standard errors, normal tests, and intervals;
- Hessian rank, eigenvalues, condition number, score, and boundary diagnostics;
- strict indefinite/rank-deficient behavior and explicit diagnostic pseudoinverse.

## Completed in the current 0.0.11 step

### Reusable matrix-polynomial diagnostics

- added `compose_lag_operators()` for temporal-lag matrices built from ordered
  spatial weights;
- added immutable `PolynomialAdmissibility` results;
- added `autoregressive_diagnostics()` with the standard AR block companion;
- added `moving_average_diagnostics()` with the positive-sign inverse recursion
  companion top row `[-B1, ..., -Bq]`;
- added `STARMAAdmissibility` for joint stationarity and invertibility;
- added low-level AR and inverse-MA spectral-radius helpers;
- preserved non-symmetric spatial-weight orientation;
- represented zero-order AR and MA polynomials by empty companions with radius 0;
- reported operators, companion matrices, complex eigenvalues, limits, signed
  boundary distances, and admissibility decisions;
- exported all diagnostic APIs from the top-level package.

### Maximum-likelihood enforcement

- added `enforce_invertibility=True` and `invertibility_margin=1e-6` defaults;
- validated AR and MA margins independently;
- shrank automatic conditional-estimator AR and MA starts into their configured
  regions;
- applied a joint feasibility penalty that accumulates squared AR and MA excess;
- hard-checked both dynamic blocks after optimization;
- preserved explicit research behavior when either enforcement flag is disabled;
- added fitted `KalmanSTARMA.admissibility()` diagnostics;
- extended `KalmanSTARMAResult` with both radii, limits, enforcement flags,
  distances, and joint admissibility;
- extended result summaries without hiding a disabled or failed criterion.

### Likelihood inference

- reconstructed both admissibility criteria in every finite-difference objective;
- rejected stencils crossing either enabled penalty boundary;
- added inverse-MA boundary distance and minimum admissibility distance to
  `LikelihoodInferenceResult`;
- retained raw optimizer-scale uncertainty and strict Hessian policies.

### Tests and documentation

- added scalar AR(1), MA(1), and MA(2) polynomial-root references;
- added asymmetric spatial-weight orientation and zero-order tests;
- added explicit joint-boundary and validation tests;
- added fitted MA recovery and admissibility tests;
- added forced non-invertible final-candidate tests with enforcement enabled and
  disabled;
- updated likelihood-inference boundary tests;
- added `docs/admissibility.md` and `examples/admissibility.py`;
- updated model, maximum-likelihood, inference, index, roadmap, citation, and
  navigation documentation;
- updated package version and citation metadata to 0.0.11.

## Mathematical convention

For

\[
z_t = c + \sum_i A_i z_{t-i} + \varepsilon_t
      + \sum_j B_j\varepsilon_{t-j},
\]

stationarity uses the block companion with top row `[A1, ..., Ap]`.

Because the MA sign is positive, the innovation inverse recursion is

\[
\varepsilon_t = r_t - \sum_j B_j\varepsilon_{t-j},
\]

so invertibility uses the block companion with top row `[-B1, ..., -Bq]`.
Each criterion requires its spectral radius to be strictly below
`1 - configured_margin`.

A signed distance is positive inside the region, zero on the configured limit,
and negative outside. The public result retains the full eigensystem rather than
reducing the diagnosis to a Boolean.

## Final validation for 0.0.11

GitHub Actions CI run #269 completed successfully on the final implementation,
test, example, and documentation head:

- 103 tests passed without Python test warnings;
- total branch coverage was 87.48%, above the configured 80% threshold;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic reference generation produced a clean diff;
- strict MkDocs construction passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14;
- the PR surface contained exactly 20 formal source, test, example,
  documentation, metadata, and navigation files;
- `.github/workflows/ci.yml` was absent from the final PR difference;
- no temporary formatting workflow or diagnostic file remained.

The validation-record edits after run #269 are documentation only. Numerical
code, tests, examples, public exports, package metadata, and CI configuration are
unchanged from the fully validated head. One final documentation-head CI repeat
is required before PR #11 is marked ready and merged.

## Design principles

1. Keep conditional and maximum-likelihood estimators separate and explicit.
2. Preserve one `(time, location)` convention.
3. Never impute missing observations inside a likelihood evaluation.
4. Preserve supplied non-symmetric spatial-matrix orientation.
5. Use the model's positive MA sign consistently in inverse-recursion diagnostics.
6. Report complete eigensystems and signed distances, not only booleans.
7. Treat spectral-radius penalties as feasibility controls, not smooth constraints.
8. Do not differentiate through AR or MA feasibility-penalty regions.
9. Preserve explicit behavior when enforcement is disabled.
10. Refuse indefinite or rank-deficient observed information by default.
11. Keep optimizer-scale and natural-scale uncertainty explicitly distinguished.
12. Use immutable public result arrays and stable linear solves.
13. Add independent scalar and matrix-polynomial references before broad claims.

## Immediate next tasks

1. Complete the final documentation-head CI repeat for PR #11.
2. Mark PR #11 ready and squash-merge it into `main`.
3. Start delta-method innovation-covariance inference unless a general, auditable
   matrix-polynomial stability/invertibility parameterization is established
   first.
4. Add integrated and multiplicative seasonal maximum-likelihood wrappers.
5. Add state and disturbance smoothing.
6. Add sparse matrices, ecosystem adapters, order selection, and exogenous inputs.
7. Add cross-language estimator fixtures and prepare the first PyPI pre-release.

## Known limitations

- feasibility is enforced by a large penalty, not a smooth bijection;
- arbitrary user starts are not projected to a nearest admissible parameter vector;
- dense eigendecomposition is used for every dynamic candidate;
- no covariance-factor delta-method transformation;
- no robust, sandwich, profile-likelihood, or likelihood-ratio inference;
- approximate diffuse initialization is not exact diffuse likelihood;
- no state or disturbance smoothing;
- maximum likelihood currently covers stationary non-seasonal STARMA only;
- multiplicative seasonal factor admissibility is not yet implemented;
- integrated and seasonal wrappers still use conditional estimation;
- conditional/bootstrap rolling refits execute serially;
- model order and spatial weights remain fixed across bootstrap replications;
- exogenous regressors and intervention variables are unsupported.

## Handoff instruction

Before the next substantial step, read this file, `docs/model.md`,
`docs/admissibility.md`, `docs/state_space.md`, `docs/maximum_likelihood.md`,
`docs/likelihood_inference.md`, and the latest development handoff. Update
repository state, validation, next tasks, and limitations after every completed
stage.
