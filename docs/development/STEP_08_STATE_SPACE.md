# Step 08: state-space filtering and missing observations

## Goal

Introduce an auditable linear Gaussian state-space route without changing the
validated conditional STARMA estimator. The first state-space stage must support
fixed-parameter likelihood evaluation and incomplete observation matrices before
adding direct likelihood optimization.

## Delivered public API

- `StateSpaceModel`
- `KalmanFilterResult`
- `build_starma_state_space()`
- `fitted_starma_state_space()`
- `kalman_filter()`
- `kalman_loglikelihood()`
- `STAR.to_state_space()`
- `STAR.filter_state_space()`
- `STARMA.to_state_space()`
- `STARMA.filter_state_space()`

## State construction

For temporal AR order `p`, temporal MA order `q`, and `N` locations, the state
contains `max(1, p)` observation blocks followed by `q` innovation-history
blocks. The top transition row contains the assembled AR operators and lagged MA
operators. Shift blocks copy prior observations and innovations down the state.

The current innovation enters both the current observation state and, for `q>0`,
the first innovation-history state. This preserves the package sign convention

`+ theta * W * epsilon`.

Every temporal operator is assembled directly from the supplied ordered spatial
matrices. Product projection, symmetry, and commutativity are never assumed.

## Filtering convention

The implementation stores the state after the current process innovation. Each
time step therefore performs:

1. state prediction;
2. covariance prediction;
3. observation selection for finite cells;
4. innovation and innovation-covariance construction;
5. Cholesky likelihood evaluation;
6. Kalman update;
7. covariance symmetrization and numerical PSD verification.

A fully missing row skips steps 4-6 and contributes exactly zero to the
likelihood.

## Initialization

- `stationary`: solves the unconditional state mean and discrete Lyapunov
  covariance; rejects spectral radius at or above one;
- `known`: requires a complete user-supplied state mean and covariance;
- `diffuse`: uses a zero mean and large diagonal covariance and is explicitly
  documented as approximate rather than exact diffuse likelihood.

## Numerical safeguards

- positive-semidefinite validation for innovation and initial covariance;
- Cholesky solves rather than explicit inverses;
- scale-aware diagonal jitter with per-time recording;
- covariance symmetrization after prediction and update;
- clipping only of floating-point-scale negative covariance eigenvalues;
- failure on materially indefinite filtered covariance.

## Verification

Focused tests cover:

- non-symmetric spatial operator orientation;
- AR and MA companion blocks;
- exact agreement with the closed-form stationary scalar AR(1) likelihood;
- partial-location missing observations;
- fully missing rows;
- stationary rejection of a unit-root transition;
- approximate diffuse filtering;
- known initialization;
- white-noise state construction;
- fitted `STAR` conversion and incomplete-data filtering;
- covariance and dimensional validation;
- immutability of public result arrays.

GitHub Actions CI run #179 completed successfully:

- 77 tests passed;
- total branch coverage was 88.25%, above the configured 80% threshold;
- Black, isort, Ruff, and mypy passed;
- the independent exact diagnostic fixture regenerated with a clean Git diff;
- strict MkDocs construction passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11-3.14.

Black 26.5.1 and the mypy type-alias correction were applied through temporary
single-purpose workflows during development. Both workflows were removed after
their changes were committed. The final PR contains only 14 formal files and no
temporary diagnostics, formatters, or repair workflows.

## Explicit exclusions

This stage does not include:

- direct maximization of the Kalman likelihood;
- covariance optimization or Cholesky covariance parameterization;
- parameter standard errors from the likelihood Hessian;
- exact diffuse initialization;
- filtering/smoothing wrappers for integrated or multiplicative seasonal models;
- Kalman forecast intervals;
- sparse state matrices.

## Next step

Implement a dedicated maximum-likelihood estimator over the state-space core.
The first optimization route should support scalar and diagonal innovation
covariance, stable parameter packing/unpacking, optimizer diagnostics, and
comparison against independently generated scalar and multivariate references.
Only after that route is stable should full covariance, constrained
stationarity/invertibility, smoothing, and integrated-seasonal wrappers be
added.
