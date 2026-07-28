# pySTARMAx project status

Updated: 2026-07-28

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 was squash-merged into `main` as commit `795c7e45`.
- PR #2 was squash-merged into `main` as commit `689f9927`.
- PR #3 was squash-merged into `main` as commit `d89fa7ce`.
- Current development branch: `agent/seasonal-starima`.
- Current draft pull request: PR #4.
- Package version under development: `0.0.4`.

## Completed baseline

- repository metadata, MIT licence, citation file, contribution and security policies;
- `src/` package layout and typed public API;
- immutable spatial-weight collection;
- lattice, distance, row-standardized, and higher-order weights;
- deterministic STARMA simulation;
- STAR ordinary least-squares estimator;
- STARMA iterative conditional least-squares estimator;
- coefficient uncertainty, residual covariance, log likelihood, AIC, and BIC;
- recursive multi-step forecast;
- classical STACF, nested Yule–Walker STPACF, and residual portmanteau test;
- exact-rational diagnostic reference fixture;
- ordinary STARIMA differencing, simulation, and original-scale forecasting;
- unit tests, examples, MkDocs pages, packaging, and multi-platform CI.

## Completed in the current step

- added `seasonal_difference()` for `(1-B^s)^D` temporal differencing;
- added immutable `SeasonalDifferencingState` rolling seasonal histories;
- added `combined_difference()` and `CombinedDifferencingState` for `(d,D,s)`;
- added public `LagOperator` and `expand_multiplicative_operators()` auditing;
- fixed the matrix-polynomial convention as seasonal factor left of ordinary factor;
- retained non-commuting cross terms as ordered matrix products without basis projection;
- added `SeasonalSTARIMA(p,d,q)x(P,D,Q)_s`;
- used nonlinear conditional least squares so cross-lag coefficients remain factor products;
- delegated `P=Q=0` models to the existing STARMA core for numerical compatibility;
- added stationary and integrated seasonal simulation;
- added separate AR-factor and MA-factor recovery tests;
- increased the suite from 32 to 45 passing tests;
- retained about 90.8% local branch coverage.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep numerical methods independent and auditable.
3. Separate data transforms, weights, estimation, diagnostics, simulation, and results.
4. Validate dimensions and assumptions before numerical work.
5. Treat current conditional estimation as a baseline, not exact likelihood.
6. Add reference fixtures before claiming cross-language equivalence.
7. Make matrix orientation explicit whenever non-symmetric weights matter.
8. Keep transformed-scale inference distinct from original-scale reconstruction.
9. Do not label freely estimated cross lags as a multiplicative seasonal model.
10. Do not assume the spatial-weight basis is closed under matrix multiplication.
11. Keep future sparse/state-space acceleration behind stable interfaces.

## Immediate next tasks

1. Reconstruct in-sample fitted values on the original scale.
2. Add forecast intervals and propagate uncertainty through ordinary-seasonal integration.
3. Implement state-space/Kalman maximum likelihood.
4. Support missing observations.
5. Add diagonal/full innovation covariance estimation and tests.
6. Add sparse matrices and NetworkX/libpysal adapters.
7. Add automatic order selection and rolling-origin evaluation.
8. Add exogenous regressors and interventions.
9. Add stationarity/invertibility diagnostics and optional constrained parameterization.
10. Prepare the first PyPI pre-release after seasonal STARIMA is merged.

## Known limitations

- MA estimation is conditional and uses recursively estimated innovations.
- Seasonal nonlinear estimation uses zero pre-sample innovations.
- Stationarity and invertibility constraints are not imposed during optimization.
- Standard errors use a local nonlinear least-squares Jacobian.
- The likelihood currently uses a scalar innovation variance for information criteria.
- STARIMA fitted values, residuals, likelihood, AIC, and BIC remain on the transformed scale.
- Forecast uncertainty is not yet propagated through inverse differencing.
- Dense matrices are used throughout.
- No missing-value handling, exogenous regressors, or interventions yet.
- The exact-rational fixture validates diagnostics; seasonal estimator fixtures remain later work.

## Handoff instruction

Before every substantial development step, read this file and the model
convention in `docs/model.md`. After completing a step, update the completed,
next-task, and limitation sections so another conversation can continue without
reconstructing the project history.
