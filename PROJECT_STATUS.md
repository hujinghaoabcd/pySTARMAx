# pySTARMAx project status

Updated: 2026-07-29

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 was squash-merged into `main` as commit `795c7e45`.
- PR #2 was squash-merged into `main` as commit `689f9927`.
- PR #3 was squash-merged into `main` as commit `d89fa7ce`.
- PR #4 was squash-merged into `main` as commit `6bd29a63`.
- Current development target: version `0.0.5`.
- Current stage: original-scale fitted values and conditional forecast intervals.
- Formal implementation surface: 19 changed files with no temporary workflow or payload files.
- GitHub-hosted runners are available again; the first real matrix run exposed and
  fixed a one-token `nspatial`/`n_spatial` typo in seasonal parameter splitting.

## Completed baseline

- repository metadata, MIT licence, citation file, contribution and security policies;
- `src/` package layout and typed public API;
- immutable spatial-weight collections and constructors;
- STAR ordinary least squares and STARMA conditional least squares;
- coefficient uncertainty, innovation covariance, likelihood, AIC, and BIC;
- recursive point forecasts and deterministic simulation;
- classical STACF, nested Yule–Walker STPACF, and residual portmanteau tests;
- exact-rational diagnostic reference fixture;
- ordinary STARIMA differencing and forecast inversion;
- multiplicative seasonal STARIMA with constrained matrix-polynomial factors;
- ordinary-seasonal simulation, documentation, packaging, and multi-platform CI.

## Completed in the current step

- added `differencing_coefficients()` for `(1-B)^d(1-B^s)^D`;
- added `restore_fitted_values()` for aligned one-step original-scale fits;
- added `STARIMA.fitted_original()` and `SeasonalSTARIMA.fitted_original()`;
- added immutable `ForecastInterval` with mean, bounds, level, and simulation count;
- added conditional innovation simulation to STAR/STARMA;
- added original-scale interval propagation to ordinary STARIMA;
- added pathwise ordinary-seasonal inverse differencing to seasonal STARIMA;
- retained cross-horizon dependence before empirical quantiles are computed;
- stabilized fitted covariance simulation by symmetric eigenvalue clipping;
- documented that intervals condition on estimated parameters;
- increased the local suite from 45 to 52 passing tests;
- retained approximately 91.3% local branch coverage;
- fixed seasonal AR parameter reshaping to consistently use `n_spatial`.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep numerical methods independent and auditable.
3. Separate data transforms, weights, estimation, diagnostics, forecasting, simulation, and results.
4. Validate dimensions and assumptions before numerical work.
5. Treat conditional estimation as a baseline, not exact likelihood.
6. Add reference fixtures before claiming cross-language equivalence.
7. Make matrix orientation explicit whenever non-symmetric weights matter.
8. Keep transformed-scale inference distinct from original-scale reconstruction.
9. Use observed history for one-step fitted-value inversion.
10. Propagate full simulated paths before inverse-differencing interval quantiles.
11. Do not label innovation-only intervals as parameter-uncertainty intervals.
12. Keep future sparse/state-space acceleration behind stable interfaces.

## Immediate next tasks

1. Add parameter-uncertainty intervals by bootstrap or asymptotic draws.
2. Implement exact state-space/Kalman maximum likelihood.
3. Support missing observations.
4. Add diagonal and full innovation covariance likelihoods.
5. Add stationarity and invertibility checks with optional constrained parameterization.
6. Add sparse matrices and NetworkX/libpysal adapters.
7. Add automatic order selection and rolling-origin evaluation.
8. Add exogenous regressors and interventions.
9. Add cross-language estimator fixtures.
10. Prepare the first PyPI pre-release after validation.

## Known limitations

- MA estimation is conditional and uses recursively estimated innovations.
- Seasonal nonlinear estimation uses zero pre-sample innovations.
- Stationarity and invertibility constraints are not imposed during optimization.
- Seasonal standard errors use a local nonlinear least-squares Jacobian.
- Information criteria use a scalar innovation-variance approximation.
- `STARMAResult` for integrated models remains on the transformed scale.
- `fitted_original()` is a one-step reconstruction, not a recursively integrated trajectory.
- forecast intervals include future innovation uncertainty but not parameter or model-order uncertainty.
- dense matrices are used throughout.
- no missing-value handling, exogenous regressors, or interventions yet.

## Handoff instruction

Before every substantial development step, read this file, `docs/model.md`, and
`docs/forecasting.md`. After completing a step, update the completed, next-task,
and limitation sections so another conversation can continue without rebuilding
the project history.
