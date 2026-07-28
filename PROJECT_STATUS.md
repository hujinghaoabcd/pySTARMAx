# pySTARMAx project status

Updated: 2026-07-28

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 was squash-merged into `main` as commit `795c7e45`.
- PR #2 was squash-merged into `main` as commit `689f9927`.
- Current development branch: `agent/starima-differencing`.
- Current draft pull request: PR #3.
- Package version under development: `0.0.3`.

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
- unit tests, examples, MkDocs pages, packaging, and multi-platform CI.

## Completed in the current step

- added `ordinary_difference()` for `(1-B)^d` temporal differencing;
- added immutable `DifferencingState` end-of-sample anchors;
- implemented recursive forecast inversion for arbitrary ordinary order `d`;
- added compositional `STARIMA(p, d, q)` over the existing STARMA estimator;
- kept `STARMAResult` explicitly on the differenced scale;
- added `predict_differenced()` and original-scale `predict()`;
- added `simulate_starima()` with zero or user-supplied integration anchors;
- verified `d=0` produces the same parameters and forecasts as STARMA;
- increased the local suite from 22 to 32 passing tests;
- reached about 91.7% local branch coverage.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep numerical methods independent and auditable.
3. Separate data transforms, weights, estimation, diagnostics, simulation, and results.
4. Validate dimensions and assumptions before numerical work.
5. Treat current conditional estimation as a baseline, not exact likelihood.
6. Add reference fixtures before claiming cross-language equivalence.
7. Make matrix orientation explicit whenever non-symmetric weights matter.
8. Keep differenced-scale inference distinct from original-scale reconstruction.
9. Keep future sparse/state-space acceleration behind stable interfaces.

## Immediate next tasks

1. Add seasonal differencing and multiplicative seasonal lag specifications.
2. Reconstruct in-sample fitted values on the original scale.
3. Add forecast intervals and propagate uncertainty through integration.
4. Implement state-space/Kalman maximum likelihood.
5. Support missing observations.
6. Add diagonal/full innovation covariance estimation and tests.
7. Add sparse matrices and NetworkX/libpysal adapters.
8. Add automatic order selection and rolling-origin evaluation.
9. Add exogenous regressors and interventions.
10. Prepare the first PyPI pre-release after seasonal STARIMA is validated.

## Known limitations

- MA estimation is conditional and uses estimated innovations.
- Standard errors for MA models are approximate conditional OLS values.
- The likelihood currently uses a scalar innovation variance for information criteria.
- STARIMA fitted values, residuals, likelihood, AIC, and BIC are on the differenced scale.
- Forecast uncertainty is not yet propagated through inverse differencing.
- Dense matrices are used throughout.
- No missing-value handling, seasonal operators, exogenous regressors, or interventions yet.
- The exact-rational fixture validates diagnostics; estimator cross-language fixtures remain later work.

## Handoff instruction

Before every substantial development step, read this file and the model
convention in `docs/model.md`. After completing a step, update the completed,
next-task, and limitation sections so another conversation can continue without
reconstructing the project history.
