# pySTARMAx project status

Updated: 2026-07-28

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

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
- STACF, regression-based STPACF, and residual portmanteau test;
- unit tests, example, MkDocs pages, and multi-platform CI.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep numerical methods independent and auditable.
3. Separate data/weights, estimation, diagnostics, simulation, and results.
4. Validate dimensions and assumptions before numerical work.
5. Treat current conditional estimation as a baseline, not exact likelihood.
6. Add reference fixtures before claiming cross-language equivalence.
7. Keep future sparse/state-space acceleration behind stable interfaces.

## Immediate next tasks

1. Add static numerical fixtures against R `starma` for STAR and STACF.
2. Implement true Yule-Walker STPACF.
3. Implement STARIMA differencing and forecast inversion.
4. Add seasonal STARIMA operators.
5. Implement state-space/Kalman maximum likelihood.
6. Support missing observations.
7. Add diagonal/full innovation covariance estimation and tests.
8. Add sparse matrices and NetworkX/libpysal adapters.
9. Add automatic order selection and rolling-origin evaluation.
10. Publish API documentation and prepare the first PyPI pre-release.

## Known limitations

- MA estimation is conditional and uses estimated innovations.
- Standard errors for MA models are approximate conditional OLS values.
- The likelihood currently uses a scalar innovation variance for information criteria.
- STPACF is regression-based rather than the final Yule-Walker implementation.
- Dense matrices are used throughout.
- No missing-value handling, seasonal differencing, exogenous regressors, or interventions yet.

## Handoff instruction

Before every substantial development step, read this file and the model
convention in `docs/model.md`. After completing a step, update the completed,
next-task, and limitation sections so another conversation can continue without
reconstructing the project history.
