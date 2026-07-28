# pySTARMAx project status

Updated: 2026-07-28

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 was squash-merged into `main` as commit `795c7e45`.
- Current development branch: `agent/classic-stpacf-reference`.
- Package version under development: `0.0.2`.

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
- residual portmanteau test;
- unit tests, examples, MkDocs pages, packaging, and multi-platform CI.

## Completed in the current step

- added public `stcov` with explicit past/future spatial-lag orientation;
- corrected STACF orientation for non-symmetric row-standardized weights;
- implemented the classical nested Yule–Walker STPACF;
- retained the former regression diagnostic as `stpacf_regression` and
  `stpacf(..., method="regression")`;
- added `auto`, `solve`, and `lstsq` policies for singular Yule–Walker systems;
- added an exact-rational fixture generated without importing pySTARMAx;
- added a deliberately non-symmetric reference weight to expose transpose errors;
- increased local tests from 15 to 22 and local branch coverage to about 90%.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep numerical methods independent and auditable.
3. Separate data/weights, estimation, diagnostics, simulation, and results.
4. Validate dimensions and assumptions before numerical work.
5. Treat current conditional estimation as a baseline, not exact likelihood.
6. Add reference fixtures before claiming cross-language equivalence.
7. Make matrix orientation explicit whenever non-symmetric weights matter.
8. Keep future sparse/state-space acceleration behind stable interfaces.

## Immediate next tasks

1. Implement a reversible differencing operator for ordinary STARIMA.
2. Add forecast inversion with stored initial conditions.
3. Extend simulation and tests to integrated processes.
4. Add seasonal differencing and multiplicative seasonal lag specifications.
5. Implement state-space/Kalman maximum likelihood.
6. Support missing observations.
7. Add diagonal/full innovation covariance estimation and tests.
8. Add sparse matrices and NetworkX/libpysal adapters.
9. Add automatic order selection and rolling-origin evaluation.
10. Prepare the first PyPI pre-release after the STARIMA layer is validated.

## Known limitations

- MA estimation is conditional and uses estimated innovations.
- Standard errors for MA models are approximate conditional OLS values.
- The likelihood currently uses a scalar innovation variance for information criteria.
- Dense matrices are used throughout.
- No missing-value handling, differencing, seasonal operators, exogenous regressors,
  or interventions yet.
- The exact-rational fixture validates diagnostics; estimator cross-language
  fixtures remain a later task.

## Handoff instruction

Before every substantial development step, read this file and the model
convention in `docs/model.md`. After completing a step, update the completed,
next-task, and limitation sections so another conversation can continue without
reconstructing the project history.
