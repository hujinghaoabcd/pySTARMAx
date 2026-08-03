# pySTARMAx project status

Updated: 2026-08-03

## Purpose

Build a modern Python implementation of classical and extended STARMA models,
using the transparent statistical workflow of the original literature and the
engineering conventions established in pyGWRx and pyKDEX.

## Repository state

- PR #1 was squash-merged into `main` as commit `795c7e45`.
- PR #2 was squash-merged into `main` as commit `689f9927`.
- PR #3 was squash-merged into `main` as commit `d89fa7ce`.
- PR #4 was squash-merged into `main` as commit `6bd29a63`.
- PR #5 was squash-merged into `main` as commit `cc1e8ac7`.
- PR #6 was squash-merged into `main` as commit `cd7ac0c9` after full quality,
  coverage, distribution, and Ubuntu/Windows/macOS validation on Python 3.11–3.14.
- Current branch: `agent/rolling-origin-interval-evaluation`.
- Current draft pull request: PR #7, `Add rolling-origin interval evaluation`.
- Current development target: version `0.0.7`.
- Current stage: out-of-sample interval calibration and sharpness diagnostics.
- Authoritative validation: GitHub Actions CI run #155 completed successfully.

## Completed baseline through 0.0.6

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
- aligned original-scale one-step fitted values;
- conditional future-innovation intervals with pathwise inverse differencing;
- residual and Gaussian parametric bootstrap intervals with model refitting;
- parameter-only and full predictive bootstrap uncertainty;
- packaging, strict documentation, and multi-platform CI.

## Completed in the current 0.0.7 step

- added `pystarmax.evaluation` without modifying model estimation cores;
- added the central Winkler `interval_score()` for arbitrary finite array shapes;
- added immutable `IntervalMetrics` with nominal and empirical coverage, signed
  coverage gap, absolute coverage error, average width, mean interval score,
  point-forecast MAE, RMSE, and forecast count;
- added immutable `RollingOriginResult` with read-only arrays shaped
  `(origins, horizon, locations)`;
- added derived coverage, width, and interval-score arrays;
- added pooled and horizon-specific metric summaries;
- added expanding-window rolling-origin evaluation;
- added fixed-length rolling training windows through `window_size`;
- added conditional and bootstrap interval dispatch through one public function;
- added deterministic per-origin seeds derived from one NumPy generator;
- prevented interval keyword arguments from overriding evaluator-controlled
  `steps`, `level`, and `random_state`;
- added focused tests for scores, validation, aggregation, dispatch, windows,
  reproducibility, and unsupported model interfaces;
- added a runnable rolling-origin example and complete evaluation documentation;
- updated package exports, documentation navigation, and citation metadata to 0.0.7.

## Statistical interpretation

`rolling_origin_evaluate()` performs genuine out-of-sample evaluation. At each
origin it constructs a new model and fits only the observations available before
that origin. It supports either all prior observations or a fixed-length recent
window.

Empirical coverage is interpreted jointly with average width and the central
interval score. High coverage alone is not sufficient evidence of a useful
interval because arbitrarily wide endpoints can cover nearly all observations.
MAE and RMSE remain point-forecast diagnostics; interval score and coverage assess
the probabilistic output.

Metrics can be pooled across all horizons and locations or reported separately
by forecast horizon. Version 0.0.7 diagnoses calibration but does not
conformalize, rescale, or otherwise alter interval endpoints.

## Design principles

1. Preserve one explicit `(time, location)` convention.
2. Keep numerical methods independent and auditable.
3. Separate transforms, weights, estimation, diagnostics, forecasting,
   bootstrap, evaluation, simulation, and results.
4. Validate dimensions and assumptions before numerical work.
5. Treat conditional estimation as a baseline, not exact likelihood.
6. Add reference fixtures before claiming cross-language equivalence.
7. Make matrix orientation explicit whenever non-symmetric weights matter.
8. Keep transformed-scale inference distinct from original-scale reconstruction.
9. Use fresh estimators and origin-limited histories for out-of-sample evaluation.
10. Report calibration and sharpness together.
11. Preserve complete stochastic paths before inverse-differencing quantiles.
12. Keep origin-specific random streams reproducible without reusing one seed.
13. Do not hide failed bootstrap replications or incomplete forecast horizons.
14. Keep future sparse, parallel, and state-space acceleration behind stable APIs.

## Final validation for 0.0.7

GitHub Actions CI run #155 completed successfully on the documented branch head:

- 69 tests passed;
- total branch coverage: 89.33%, above the configured 80% threshold;
- Black passed;
- isort passed;
- Ruff passed;
- mypy passed;
- exact diagnostic fixture regeneration produced a clean diff;
- strict MkDocs build passed;
- source distribution and wheel built successfully;
- Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

## Immediate next tasks

1. Keep PR #7 as draft until explicitly requested for review or merge.
2. Begin exact state-space/Kalman likelihood and missing-observation support.
3. Add diagonal and full contemporaneous innovation covariance likelihoods.
4. Add stationarity and invertibility checks with optional constrained fitting.
5. Add sparse spatial matrices and large-network computation.
6. Add automatic order selection and exogenous regressors/interventions.
7. Add NetworkX/libpysal/GeoPandas/OSMnx adapters.
8. Add cross-language estimator fixtures and prepare the first PyPI pre-release.
9. Add parallel execution and advanced bootstrap schemes as optional extensions.
10. Add calibrated or conformal interval post-processing after benchmark evidence.

## Remaining release estimate

- A mature first PyPI pre-release is expected after approximately five additional
  stages following 0.0.7: state-space/missing data; covariance and constraints;
  sparse/ecosystem support; selection/exogenous inputs; and reference/release
  hardening.
- Completing most advanced roadmap extensions is expected to require roughly
  eight to ten additional stages after 0.0.7, depending on whether parallel,
  advanced bootstrap, calibration, and generalized time-varying models are split.

## Known limitations

- MA estimation is conditional and uses recursively estimated innovations;
- seasonal nonlinear estimation uses zero pre-sample innovations;
- stationarity and invertibility constraints are not imposed during optimization;
- seasonal standard errors use a local nonlinear least-squares Jacobian;
- information criteria use a scalar innovation-variance approximation;
- residual bootstrap assumes complete innovation vectors are exchangeable over time;
- bootstrap and rolling-origin refits execute serially;
- no block, wild, predictive-residual, studentized, or bias-corrected bootstrap;
- no automatic interval recalibration or uncertainty for empirical score means;
- overlapping forecast origins can produce dependent forecast errors;
- model order and spatial weights remain fixed across bootstrap replications;
- `STARMAResult` for integrated models remains on the transformed scale;
- dense matrices are used throughout;
- missing values, exogenous regressors, and interventions are unsupported.

## Handoff instruction

Before every substantial development step, read this file, `docs/model.md`,
`docs/forecasting.md`, `docs/bootstrap.md`, `docs/evaluation.md`, and the latest
file under `docs/development/`. After completing a step, update the completed,
validation, next-task, estimate, and limitation sections so another conversation
can continue without reconstructing project history.
