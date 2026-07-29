# Step 06 handoff: bootstrap predictive inference

Updated: 2026-07-29

## Goal

Extend the conditional future-innovation intervals introduced in 0.0.5 with a
separate direct-bootstrap API that propagates parameter-estimation variation by
generating and refitting same-length pseudo-samples.

## Public API

All public model families expose:

```python
model.predict_bootstrap_interval(
    steps=1,
    level=0.95,
    n_bootstrap=200,
    bootstrap_method="residual",
    include_future_innovations=True,
    require_convergence=True,
    max_attempts=None,
    random_state=None,
)
```

The return type remains `ForecastInterval`.

## Algorithm implemented

For each accepted replication:

1. extract the fitted innovation history;
2. draw a same-length innovation sequence by residual-vector resampling or from
   the fitted Gaussian covariance;
3. retain the initial data and innovation rows required by the maximum model lag;
4. recursively generate a transformed-scale pseudo-series under the fitted model;
5. for integrated models, recursively restore the original-scale pseudo-series
   from its own pseudo-history using `(1-B)^d(1-B^s)^D`;
6. instantiate a new estimator with the same orders and numerical settings;
7. refit the pseudo-series using the original spatial weights;
8. reject a non-converged result when `require_convergence=True`;
9. record either the refitted conditional mean or one refitted future path;
10. continue until exactly `n_bootstrap` successful replications are available;
11. compute central empirical quantiles on the original scale.

## Innovation methods

### Residual

- remove rows with any non-finite location value;
- center each location column;
- sample complete rows with replacement;
- preserve contemporaneous dependence between locations;
- assume exchangeability over time.

### Parametric

- symmetrize the fitted covariance;
- reject materially indefinite matrices;
- clip numerical negative eigenvalues;
- draw Gaussian vectors through the stable covariance factor.

## File structure

- `src/pystarmax/bootstrap.py`
  - argument validation;
  - residual preprocessing;
  - residual/parametric innovation draws;
  - recursive ordinary-seasonal pseudo-series restoration.
- `src/pystarmax/models/bootstrap_starma.py`
  - stationary pseudo-series recursion;
  - stationary future paths;
  - public bootstrap-enabled `STAR` and `STARMA`.
- `src/pystarmax/models/bootstrap_starima.py`
  - ordinary integrated pseudo-series reconstruction and refitting;
  - public bootstrap-enabled `STARIMA`.
- `src/pystarmax/models/seasonal_bootstrap.py`
  - linear-core reuse;
  - multiplicative seasonal `LagOperator` pseudo-series and future recursion.
- `src/pystarmax/models/bootstrap_seasonal.py`
  - combined ordinary-seasonal reconstruction;
  - public bootstrap-enabled `SeasonalSTARIMA`.
- `tests/test_bootstrap.py`
  - nine focused tests across utilities and all public model families.
- `docs/bootstrap.md`
  - user-facing statistical interpretation and limitations.

The original estimator modules remain unchanged. Public exports point to thin
subclasses that add bootstrap methods, limiting regression risk in the fitting
core.

## Important conventions

- `ForecastInterval.mean` is always the deterministic point forecast from the
  model fitted to the observed sample.
- `include_future_innovations=False` creates a parameter-only interval from
  refitted conditional means.
- `include_future_innovations=True` combines parameter and future-innovation
  variation by recording one stochastic future path per refit.
- model order and spatial weights are fixed across replications.
- residual resampling operates on complete location vectors, never individual
  cells.
- pseudo-series restoration uses observed values only for required initial
  conditions; later values are recursively generated from pseudo-history.
- the method raises rather than silently returning fewer successful paths.

## Validation completed before final CI

- project-pinned Black and isort applied;
- Ruff passed;
- mypy passed;
- nine focused bootstrap tests passed;
- first PR matrix job passed on macOS/Python 3.11;
- source and wheel distributions built and passed Twine checks.

## Remaining work in this stage

1. trigger a user-authored final CI run after documentation changes;
2. verify Black, isort, Ruff, mypy, exact fixture regeneration, and strict MkDocs;
3. verify all 61 tests with branch coverage;
4. verify Ubuntu, Windows, and macOS on Python 3.11–3.14;
5. update `PROJECT_STATUS.md` and PR #6 with exact final results;
6. leave PR #6 as draft unless the user explicitly requests merge.

## Next scientific extensions

- rolling-origin interval scoring and empirical coverage calibration;
- parallel replication execution with deterministic seed partitioning;
- block bootstrap for residual serial dependence;
- wild bootstrap for heteroskedastic innovations;
- predictive-residual bootstrap;
- studentized, bias-corrected, or calibrated intervals;
- order-selection and spatial-weight uncertainty;
- state-space likelihood and missing-observation bootstrap.
