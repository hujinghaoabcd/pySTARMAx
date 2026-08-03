# Bootstrap forecast intervals

Version 0.0.6 adds direct-bootstrap predictive inference for `STAR`, `STARMA`,
`STARIMA`, and `SeasonalSTARIMA` through `predict_bootstrap_interval()`.

## Why a separate API?

`predict_interval()` conditions on the fitted coefficients and simulates only
future innovations. `predict_bootstrap_interval()` regenerates and refits the
historical sample, so parameter-estimation variation enters the forecast
distribution.

The direct-bootstrap workflow is:

1. obtain fitted innovations;
2. generate innovations for a same-length pseudo-series;
3. retain the initial observations and innovation history required by the model;
4. recursively generate the pseudo-series under the fitted coefficients;
5. refit the same model specification;
6. record the refitted conditional forecast, or simulate one future path;
7. repeat until the requested number of successful replications is reached;
8. compute central empirical quantiles on the original observation scale.

## Basic usage

```python
interval = model.predict_bootstrap_interval(
    steps=12,
    level=0.95,
    n_bootstrap=500,
    bootstrap_method="residual",
    include_future_innovations=True,
    random_state=42,
)
```

The returned object is the same immutable `ForecastInterval` used by
`predict_interval()`. Its `mean` remains the deterministic point forecast from
the model fitted to the observed data. Its bounds are empirical quantiles of
successful bootstrap forecast paths.

## Innovation methods

### Residual bootstrap

The residual method centers complete fitted-innovation rows by location and
samples whole rows with replacement. Sampling a complete vector rather than
individual cells preserves contemporaneous dependence between locations.

This method assumes the innovation vectors are exchangeable over time. It does
not preserve remaining serial dependence or conditional heteroskedasticity.

### Parametric bootstrap

The parametric method draws Gaussian innovation vectors from the fitted
contemporaneous covariance matrix. The covariance is symmetrized and small
negative eigenvalues attributable to floating-point error are clipped by the
same numerical policy used by conditional innovation intervals.

```python
interval = model.predict_bootstrap_interval(
    steps=12,
    n_bootstrap=500,
    bootstrap_method="parametric",
    random_state=42,
)
```

## Parameter-only and full predictive intervals

With `include_future_innovations=False`, every accepted replication contributes
the conditional mean from the refitted model. The interval therefore represents
parameter-estimation variation under the selected bootstrap scheme.

```python
parameter_only = model.predict_bootstrap_interval(
    steps=12,
    n_bootstrap=500,
    include_future_innovations=False,
    random_state=42,
)
```

With the default `include_future_innovations=True`, each refitted model
contributes one future path generated with a new innovation sequence. The
resulting interval combines parameter-estimation and future-innovation
uncertainty.

## Integrated models

For `STARIMA`, bootstrap pseudo-data are generated on the ordinary differenced
scale and recursively restored to the original scale using the observed initial
conditions. The restored pseudo-series is then refitted by the full `STARIMA`
workflow.

For `SeasonalSTARIMA`, the same procedure uses the complete combined polynomial

\[
(1-B)^d(1-B^s)^D.
\]

The implementation recursively reconstructs each pseudo-series from its own
pseudo-history. It does not repeatedly substitute observed lagged values after
the initial rows. Future paths are also inverse-differenced separately after
each refit, so dependence across forecast horizons is retained.

## Failed refits and convergence

Bootstrap estimation can fail for numerically difficult pseudo-samples. The API
therefore distinguishes requested successful replications from attempted
replications:

```python
interval = model.predict_bootstrap_interval(
    steps=12,
    n_bootstrap=500,
    max_attempts=1500,
    require_convergence=True,
    random_state=42,
)
```

The method never silently returns fewer paths than requested. If the successful
count is still below `n_bootstrap` after `max_attempts`, it raises a
`RuntimeError` reporting the accepted and attempted counts.

`require_convergence=True` rejects nonlinear or iterative fits whose result does
not report convergence. Setting it to `False` can be useful for controlled
sensitivity analysis but should be disclosed in research reporting.

## Reproducibility

An integer `random_state` reproduces the complete pseudo-sample, refit, and
future-path sequence. A NumPy `Generator` is consumed in place.

## Interpretation and limitations

The current direct bootstrap keeps the model order and spatial weights fixed.
It adds parameter-estimation variation conditional on that specification; it
does not integrate uncertainty from order selection or weight construction.

Current limitations are:

- residual innovation vectors are sampled independently over time;
- no block, wild, robust, predictive-residual, studentized, or bias-corrected
  bootstrap is implemented yet;
- bootstrap refits execute serially;
- finite-sample coverage calibration requires external rolling-origin or Monte
  Carlo evaluation;
- the underlying conditional-estimation assumptions remain unchanged.

The next bootstrap priorities are rolling-origin interval scoring, empirical
coverage examples, and optional parallel execution behind the same API.
