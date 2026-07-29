# Step 05: original-scale fits and forecast intervals

Date: 2026-07-28  
Target version: 0.0.5

## Goal

Expose original-scale fitted values for integrated models and produce forecast
intervals that correctly propagate future innovations through STARMA dynamics
and ordinary-seasonal inverse differencing.

## Mathematical decisions

### One-step fitted values

For `C(B) z_t = w_t`, where

`C(B) = (1-B)^d (1-B^s)^D = 1 + c_1 B + ... + c_K B^K`,

the package reconstructs

`z_hat_t = w_hat_t - sum(c_k z_{t-k})`.

Observed lagged values are used. Recursively reconstructed fitted values are not
used because they would turn one-step fits into a drifting pseudo-trajectory.

### Forecast intervals

Future innovations are drawn jointly across locations from the fitted covariance.
Each path is propagated recursively through AR and MA terms. Integrated models
then invert each complete path before quantiles are computed. This preserves
cross-horizon dependence and avoids integrating lower and upper endpoints as if
they were independent paths.

### Uncertainty scope

The interval is conditional on fitted coefficients and weights. It includes
future innovation uncertainty only. Parameter, order-selection, and spatial-weight
uncertainty remain future work.

## New public API

- `ForecastInterval`
- `differencing_coefficients()`
- `restore_fitted_values()`
- `STARMA.predict_interval()`
- `STARIMA.fitted_original()`
- `STARIMA.predict_interval()`
- `SeasonalSTARIMA.fitted_original()`
- `SeasonalSTARIMA.predict_interval()`

## Validation

- 52 tests pass locally;
- branch coverage is approximately 91.3%;
- fixed-seed interval reproducibility is tested;
- exact pathwise quantiles are compared against manual inverse-differencing;
- linear STARMA, ordinary STARIMA, seasonal-difference delegation, and nonlinear
  multiplicative seasonal cores are covered;
- singular positive-semidefinite fitted covariance matrices are supported by
  eigenvalue clipping.

## Next development step

Add parameter-uncertainty intervals, preferably behind a method option that keeps
the current fast conditional-innovation interval as the default.
