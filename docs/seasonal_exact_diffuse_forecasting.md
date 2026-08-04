# Seasonal exact-diffuse forecasting

Version 0.0.30 extends `SeasonalExactDiffuseKalmanSTARIMA` with fixed-parameter
forecast paths and simulation intervals on both the original and combined
ordinary-seasonal transformed scales.

## Fitted-model interface

```python
original_paths = model.simulate_forecast_paths(
    steps=12,
    n_simulations=5000,
    random_state=7,
)
transformed_paths = model.simulate_differenced_forecast_paths(
    steps=12,
    n_simulations=5000,
    random_state=7,
)
original_interval = model.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=7,
)
transformed_interval = model.predict_differenced_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=7,
)
```

## Uncertainty scope

The terminal finite filtered state covariance and future fitted innovation
covariance are simulated. Fitted-parameter uncertainty is not included.

Forecasting requires a proper terminal posterior. If the exact-diffuse filter
retains a nonzero final diffuse rank, path and interval functions raise an
error rather than substituting a large finite covariance.

## Pathwise level restoration

For

\[
x_t=(1-B)^d(1-B^s)^D y_t,
\]

the exact seasonal integrated state contains the lag coordinates required to
invert the complete polynomial. Every simulated augmented state path therefore
produces its own original-level path. Original-level quantiles are computed only
after this pathwise restoration; transformed-scale quantiles are obtained by
projecting the stationary state block from the same simulation contract.

A complete mathematical reference, validation record, and functional API table
will be added after the implementation CI is clean.
