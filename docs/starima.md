# Ordinary STARIMA

pySTARMAx 0.0.5 supports ordinary temporal integration while keeping the spatial
weight system and STARMA estimator unchanged. For integration order `d`, the
model is fitted to

\[
w_t = (1-B)^d z_t,
\]

where `B` is the one-step temporal backshift operator. The transformed process
`w_t` follows the same STARMA convention documented in
[Model convention](model.md). Differencing is applied independently along the
time axis for every location, without transforming the spatial weights.

## Fit and forecast

```python
import numpy as np
from pystarmax import STARIMA, SpatialWeights, lattice_weights

weights = SpatialWeights.from_adjacency(lattice_weights(2, 2))
model = STARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
)
result = model.fit(observations, weights)

# Result arrays and residual diagnostics are on the differenced scale.
print(result.summary())

# Public forecasts are restored to the original observation scale.
forecast = model.predict(steps=6)

# One-step fitted values are aligned to the original observation matrix.
fitted = model.fitted_original()

# Conditional innovation uncertainty is propagated through integration.
interval = model.predict_interval(steps=6, random_state=42)

# The stationary-scale forecast remains available explicitly.
difference_forecast = model.predict_differenced(steps=6)
```

`STARIMA.fit()` returns the existing `STARMAResult`. Its coefficients,
fitted values, residuals, likelihood, AIC, and BIC describe the differenced
process. `STARIMA.predict()` integrates future differences recursively and
returns original-scale observations. `fitted_original()` reconstructs aligned
one-step fitted values with observed historical anchors. `predict_interval()`
inverts complete simulated paths before empirical quantiles are computed.

## Differencing state

`ordinary_difference()` returns both the transformed observations and an
immutable end-of-sample state:

```python
from pystarmax import ordinary_difference

transformed, state = ordinary_difference(observations, order=2)
restored_forecast = state.inverse_forecast(predicted_second_differences)
```

For order `d`, the state stores the final observed value at each lower-order
level: the original series, first difference, and so on through order `d-1`.
Forecast inversion updates those levels from highest to lowest for every future
time step. The state is copied internally, so repeated inversion calls are
deterministic.

## Intercept interpretation

The underlying STARMA intercept is estimated on the differenced scale. With
`d=1`, a non-zero intercept therefore acts as a drift term on the original
scale. pySTARMAx does not silently remove or reinterpret this term.

## Simulation

`simulate_starima()` first simulates the stationary highest-difference process
and then integrates it:

```python
from pystarmax import simulate_starima

series = simulate_starima(
    phi=np.array([[0.4, 0.1]]),
    theta=np.zeros((0, 2)),
    weights=weights,
    n_steps=300,
    integration_order=1,
    random_state=42,
)
```

By default, all lower-order levels are anchored at zero immediately before the
returned sample. A custom `DifferencingState` can supply alternative starting
levels.

## Current limits

- coefficient inference and the stored `STARMAResult` remain on the differenced scale;
- forecast intervals condition on estimated parameters and do not include parameter uncertainty;
- state-space likelihood and missing observations remain future work.
