# pySTARMAx

pySTARMAx provides transparent, typed building blocks for classical STARMA,
ordinary STARIMA, and multiplicative seasonal STARIMA modelling in Python. The
current workflow covers spatial-weight construction, simulation, conditional
model fitting, diagnostics, temporal differencing, original-scale forecast
inversion, aligned one-step fitted values, conditional and bootstrap intervals,
rolling-origin calibration diagnostics, and fixed-parameter Gaussian Kalman
filtering with partial missing observations.

```python
import numpy as np

from pystarmax import STARMA, rolling_origin_evaluate

model = STARMA(ar_order=1, ma_order=1)
result = model.fit(y, weights)
print(result.summary())
print(model.predict(steps=6))
print(model.predict_interval(steps=6, random_state=42))

incomplete = y.copy()
incomplete[20, 1] = np.nan
incomplete[80, :] = np.nan
filtered = model.filter_state_space(incomplete)
print(filtered.log_likelihood)
print(filtered.filtered_observations[-1])

evaluation = rolling_origin_evaluate(
    lambda: STARMA(ar_order=1, ma_order=1),
    y,
    weights,
    initial_window=120,
    horizon=6,
    step=6,
    interval_kwargs={"n_simulations": 500},
    random_state=42,
)
print(evaluation.metrics())
```

See [Model convention](model.md) for the STARMA equation and data orientation,
[Ordinary STARIMA](starima.md) plus [Seasonal STARIMA](seasonal.md) for
differencing and multiplicative factors, [State-space filtering](state_space.md)
for Kalman likelihood and missing observations, [Forecasting](forecasting.md) for
conditional innovation intervals, [Bootstrap intervals](bootstrap.md) for
parameter-aware predictive inference, and [Rolling evaluation](evaluation.md)
for coverage, width, interval scores, MAE, and RMSE.
