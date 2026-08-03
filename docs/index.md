# pySTARMAx

pySTARMAx provides transparent, typed building blocks for classical STARMA,
ordinary STARIMA, and multiplicative seasonal STARIMA modelling in Python. The
current workflow covers spatial-weight construction, simulation, conditional and
Gaussian Kalman maximum-likelihood estimation, diagnostics, temporal
differencing, original-scale forecast inversion, conditional and bootstrap
intervals, rolling-origin calibration diagnostics, and missing-observation
state-space filtering.

```python
import numpy as np

from pystarmax import KalmanSTARMA, STARMA, rolling_origin_evaluate

conditional = STARMA(ar_order=1, ma_order=1)
conditional_result = conditional.fit(y, weights)
print(conditional_result.summary())

incomplete = y.copy()
incomplete[20, 1] = np.nan
incomplete[80, :] = np.nan

mle = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
)
mle_result = mle.fit(incomplete, weights)
print(mle_result.summary())
print(mle.predict(steps=6))

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
for fixed-parameter filtering, [Maximum likelihood](maximum_likelihood.md) for
direct Gaussian Kalman estimation, [Forecasting](forecasting.md) and
[Bootstrap intervals](bootstrap.md) for predictive uncertainty, and
[Rolling evaluation](evaluation.md) for calibration and sharpness diagnostics.
