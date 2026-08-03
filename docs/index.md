# pySTARMAx

pySTARMAx provides transparent, typed building blocks for classical STARMA,
ordinary STARIMA, and multiplicative seasonal STARIMA modelling in Python. The
current workflow covers spatial-weight construction, simulation, model fitting,
diagnostics, temporal differencing, original-scale forecast inversion, aligned
one-step fitted values, conditional innovation intervals, parameter-aware
bootstrap intervals, constrained seasonal matrix-polynomial expansion, and
reproducible numerical validation.

```python
from pystarmax import SeasonalSTARIMA

model = SeasonalSTARIMA(
    ar_order=1,
    integration_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_period=12,
)
result = model.fit(y, weights)
print(result.summary())
print(model.predict(steps=6))
print(model.fitted_original())
print(model.predict_interval(steps=6, random_state=42))
print(
    model.predict_bootstrap_interval(
        steps=6,
        n_bootstrap=200,
        random_state=42,
    )
)
```

See [Model convention](model.md) for the STARMA equation and data orientation,
[Ordinary STARIMA](starima.md) plus [Seasonal STARIMA](seasonal.md) for
differencing and multiplicative factors, [Forecasting](forecasting.md) for
conditional innovation intervals, and [Bootstrap intervals](bootstrap.md) for
parameter-aware predictive inference.
