# pySTARMAx

pySTARMAx provides transparent, typed building blocks for classical STARMA, ordinary STARIMA, and multiplicative seasonal STARIMA modelling in Python. The current workflow covers spatial-weight
construction, simulation, model fitting, diagnostics, temporal differencing,
original-scale forecast inversion, constrained seasonal matrix-polynomial
expansion, and reproducible numerical validation.

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
```

See [Model convention](model.md) for the STARMA equation and data orientation,
and [Ordinary STARIMA](starima.md) plus [Seasonal STARIMA](seasonal.md) for
differencing, multiplicative factors, and forecast inversion.
