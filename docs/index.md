# pySTARMAx

pySTARMAx provides transparent, typed building blocks for classical STARMA and
ordinary STARIMA modelling in Python. The current workflow covers spatial-weight
construction, simulation, model fitting, diagnostics, temporal differencing,
original-scale forecast inversion, and reproducible numerical validation.

```python
from pystarmax import STARIMA

model = STARIMA(ar_order=1, integration_order=1, ma_order=1)
result = model.fit(y, weights)
print(result.summary())
print(model.predict(steps=6))
```

See [Model convention](model.md) for the STARMA equation and data orientation,
and [Ordinary STARIMA](starima.md) for differencing and forecast inversion.
