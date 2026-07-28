# pySTARMAx

pySTARMAx provides transparent, typed building blocks for classical STARMA
modelling in Python. The first baseline contains a complete simulation-to-fit-
diagnose-forecast workflow and deliberately keeps the numerical route easy to
audit.

```python
from pystarmax import STARMA

result = STARMA(ar_order=1, ma_order=1).fit(y, weights)
print(result.summary())
```

See [Model convention](model.md) for the exact equation and data orientation.
