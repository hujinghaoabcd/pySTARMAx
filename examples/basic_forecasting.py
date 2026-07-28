"""Original-scale fitted values and forecast intervals."""

import numpy as np

from pystarmax import STARIMA, SpatialWeights, simulate_starima

weights = SpatialWeights.from_matrices([np.eye(2)], names=["W0"])
series = simulate_starima(
    phi=np.array([[0.35]]),
    theta=np.array([[0.15]]),
    weights=weights,
    n_steps=300,
    integration_order=1,
    random_state=42,
)

model = STARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    include_intercept=False,
)
model.fit(series, weights)

print(model.fitted_original()[-5:])
interval = model.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=2000,
    random_state=42,
)
print(interval.mean)
print(interval.lower)
print(interval.upper)
