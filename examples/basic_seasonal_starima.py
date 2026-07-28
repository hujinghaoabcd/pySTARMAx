"""Fit a multiplicative seasonal STARIMA model to simulated data."""

import numpy as np

from pystarmax import (
    SeasonalSTARIMA,
    SpatialWeights,
    lattice_weights,
    simulate_seasonal_starima,
)

weights = SpatialWeights.from_adjacency(lattice_weights(2, 2), max_order=1)
series = simulate_seasonal_starima(
    phi=np.array([[0.25, 0.05]]),
    seasonal_phi=np.array([[0.30, 0.04]]),
    theta=np.zeros((0, 2)),
    seasonal_theta=np.zeros((0, 2)),
    weights=weights,
    seasonal_period=12,
    n_steps=500,
    integration_order=1,
    seasonal_integration_order=1,
    random_state=42,
)

model = SeasonalSTARIMA(
    ar_order=1,
    integration_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_period=12,
    include_intercept=False,
)
result = model.fit(series, weights)

print(result.summary())
print(model.predict(steps=12))
