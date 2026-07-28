"""Fit and forecast a small ordinary STARIMA model."""

import numpy as np

from pystarmax import STARIMA, SpatialWeights, lattice_weights, simulate_starima

weights = SpatialWeights.from_adjacency(lattice_weights(2, 2))
series = simulate_starima(
    phi=np.array([[0.40, 0.10]]),
    theta=np.zeros((0, 2)),
    weights=weights,
    n_steps=300,
    integration_order=1,
    random_state=42,
)

model = STARIMA(ar_order=1, integration_order=1, include_intercept=False)
result = model.fit(series, weights)

print(result.summary())
print(model.predict(steps=6))
