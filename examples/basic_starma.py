"""Minimal end-to-end STARMA example."""

import numpy as np

from pystarmax import STARMA, SpatialWeights, lattice_weights, simulate_starma

weights = SpatialWeights.from_adjacency(lattice_weights(2, 3))
data = simulate_starma(
    phi=np.array([[0.45, 0.20]]),
    theta=np.array([[0.15, 0.05]]),
    weights=weights,
    n_steps=300,
    random_state=42,
)
model = STARMA(ar_order=1, ma_order=1)
result = model.fit(data, weights)
print(result.summary())
print("Forecasts:\n", model.predict(steps=6))
