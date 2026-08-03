"""Estimate a stationary STARMA model by Gaussian Kalman likelihood."""

from __future__ import annotations

import numpy as np

from pystarmax import (
    KalmanSTARMA,
    SpatialWeights,
    lattice_weights,
    simulate_starma,
)

weights = SpatialWeights.from_adjacency(
    lattice_weights(2, 2),
    max_order=1,
)

series = simulate_starma(
    phi=np.array([[0.45, 0.15]], dtype=float),
    theta=np.array([[0.20, 0.05]], dtype=float),
    weights=weights,
    n_steps=320,
    innovation_covariance=np.array(
        [
            [1.0, 0.20, 0.00, 0.00],
            [0.20, 1.2, 0.10, 0.00],
            [0.00, 0.10, 0.8, 0.15],
            [0.00, 0.00, 0.15, 1.1],
        ],
        dtype=float,
    ),
    random_state=42,
)

incomplete = series.copy()
incomplete[20:50:3, 1] = np.nan
incomplete[120, :] = np.nan
incomplete[210:215, 3] = np.nan

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=False,
    max_iter=500,
)
result = model.fit(incomplete, weights)

print(result.summary())
print("\nSix-step conditional-mean forecast:")
print(model.predict(steps=6))
