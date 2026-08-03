"""Evaluate fitted STARMA parameters with missing observations."""

from __future__ import annotations

import numpy as np

from pystarmax import (
    STARMA,
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
    n_steps=240,
    innovation_covariance=np.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.2, 1.0, 0.1, 0.0],
            [0.0, 0.1, 1.0, 0.2],
            [0.0, 0.0, 0.2, 1.0],
        ],
        dtype=float,
    ),
    random_state=42,
)

model = STARMA(
    ar_order=1,
    ma_order=1,
    include_intercept=False,
)
conditional = model.fit(series, weights)

incomplete = series.copy()
incomplete[20:30:2, 1] = np.nan
incomplete[80, :] = np.nan
incomplete[150:155, 3] = np.nan

filtered_complete = model.filter_state_space()
filtered_missing = model.filter_state_space(incomplete)

print(f"conditional likelihood: {conditional.log_likelihood:.3f}")
print(f"Kalman likelihood, complete: {filtered_complete.log_likelihood:.3f}")
print(f"Kalman likelihood, missing: {filtered_missing.log_likelihood:.3f}")
print(
    "observed cells:",
    filtered_missing.n_observations,
    "/",
    incomplete.size,
)
print("last filtered observation:")
print(filtered_missing.filtered_observations[-1])
