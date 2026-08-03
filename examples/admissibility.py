"""Inspect STARMA stationarity and invertibility before and after fitting."""

from __future__ import annotations

import numpy as np

from pystarmax import (
    KalmanSTARMA,
    SpatialWeights,
    autoregressive_diagnostics,
    lattice_weights,
    moving_average_diagnostics,
    simulate_starma,
    starma_admissibility,
)


weights = SpatialWeights.from_adjacency(lattice_weights(2, 2), max_order=1)

ar_parameters = np.array([[0.42, 0.12]], dtype=float)
ma_parameters = np.array([[0.28, 0.08]], dtype=float)

ar_diagnostic = autoregressive_diagnostics(ar_parameters, weights, margin=1e-6)
ma_diagnostic = moving_average_diagnostics(ma_parameters, weights, margin=1e-6)
joint = starma_admissibility(ar_parameters, ma_parameters, weights)

print(ar_diagnostic.summary())
print()
print(ma_diagnostic.summary())
print()
print(joint.summary())

series = simulate_starma(
    phi=ar_parameters,
    theta=ma_parameters,
    weights=weights,
    n_steps=500,
    burnin=300,
    innovation_covariance=0.4,
    random_state=42,
)

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="scalar",
    include_intercept=False,
    enforce_stationarity=True,
    enforce_invertibility=True,
    max_iter=700,
)
result = model.fit(series, weights)
fitted_diagnostic = model.admissibility()

print()
print(result.summary())
print()
print(fitted_diagnostic.summary())
print("AR eigenvalues:", fitted_diagnostic.autoregressive.eigenvalues)
print("Inverse-MA eigenvalues:", fitted_diagnostic.moving_average.eigenvalues)
