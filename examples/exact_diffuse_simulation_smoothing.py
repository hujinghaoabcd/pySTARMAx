"""Exact diffuse conditional simulation of complete latent state paths."""

from __future__ import annotations

import numpy as np

from pystarmax import ExactDiffuseKalmanSTARIMA, SpatialWeights

rng = np.random.default_rng(2026)
increments = 0.12 + rng.normal(scale=0.35, size=120)
levels = np.cumsum(increments)[:, None]
levels[45:50, 0] = np.nan

weights = SpatialWeights(
    matrices=(np.eye(1, dtype=float),),
    names=("W0",),
)
model = ExactDiffuseKalmanSTARIMA(
    ar_order=0,
    integration_order=1,
    ma_order=0,
    covariance_type="scalar",
    include_intercept=True,
    max_iter=300,
)
model.fit(levels, weights)

result = model.simulate_smoothing_paths(
    n_simulations=2000,
    random_state=2026,
)

print("simulations:", result.n_simulations)
print("diffuse rank:", result.diffuse_rank)
print("conditioning rank:", result.conditioning_rank)
print("maximum observation residual:", result.maximum_constraint_residual)
print("maximum mean discrepancy:", result.maximum_mean_discrepancy)
print("maximum covariance discrepancy:", result.maximum_covariance_discrepancy)
print("conditional means for missing rows:")
print(result.state_paths[:, 45:50, 0].mean(axis=0))
print("conditional standard deviations for missing rows:")
print(result.state_paths[:, 45:50, 0].std(axis=0))
