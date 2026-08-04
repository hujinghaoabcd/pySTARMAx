"""Exact diffuse original-level and differenced forecast intervals."""

from __future__ import annotations

import numpy as np

from pystarmax import ExactDiffuseKalmanSTARIMA, SpatialWeights


rng = np.random.default_rng(2026)
increments = 0.15 + rng.normal(scale=0.45, size=180)
levels = np.cumsum(increments)[:, None]

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
result = model.fit(levels, weights)

original = model.predict_interval(
    steps=8,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
differenced = model.predict_differenced_interval(
    steps=8,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)

print(result.summary())
print("Final diffuse rank:", result.filter_result.final_diffuse_rank)
print("\nOriginal-level forecast interval")
for horizon, (mean, lower, upper) in enumerate(
    zip(original.mean[:, 0], original.lower[:, 0], original.upper[:, 0]),
    start=1,
):
    print(f"h={horizon:2d}: mean={mean: .4f}, [{lower: .4f}, {upper: .4f}]")

print("\nHighest-difference forecast interval")
for horizon, (mean, lower, upper) in enumerate(
    zip(
        differenced.mean[:, 0],
        differenced.lower[:, 0],
        differenced.upper[:, 0],
    ),
    start=1,
):
    print(f"h={horizon:2d}: mean={mean: .4f}, [{lower: .4f}, {upper: .4f}]")
