"""Estimate a random walk with drift by exact diffuse maximum likelihood."""

from __future__ import annotations

import numpy as np

from pystarmax import ExactDiffuseKalmanSTARIMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    true_drift = 0.15
    true_scale = 0.45
    increments = true_drift + rng.normal(scale=true_scale, size=240)
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
        max_iter=400,
    )
    result = model.fit(levels, weights)

    print(result.summary())
    print("Estimated drift:", result.intercept)
    print("Estimated innovation variance:", result.innovation_covariance[0, 0])
    print("Diffuse observations:", result.n_diffuse_observations)
    print("Diffuse end time:", result.filter_result.diffuse_end_time)
    print("Original-level forecasts:")
    print(model.predict(steps=6))
    print("Highest-difference forecasts:")
    print(model.predict_differenced(steps=6))


if __name__ == "__main__":
    main()
