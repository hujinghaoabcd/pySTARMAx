"""Smooth missing levels with exact diffuse initialization."""

from __future__ import annotations

import numpy as np

from pystarmax import ExactDiffuseKalmanSTARIMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    increments = 0.12 + rng.normal(scale=0.35, size=180)
    levels = np.cumsum(increments)[:, None]
    incomplete = levels.copy()
    incomplete[40:45, 0] = np.nan
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
    model.fit(incomplete, weights)
    smoothed = model.smooth()

    print("Diffuse end time:", smoothed.diffuse_end_time)
    print("Final diffuse rank:", smoothed.final_diffuse_rank)
    print(
        "Maximum filter reconstruction error:",
        smoothed.maximum_filter_reconstruction_error,
    )
    print(
        "Maximum covariance correction:",
        smoothed.maximum_covariance_correction,
    )
    print("Smoothed missing levels:")
    print(smoothed.smoothed_observations[40:45])
    print("Smoothed missing-level variances:")
    print(smoothed.smoothed_observation_covariance[40:45, 0, 0])


if __name__ == "__main__":
    main()
