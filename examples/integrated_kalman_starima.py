"""Fit a conditional Kalman STARIMA model and forecast both scales."""

from __future__ import annotations

import numpy as np

from pystarmax import KalmanSTARIMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    n_time = 240
    drift = 0.18
    phi = 0.45
    innovation_scale = 0.65

    differences = np.zeros(n_time - 1, dtype=float)
    differences[0] = rng.normal(loc=drift, scale=innovation_scale)
    for time_index in range(1, differences.size):
        differences[time_index] = (
            drift
            + phi * differences[time_index - 1]
            + rng.normal(scale=innovation_scale)
        )
    levels = np.concatenate([[10.0], 10.0 + np.cumsum(differences)])[:, None]

    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )
    model = KalmanSTARIMA(
        ar_order=1,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
    )
    result = model.fit(levels, weights)

    print(result.summary())
    print("Differenced-scale forecast:")
    print(model.predict_differenced(steps=8)[:, 0])
    print("Original-scale forecast:")
    print(model.predict(steps=8)[:, 0])

    incomplete = levels[-48:].copy()
    incomplete[18:21, 0] = np.nan
    filtered = model.filter(incomplete)
    smoothed = model.smooth(incomplete)
    print("Transformed observed-mask count:", filtered.n_observations)
    print("Smoothed transformed observations around the gap:")
    print(smoothed.smoothed_observations[15:23, 0])


if __name__ == "__main__":
    main()
