"""Fit a Kalman STARMA model and smooth a missing observation block."""

from __future__ import annotations

import numpy as np

from pystarmax import KalmanSTARMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    n_time = 240
    phi = 0.65
    series = np.empty((n_time, 1), dtype=float)
    series[0, 0] = rng.normal(scale=0.8)
    for time_index in range(1, n_time):
        series[time_index, 0] = (
            phi * series[time_index - 1, 0] + rng.normal(scale=0.8)
        )

    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )
    model = KalmanSTARMA(
        ar_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=500,
    )
    model.fit(series, weights)

    incomplete = series.copy()
    incomplete[80:90, 0] = np.nan
    smoothed = model.smooth(incomplete)

    print("Smoothed missing block:")
    print(smoothed.smoothed_observations[80:90, 0])
    print("Posterior standard deviations:")
    print(np.sqrt(smoothed.smoothed_observation_covariance[80:90, 0, 0]))
    print("State disturbance means:")
    print(smoothed.state_disturbance_mean[79:90, 0])
    print("Pseudoinverse transitions:", int(smoothed.used_pseudoinverse.sum()))


if __name__ == "__main__":
    main()
