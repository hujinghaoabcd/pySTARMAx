"""Fit a multiplicative seasonal Kalman STARIMA model."""

from __future__ import annotations

import numpy as np

from pystarmax import SeasonalKalmanSTARIMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    seasonal_period = 12
    n_time = 300
    ordinary_ar = 0.35
    seasonal_ar = 0.45
    innovation_scale = 0.6

    transformed = np.zeros(n_time - seasonal_period, dtype=float)
    innovations = rng.normal(scale=innovation_scale, size=transformed.size)
    for time_index in range(seasonal_period, transformed.size):
        transformed[time_index] = (
            ordinary_ar * transformed[time_index - 1]
            + seasonal_ar * transformed[time_index - seasonal_period]
            - ordinary_ar * seasonal_ar * transformed[time_index - seasonal_period - 1]
            + innovations[time_index]
        )

    initial_cycle = 10.0 + rng.normal(scale=0.5, size=seasonal_period)
    levels = list(initial_cycle)
    for difference in transformed:
        levels.append(levels[-seasonal_period] + difference)
    observations = np.asarray(levels, dtype=float)[:, None]

    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )
    model = SeasonalKalmanSTARIMA(
        ar_order=1,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=seasonal_period,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=500,
    )
    result = model.fit(observations, weights)

    print(result.summary())
    print(model.admissibility().summary())
    print("Expanded AR lags:", result.ar_lags)
    print("Transformed forecast:")
    print(model.predict_differenced(steps=18)[:, 0])
    print("Original-scale forecast:")
    print(model.predict(steps=18)[:, 0])

    incomplete = observations[-72:].copy()
    incomplete[30:33, 0] = np.nan
    smoothed = model.smooth(incomplete)
    print("Smoothed transformed observations near the gap:")
    print(smoothed.smoothed_observations[16:28, 0])


if __name__ == "__main__":
    main()
