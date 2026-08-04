"""Seasonal exact diffuse state and innovation smoothing example."""

from __future__ import annotations

import numpy as np

from pystarmax import SeasonalExactDiffuseKalmanSTARIMA, SpatialWeights


def main() -> None:
    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )
    observations = np.array(
        [
            [0.0],
            [10.0],
            [2.0],
            [12.0],
            [np.nan],
            [np.nan],
            [7.0],
            [17.0],
            [9.0],
            [20.0],
        ]
    )

    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=300,
    )
    result = model.fit(
        observations,
        weights,
        start_covariance=np.array([[0.5]]),
    )
    smoothed = model.smooth()
    disturbances = model.smooth_innovation_disturbances()

    print(result.summary())
    print("smoothed observations:")
    print(smoothed.smoothed_observations[:, 0])
    print("primitive innovation means:")
    print(disturbances.innovation_mean[:, 0])
    print("primitive innovation variances:")
    print(disturbances.innovation_covariance[:, 0, 0])
    print("maximum filter reconstruction error:")
    print(smoothed.maximum_filter_reconstruction_error)


if __name__ == "__main__":
    main()
