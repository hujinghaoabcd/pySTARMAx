"""Recover original STARMA innovations from fixed-interval state moments."""

from __future__ import annotations

import numpy as np

from pystarmax import KalmanSTARMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    n_time = 220
    phi = 0.55
    theta = 0.30
    standard_deviation = 0.7
    innovations = rng.normal(scale=standard_deviation, size=n_time)
    series = np.zeros((n_time, 1), dtype=float)
    for time_index in range(1, n_time):
        series[time_index, 0] = (
            phi * series[time_index - 1, 0]
            + innovations[time_index]
            + theta * innovations[time_index - 1]
        )

    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )
    model = KalmanSTARMA(
        ar_order=1,
        ma_order=1,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=500,
    )
    model.fit(series, weights)

    incomplete = series.copy()
    incomplete[80:90, 0] = np.nan
    result = model.smooth_innovation_disturbances(incomplete)

    posterior_standard_deviation = np.sqrt(
        result.innovation_covariance[:, 0, 0]
    )
    print("Innovation posterior means around the missing block:")
    print(result.innovation_mean[76:94, 0])
    print("Innovation posterior standard deviations:")
    print(posterior_standard_deviation[76:94])
    print("Process rank:", result.process_rank)
    print("Used pseudoinverse:", result.used_pseudoinverse)
    print("Unresolved variance:", result.unresolved_variance)
    print("Largest mean support residual:", result.mean_support_residual.max())


if __name__ == "__main__":
    main()
