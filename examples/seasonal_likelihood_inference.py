"""Estimate seasonal factors and inspect observed-information uncertainty."""

from __future__ import annotations

import numpy as np

from pystarmax import SeasonalKalmanSTARIMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    period = 4
    n_time = 260
    ordinary_ar = 0.25
    seasonal_ar = 0.40
    innovation_scale = 0.55
    series = np.zeros(n_time, dtype=float)
    innovations = rng.normal(scale=innovation_scale, size=n_time)
    for time_index in range(period + 1, n_time):
        series[time_index] = (
            ordinary_ar * series[time_index - 1]
            + seasonal_ar * series[time_index - period]
            - ordinary_ar * seasonal_ar * series[time_index - period - 1]
            + innovations[time_index]
        )

    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )
    model = SeasonalKalmanSTARIMA(
        ar_order=1,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=period,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=500,
    )
    model.fit(series[:, None], weights)
    inference = model.infer(relative_step=2e-4, absolute_step=1e-6)
    natural = inference.innovation_covariance_inference()

    print(inference.coefficient_table)
    print(inference.optimizer_table)
    print(inference.confidence_intervals(level=0.95))
    print("Hessian eigenvalues:", inference.eigenvalues)
    print("Condition number:", inference.condition_number)
    print(
        "Minimum admissibility distance:",
        inference.minimum_admissibility_distance,
    )
    print(natural.table)
    print("Factor/covariance cross covariance:")
    print(natural.dynamic_cross_covariance)


if __name__ == "__main__":
    main()
