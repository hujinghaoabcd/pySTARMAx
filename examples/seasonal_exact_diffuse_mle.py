from __future__ import annotations

import numpy as np

from pystarmax import SeasonalExactDiffuseKalmanSTARIMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2027)
    period = 4
    transformed = np.zeros(180, dtype=float)
    innovations = rng.normal(scale=0.3, size=transformed.size)
    for time_index in range(period + 1, transformed.size):
        transformed[time_index] = (
            0.25 * transformed[time_index - 1]
            + 0.18 * transformed[time_index - period]
            - 0.25 * 0.18 * transformed[time_index - period - 1]
            + innovations[time_index]
        )

    history = list(np.linspace(8.0, 9.0, period))
    for value in transformed:
        history.append(history[-period] + float(value))
    observations = np.asarray(history, dtype=float)[:, None]
    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )

    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=1,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=period,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=400,
    )
    result = model.fit(
        observations,
        weights,
        start_params=np.array([0.15, 0.1]),
        start_covariance=np.array([[0.15]]),
    )

    print(result.summary())
    print("Expanded AR lags:", result.ar_lags)
    print("Original-level forecast:")
    print(model.predict(steps=8))
    print("Highest-difference forecast:")
    print(model.predict_differenced(steps=8))


if __name__ == "__main__":
    main()
