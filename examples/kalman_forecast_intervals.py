"""Generate transformed and original-scale Kalman forecast intervals."""

from __future__ import annotations

import numpy as np

from pystarmax import KalmanSTARIMA, SpatialWeights


def main() -> None:
    rng = np.random.default_rng(2026)
    increments = rng.normal(loc=0.15, scale=0.45, size=240)
    levels = np.cumsum(increments)[:, None]
    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )

    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=400,
    )
    model.fit(levels, weights)

    transformed = model.predict_differenced_interval(
        steps=12,
        level=0.95,
        n_simulations=5000,
        random_state=17,
    )
    original = model.predict_interval(
        steps=12,
        level=0.95,
        n_simulations=5000,
        random_state=17,
    )

    print("Highest-difference scale")
    print(np.column_stack((transformed.mean, transformed.lower, transformed.upper)))
    print("Original level scale")
    print(np.column_stack((original.mean, original.lower, original.upper)))
    print("Method:", original.method)
    print("Simulations:", original.n_simulations)


if __name__ == "__main__":
    main()
